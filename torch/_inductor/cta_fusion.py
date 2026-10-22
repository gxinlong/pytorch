from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import time
from collections import deque
from itertools import zip_longest
from pathlib import Path
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Min

from . import config, ir
from .dependencies import WeakDep
from .runtime.runtime_utils import next_power_of_2
from .utils import sympy_product
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .scheduler import BaseSchedulerNode, Scheduler


log = logging.getLogger(__name__)
GEMM_ANNOTATION = "cta_fusion_gemm"


class UnsupportedCtaFusion(Exception):
    """A region cannot be implemented with the supported CTA schedules."""


@dataclasses.dataclass(frozen=True)
class GemmDescriptor:
    lhs: ir.IRNode
    rhs: ir.IRNode
    dtype: torch.dtype
    allow_tf32: bool

    @property
    def batched(self) -> bool:
        return len(self.lhs.get_size()) == 3


def annotate_gemm(result: ir.TensorBox, lhs: ir.IRNode, rhs: ir.IRNode) -> None:
    if config.cta_fusion.enabled:
        m, k = lhs.get_size()[-2:]
        n = rhs.get_size()[-1]
        # Match the standalone MM template's shape-dependent precision policy.
        allow_tf32 = (
            torch.backends.cuda.matmul.fp32_precision == "tf32"
            and V.graph.sizevars.statically_known_true(
                sympy.And(sympy.Ge(m, 16), sympy.Ge(Min(n, k), 512))
            )
        )
        result.data.data.annotations[GEMM_ANNOTATION] = GemmDescriptor(
            lhs, rhs, result.get_dtype(), allow_tf32
        )


@dataclasses.dataclass(frozen=True)
class CtaConfig:
    rows: int
    block_k: int
    num_warps: int
    num_stages: int
    use_dot: bool


@dataclasses.dataclass(frozen=True)
class CtaStage:
    node: BaseSchedulerNode
    output: ir.Buffer
    gemm: GemmDescriptor | None
    elements_per_unit: int
    tile_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class CtaFusionPlan:
    stages: tuple[CtaStage, ...]
    units: sympy.Expr
    batched: bool
    outputs: frozenset[str]
    original_groups: tuple[BaseSchedulerNode, ...]

    def configs(self) -> Sequence[CtaConfig]:
        rows = (1,) if self.batched or self.units == 1 else (1, 4, 16)
        choices = [
            CtaConfig(r, k, w, s, dot)
            for dot in (False, True)
            for r in rows
            for k, w, s in ((32, 4, 1), (64, 4, 1), (32, 8, 2))
            if dot or r <= 4
        ]
        # Interleave implementations so a short budget still considers both.
        choices.sort(key=lambda c: (c.block_k, c.num_warps, c.rows, c.use_dot))
        return choices[: config.cta_fusion.max_configs_per_pair]


@dataclasses.dataclass(frozen=True)
class Timing:
    elapsed_ms: float
    iterations: int

    @property
    def avg_us(self) -> float:
        if self.iterations <= 0:
            return math.inf
        return self.elapsed_ms * 1000 / self.iterations

    @property
    def valid(self) -> bool:
        return math.isfinite(self.avg_us) and self.avg_us > 0


class TimingCache:
    protocol = "cuda_graph_mean_v2_100_calls_5_warmups_5_interleaved_rounds"

    def __init__(self, key: str) -> None:
        from .runtime.runtime_utils import cache_dir

        self.path = Path(cache_dir()) / "cta_fusion" / f"{key}.json"

    def read(self):
        if config.force_disable_caches:
            return None
        try:
            data = json.loads(self.path.read_text())
            if data["protocol"] != self.protocol:
                return None
            samples = tuple(
                [Timing(**sample) for sample in data[name]]
                for name in ("baseline", "fused")
            )
            if all(
                len(values) == 5 and all(t.valid for t in values) for values in samples
            ):
                return samples
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def write(self, baseline, fused):
        from .codecache import write_atomic

        if config.force_disable_caches or not all(t.valid for t in (*baseline, *fused)):
            return
        write_atomic(
            str(self.path),
            json.dumps(
                {
                    "protocol": self.protocol,
                    "baseline": [dataclasses.asdict(t) for t in baseline],
                    "fused": [dataclasses.asdict(t) for t in fused],
                }
            ),
            make_dirs=True,
        )


def profitable(baseline: Sequence[Timing], fused: Sequence[Timing]) -> bool:
    if len(baseline) != 5 or len(fused) != 5:
        return False
    if not all(t.valid for t in (*baseline, *fused)):
        return False
    baseline_us = sum(t.elapsed_ms for t in baseline) / sum(
        t.iterations for t in baseline
    )
    fused_us = sum(t.elapsed_ms for t in fused) / sum(t.iterations for t in fused)
    return (
        baseline_us > fused_us
        and baseline_us / fused_us >= config.cta_fusion.min_speedup
        and sum(a.avg_us > b.avg_us for a, b in zip(baseline, fused)) >= 4
    )


def measure(call: Callable[[], None], iterations: int = 100) -> Timing:
    from .runtime.benchmarking import maybe_gpu_benchmark_lock

    with maybe_gpu_benchmark_lock():
        reset_inputs = getattr(call, "reset", lambda: None)
        for _ in range(5):
            reset_inputs()
            call()
        reset_inputs()
        calls, reset = (
            call.prepare(iterations)
            if hasattr(call, "prepare")
            else ([call] * iterations, lambda: None)
        )
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for replay_call in calls:
                    replay_call()
            reset()
            graph.replay()
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            return Timing(start.elapsed_time(end), iterations)
        finally:
            reset_inputs()


def report(nodes: Sequence[BaseSchedulerNode], reason: str, **data: Any) -> None:
    payload = {"nodes": [n.get_name() for n in nodes], "reason": reason, **data}
    log.debug("CTA fusion: %s", payload)
    V.debug.cta_fusion(payload)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": "cta_fusion", "encoding": "json"},
        payload_fn=lambda: json.dumps(payload),
    )


def static_int(value: sympy.Expr | int) -> int:
    value = V.graph.sizevars.simplify(value)
    if not isinstance(value, (int, sympy.Integer)):
        raise UnsupportedCtaFusion(f"unbounded tile dimension: {value}")
    return int(value)


def descriptor(node: BaseSchedulerNode) -> GemmDescriptor | None:
    return node.node.annotations.get(GEMM_ANNOTATION) if node.node is not None else None


def make_plan(
    scheduler: Scheduler,
    first: BaseSchedulerNode,
    second: BaseSchedulerNode,
    *,
    single_cta: bool = False,
) -> CtaFusionPlan:
    from .scheduler import FusedCtaSchedulerNode, GroupedSchedulerNode, SchedulerNode

    if first is second or first.get_device() != second.get_device():
        raise UnsupportedCtaFusion("different devices or identical nodes")
    if isinstance(first, GroupedSchedulerNode) or isinstance(
        second, GroupedSchedulerNode
    ):
        raise UnsupportedCtaFusion("explicit scheduler group")
    device = first.get_device()
    if device is None or device.type != "cuda" or torch.version.hip:
        raise UnsupportedCtaFusion("requires CUDA")
    if scheduler.get_node_stream(first) != scheduler.get_node_stream(second):
        raise UnsupportedCtaFusion("different streams")
    if scheduler.node_to_mempool.get(first) != scheduler.node_to_mempool.get(second):
        raise UnsupportedCtaFusion("different memory pools")
    if first.has_aliasing_or_mutation() or second.has_aliasing_or_mutation():
        raise UnsupportedCtaFusion("aliasing or mutation")
    if first.has_strict_sum() or second.has_strict_sum():
        raise UnsupportedCtaFusion("strict sum")
    nodes = tuple(first.get_nodes()) + tuple(second.get_nodes())
    if any(n.has_side_effects() for n in nodes):
        raise UnsupportedCtaFusion("side effects")
    if len(nodes) > config.max_fusion_size:
        raise UnsupportedCtaFusion("max_fusion_size")
    names = first.get_operation_names() | second.get_operation_names()
    buffers = first.get_buffer_names() | second.get_buffer_names()
    if buffers & V.graph.no_fuse_buffer_names:
        raise UnsupportedCtaFusion("explicit no_fuse")
    if not first.get_buffer_names() & OrderedSet(
        d.name for d in second.read_writes.reads if not isinstance(d, WeakDep)
    ):
        raise UnsupportedCtaFusion("not a producer-consumer pair")
    if second.get_operation_names() & first.ancestors:
        raise UnsupportedCtaFusion("reversed dependency")
    # Contracting a non-convex region would skip an intermediate producer.
    for n in nodes:
        for dep in n.unmet_dependencies:
            if dep.name in buffers:
                continue
            producer = scheduler.name_to_buf.get(dep.name)
            if producer is not None and producer.defining_op is not None:
                producer_node = scheduler.get_fused_node(producer.defining_op)
                if names & producer_node.ancestors:
                    raise UnsupportedCtaFusion("external intermediate dependency")
    if scheduler.will_fusion_create_cycle(first, second):
        raise UnsupportedCtaFusion("cycle")
    gemms = [g for n in nodes if (g := descriptor(n)) is not None]
    if not gemms:
        raise UnsupportedCtaFusion("no GEMM stage")
    batched = any(g.batched for g in gemms)
    if single_cta and batched:
        raise UnsupportedCtaFusion("single CTA cannot combine batch domains")
    anchor = next(g for g in gemms if g.batched) if batched else gemms[0]
    units = sympy.S.One if single_cta else anchor.lhs.get_size()[0]
    stages = []
    for node in nodes:
        gemm = descriptor(node)
        outputs = node.get_outputs()
        if len(outputs) != 1:
            raise UnsupportedCtaFusion("multiple-output operation")
        output = outputs[0].node
        if gemm is not None:
            if output.get_dtype() not in (torch.float32, torch.float16, torch.bfloat16):
                raise UnsupportedCtaFusion("GEMM dtype")
            if gemm.lhs.get_dtype() != gemm.rhs.get_dtype():
                raise UnsupportedCtaFusion("mixed GEMM input dtypes")
            if output.get_dtype() != gemm.lhs.get_dtype():
                raise UnsupportedCtaFusion("GEMM out_dtype")
            if gemm.batched and not V.graph.sizevars.statically_known_equals(
                gemm.lhs.get_size()[0], units
            ):
                raise UnsupportedCtaFusion("incompatible batch domains")
        elif not (
            isinstance(node, SchedulerNode)
            and isinstance(output, ir.ComputedBuffer)
            and type(output.data)
            in (ir.Pointwise, ir.Reduction, ir.MultiOutputReduction)
            and output.get_reduction_type()
            in (None, "sum", "max", "min", "any", "all", "online_softmax_reduce")
        ):
            raise UnsupportedCtaFusion(f"unsupported stage: {type(output).__name__}")
        count = static_int(sympy_product(output.get_size()) / units)
        if (
            count <= 0
            or next_power_of_2(count) > config.cta_fusion.max_intermediate_tile_elements
        ):
            raise UnsupportedCtaFusion("intermediate tile size")
        sizes = output.get_size()
        try:
            tile_shape = (
                (static_int(sizes[0] / units), *map(static_int, sizes[1:]))
                if sizes
                else (count,)
            )
        except UnsupportedCtaFusion:
            tile_shape = (count,)
        if any(
            next_power_of_2(s) > config.cta_fusion.max_intermediate_dim
            for s in tile_shape
        ):
            raise UnsupportedCtaFusion("intermediate dimension")
        stages.append(CtaStage(node, output, gemm, count, tile_shape))
    external_outputs = frozenset(
        out.get_name()
        for n in nodes
        for out in n.get_outputs()
        if any(user.node.get_name() not in names for user in out.users)
    )
    groups = tuple(
        group
        for n in (first, second)
        for group in (
            n.plan.original_groups if isinstance(n, FusedCtaSchedulerNode) else (n,)
        )
    )
    return CtaFusionPlan(tuple(stages), units, batched, external_outputs, groups)


def fuse_small_kernels(scheduler: Scheduler) -> None:
    from .codegen.triton_cta import CtaBenchmark, CtaKernel
    from .scheduler import BaseSchedulerNode, FusedCtaSchedulerNode

    remaining = OrderedSet(scheduler.nodes)
    measured: dict[BaseSchedulerNode, Timing] = {}
    seen: set[tuple[BaseSchedulerNode, BaseSchedulerNode]] = set()
    queue: deque[tuple[BaseSchedulerNode, BaseSchedulerNode]] = deque()

    def enqueue(node: BaseSchedulerNode) -> None:
        pairs = []
        for dep in node.unmet_dependencies:
            producer = scheduler.name_to_buf.get(dep.name)
            if producer is not None and producer.defining_op is not None:
                pairs.append((scheduler.get_fused_node(producer.defining_op), node))
        for output in node.get_outputs():
            for user in output.users:
                if isinstance(user.node, BaseSchedulerNode):
                    pairs.append((node, scheduler.get_fused_node(user.node)))
        for a, b in pairs:
            if (
                a is not b
                and a in remaining
                and b in remaining
                and (a, b) not in seen
                and a.get_buffer_names()
                & OrderedSet(
                    d.name for d in b.read_writes.reads if not isinstance(d, WeakDep)
                )
            ):
                seen.add((a, b))
                queue.append((a, b))

    for node in remaining:
        enqueue(node)
    while queue:
        first, second = queue.popleft()
        if first not in remaining or second not in remaining:
            continue
        stack = contextlib.ExitStack()
        try:
            plans = []
            for single_cta in (False, True):
                try:
                    trial = make_plan(scheduler, first, second, single_cta=single_cta)
                    if not plans or trial.units != plans[0].units:
                        plans.append(trial)
                except UnsupportedCtaFusion as exc:
                    report((first, second), str(exc), single_cta=single_cta)
            if not plans:
                continue
            stack.enter_context(torch.cuda.device(first.get_device()))
            stack.enter_context(V.graph.set_current_device(first.get_device()))
            candidates = []
            schedules = [
                [(plan, candidate) for candidate in plan.configs()] for plan in plans
            ]
            for alternatives in zip_longest(*schedules):
                for alternative in alternatives:
                    if alternative is None:
                        continue
                    candidate_plan, candidate = alternative
                    try:
                        with V.graph.set_current_device(first.get_device()):
                            candidates.append(CtaKernel(candidate_plan, candidate))
                    except UnsupportedCtaFusion as exc:
                        report(
                            (first, second),
                            str(exc),
                            config=dataclasses.asdict(candidate),
                        )
                    if len(candidates) >= config.cta_fusion.max_configs_per_pair:
                        break
                if len(candidates) >= config.cta_fusion.max_configs_per_pair:
                    break
            if not candidates:
                continue
            plan = plans[0]
            with V.graph.set_current_device(first.get_device()):
                benchmark = CtaBenchmark(scheduler, (first, second), plan)
            for group in plan.original_groups:
                if group not in measured:
                    measured[group] = measure(benchmark.group_call(group))
                timing = measured[group]
                report((group,), "eligibility", timing=dataclasses.asdict(timing))
                if (
                    not timing.valid
                    or timing.avg_us >= config.cta_fusion.max_kernel_duration_us
                ):
                    raise UnsupportedCtaFusion("kernel duration")
            best = None
            best_time = math.inf
            best_ratio = 0.0
            for kernel in candidates:
                try:
                    with V.graph.set_current_device(first.get_device()):
                        started = time.perf_counter()
                        call = benchmark.compile(kernel)
                        compile_seconds = time.perf_counter() - started
                    cache = TimingCache(benchmark.cache_key(kernel))
                    cached = cache.read()
                    if cached is None:
                        baseline_times, fused_times = [], []
                        for i in range(5):
                            if i % 2:
                                fused_times.append(measure(call))
                                baseline_times.append(measure(benchmark.call))
                            else:
                                baseline_times.append(measure(benchmark.call))
                                fused_times.append(measure(call))
                        cache.write(baseline_times, fused_times)
                    else:
                        baseline_times, fused_times = cached
                    elapsed = sum(t.avg_us for t in fused_times)
                    if profitable(baseline_times, fused_times) and elapsed < best_time:
                        best, best_time = kernel, elapsed
                        best_ratio = sum(t.avg_us for t in baseline_times) / elapsed
                    report(
                        (first, second),
                        "candidate",
                        config=dataclasses.asdict(kernel.candidate),
                        units=str(kernel.plan.units),
                        baseline=[dataclasses.asdict(t) for t in baseline_times],
                        fused=[dataclasses.asdict(t) for t in fused_times],
                        resources=kernel.resources,
                        peak_live_elements=kernel.peak_live_elements,
                        compile_seconds=compile_seconds,
                        cached=cached is not None,
                        cache_key=cache.path.stem,
                    )
                except Exception as exc:
                    report(
                        (first, second),
                        "candidate failed",
                        error=str(exc),
                        config=dataclasses.asdict(kernel.candidate),
                        resources=kernel.resources,
                    )
            if best is None:
                report((first, second), "no profitable candidate")
                continue
            plan = make_plan(
                scheduler,
                first,
                second,
                single_cta=not best.plan.batched and best.plan.units == 1,
            )
            if plan.outputs != best.plan.outputs:
                raise UnsupportedCtaFusion("region outputs changed during search")
            fused = FusedCtaSchedulerNode(scheduler, plan, best)
            remaining.remove(first)
            remaining.remove(second)
            remaining.add(fused)
            scheduler.name_to_fused_node.update(
                {n.get_name(): fused for n in fused.get_nodes()}
            )
            scheduler.node_to_stream[fused] = scheduler.get_node_stream(first)
            scheduler.node_to_mempool[fused] = scheduler.node_to_mempool.get(first)
            enqueue(fused)
            report(
                (first, second),
                "accepted",
                avg_us=best_time / 5,
                speedup=best_ratio,
                config=dataclasses.asdict(best.candidate),
                gemm_stages=sum(stage.gemm is not None for stage in plan.stages),
            )
        except Exception as exc:
            report((first, second), str(exc))
        finally:
            stack.close()
    scheduler.nodes = scheduler.topological_sort_schedule(list(remaining))
