from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .codegen.triton_cta import (
    BenchmarkCall,
    CtaBenchmark,
    CtaKernel,
    preserve_codegen_state,
)
from .codegen.triton_region_combo import (
    RegionComboKernel,
    RegionKernel,
    UnsupportedRegionCombo,
)
from .cta_fusion import measure, Timing
from .dependencies import MemoryDep
from .runtime.benchmarking import gpu_benchmark_lock
from .select_algorithm import (
    ExternKernelCaller,
    TritonTemplateCaller,
    TritonTemplateKernel,
)
from .virtualized import V


if TYPE_CHECKING:
    from .scheduler import BaseSchedulerNode


log = logging.getLogger(__name__)


def report(nodes, reason, **data):
    payload = {"nodes": [n.get_name() for n in nodes], "reason": reason, **data}
    log.debug("Region combo: %s", payload)
    V.debug.region_combo(payload)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": "region_combo", "encoding": "json"},
        payload_fn=lambda: json.dumps(payload),
    )


def profitable(baseline, combined):
    if len(baseline) < 20 or len(baseline) != len(combined):
        return False
    if not all(t.valid for t in (*baseline, *combined)):
        return False
    differences = [a.avg_us - b.avg_us for a, b in zip(baseline, combined)]
    lower = statistics.mean(differences) - 2.1 * statistics.stdev(
        differences
    ) / math.sqrt(len(differences))
    return (
        lower > 0
        and statistics.median(a.avg_us / b.avg_us for a, b in zip(baseline, combined))
        >= config.region_combo.min_speedup
    )


@gpu_benchmark_lock
def measure_ordinary(call, iterations):
    for _ in range(5):
        call.reset()
        call()
    call.reset()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return Timing(start.elapsed_time(end), iterations)


@dataclass(frozen=True)
class Resources:
    registers: int
    shared: int
    spills: int
    active_threads: int


def resources(autotuner, device, warps):
    if len(autotuner.launchers) != 1 or len(autotuner.compile_results) != 1:
        raise UnsupportedRegionCombo("multiple launchers")
    launcher = autotuner.launchers[0]
    registers, shared, spills = launcher.n_regs, launcher.shared, launcher.n_spills
    if any(value is None for value in (registers, shared, spills)):
        raise UnsupportedRegionCombo("missing compiled resources")
    if (
        spills
        or shared
        > torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    ):
        raise UnsupportedRegionCombo("compiled spill or shared-memory limit")
    from cuda.bindings import driver  # pyrefly: ignore[missing-import]

    error, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        autotuner.compile_results[0].kernel.function, warps * 32, shared
    )
    if int(error) != 0 or blocks <= 0:
        raise UnsupportedRegionCombo("occupancy query failed")
    return Resources(registers, shared, spills, blocks * warps * 32)


def check_resources(members, combined):
    if any(
        combined.active_threads
        < r.active_threads * (1 - config.region_combo.max_occupancy_drop)
        for r in members
    ):
        raise UnsupportedRegionCombo("occupancy drop")


@dataclass(frozen=True)
class PreparedRegion:
    region: RegionKernel
    choice: TritonTemplateCaller | None
    call: BenchmarkCall
    resources: Resources


def resource_clusters(candidates, device, launch):
    threads = launch[0] * 32
    max_threads = max((c.resources.active_threads for c in candidates), default=0)

    def profile(candidate):
        resource = candidate.resources
        # Shared-memory residency is an upper bound; the compiled combo is checked too.
        shared_threads = (
            device.shared_memory_per_multiprocessor // max(resource.shared, 1) * threads
        )
        return (
            resource.registers,
            min(max_threads, shared_threads),
            resource.active_threads,
        )

    def distance(low, high):
        return max(
            high[0] / max(low[0], 1) / config.region_combo.max_register_ratio,
            *(
                high[i] * (1 - config.region_combo.max_occupancy_drop) / max(low[i], 1)
                for i in (1, 2)
            ),
        )

    clusters = []
    for candidate in sorted(
        candidates,
        key=lambda c: (profile(c), c.region.node.get_name(), c.region.choice),
    ):
        values = profile(candidate)
        matches = []
        for i, (_, low, high) in enumerate(clusters):
            # Bound the full range so a chain of similar neighbors cannot hide an outlier.
            lower = tuple(min(a, b) for a, b in zip(low, values))
            upper = tuple(max(a, b) for a, b in zip(high, values))
            score = distance(lower, upper)
            if score <= 1:
                matches.append((score, i, lower, upper))
        if matches:
            _, index, low, high = min(matches)
            members, _, _ = clusters[index]
            members.append(candidate)
            clusters[index] = members, low, high
        else:
            clusters.append(([candidate], values, values))
    return [members for members, _, _ in clusters]


def check_group(scheduler, nodes):
    first = nodes[0]
    device = first.get_device()
    if device is None or device.type != "cuda" or torch.version.hip:
        raise UnsupportedRegionCombo("requires NVIDIA CUDA")
    operations, ancestors = OrderedSet(), OrderedSet()
    for node in nodes:
        if node.get_device() != device:
            raise UnsupportedRegionCombo("different devices")
        if scheduler.get_node_stream(node) != scheduler.get_node_stream(first):
            raise UnsupportedRegionCombo("different streams")
        if scheduler.node_to_mempool.get(node) != scheduler.node_to_mempool.get(first):
            raise UnsupportedRegionCombo("different memory pools")
        if node.has_aliasing_or_mutation() or node.has_strict_sum():
            raise UnsupportedRegionCombo("aliasing, mutation or strict sum")
        names = node.get_operation_names()
        if names & ancestors or operations & node.ancestors:
            raise UnsupportedRegionCombo("dependent regions")
        operations.update(names)
        ancestors.update(node.ancestors)
        if any(
            isinstance(dep, MemoryDep) and dep.is_indirect()
            for dep in node.read_writes.reads_and_writes()
        ):
            raise UnsupportedRegionCombo("indirect benchmark inputs")


@contextlib.contextmanager
def speculative_codegen(nodes):
    buffers = list(V.graph.name_to_buffer.values())
    layouts = [(buf, buf.layout) for buf in buffers if isinstance(buf, ir.Buffer)]
    constraints = V.graph.buffer_layout_constraints.copy()
    with preserve_codegen_state([n for node in nodes for n in node.get_nodes()]):
        try:
            yield
            if any(buf.layout != layout for buf, layout in layouts):
                raise UnsupportedRegionCombo("speculative layout change")
            if V.graph.buffer_layout_constraints != constraints:
                raise UnsupportedRegionCombo("new speculative layout constraints")
        finally:
            for buf, layout in layouts:
                buf.layout = layout
            V.graph.buffer_layout_constraints = constraints


def template_parts(node):
    return node.get_prologue_template_epilogue(list(node.get_nodes()))


def choices_for(scheduler, node):
    from .scheduler import FusedCtaSchedulerNode

    if isinstance(node, FusedCtaSchedulerNode):
        return [None]
    _, template, _ = template_parts(node)
    buf = template.node
    if not isinstance(buf, ir.TritonTemplateBuffer):
        raise UnsupportedRegionCombo("non-Triton template")
    if isinstance(buf, ir.MultiTemplateBuffer):
        if (
            config.test_configs.force_extern_kernel_in_multi_template
            or scheduler._has_layout_conflict_for_template(buf)
        ):
            raise UnsupportedRegionCombo("extern required by layout or configuration")
        if buf._make_kernel_renders or config.multi_kernel_hints:
            raise UnsupportedRegionCombo("runtime multi-kernel dispatch")
        if buf._render_caller is not None:
            return [buf._render_caller]
        timings = buf.choice_timings()
        choices = sorted(
            (
                c
                for c in buf.choices
                if isinstance(c, TritonTemplateCaller)
                and math.isfinite(timings.get(c, math.inf))
            ),
            key=timings.__getitem__,
        )
        counts, retained = {}, []
        for choice in choices:
            key = (choice.bmreq.num_warps, choice.bmreq.num_stages)
            if counts.get(key, 0) < config.region_combo.max_choices_per_member:
                retained.append(choice)
                counts[key] = counts.get(key, 0) + 1
        return retained
    return [None]


def render_region(backend, node, choice, *, combo_mode, launch=None):
    from .scheduler import FusedCtaSchedulerNode

    if isinstance(node, FusedCtaSchedulerNode):
        candidate = node.kernel.candidate
        if launch is not None:
            candidate = replace(candidate, num_warps=launch[0], num_stages=launch[1])
        kernel = CtaKernel(node.plan, candidate, combo_mode=combo_mode)
        # Internal CTA outputs have no global storage in either execution mode.
        kernel.removed_buffers.update(
            out.get_name()
            for out in node.get_outputs()
            if out.get_name() not in node.plan.outputs
        )
        return RegionKernel(node, kernel, kernel.source, repr(node.kernel.candidate))
    prologue, template, epilogue = template_parts(node)
    buf = template.node
    factory = (
        choice.get_make_kernel_render()
        if choice is not None
        else buf.make_kernel_render
    )
    if factory is None:
        raise UnsupportedRegionCombo("no template renderer")
    kernel, render = factory(buf)
    if combo_mode and not kernel.combo_supported:
        raise UnsupportedRegionCombo("template has no combo renderer")
    if type(kernel) is not TritonTemplateKernel:
        raise UnsupportedRegionCombo("custom template launch contract")
    kernel.combo_mode, kernel.use_jit = combo_mode, combo_mode
    if launch is not None:
        kernel.num_warps, kernel.num_stages = launch
    source = backend._codegen_single_template(
        kernel, render, template, epilogue, prologue, only_gen_src_code=True
    )
    return RegionKernel(
        node, kernel, source, choice.hash_key() if choice is not None else "bound"
    )


class RegionBenchmark(CtaBenchmark):
    """Cache member preparation and baseline calls within one independent region group."""

    def __init__(self, scheduler, nodes):
        self.scheduler, self.nodes = scheduler, nodes
        self.tensors = {}
        self.calls = {}
        self.baseline_choices = {}
        self.modules = []
        self.regions = {}
        self.compiled = {}
        self.prepared = {}
        self.sequences = {}
        self.counts = dict(
            renders=0, member_compiles=0, resource_queries=0, combo_compiles=0
        )
        self.backend = scheduler.get_backend(nodes[0].get_device())._triton_scheduling
        for node in nodes:
            self.group_call(node)
        self.call = self.sequence(nodes)

    def sequence(self, nodes):
        key = tuple(nodes)
        if key not in self.sequences:
            self.sequences[key] = BenchmarkCall(
                [factory for node in nodes for factory in self.calls[node].factories],
                self.tensors,
            )
        return self.sequences[key]

    def launch_for(self, node, choice):
        if choice is not None:
            return choice.bmreq.num_warps, choice.bmreq.num_stages
        if hasattr(node, "kernel"):
            return node.kernel.num_warps, node.kernel.num_stages
        kernel = self.render(node, choice, combo_mode=False).kernel
        return kernel.num_warps, kernel.num_stages

    def render(self, node, choice, *, combo_mode, launch=None):
        if launch is not None and launch == self.launch_for(node, choice):
            launch = None
        key = (
            node,
            choice.hash_key() if choice is not None else "bound",
            combo_mode,
            launch,
        )
        if key not in self.regions:
            try:
                self.counts["renders"] += 1
                with preserve_codegen_state(node.get_nodes()):
                    self.regions[key] = render_region(
                        self.backend, node, choice, combo_mode=combo_mode, launch=launch
                    )
            except Exception as exc:
                self.regions[key] = exc
        region = self.regions[key]
        if isinstance(region, Exception):
            raise region
        return region

    def compile_member(self, node, choice, launch=None):
        region = self.render(node, choice, combo_mode=False, launch=launch)
        if region not in self.compiled:
            try:
                self.counts["member_compiles"] += 1
                self.compiled[region] = self.compile_region(region)
            except Exception as exc:
                self.compiled[region] = exc
        compiled = self.compiled[region]
        if isinstance(compiled, Exception):
            raise compiled
        return compiled

    def prepare(self, node, choice, launch):
        region = self.render(node, choice, combo_mode=False, launch=launch)
        if region not in self.prepared:
            try:
                call, autotuner = self.compile_member(node, choice, launch)
                self.counts["resource_queries"] += 1
                resource = resources(
                    autotuner, node.get_device(), region.kernel.num_warps
                )
                self.prepared[region] = PreparedRegion(region, choice, call, resource)
                report(
                    [node],
                    "member prepared",
                    choice=region.choice,
                    launch=(region.kernel.num_warps, region.kernel.num_stages),
                    registers=resource.registers,
                    shared=resource.shared,
                    active_threads=resource.active_threads,
                )
            except Exception as exc:
                self.prepared[region] = exc
        prepared = self.prepared[region]
        if isinstance(prepared, Exception):
            raise prepared
        return prepared

    def group_call(self, node):
        """Bind the member execution that normal template finalization would select."""
        from .scheduler import FusedCtaSchedulerNode

        if node in self.calls:
            return self.calls[node]
        if isinstance(node, FusedCtaSchedulerNode):
            self.baseline_choices[node.get_name()] = {
                "kind": "cta",
                "selected": repr(node.kernel.candidate),
            }
            call = self.compile_member(node, None)[0]
        else:
            _, template, _ = template_parts(node)
            buf = template.node
            choice = None
            if isinstance(buf, ir.MultiTemplateBuffer):
                if buf._make_kernel_renders or config.multi_kernel_hints:
                    raise UnsupportedRegionCombo("runtime multi-kernel baseline")
                choice = buf._render_caller
                if choice is None:
                    if len(node.get_nodes()) != 1:
                        raise UnsupportedRegionCombo("unbound fused template baseline")
                    if (
                        config.test_configs.force_extern_kernel_in_multi_template
                        or self.scheduler._has_layout_conflict_for_template(buf)
                    ):
                        externs = [
                            c for c in buf.choices if isinstance(c, ExternKernelCaller)
                        ]
                        if not externs:
                            raise UnsupportedRegionCombo("no extern baseline")
                        timings = (
                            buf.choice_timings()
                            if len(externs) > 1
                            else {externs[0]: 0}
                        )
                        choice = min(externs, key=timings.__getitem__)
                    else:
                        choice, _ = buf.get_min_choice()
            self.baseline_choices[node.get_name()] = {
                "kind": type(choice).__name__
                if choice is not None
                else type(buf).__name__,
                "selected": choice.hash_key() if choice is not None else "bound",
                "candidates": [
                    {"name": caller.name, "kind": type(caller).__name__}
                    for caller in buf.choices
                ]
                if isinstance(buf, ir.MultiTemplateBuffer)
                else [],
            }
            if isinstance(choice, ExternKernelCaller):
                if not choice.has_out_variant:
                    raise UnsupportedRegionCombo("extern baseline without out variant")
                fn = choice.to_callable()
                for inp in choice.input_nodes:
                    self.tensor(inp.get_name())
                self.tensor(buf.get_name())

                layouts = [
                    (
                        inp.get_name(),
                        V.graph.sizevars.optimization_hints(inp.get_size()),
                        V.graph.sizevars.optimization_hints(inp.get_stride()),
                        V.graph.sizevars.optimization_hint(inp.get_layout().offset),
                    )
                    for inp in (*choice.input_nodes, buf)
                ]

                def factory(tensors):
                    *args, out = [
                        tensors[name].as_strided(size, stride, offset)
                        for name, size, stride, offset in layouts
                    ]
                    return lambda: fn(*args, out=out)

                call = BenchmarkCall((factory,), self.tensors, key=(choice.hash_key(),))
            elif choice is None or isinstance(choice, TritonTemplateCaller):
                call = self.compile_member(node, choice)[0]
            else:
                raise UnsupportedRegionCombo("unsupported baseline caller")
        call()
        self.calls[node] = call
        return call

    def compile_region(self, region):
        kernel = region.kernel
        with V.set_kernel_handler(kernel):
            source = (
                kernel.imports_for_benchmark_kernel()
                + "\n"
                + region.source
                + "\n"
                + kernel.codegen_kernel_benchmark(None).getvalue()
            )
        from .utils import Placeholder

        call = self.bind(
            kernel, source.replace(str(Placeholder.KERNEL_NAME), "triton_")
        )
        launchers = self.modules[-1].triton_.launchers
        if len(launchers) != 1:
            raise UnsupportedRegionCombo("multiple launchers")
        return call, self.modules[-1].triton_


def compare_calls(baseline_call, call):
    samples = {}
    for mode, timer in (("ordinary", measure_ordinary), ("cuda_graph", measure)):
        baseline, combined = [], []
        for i in range(max(20, config.region_combo.benchmark_rounds)):
            order = ((baseline_call, baseline), (call, combined))
            for fn, values in order[:: -1 if i % 2 else 1]:
                values.append(timer(fn, config.region_combo.benchmark_iterations))
        samples[mode] = {
            "baseline_us": [t.avg_us for t in baseline],
            "combo_us": [t.avg_us for t in combined],
            "accepted": profitable(baseline, combined),
        }
    return samples


def benchmark_plan(benchmark, kernel, member_resources, baseline):
    benchmark.counts["combo_compiles"] += 1
    call = benchmark.bind(kernel, kernel.benchmark_source())
    launchers = benchmark.modules[-1].triton_.launchers
    if len(launchers) != 1:
        raise UnsupportedRegionCombo("multiple combo launchers")
    device = kernel.output_node.get_device()
    resource = resources(benchmark.modules[-1].triton_, device, kernel.num_warps)
    check_resources(member_resources, resource)
    return call, resource, compare_calls(baseline, call)


@dataclass
class MeasuredCombo:
    nodes: tuple
    kernel: RegionComboKernel
    call: BenchmarkCall
    savings: float


def search_groups(benchmark):
    choices, pools = {}, {}
    for node in benchmark.nodes:
        try:
            candidates = choices_for(benchmark.scheduler, node)
            pool = defaultdict(list)
            for choice in candidates:
                pool[benchmark.launch_for(node, choice)].append(choice)
            choices[node], pools[node] = candidates, pool
        except Exception as exc:
            report(
                [node],
                "member candidates rejected",
                error=f"{type(exc).__name__}: {exc}",
            )
    report(
        benchmark.nodes,
        "candidates",
        choices={
            node.get_name(): [
                c.hash_key() if c is not None else "bound" for c in candidates
            ]
            for node, candidates in choices.items()
        },
    )
    launches = sorted(OrderedSet(launch for pool in pools.values() for launch in pool))
    device = torch.cuda.get_device_properties(benchmark.nodes[0].get_device())
    groups = defaultdict(list)
    for launch in launches:
        prepared = []
        for node, candidates in choices.items():
            for choice in pools[node].get(
                launch, candidates[: config.region_combo.max_choices_per_member]
            ):
                try:
                    prepared.append(benchmark.prepare(node, choice, launch))
                except Exception as exc:
                    report(
                        [node],
                        "member preparation rejected",
                        launch=launch,
                        choice=choice.hash_key() if choice is not None else "bound",
                        error=f"{type(exc).__name__}: {exc}",
                    )
        for cluster in resource_clusters(prepared, device, launch):
            members = defaultdict(list)
            for candidate in cluster:
                members[candidate.region.node].append(candidate)
            nodes = sorted(members, key=lambda node: node.get_name())
            report(
                nodes,
                "resource cluster",
                launch=launch,
                candidates=[
                    {
                        "node": c.region.node.get_name(),
                        "choice": c.region.choice,
                        "registers": c.resources.registers,
                        "shared": c.resources.shared,
                        "active_threads": c.resources.active_threads,
                    }
                    for c in cluster
                ],
            )
            for offset in range(0, len(nodes), config.combo_kernel_max_num_nodes):
                group = tuple(
                    nodes[offset : offset + config.combo_kernel_max_num_nodes]
                )
                if len(group) < 2:
                    report(group, "resource singleton", launch=launch)
                    continue
                groups[group].append((launch, [members[n] for n in group]))
    return groups


def select_group(benchmark, nodes, searches):
    check_group(benchmark.scheduler, nodes)
    best, best_time = None, math.inf
    baseline = benchmark.sequence(nodes)
    iterators = [
        zip(itertools.repeat(launch), itertools.product(*pools))
        for launch, pools in searches
    ]
    combinations = (
        c for batch in itertools.zip_longest(*iterators) for c in batch if c is not None
    )
    attempts = 0
    for launch, combination in itertools.islice(
        combinations, config.region_combo.max_configs
    ):
        attempts += 1
        try:
            regions, member_resources = [], []
            for candidate in combination:
                node, choice = candidate.region.node, candidate.choice
                original = benchmark.prepare(
                    node, choice, benchmark.launch_for(node, choice)
                )
                member_resources.append(original.resources)
                regions.append(
                    benchmark.render(node, choice, combo_mode=True, launch=launch)
                )
            kernel = RegionComboKernel(tuple(regions))
            call, resource, samples = benchmark_plan(
                benchmark, kernel, member_resources, baseline
            )
            report(
                nodes,
                "measured",
                choices=[r.choice for r in regions],
                samples=samples,
                registers=resource.registers,
                shared=resource.shared,
                active_threads=resource.active_threads,
                warps=kernel.num_warps,
                stages=kernel.num_stages,
            )
            elapsed = statistics.median(samples["cuda_graph"]["combo_us"])
            if all(s["accepted"] for s in samples.values()) and elapsed < best_time:
                savings = min(
                    statistics.median(s["baseline_us"])
                    - statistics.median(s["combo_us"])
                    for s in samples.values()
                )
                best, best_time = MeasuredCombo(nodes, kernel, call, savings), elapsed
        except Exception as exc:
            report(
                nodes,
                "candidate rejected",
                launch=launch,
                choices=[c.region.choice for c in combination],
                error=f"{type(exc).__name__}: {exc}",
            )
    candidate_count = sum(
        math.prod(len(pool) for pool in pools) for _, pools in searches
    )
    report(
        nodes,
        "candidate search complete",
        attempts=attempts,
        candidates=candidate_count,
        max_configs=config.region_combo.max_configs,
    )
    return best


def candidate_groups(scheduler):
    from .scheduler import FusedCtaSchedulerNode

    for level, nodes in enumerate(scheduler._topological_sort_nodes()):
        groups = defaultdict(list)
        for node in nodes:
            kind = (
                "cta"
                if isinstance(node, FusedCtaSchedulerNode) and config.region_combo.cta
                else "template"
                if node.is_template() and config.region_combo.template
                else None
            )
            if kind is None:
                continue
            try:
                check_group(scheduler, [node])
            except UnsupportedRegionCombo as exc:
                report([node], "ineligible member", error=str(exc), level=level)
                continue
            key = (
                kind,
                node.get_device(),
                scheduler.get_node_stream(node),
                scheduler.node_to_mempool.get(node),
            )
            groups[key].append(node)
        for key, group in groups.items():
            report(group, "topological candidates", level=level, kind=key[0])
            yield group


def select_partitions(scheduler, nodes):
    from .scheduler import RegionComboSchedulerNode

    check_group(scheduler, nodes)
    with (
        torch.cuda.device(nodes[0].get_device()),
        V.graph.set_current_device(nodes[0].get_device()),
        speculative_codegen(nodes),
    ):
        started = time.perf_counter()
        benchmark = RegionBenchmark(scheduler, nodes)
        report(nodes, "baseline selected", choices=benchmark.baseline_choices)
        groups = search_groups(benchmark)
        prepared = time.perf_counter()
        candidates = []
        for members, searches in groups.items():
            try:
                selected = select_group(benchmark, members, searches)
                if selected is not None:
                    candidates.append(selected)
            except Exception as exc:
                report(members, "group rejected", error=f"{type(exc).__name__}: {exc}")
        replacements: dict[BaseSchedulerNode, RegionComboSchedulerNode] = {}
        calls = {}
        for candidate in sorted(
            candidates, key=lambda c: (-c.savings, tuple(n.get_name() for n in c.nodes))
        ):
            if any(node in replacements for node in candidate.nodes):
                report(
                    candidate.nodes,
                    "overlapping measured group retained another choice",
                )
                continue
            combo = RegionComboSchedulerNode(
                scheduler, candidate.nodes, candidate.kernel
            )
            replacements.update(dict.fromkeys(candidate.nodes, combo))
            calls[combo] = candidate.call
        partitions = list(dict.fromkeys(replacements.get(node, node) for node in nodes))
        if calls and len(partitions) > 1:
            partition_calls = [
                calls[p] if p in calls else benchmark.group_call(p) for p in partitions
            ]
            call = BenchmarkCall(
                [f for c in partition_calls for f in c.factories], benchmark.tensors
            )
            samples = compare_calls(benchmark.call, call)
            members = [
                [n.get_name() for n in p.members] if p in calls else [p.get_name()]
                for p in partitions
            ]
            report(
                nodes,
                "partition sequence measured",
                samples=samples,
                partitions=members,
            )
            if not all(s["accepted"] for s in samples.values()):
                partitions = list(nodes)
        report(
            nodes,
            "search work",
            **benchmark.counts,
            prepare_seconds=prepared - started,
            search_seconds=time.perf_counter() - prepared,
            resource_groups=len(groups),
            measured_groups=len(candidates),
        )
        return partitions


def create_region_combos(scheduler):
    from .scheduler import RegionComboSchedulerNode

    if config.deterministic or not config.benchmark_combo_kernel:
        report((), "benchmark disabled or deterministic")
        return
    replacements = {}

    def flush(group):
        if len(group) < 2:
            return
        try:
            partitions = select_partitions(scheduler, group)
        except Exception as exc:
            report(
                group,
                "partition sequence rejected",
                error=f"{type(exc).__name__}: {exc}",
            )
            partitions = list(group)
        combos = [p for p in partitions if isinstance(p, RegionComboSchedulerNode)]
        for selected in combos:
            replacements.update(dict.fromkeys(selected.members, selected))
            scheduler.name_to_fused_node.update(
                {n.get_name(): selected for n in selected.get_nodes()}
            )
            scheduler.node_to_stream[selected] = scheduler.get_node_stream(
                selected.members[0]
            )
            scheduler.node_to_mempool[selected] = scheduler.node_to_mempool.get(
                selected.members[0]
            )
            report(selected.members, "accepted")
        if not combos:
            report(group, "original execution retained")

    for group in candidate_groups(scheduler):
        flush(group)
    # Preserve the baseline order unless an accepted combo requires moving its inputs.
    result = list(
        dict.fromkeys(replacements.get(node, node) for node in scheduler.nodes)
    )
    scheduler.nodes = scheduler.topological_sort_schedule(result)
    for node in scheduler.nodes:
        ancestors = OrderedSet()
        for producer in scheduler._get_unmet_dep_nodes(node):
            if producer is node:
                continue
            ancestors.update(producer.get_operation_names())
            ancestors.update(producer.ancestors)
        node.ancestors = ancestors
