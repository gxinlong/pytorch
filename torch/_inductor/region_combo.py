from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import statistics
from dataclasses import dataclass, replace

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
from .select_algorithm import ExternKernelCaller, TritonTemplateCaller, TritonTemplateKernel
from .virtualized import V


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
    lower = statistics.mean(differences) - 2.1 * statistics.stdev(differences) / math.sqrt(
        len(differences)
    )
    return lower > 0 and statistics.median(
        a.avg_us / b.avg_us for a, b in zip(baseline, combined)
    ) >= config.region_combo.min_speedup


@gpu_benchmark_lock
def measure_ordinary(call, iterations):
    for _ in range(5):
        call.reset()
        call()
    call.reset()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
    if spills or shared > torch.cuda.get_device_properties(device).shared_memory_per_block_optin:
        raise UnsupportedRegionCombo("compiled spill or shared-memory limit")
    from cuda.bindings import driver

    error, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        autotuner.compile_results[0].kernel.function, warps * 32, shared
    )
    if int(error) != 0 or blocks <= 0:
        raise UnsupportedRegionCombo("occupancy query failed")
    return Resources(registers, shared, spills, blocks * warps * 32)


def check_resources(members, combined):
    median_regs = statistics.median(r.registers for r in members)
    if any(r.registers > median_regs * config.region_combo.max_register_ratio for r in members):
        raise UnsupportedRegionCombo("register outlier")
    if any(
        combined.active_threads < r.active_threads * (1 - config.region_combo.max_occupancy_drop)
        for r in members
    ):
        raise UnsupportedRegionCombo("occupancy drop")


def check_group(scheduler, nodes):
    first = nodes[0]
    device = first.get_device()
    if device is None or device.type != "cuda" or torch.version.hip:
        raise UnsupportedRegionCombo("requires NVIDIA CUDA")
    for i, node in enumerate(nodes):
        if node.get_device() != device:
            raise UnsupportedRegionCombo("different devices")
        if scheduler.get_node_stream(node) != scheduler.get_node_stream(first):
            raise UnsupportedRegionCombo("different streams")
        if scheduler.node_to_mempool.get(node) != scheduler.node_to_mempool.get(first):
            raise UnsupportedRegionCombo("different memory pools")
        if node.has_aliasing_or_mutation() or node.has_strict_sum():
            raise UnsupportedRegionCombo("aliasing, mutation or strict sum")
        for other in nodes[:i]:
            if (
                node.get_operation_names() & other.ancestors
                or other.get_operation_names() & node.ancestors
            ):
                raise UnsupportedRegionCombo("dependent regions")
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
                c for c in buf.choices
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
            out.get_name() for out in node.get_outputs() if out.get_name() not in node.plan.outputs
        )
        return RegionKernel(node, kernel, kernel.source, repr(node.kernel.candidate))
    prologue, template, epilogue = template_parts(node)
    buf = template.node
    factory = choice.get_make_kernel_render() if choice is not None else buf.make_kernel_render
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
    return RegionKernel(node, kernel, source, choice.hash_key() if choice is not None else "bound")


class RegionBenchmark(CtaBenchmark):
    def __init__(self, scheduler, nodes):
        self.scheduler, self.nodes = scheduler, nodes
        self.tensors = {}
        self.calls = {}
        self.baseline_choices = {}
        self.modules = []
        self.backend = scheduler.get_backend(nodes[0].get_device())._triton_scheduling
        for node in nodes:
            self.group_call(node)
        self.call = BenchmarkCall(
            [factory for node in nodes for factory in self.calls[node].factories], self.tensors
        )

    def group_call(self, node):
        from .scheduler import FusedCtaSchedulerNode

        if node in self.calls:
            return self.calls[node]
        if isinstance(node, FusedCtaSchedulerNode):
            self.baseline_choices[node.get_name()] = {
                "kind": "cta", "selected": repr(node.kernel.candidate)
            }
            call = self.compile_region(render_region(self.backend, node, None, combo_mode=False))[0]
        else:
            _, template, _ = template_parts(node)
            buf = template.node
            choice = None
            if isinstance(buf, ir.MultiTemplateBuffer):
                choice = buf._render_caller
                if choice is None:
                    if len(node.get_nodes()) != 1:
                        raise UnsupportedRegionCombo("unbound fused template baseline")
                    choice, _ = buf.get_min_choice()
            self.baseline_choices[node.get_name()] = {
                "kind": type(choice).__name__ if choice is not None else type(buf).__name__,
                "selected": choice.hash_key() if choice is not None else "bound",
                "candidates": [
                    {"name": caller.name, "kind": type(caller).__name__}
                    for caller in buf.choices
                ] if isinstance(buf, ir.MultiTemplateBuffer) else [],
            }
            if isinstance(choice, ExternKernelCaller):
                if not choice.has_out_variant:
                    raise UnsupportedRegionCombo("extern baseline without out variant")
                fn = choice.to_callable()
                for inp in choice.input_nodes:
                    self.tensor(inp.get_name())
                self.tensor(buf.get_name())

                layouts = [
                    (inp.get_name(), V.graph.sizevars.optimization_hints(inp.get_size()),
                     V.graph.sizevars.optimization_hints(inp.get_stride()),
                     V.graph.sizevars.optimization_hint(inp.get_layout().offset))
                    for inp in (*choice.input_nodes, buf)
                ]

                def factory(tensors):
                    *args, out = [tensors[name].as_strided(size, stride, offset)
                                  for name, size, stride, offset in layouts]
                    return lambda: fn(*args, out=out)

                call = BenchmarkCall((factory,), self.tensors, key=(choice.hash_key(),))
            elif choice is None or isinstance(choice, TritonTemplateCaller):
                region = render_region(self.backend, node, choice, combo_mode=False)
                call = self.compile_region(region)[0]
            else:
                raise UnsupportedRegionCombo("unsupported baseline caller")
        call()
        self.calls[node] = call
        return call

    def compile_region(self, region):
        kernel = region.kernel
        with V.set_kernel_handler(kernel):
            source = (
                kernel.imports_for_benchmark_kernel() + "\n" + region.source + "\n"
                + kernel.codegen_kernel_benchmark(None).getvalue()
            )
        from .utils import Placeholder

        call = self.bind(kernel, source.replace(str(Placeholder.KERNEL_NAME), "triton_"))
        launchers = self.modules[-1].triton_.launchers
        if len(launchers) != 1:
            raise UnsupportedRegionCombo("multiple launchers")
        device = kernel.output_node.get_device()
        return call, resources(self.modules[-1].triton_, device, kernel.num_warps)


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


def benchmark_plan(benchmark, kernel, member_resources):
    call = benchmark.bind(kernel, kernel.benchmark_source())
    launchers = benchmark.modules[-1].triton_.launchers
    if len(launchers) != 1:
        raise UnsupportedRegionCombo("multiple combo launchers")
    device = kernel.output_node.get_device()
    resource = resources(benchmark.modules[-1].triton_, device, kernel.num_warps)
    check_resources(member_resources, resource)
    return resource, compare_calls(benchmark.call, call)


def select_group(scheduler, nodes, pressures):
    from .scheduler import RegionComboSchedulerNode

    check_group(scheduler, nodes)
    best, best_time = None, math.inf
    with (
        torch.cuda.device(nodes[0].get_device()),
        V.graph.set_current_device(nodes[0].get_device()),
        speculative_codegen(nodes),
    ):
        choices = [choices_for(scheduler, node) for node in nodes]
        report(nodes, "candidates", choices=[
            [choice.hash_key() if choice is not None else "bound" for choice in candidates]
            for candidates in choices
        ])
        if not all(choices):
            raise UnsupportedRegionCombo("no compatible Triton candidate")
        benchmark = RegionBenchmark(scheduler, nodes)
        report(nodes, "baseline selected", choices=benchmark.baseline_choices)
        pools = []
        for node, candidates in zip(nodes, choices):
            pool = {}
            for choice in candidates:
                if choice is not None:
                    key = (choice.bmreq.num_warps, choice.bmreq.num_stages)
                elif hasattr(node, "kernel"):
                    key = (node.kernel.num_warps, node.kernel.num_stages)
                else:
                    region = render_region(benchmark.backend, node, choice, combo_mode=False)
                    key = (region.kernel.num_warps, region.kernel.num_stages)
                pool.setdefault(key, []).append(choice)
            pools.append(pool)
        launches = sorted(set.union(*(set(p) for p in pools)))
        searches = [
            zip(itertools.repeat(key), itertools.product(*(
                p.get(key, candidates[:config.region_combo.max_choices_per_member])
                for p, candidates in zip(pools, choices)
            )))
            for key in launches
        ]
        combinations = (
            c for batch in itertools.zip_longest(*searches) for c in batch if c is not None
        )
        for launch, combination in itertools.islice(combinations, config.region_combo.max_configs):
            try:
                regions, member_resources = [], []
                for node, choice in zip(nodes, combination):
                    original = render_region(benchmark.backend, node, choice, combo_mode=False)
                    try:
                        _, resource = benchmark.compile_region(original)
                    except Exception:
                        pressures[node] = (math.inf, math.inf)
                        raise
                    pressures[node] = (resource.registers, resource.shared)
                    member_resources.append(resource)
                    regions.append(render_region(
                        benchmark.backend, node, choice, combo_mode=True, launch=launch
                    ))
                kernel = RegionComboKernel(tuple(regions))
                resource, samples = benchmark_plan(benchmark, kernel, member_resources)
                report(
                    nodes, "measured", choices=[r.choice for r in regions], samples=samples,
                    registers=resource.registers, shared=resource.shared,
                    active_threads=resource.active_threads,
                    warps=kernel.num_warps, stages=kernel.num_stages,
                )
                elapsed = statistics.median(samples["cuda_graph"]["combo_us"])
                if all(s["accepted"] for s in samples.values()) and elapsed < best_time:
                    best, best_time = kernel, elapsed
            except Exception as exc:
                report(nodes, "candidate rejected", error=f"{type(exc).__name__}: {exc}")
    return RegionComboSchedulerNode(scheduler, nodes, best) if best is not None else None


def candidate_groups(scheduler):
    from .scheduler import FusedCtaSchedulerNode

    max_nodes = config.combo_kernel_max_num_nodes
    for level, nodes in enumerate(scheduler._topological_sort_nodes()):
        groups = {}
        for node in nodes:
            kind = (
                "cta" if isinstance(node, FusedCtaSchedulerNode) and config.region_combo.cta
                else "template" if node.is_template() and config.region_combo.template
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
                kind, node.get_device(), scheduler.get_node_stream(node),
                scheduler.node_to_mempool.get(node),
            )
            groups.setdefault(key, []).append(node)
        for key, group in groups.items():
            report(group, "topological candidates", level=level, kind=key[0])
            for i in range(0, len(group), max_nodes):
                yield group[i:i + max_nodes]


def create_region_combos(scheduler):
    from .scheduler import RegionComboSchedulerNode

    if config.deterministic or not config.benchmark_combo_kernel:
        report((), "benchmark disabled or deterministic")
        return
    replacements = {}

    def select_partitions(nodes):
        if len(nodes) < 2:
            return list(nodes)
        pressures = {}
        try:
            selected = select_group(scheduler, nodes, pressures)
        except Exception as exc:
            report(nodes, "group rejected", error=f"{type(exc).__name__}: {exc}")
            selected = None
        if selected is not None:
            return [selected]
        if len(nodes) > 2:
            if pressures:
                index = max(range(len(nodes)), key=lambda i: pressures.get(nodes[i], (0, 0)))
                report(nodes, "isolate resource-heavy member", member=nodes[index].get_name())
                return [
                    *select_partitions(nodes[:index]), nodes[index],
                    *select_partitions(nodes[index + 1:]),
                ]
            middle = len(nodes) // 2
            return [*select_partitions(nodes[:middle]), *select_partitions(nodes[middle:])]
        return list(nodes)

    def flush(group):
        if len(group) < 2:
            return
        partitions = select_partitions(group)
        combos = [p for p in partitions if isinstance(p, RegionComboSchedulerNode)]
        if combos and len(partitions) > 1:
            try:
                with (torch.cuda.device(group[0].get_device()),
                      V.graph.set_current_device(group[0].get_device()),
                      speculative_codegen(group)):
                    benchmark = RegionBenchmark(scheduler, group)
                    calls = [
                        benchmark.bind(p.kernel, p.kernel.benchmark_source())
                        if isinstance(p, RegionComboSchedulerNode) else benchmark.group_call(p)
                        for p in partitions
                    ]
                    call = BenchmarkCall([f for c in calls for f in c.factories], benchmark.tensors)
                    samples = compare_calls(benchmark.call, call)
                    members = [
                        [n.get_name() for n in p.members]
                        if isinstance(p, RegionComboSchedulerNode) else [p.get_name()]
                        for p in partitions
                    ]
                    report(
                        group, "partition sequence measured", samples=samples, partitions=members
                    )
                    if not all(s["accepted"] for s in samples.values()):
                        partitions, combos = list(group), []
            except Exception as exc:
                report(group, "partition sequence rejected", error=str(exc))
                partitions, combos = list(group), []
        for selected in combos:
            replacements.update({member: selected for member in selected.members})
            scheduler.name_to_fused_node.update(
                {n.get_name(): selected for n in selected.get_nodes()}
            )
            scheduler.node_to_stream[selected] = scheduler.get_node_stream(selected.members[0])
            scheduler.node_to_mempool[selected] = scheduler.node_to_mempool.get(selected.members[0])
            report(selected.members, "accepted")
        if not combos:
            report(group, "original execution retained")

    for group in candidate_groups(scheduler):
        flush(group)
    # Preserve the baseline order unless an accepted combo requires moving its inputs.
    result = list(dict.fromkeys(replacements.get(node, node) for node in scheduler.nodes))
    scheduler.nodes = scheduler.topological_sort_schedule(result)
    for node in scheduler.nodes:
        ancestors = OrderedSet()
        for producer in scheduler._get_unmet_dep_nodes(node):
            if producer is node:
                continue
            ancestors.update(producer.get_operation_names())
            ancestors.update(producer.ancestors)
        node.ancestors = ancestors
