from __future__ import annotations

import contextlib
import copy
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config, ir
from ..codecache import PyCodeCache
from ..cta_fusion import (
    CtaConfig,
    CtaFusionPlan,
    CtaStage,
    descriptor,
    static_int,
    TimingCache,
    UnsupportedCtaFusion,
)
from ..ops_handler import DefaultHandler
from ..runtime.runtime_utils import next_power_of_2
from ..select_algorithm import (
    AlgorithmSelectorCache,
    ExternKernelCaller,
    SymbolicGridFn,
    TritonTemplateKernel,
)
from ..utils import IndentedBuffer, Placeholder, sympy_product
from ..virtualized import V
from .common import InplacedBuffer, SizeArg, TensorArg
from .simd_kernel_features import SIMDKernelFeatures
from .triton import texpr, triton_store_type, TritonOverrides


@SymbolicGridFn
def cta_grid(units, meta, *, cdiv):
    return (cdiv(units, meta["rows"]), 1, 1)


@dataclass
class LocalValue:
    stage: CtaStage
    name: str
    padded_columns: int
    logical_columns: int
    size: int


class TileOps(DefaultHandler):
    def __init__(self, kernel: CtaKernel) -> None:
        self.kernel = kernel

    def _default(self, name, args, kwargs):
        if name in (
            "rand",
            "randn",
            "randint64",
            "load_seed",
            "scan",
            "sort",
            "bucketize",
            "device_assert",
            "indirect_indexing",
        ):
            raise UnsupportedCtaFusion(f"unsupported tile operation: {name}")
        try:
            result = getattr(TritonOverrides, name)(*args, **kwargs)
        except (AttributeError, NotImplementedError) as exc:
            raise UnsupportedCtaFusion(f"unsupported tile operation: {name}") from exc
        if not isinstance(result, str):
            raise UnsupportedCtaFusion(f"non-scalar operation: {name}")
        return self.kernel.emit(result)

    def load(self, name, index):
        return self.kernel.tile_load(name, index)

    def index_expr(self, expr, dtype):
        return self.kernel.emit(
            f"tl.full((), 0, {triton_store_type(dtype)}) + ({self.kernel.expr(expr)})"
        )

    value_expr = index_expr

    def to_dtype(self, value, dtype, src_dtype=None, use_compute_types=True):
        value = self.kernel.emit(f"{value}.to({triton_store_type(dtype)})")
        if use_compute_types and dtype in (torch.float16, torch.bfloat16):
            value = self.kernel.emit(f"{value}.to(tl.float32)")
        return value

    def indirect_indexing(self, *args, **kwargs):
        raise UnsupportedCtaFusion("indirect indexing")

    def masked(self, mask, body, other):
        old_mask = self.kernel.mask
        self.kernel.mask = f"({old_mask}) & ({mask})"
        try:
            value = body()
        finally:
            self.kernel.mask = old_mask
        return self.kernel.emit(f"tl.where({mask}, {value}, {other})")

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type == "online_softmax_reduce":
            if isinstance(value, (tuple, list)):
                raise UnsupportedCtaFusion("split online softmax")
            maximum = self.kernel.emit(
                f"triton_helpers.fmax2(tl.where({self.kernel.mask}, {value}, float('-inf')), 1)"
            )
            exponential = self.kernel.emit(
                TritonOverrides.exp(f"{value} - {maximum}[:, None]")
            )
            total = self.kernel.emit(
                f"tl.sum(tl.where({self.kernel.mask}, {exponential}, 0), 1)"
            )
            return maximum, total
        functions = {
            "sum": ("sum", "0"),
            "max": ("max", 'float("-inf")'),
            "min": ("min", 'float("inf")'),
            "any": ("max", "False"),
            "all": ("min", "True"),
        }
        if reduction_type not in functions or isinstance(value, tuple):
            raise UnsupportedCtaFusion(f"reduction: {reduction_type}")
        fn, neutral = functions[reduction_type]
        operation = f"tl.{fn}"
        if reduction_type in ("max", "min"):
            suffix = "_strict" if config.strict_signed_zero else ""
            operation = f"triton_helpers.{fn}2{suffix}"
        if src_dtype.is_floating_point and src_dtype.itemsize < 4:
            value = f"{value}.to(tl.float32)"
        elif src_dtype == torch.bool and reduction_type in ("max", "min"):
            neutral = "False" if reduction_type == "max" else "True"
        elif not src_dtype.is_floating_point and reduction_type in ("max", "min"):
            limits = torch.iinfo(src_dtype)
            neutral = str(limits.min if reduction_type == "max" else limits.max)
        value = self.kernel.emit(
            f"{operation}(tl.where({self.kernel.mask}, {value}, {neutral}), 1)"
        )
        return self.kernel.emit(f"{value}.to({triton_store_type(dtype)})")

    def store(self, name, index, value, mode=None):
        if mode is not None:
            raise UnsupportedCtaFusion("non-functional store")
        self.kernel.tile_store(name, index, value)

    store_reduction = store


class CtaKernel(TritonTemplateKernel):
    def __init__(
        self, plan: CtaFusionPlan, candidate: CtaConfig, *, combo_mode: bool = False
    ) -> None:
        super().__init__(
            kernel_name=str(Placeholder.KERNEL_NAME),
            input_nodes=(),
            output_node=plan.stages[-1].output,
            defines="",
            num_stages=candidate.num_stages,
            num_warps=candidate.num_warps,
            grid_fn=cta_grid,
            meta={"rows": candidate.rows},
            call_sizes=[plan.units],
            index_dtype_override="tl.int64",
        )
        self.plan = plan
        self.combo_mode = combo_mode
        self.combo_supported = True
        self.use_jit = combo_mode
        self.candidate = candidate
        self.tile_code = IndentedBuffer()
        self.counter = itertools.count()
        self.local_values: dict[str, LocalValue] = {}
        self.resources: list[dict[str, int | None]] = []
        self.peak_live_elements = 0
        self.symbols: dict[sympy.Symbol, str] = {}
        self.ranges: dict[sympy.Symbol, sympy.Expr] = {}
        self.unit = sympy.Symbol("cta_unit", integer=True, nonnegative=True)
        self.ranges[self.unit] = plan.units
        self.symbols[self.unit] = "cta_unit"
        self.mask = "True"
        self.store_mask = "True"
        self.stage: CtaStage | None = None
        self.local_size = 1
        self.local_shape = (1,)
        self.logical_columns = 1
        self.padded_columns = 1
        self.tile_code.writeline(
            f"cta_unit = {self.program_id(0)} * {candidate.rows}"
        )
        with V.set_kernel_handler(self), V.set_ops_handler(TileOps(self)):
            for i, stage in enumerate(plan.stages):
                self.stage = stage
                if stage.gemm is not None:
                    self.gemm(stage)
                else:
                    self.elementwise(stage)
                self.peak_live_elements = max(
                    self.peak_live_elements,
                    sum(value.size for value in self.local_values.values()),
                )
                future_reads = {
                    dep.name
                    for future in plan.stages[i + 1 :]
                    for dep in future.node.read_writes.reads
                }
                for name in list(self.local_values):
                    if name not in future_reads:
                        self.local_values.pop(name)
        self.source = self.source_code()

    def emit(self, expression: str) -> str:
        name = f"cta_v{next(self.counter)}"
        self.tile_code.writeline(f"{name} = {expression}")
        return name

    def symbol(self, name: str, size: int, value: str) -> sympy.Symbol:
        symbol = sympy.Symbol(
            f"cta_{name}_{next(self.counter)}", integer=True, nonnegative=True
        )
        self.ranges[symbol] = sympy.Integer(size)
        self.symbols[symbol] = value
        return symbol

    def simplify(self, expression):
        expression = sympy.sympify(expression)
        if self.plan.units == 1:
            expression = expression.xreplace({self.unit: sympy.S.Zero})
        return V.graph.sizevars.simplify_with_ranges(expression, self.ranges)

    def expr(self, expression) -> str:
        expression = self.simplify(expression)
        expression = expression.xreplace(
            {symbol: sympy.Symbol(text) for symbol, text in self.symbols.items()}
        )
        return texpr(self.rename_indexing(expression))

    @staticmethod
    def unravel(index, sizes):
        result = []
        for i, size in enumerate(sizes):
            stride = sympy_product(sizes[i + 1 :])
            result.append(ModularIndexing(index, stride, size))
        return result

    def logical_index(self, output, physical):
        sizes, strides = output.get_size(), output.get_stride()
        # A dense permutation is invertible. Overlapping or holed output layouts
        # need a different local-value representation.
        dims = sorted(
            (i for i, size in enumerate(sizes) if size != 1),
            key=lambda i: V.graph.sizevars.optimization_hint(strides[i]),
        )
        expected = sympy.S.One
        for i in dims:
            if not V.graph.sizevars.statically_known_equals(strides[i], expected):
                raise UnsupportedCtaFusion("non-dense intermediate layout")
            expected *= sizes[i]
        physical -= output.get_layout().offset
        return sum(
            ModularIndexing(physical, strides[i], sizes[i])
            * sympy_product(sizes[i + 1 :])
            for i in dims
        )

    def tile_load(self, name, index):
        if name in self.local_values:
            local = self.local_values[name]
            logical = self.logical_index(local.stage.output, index)
            offset = self.simplify(logical - self.unit * local.stage.elements_per_unit)
            if self.unit in offset.free_symbols:
                raise UnsupportedCtaFusion("intermediate load crosses CTA ownership")
            bounds = bound_sympy(
                offset,
                {
                    symbol: ValueRanges(0, size - 1)
                    for symbol, size in self.ranges.items()
                    if symbol in offset.free_symbols
                },
            )
            available = local.stage.elements_per_unit * self.candidate.rows
            if not (
                V.graph.sizevars.statically_known_leq(0, bounds.lower)
                and V.graph.sizevars.statically_known_lt(bounds.upper, available)
            ):
                raise UnsupportedCtaFusion("unproved intermediate bounds")
            offset = FloorDiv(
                offset, local.logical_columns
            ) * local.padded_columns + ModularIndexing(offset, 1, local.logical_columns)
            index_value = self.emit(
                f"tl.where({self.mask}, {self.expr(offset)}, 0).to(tl.int32)"
            )
            value = self.emit(
                f"tl.gather({local.name}.reshape(({local.size},)), "
                f"{index_value}.reshape(({index_value}.numel,)), 0).reshape({index_value}.shape)"
            )
            value = self.emit(f"tl.where({self.mask}, {value}, 0)")
            if local.stage.output.get_dtype() in (torch.float16, torch.bfloat16):
                value = self.emit(f"{value}.to(tl.float32)")
            return value
        if any(stage.output.get_name() == name for stage in self.plan.stages):
            raise UnsupportedCtaFusion("intermediate is not yet defined")
        pointer = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        value = self.emit(
            f"tl.load({pointer} + ({self.expr(index)}) + tl.zeros(({self.mask}).shape, tl.int64), "
            f"{self.mask}, other=0)"
        )
        if dtype in (torch.float16, torch.bfloat16):
            value = self.emit(f"{value}.to(tl.float32)")
        return value

    def tile_store(self, name, index, value):
        stage = self.stage
        if stage is None or name != stage.output.get_name():
            raise UnsupportedCtaFusion("unexpected stage output")
        self.logical_index(stage.output, index)
        value = self.emit(
            f"tl.broadcast_to({value}, {self.local_shape}).to({triton_store_type(stage.output.get_dtype())})"
        )
        self.local_values[name] = LocalValue(
            stage, value, self.padded_columns, self.logical_columns, self.local_size
        )
        if name in self.plan.outputs:
            pointer = self.args.output(name)
            self.tile_code.writeline(
                f"tl.store({pointer} + ({self.expr(index)}) + tl.zeros({value}.shape, tl.int64), "
                f"{value}, {self.store_mask})"
            )

    def check_tile(self, *sizes):
        if any(s > config.cta_fusion.max_intermediate_dim for s in sizes):
            raise UnsupportedCtaFusion("intermediate dimension")
        if math.prod(sizes) > config.cta_fusion.max_intermediate_tile_elements:
            raise UnsupportedCtaFusion("padded tile elements")

    def elementwise(self, stage: CtaStage):
        output = stage.output
        if not isinstance(output, ir.ComputedBuffer):
            raise UnsupportedCtaFusion("expected computed buffer")
        count = stage.elements_per_unit * self.candidate.rows
        size = next_power_of_2(count)
        if size > config.cta_fusion.max_intermediate_tile_elements:
            raise UnsupportedCtaFusion("padded tile elements")
        x = self.emit(f"tl.arange(0, {size})")
        variable = self.symbol("x", count, x)
        logical = self.unit * stage.elements_per_unit + variable
        indices = self.unravel(logical, output.data.get_pointwise_size())
        self.store_mask = f"({x} < {count}) & ({self.expr(logical)} < {self.expr(output.get_numel())})"
        self.mask = self.store_mask
        self.local_size = size
        self.local_shape = (size,)
        self.logical_columns = count
        self.padded_columns = size
        rsize = static_int(sympy_product(output.data.get_reduction_size()))
        if output.get_reduction_type() is not None:
            rblock = next_power_of_2(rsize)
            if (
                rblock > config.cta_fusion.max_intermediate_dim
                or size * rblock > config.cta_fusion.max_intermediate_tile_elements
            ):
                raise UnsupportedCtaFusion("reduction tile size")
            r = self.emit(f"tl.arange(0, {rblock})")
            rvar = self.symbol("r", rsize, f"{r}[None, :]")
            self.symbols[variable] = f"{x}[:, None]"
            self.mask = f"({self.store_mask})[:, None] & ({r}[None, :] < {rsize})"
            rindices = self.unravel(rvar, output.data.get_reduction_size())
            value = output.data.inner_fn(indices, rindices)
            value = V.ops.reduction(
                output.data.dtype,
                output.data.src_dtype,
                output.data.reduction_type,
                value,
            )
            if isinstance(output.data, ir.MultiOutputReduction):
                value = value[output.data.output_index]
            self.symbols[variable] = x
            index = output.get_layout().as_fixed().make_indexer()(indices)
            self.tile_store(output.get_name(), index, value)
        else:
            output.get_store_function()(indices)

    def gemm(self, stage: CtaStage):
        g = stage.gemm
        if g is None:
            raise AssertionError("expected GEMM descriptor")
        m, n, k = g.lhs.get_size()[-2], g.rhs.get_size()[-1], g.lhs.get_size()[-1]
        n, k = static_int(n), static_int(k)
        if g.batched:
            rows = static_int(m)
            row_base = sympy.S.Zero
        else:
            per_unit_rows = static_int(m / self.plan.units)
            rows = per_unit_rows * self.candidate.rows
            row_base = self.unit * per_unit_rows
        use_dot = self.candidate.use_dot
        bm = max(16 if use_dot else 1, next_power_of_2(rows))
        bn = max(16 if use_dot else 1, next_power_of_2(n))
        bk = self.candidate.block_k
        self.check_tile(bm, bn)
        if not use_dot:
            self.check_tile(bm, bn, bk)
        row = self.emit(f"tl.arange(0, {bm})")
        col = self.emit(f"tl.arange(0, {bn})")
        r = self.symbol("m", rows, row)
        c = self.symbol("n", n, col)
        kk = self.symbol("k", k, "cta_k")
        acc = self.emit(f"tl.full(({bm}, {bn}), 0, tl.float32)")
        loop = f"cta_loop_{next(self.counter)}"
        self.tile_code.writeline(f"for {loop} in range(tl.cdiv({k}, {bk})):")
        with self.tile_code.indent():
            self.tile_code.writeline(f"cta_k = {loop} * {bk} + tl.arange(0, {bk})")
            if use_dot:
                self.symbols[r], self.symbols[kk] = f"{row}[:, None]", "cta_k[None, :]"
            else:
                self.symbols[r], self.symbols[kk] = (
                    f"{row}[:, None, None]",
                    "cta_k[None, None, :]",
                )
            self.mask = f"({self.expr(r)} < {rows}) & ({self.expr(row_base + r)} < {self.expr(m)}) & ({self.expr(kk)} < {k})"
            lhs_indices = [row_base + r, kk]
            if g.batched:
                lhs_indices.insert(0, self.unit)
                self.mask += f" & (cta_unit < {self.expr(self.plan.units)})"
            a = g.lhs.make_loader()(lhs_indices)
            if use_dot:
                self.symbols[kk], self.symbols[c] = "cta_k[:, None]", f"{col}[None, :]"
            else:
                self.symbols[kk], self.symbols[c] = (
                    "cta_k[None, None, :]",
                    f"{col}[None, :, None]",
                )
            self.mask = f"({self.expr(kk)} < {k}) & ({self.expr(c)} < {n})"
            rhs_indices = [kk, c]
            if g.batched:
                rhs_indices.insert(0, self.unit)
                self.mask += f" & (cta_unit < {self.expr(self.plan.units)})"
            b = g.rhs.make_loader()(rhs_indices)
            if use_dot:
                dtype = triton_store_type(g.lhs.get_dtype())
                precision = "tf32" if g.allow_tf32 else "ieee"
                self.tile_code.writeline(
                    f"{acc} += tl.dot({a}.to({dtype}), {b}.to({dtype}), input_precision='{precision}')"
                )
            else:
                self.tile_code.writeline(f"{acc} += tl.sum({a} * {b}, 2)")
        self.symbols[r], self.symbols[c] = f"{row}[:, None]", f"{col}[None, :]"
        indices = [row_base + r, c]
        self.store_mask = f"({row}[:, None] < {rows}) & ({self.expr(row_base + r)} < {self.expr(m)}) & ({col}[None, :] < {n})"
        if g.batched:
            indices.insert(0, self.unit)
            self.store_mask += f" & (cta_unit < {self.expr(self.plan.units)})"
        self.local_size, self.logical_columns, self.padded_columns = bm * bn, n, bn
        self.local_shape = (bm, bn)
        index = stage.output.get_layout().as_fixed().make_indexer()(indices)
        self.tile_store(stage.output.get_name(), index, acc)

    def source_code(self):
        code = IndentedBuffer()
        with V.set_kernel_handler(self):
            code.splice(self.gen_common_triton_imports())
            code.splice(self.jit_lines())
            args, _, _, _ = self.args.python_argdefs()
            code.writeline(
                f"def {self.kernel_name}({', '.join(a.full_name() for a in args)}):"
            )
            with code.indent():
                code.splice(self.tile_code.getvalue())
        return code.getvalue()

    def benchmark_source(self):
        with V.set_kernel_handler(self):
            return (
                self.imports_for_benchmark_kernel()
                + "\n"
                + self.source
                + "\n"
                + self.codegen_kernel_benchmark(None).getvalue()
            ).replace(str(Placeholder.KERNEL_NAME), "triton_")

    def codegen(self, node):
        backend = node.scheduler.get_backend(node.get_device())._triton_scheduling
        with V.set_kernel_handler(self):
            for output in node.get_outputs():
                if output.get_name() in self.plan.outputs:
                    output.allocate()
                else:
                    self.removed_buffers.add(output.get_name())
            # Scheduling disables cubin storage; refresh metadata for AOT codegen.
            source = self.source_code()
            name = backend.define_kernel(source, list(node.get_nodes()), self)
            backend.codegen_comment(list(node.get_nodes()), name)
            self.call_kernel(name)
            V.graph.removed_buffers.update(self.removed_buffers)
        backend.free_buffers_in_scheduler()


class BenchmarkCall:
    def __init__(self, factories, tensors, mutable=(), key=()):
        self.factories = tuple(factories)
        self.tensors = tensors
        self.mutable = frozenset(mutable)
        self.key = key
        self.bound = tuple(factory(tensors) for factory in self.factories)
        self.originals = {id(tensors[name]): tensors[name] for name in self.mutable}
        self.pristine = {key: value.clone() for key, value in self.originals.items()}

    def __call__(self):
        for call in self.bound:
            call()

    def reset(self):
        for key, value in self.originals.items():
            value.copy_(self.pristine[key])

    def prepare(self, iterations):
        if not self.mutable:
            return [self] * iterations, lambda: None
        calls, copies = [], []
        for _ in range(iterations):
            replacements = {}
            for key, tensor in self.originals.items():
                value = torch.empty_strided(
                    tensor.size(),
                    tensor.stride(),
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                replacements[key] = value
                copies.append((value, self.pristine[key]))
            tensors = {
                name: replacements.get(id(tensor), tensor)
                for name, tensor in self.tensors.items()
            }
            calls.append(BenchmarkCall(self.factories, tensors))

        def reset():
            for target, source in copies:
                target.copy_(source)

        reset()
        return calls, reset


@contextlib.contextmanager
def preserve_codegen_state(nodes):
    from ..scheduler import SchedulerNode

    states = [
        (node, node.snapshot_loop_state())
        for node in nodes
        if isinstance(node, SchedulerNode)
    ]
    removed = V.graph.removed_buffers.copy()
    inplaced = V.graph.inplaced_to_remove.copy()
    unaligned = V.graph.unaligned_buffers.copy()
    try:
        for node, _ in states:
            node._body = copy.deepcopy(node._body)
        yield
    finally:
        for node, state in states:
            node.restore_loop_state(state)
        V.graph.removed_buffers = removed
        V.graph.inplaced_to_remove = inplaced
        V.graph.unaligned_buffers = unaligned


class CtaBenchmark:
    def __init__(self, scheduler, nodes, plan):
        self.scheduler = scheduler
        self.nodes = nodes
        self.plan = plan
        self.tensors: dict[str, torch.Tensor] = {}
        self.calls: dict[Any, BenchmarkCall] = {}
        self.backend = scheduler.get_backend(nodes[0].get_device())._triton_scheduling
        self.modules: list[Any] = []
        self.group_call(nodes[0])
        self.group_call(nodes[1])
        self.call = BenchmarkCall(
            [factory for node in nodes for factory in self.calls[node].factories],
            self.tensors,
            {name for node in nodes for name in self.calls[node].mutable},
            key=tuple(self.calls[node].key for node in nodes),
        )

    def cache_key(self, kernel):
        from ..codecache import CacheBase, code_hash, torch_key

        device = torch.cuda.get_device_properties(kernel.output_node.get_device())
        aliases = {}
        for name, tensor in sorted(self.tensors.items()):
            aliases.setdefault(id(tensor), []).append(name)
        return code_hash(
            json.dumps(
                {
                    "protocol": TimingCache.protocol,
                    "compiler": torch_key().hex(),
                    "system": CacheBase.get_system(),
                    "device": str(device),
                    "baseline": [self.calls[node].key for node in self.nodes],
                    "fused": kernel.benchmark_source(),
                    "inputs": [
                        (
                            name,
                            tuple(tensor.shape),
                            tensor.stride(),
                            str(tensor.dtype),
                            tensor.storage_offset(),
                        )
                        for name, tensor in sorted(self.tensors.items())
                    ],
                    "aliases": list(aliases.values()),
                    "precision": (
                        torch.get_float32_matmul_precision(),
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    ),
                },
                sort_keys=True,
            )
        )

    def tensor(self, name):
        if name not in self.tensors:
            self.tensors[name] = AlgorithmSelectorCache.benchmark_example_value(
                V.graph.get_buffer(name)
            )
        return self.tensors[name]

    def view(self, node, tensors=None):
        tensor = (
            self.tensor(node.get_name())
            if tensors is None
            else tensors[node.get_name()]
        )
        sizes = V.graph.sizevars.optimization_hints(node.get_size())
        strides = V.graph.sizevars.optimization_hints(node.get_stride())
        offset = V.graph.sizevars.optimization_hint(node.get_layout().offset)
        return tensor.as_strided(sizes, strides, offset)

    def bind(self, kernel, source):
        module = PyCodeCache.load(source)
        self.modules.append(module)
        sample_args = list(module.get_args())
        _, _, signature, _ = kernel.args.python_argdefs()
        mutable = set()
        for value in kernel.args.inplace_buffers.values():
            if isinstance(value, InplacedBuffer):
                name = value.other_names[0]
                mutable.add(name)
                for alias in value.other_names:
                    self.tensors[alias] = self.tensor(name)
        for i, arg in enumerate(signature):
            if isinstance(arg, TensorArg):
                self.tensor(arg.buffer)
            elif isinstance(arg, SizeArg):
                sample_args[i] = V.graph.sizevars.optimization_hint(arg.expr)
            else:
                raise UnsupportedCtaFusion("unsupported benchmark argument")

        def factory(tensors):
            args = list(sample_args)
            for i, arg in enumerate(signature):
                if isinstance(arg, TensorArg):
                    args[i] = tensors[arg.buffer]
            return lambda: module.call(args)

        call = BenchmarkCall((factory,), self.tensors, mutable, key=(source,))
        call()
        call.reset()
        return call

    def compile(self, kernel):
        call = self.bind(kernel, kernel.benchmark_source())
        launchers = self.modules[-1].triton_.launchers
        kernel.resources = [
            {
                "registers": entry.n_regs,
                "spills": entry.n_spills,
                "shared": entry.shared,
            }
            for entry in launchers
        ]
        device = torch.cuda.get_device_properties(kernel.output_node.get_device())
        for entry in launchers:
            if entry.n_spills or entry.shared > device.shared_memory_per_block_optin:
                raise UnsupportedCtaFusion("compiled resource limit")
        return call

    def group_call(self, node):
        from ..scheduler import FusedCtaSchedulerNode

        if node in self.calls:
            return self.calls[node]
        if isinstance(node, FusedCtaSchedulerNode):
            call = self.compile(node.kernel)
        elif len(node.get_nodes()) == 1 and descriptor(node) is not None:
            buf = node.node
            if isinstance(buf, ir.MultiTemplateBuffer):
                if (
                    config.test_configs.force_extern_kernel_in_multi_template
                    or self.scheduler._has_layout_conflict_for_template(buf)
                ):
                    choices = [
                        c for c in buf.choices if isinstance(c, ExternKernelCaller)
                    ]
                    if not choices:
                        raise UnsupportedCtaFusion("no layout-compatible baseline")
                    timings = buf.choice_timings()
                    choice = min(choices, key=timings.__getitem__)
                else:
                    choice, _ = buf.get_min_choice()
                if isinstance(choice, ExternKernelCaller):
                    fn = choice.to_callable()
                    for x in choice.input_nodes:
                        self.tensor(x.get_name())
                    self.tensor(buf.get_name())

                    def factory(tensors):
                        args = [self.view(x, tensors) for x in choice.input_nodes]
                        out = tensors[buf.get_name()]
                        return lambda: fn(*args, out=out)
                else:
                    for x in choice.input_nodes:
                        self.tensor(x.get_name())
                    self.tensor(buf.get_name())

                    def factory(tensors):
                        args = [self.view(x, tensors) for x in choice.input_nodes]
                        return choice.bmreq.make_run_fn(
                            *args, out=tensors[buf.get_name()]
                        )

                call = BenchmarkCall((factory,), self.tensors, key=(choice.hash_key(),))
            elif isinstance(buf, ir.ExternKernel):
                g = descriptor(node)
                fn = torch.bmm if g.batched else torch.mm
                for x in (g.lhs, g.rhs, buf):
                    self.tensor(x.get_name())

                def factory(tensors):
                    a, b, out = (
                        self.view(g.lhs, tensors),
                        self.view(g.rhs, tensors),
                        tensors[buf.get_name()],
                    )
                    return lambda: fn(a, b, out=out)

                call = BenchmarkCall(
                    (factory,), self.tensors, key=(fn.__name__, str(buf.get_layout()))
                )
            else:
                call = self.codegen_group(node)
        else:
            call = self.codegen_group(node)
        call()
        call.reset()
        self.calls[node] = call
        return call

    def codegen_group(self, node):
        nodes = list(node.get_nodes())
        with (
            preserve_codegen_state(nodes),
            V.graph.set_current_device(node.get_device()),
            config.patch(benchmark_kernel=True),
        ):
            if node.is_template():
                prologue, template, epilogue = node.get_prologue_template_epilogue(
                    nodes
                )
                kernel, render = template.node.make_kernel_render(template.node)
                if not isinstance(kernel, TritonTemplateKernel):
                    raise UnsupportedCtaFusion("non-Triton fused baseline")
                source = self.backend._codegen_single_template(
                    kernel, render, template, epilogue, prologue, only_gen_src_code=True
                )
            else:
                _, (numel, rnumel) = node.group
                schedule = self.backend.generate_node_schedule(nodes, numel, rnumel)
                features = SIMDKernelFeatures(schedule, numel, rnumel)
                tiling, scores = self.backend.get_tiling_and_scores(
                    schedule, numel, rnumel, None
                )
                kernel = self.backend.kernel_type(
                    tiling, features=features, tiling_scores=scores
                )
                self.backend.codegen_node_schedule_with_kernel(schedule, kernel)
                with V.set_kernel_handler(kernel):
                    source = kernel.codegen_kernel()
        source = source.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return self.bind(kernel, source)
