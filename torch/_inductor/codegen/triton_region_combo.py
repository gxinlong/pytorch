from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import sympy

from .. import config
from ..select_algorithm import SymbolicGridFn, TritonTemplateKernel
from ..utils import IndentedBuffer, Placeholder, sympy_product
from ..virtualized import V
from .common import SizeArg, TensorArg
from .triton import texpr


if TYPE_CHECKING:
    from ..scheduler import BaseSchedulerNode


class UnsupportedRegionCombo(Exception):
    pass


@SymbolicGridFn
def region_combo_grid(total, meta):
    return (total, 1, 1)


@dataclasses.dataclass(frozen=True)
class RegionKernel:
    node: BaseSchedulerNode
    kernel: TritonTemplateKernel
    source: str
    choice: str


def export_helper(source: str, name: str) -> tuple[str, str]:
    """Export an explicitly opted-in renderer without rewriting its body."""
    module = ast.parse(source.replace(str(Placeholder.KERNEL_NAME), "region_kernel"))
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise UnsupportedRegionCombo("template must export one function")
    function = functions[0]
    imports = []
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif node is not function:
            raise UnsupportedRegionCombo("template module has extra definitions")
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith(("atomic_", "rand"))
            or node.attr in ("program_id", "num_programs", "inline_asm_elementwise")
        ):
            raise UnsupportedRegionCombo(f"unsupported template operation: {node.attr}")
    function.name = name
    function.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    function.args.args.extend(
        ast.arg(arg=f"combo_{kind}_{axis}")
        for kind in ("pid", "grid")
        for axis in range(3)
    )
    return "\n".join(imports), ast.unparse(ast.fix_missing_locations(function))


def executable_identity(source: str) -> tuple[str, ...]:
    module = ast.parse(source.replace(str(Placeholder.KERNEL_NAME), "region_kernel"))
    return tuple(
        ast.dump(node.args) + "".join(ast.dump(statement) for statement in node.body)
        for node in module.body if isinstance(node, ast.FunctionDef)
    )


class RegionComboKernel(TritonTemplateKernel):
    def __init__(self, regions: tuple[RegionKernel, ...]) -> None:
        first = regions[0].kernel
        super().__init__(
            kernel_name=str(Placeholder.KERNEL_NAME),
            input_nodes=(),
            output_node=first.output_node,
            defines="",
            num_stages=first.num_stages,
            num_warps=first.num_warps,
            grid_fn=region_combo_grid,
            meta={},
            call_sizes=[],
            index_dtype_override="tl.int64",
        )
        self.regions = regions
        self.helpers = []
        self.helper_imports = []
        self.dispatch = IndentedBuffer()
        self.resources = []
        total = sympy.S.Zero
        with V.set_kernel_handler(self):
            self.dispatch.writeline("combo_pid = tl.program_id(0).to(tl.int64)")
            for index, region in enumerate(regions):
                kernel = region.kernel
                if (kernel.num_warps, kernel.num_stages) != (
                    self.num_warps, self.num_stages
                ):
                    raise UnsupportedRegionCombo("incompatible launch configuration")
                if (
                    not kernel.combo_supported
                    or kernel.args.workspace_args
                    or kernel.args.inplace_buffers
                    or kernel.num_consumer_groups
                    or kernel.num_buffers_warp_spec
                    or kernel.tma_store
                    or kernel.tma_load_for_template_epilogue
                    or kernel.get_unfused_epilogues()
                ):
                    raise UnsupportedRegionCombo("unsupported template launch contract")
                if not isinstance(kernel.grid_fn, SymbolicGridFn):
                    raise UnsupportedRegionCombo("non-symbolic grid")
                grid = kernel.grid_fn.sympy_call(*kernel.call_sizes, kernel.meta)
                if len(grid) != 3:
                    raise UnsupportedRegionCombo("grid must have three dimensions")
                grids = [texpr(self.rename_indexing(sympy.sympify(g))) for g in grid]
                count = sympy_product(grid)
                begin = texpr(self.rename_indexing(total))
                end = texpr(self.rename_indexing(total + count))
                helper_name = f"region_combo_member_{index}"
                imports, helper = export_helper(region.source, helper_name)
                self.helper_imports.append(imports)
                self.helpers.append(helper)
                arguments = []
                _, _, signature, _ = kernel.args.python_argdefs()
                for arg in signature:
                    if isinstance(arg, TensorArg):
                        if arg.buffer in kernel.args.output_buffers:
                            arguments.append(self.args.output(arg.buffer))
                        else:
                            arguments.append(self.args.input(arg.buffer))
                    elif isinstance(arg, SizeArg):
                        arguments.append(texpr(self.rename_indexing(arg.expr)))
                    else:
                        raise UnsupportedRegionCombo("unsupported kernel argument")
                self.dispatch.writeline(
                    f"{'if' if index == 0 else 'elif'} combo_pid < {end}:"
                )
                with self.dispatch.indent():
                    self.dispatch.writeline(f"local_pid = combo_pid - ({begin})")
                    # A zero-size member never enters its branch. Clamp divisors so
                    # specialization of another member cannot introduce division by zero.
                    gx, gy, gz = grids
                    arguments.extend([
                        f"local_pid % tl.maximum({gx}, 1)",
                        f"(local_pid // tl.maximum({gx}, 1)) % tl.maximum({gy}, 1)",
                        f"local_pid // tl.maximum(({gx}) * ({gy}), 1)",
                        gx, gy, gz,
                    ])
                    self.dispatch.writeline(f"{helper_name}({', '.join(arguments)})")
                total += count
                self.removed_buffers.update(kernel.removed_buffers)
                self.inplaced_to_remove.update(kernel.inplaced_to_remove)
            self.call_sizes = [total]
            _, _, signature, _ = self.args.python_argdefs()
            # All supported parameters are pointers or scalar integers, at most 8 bytes.
            if len(signature) > min(config.combo_kernel_max_num_args, 512):
                raise UnsupportedRegionCombo("kernel parameter limit")
            if not V.graph.sizevars.statically_known_leq(total, 2147483647):
                raise UnsupportedRegionCombo("flattened grid limit is not proven")
        self.source = self.source_code()
        self.measured_identity = executable_identity(self.source)
        self.measured_launch = (self.num_warps, self.num_stages)

    def source_code(self) -> str:
        code = IndentedBuffer()
        with V.set_kernel_handler(self):
            code.splice(self.gen_common_triton_imports())
            for imports in dict.fromkeys(self.helper_imports):
                code.splice(imports)
            for helper in self.helpers:
                code.splice(helper)
            code.splice(self.jit_lines())
            args, _, _, _ = self.args.python_argdefs()
            code.writeline(
                f"def {self.kernel_name}({', '.join(a.full_name() for a in args)}):"
            )
            with code.indent():
                code.splice(self.dispatch.getvalue())
        return code.getvalue()

    def benchmark_source(self) -> str:
        with V.set_kernel_handler(self):
            return (
                self.imports_for_benchmark_kernel() + "\n" + self.source + "\n"
                + self.codegen_kernel_benchmark(None).getvalue()
            ).replace(str(Placeholder.KERNEL_NAME), "triton_")

    def codegen(self, node) -> None:
        from ..region_combo import report

        backend = node.scheduler.get_backend(node.get_device())._triton_scheduling
        with V.set_kernel_handler(self):
            for member in node.get_nodes():
                member.mark_run()
            # Only wrapper/AOT metadata changes here; the measured body is frozen.
            source = self.source_code()
            if (
                executable_identity(source) != self.measured_identity
                or (self.num_warps, self.num_stages) != self.measured_launch
            ):
                raise AssertionError("region combo changed after benchmarking")
            name = backend.define_kernel(source, list(node.get_nodes()), self)
            backend.codegen_comment(list(node.get_nodes()), name)
            self.call_kernel(name)
            V.graph.removed_buffers.update(self.removed_buffers)
            V.graph.inplaced_to_remove.update(self.inplaced_to_remove)
        report(node.members, "emitted", kernel=name)
        backend.free_buffers_in_scheduler()
