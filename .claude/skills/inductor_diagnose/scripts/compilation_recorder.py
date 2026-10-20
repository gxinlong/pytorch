from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import sympy
from graph_records import graph_record, RegionError, save_graph, write_json

import torch


def describe_value(value):
    from torch._inductor import ir

    while isinstance(value, (ir.TensorBox, ir.StorageBox)):
        value = value.data
    if value is None:
        return {"kind": "literal", "value": None}
    if isinstance(value, (sympy.Expr, int, float, bool)):
        return {"kind": "scalar", "expression": str(value)}
    if isinstance(value, ir.BaseView) and not isinstance(value, ir.ReinterpretView):
        return {"kind": "unsupported_view", "type": type(value).__name__}
    if not isinstance(value, (ir.Buffer, ir.ReinterpretView)):
        return {"kind": "unmaterialized", "type": type(value).__name__}
    layout = value.get_layout()
    if not hasattr(layout, "stride") or not value.get_name():
        return {"kind": "unmaterialized", "type": type(value).__name__}
    return {
        "kind": "tensor",
        "buffer": value.get_name(),
        "shape": [str(v) for v in layout.size],
        "stride": [str(v) for v in layout.stride],
        "offset": str(layout.offset),
        "dtype": str(layout.dtype),
        "device": str(layout.device),
        "layout": type(layout).__name__,
    }


def scheduler_records(nodes):
    from torch._inductor.virtualized import V

    rows = []
    for group in nodes:
        members = []
        for node in group.get_nodes():
            operation = getattr(node, "node", None)
            members.append(
                {
                    "operation": node.get_name(),
                    "buffers": sorted(node.get_buffer_names()),
                    "origins": sorted(
                        origin.name
                        for origin in getattr(operation, "origins", ())
                        if isinstance(origin, torch.fx.Node)
                    ),
                    "exact_origin": getattr(
                        getattr(operation, "origin_node", None), "name", None
                    ),
                    "type": type(node).__name__,
                }
            )
        coverage = {}
        for dep in group.read_writes.writes:
            buffer = V.graph.name_to_buffer.get(dep.name)
            if getattr(dep, "mode", None) is not None:
                coverage[dep.name] = "partial_or_atomic"
            elif buffer is not None and hasattr(dep, "size"):
                difference = sympy.simplify(
                    sympy.prod(dep.size) - sympy.prod(buffer.get_size())
                )
                coverage[dep.name] = (
                    "full" if difference == 0 else "partial_or_unresolved"
                )
            else:
                coverage[dep.name] = "opaque"
        rows.append(
            {
                "type": type(group).__name__,
                "members": members,
                "reads": sorted({dep.name for dep in group.read_writes.reads}),
                "writes": sorted({dep.name for dep in group.read_writes.writes}),
                "write_coverage": coverage,
            }
        )
    return rows


class CompilationRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.values = defaultdict(dict)
        self.calls = defaultdict(list)
        self.views = defaultdict(dict)
        self.graphs = []
        self.stack = ExitStack()
        self.current_call = {}
        self.lowered = {}
        self.invocations = 0
        self.schedule_by_key = {}

    def __enter__(self):
        from torch._inductor import debug
        from torch._inductor.codegen import cpp as cpp_codegen, triton as triton_codegen
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.codegen.debug_utils import DebugPrinterManager
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V

        self.directory.mkdir(parents=True, exist_ok=False)
        original_graph = debug.DebugFormatter.fx_graph_transformed
        original_run = GraphLowering.run_node
        original_codegen = GraphLowering.codegen
        original_args = DebugPrinterManager.set_printer_args
        original_view = CppWrapperCpu.codegen_reinterpret_view
        original_provenance = debug.set_kernel_post_grad_provenance_tracing

        def provenance(node_schedule, kernel_name, is_extern=False):
            handle = original_provenance(node_schedule, kernel_name, is_extern)
            if handle is not None:
                if is_extern:
                    rw = node_schedule.get_read_writes()
                    rows = [
                        {
                            "type": type(node_schedule).__name__,
                            "members": [
                                {
                                    "operation": node_schedule.get_operation_name(),
                                    "buffers": [
                                        buffer.get_name()
                                        for buffer in node_schedule.get_outputs()
                                    ],
                                    "origins": sorted(
                                        node.name for node in node_schedule.origins
                                    ),
                                    "exact_origin": getattr(
                                        node_schedule.origin_node, "name", None
                                    ),
                                }
                            ],
                            "reads": sorted({dep.name for dep in rw.reads}),
                            "writes": sorted({dep.name for dep in rw.writes}),
                        }
                    ]
                else:
                    nodes = [
                        node for node in node_schedule if hasattr(node, "read_writes")
                    ]
                    rows = scheduler_records(nodes)
                self.schedule_by_key[f"{kernel_name}:{handle}"] = rows
            return handle

        def transformed(formatter, gm, inputs):
            original_graph(formatter, gm, inputs)
            graph_dir = self.directory / f"graph_{len(self.graphs)}"
            identity = save_graph(gm, inputs, graph_dir)
            self.graphs.append(
                {
                    "directory": str(graph_dir),
                    "identity": identity,
                    "record": graph_record(gm),
                }
            )

        def run_node(graph, node):
            result = original_run(graph, node)
            self.values[id(graph)][node.name] = result
            return result

        def reinterpret(wrapper, data, size, stride, offset, writeline, dtype=None):
            result = original_view(
                wrapper, data, size, stride, offset, writeline, dtype=dtype
            )
            self.views[id(V.graph)][result] = {
                "kind": "tensor",
                "buffer": data.get_name(),
                "shape": list(map(str, size)),
                "stride": list(map(str, stride)),
                "offset": str(offset),
                "dtype": str(dtype or data.get_dtype()),
                "device": str(data.get_device()),
            }
            return result

        def set_args(
            printer, args, kernel_name, arg_signatures, kernel, kernel_type=None
        ):
            original_args(
                printer, args, kernel_name, arg_signatures, kernel, kernel_type
            )
            graph = V.graph
            if not graph.cpp_wrapper:
                return
            rows = self.calls[id(graph)]
            arguments = []
            for index, arg in enumerate(printer.args_to_print_or_save):
                expression = str(arg)
                value = self.views[id(graph)].get(expression)
                if value is None:
                    buffer = graph.name_to_buffer.get(expression)
                    if buffer is None:
                        buffer = graph.graph_inputs.get(expression)
                    if buffer is not None:
                        value = describe_value(buffer)
                arguments.append(
                    {
                        "expression": expression,
                        **(value or {"kind": "scalar"}),
                        "signature": str(arg_signatures[index])
                        if arg_signatures and index < len(arg_signatures)
                        else None,
                    }
                )
            call = {
                "id": f"call_{len(rows):05d}",
                "ordinal": len(rows),
                "name": kernel_name,
                "kernel_type": kernel_type,
                "args": arguments,
            }
            rows.append(call)
            self.current_call[id(printer)] = call

        def codegen(graph):
            self.calls[id(graph)] = []
            self.views[id(graph)] = {}
            result = original_codegen(graph)
            if not graph.cpp_wrapper:
                return result
            self.invocations += 1
            provenance = copy.deepcopy(debug.dump_inductor_provenance_info())
            buckets = defaultdict(list)
            for key in provenance.get("cppCodeToPost", {}):
                base, separator, handle = key.rpartition(":")
                buckets[base if separator and handle.isdecimal() else key].append(key)
            used = defaultdict(int)
            calls = self.calls[id(graph)]
            for call in calls:
                candidates = buckets[call["name"]]
                index = used[call["name"]]
                if index < len(candidates):
                    call["provenance_key"] = candidates[index]
                    members = self.schedule_by_key.get(candidates[index], [])
                    call["reads"] = sorted(
                        {name for row in members for name in row["reads"]}
                    )
                    call["writes"] = sorted(
                        {name for row in members for name in row["writes"]}
                    )
                    call["write_coverage"] = {
                        name: coverage
                        for row in members
                        for name, coverage in row.get("write_coverage", {}).items()
                    }
                used[call["name"]] += 1
            bodies = getattr(graph.wrapper_code, "_kernel_name_to_body", {})
            fingerprints = {}
            (self.directory / "kernels").mkdir(exist_ok=True)
            for name, body in bodies.items():
                normalized = re.sub(rf"\b{re.escape(name)}\b", "KERNEL", body)
                body_path = self.directory / "kernels" / f"{name}.py"
                body_path.write_text(body)
                fingerprints[name] = {
                    "source": str(body_path),
                    "fingerprint_version": 2,
                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "normalized_sha256": hashlib.sha256(
                        normalized.encode()
                    ).hexdigest(),
                    "configuration_lines": [
                        line.strip()
                        for line in body.splitlines()
                        if any(
                            key in line
                            for key in (
                                "BLOCK_",
                                "num_warps",
                                "num_stages",
                                "tl.constexpr =",
                            )
                        )
                    ],
                }
            records = {
                name: describe_value(value)
                for name, value in self.values[id(graph)].items()
            }
            wrapper = result[0].value
            source_path = self.directory / f"wrapper_{self.invocations}.cpp"
            source_path.write_text(wrapper)
            self.lowered = {
                "provenance": provenance,
                "calls": calls,
                "values": records,
                "scheduler": scheduler_records(graph.scheduler.nodes),
                "kernels": fingerprints,
                "wrapper": str(source_path),
                "mutated_inputs": sorted(graph.mutated_inputs),
                "removed_buffers": sorted(graph.removed_buffers),
                "input_names": list(graph.graph_inputs),
                "graph_inputs": {
                    name: describe_value(value)
                    for name, value in graph.graph_inputs.items()
                },
                "graph_outputs": [
                    describe_value(value) for value in graph.graph_outputs
                ],
                "range_constraints": {
                    str(symbol): {"min": str(bounds.lower), "max": str(bounds.upper)}
                    for symbol, bounds in graph.sizevars.shape_env.var_to_range.items()
                },
                "call_schedules": {
                    key: self.schedule_by_key.get(key, [])
                    for key in provenance.get("cppCodeToPost", {})
                },
            }
            return result

        for owner, name, replacement in (
            (debug.DebugFormatter, "fx_graph_transformed", transformed),
            (GraphLowering, "run_node", run_node),
            (GraphLowering, "codegen", codegen),
            (CppWrapperCpu, "codegen_reinterpret_view", reinterpret),
            (DebugPrinterManager, "set_printer_args", set_args),
            (debug, "set_kernel_post_grad_provenance_tracing", provenance),
            (triton_codegen, "set_kernel_post_grad_provenance_tracing", provenance),
            (cpp_codegen, "set_kernel_post_grad_provenance_tracing", provenance),
        ):
            self.stack.enter_context(patch.object(owner, name, replacement))
        return self

    def __exit__(self, kind, value, traceback):
        self.stack.close()
        if kind is None:
            if len(self.graphs) != 1 or not self.lowered:
                raise RegionError(
                    "COMPILATION_RECORD",
                    "Expected one recorded post-grad graph and C++ lowering",
                    graphs=len(self.graphs),
                    lowerings=self.invocations,
                )
            record = {"graph": self.graphs[0], **self.lowered}
            write_json(self.directory / "compilation.json", record)
