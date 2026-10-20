import ast
import sys
from pathlib import Path


class ComboTrace:
    """Record executed statements and list transitions without replaying predicates."""

    TARGETS = {
        "ForeachKernelSchedulerNode.combinable_nodes",
        "ForeachKernelSchedulerNode._default_group_nodes_for_combo_kernels",
        "Scheduler.create_combo_kernel_nodes",
        "Scheduler._distance_windows",
        "Scheduler.speedup_by_combo_kernel",
        "ComboKernel._update_partition",
        "ComboKernel._base_horizontal_partition",
        "_default_custom_combo_kernel_horizontal_partition",
    }
    KEYS = {
        "nodes",
        "members",
        "window",
        "ordered",
        "node",
        "n",
        "num",
        "device",
        "context_nodes",
        "context_groups",
        "max_num_nodes",
        "max_distance",
        "idx",
        "window_min",
        "memory_check",
        "abs_thr_gb",
        "pct_thr",
        "partition_state",
        "node_rw_count",
        "node_info",
        "tiled_groups",
        "ndim",
        "mixed_sizes",
        "y_elem",
        "read_write_count",
        "ms1",
        "ms2",
        "ms2_clone",
        "small_kernel",
        "min_ms_fused",
        "group_per_dim",
        "i",
        "max_dims",
    }

    def __init__(self, source, emit, serialize, errors, parent_id):
        self.source = Path(source)
        self.emit = emit
        self.serialize = serialize
        self.errors = errors
        self.parent_id = parent_id
        self.previous = None
        self.pending = {}
        self.points = {}
        self.missing_targets = set(self.TARGETS)
        for relative in (
            "torch/_inductor/scheduler.py",
            "torch/_inductor/codegen/triton_combo_kernel.py",
        ):
            path = self.source / relative
            text = path.read_text()
            module = ast.parse(text)
            for owner in module.body:
                functions = owner.body if isinstance(owner, ast.ClassDef) else [owner]
                for fn in functions:
                    if not isinstance(fn, ast.FunctionDef):
                        continue
                    name = (
                        f"{owner.name}.{fn.name}"
                        if isinstance(owner, ast.ClassDef)
                        else fn.name
                    )
                    if name not in self.TARGETS:
                        continue
                    self.points[(str(path), name)] = self.index(fn, text)
                    self.missing_targets.discard(name)

    def index(self, function, text):
        points = {}
        for node in ast.walk(function):
            point = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                variable = node.targets[0].id
                if variable == "filtered_nodes" or (
                    isinstance(node.value, ast.ListComp)
                    and function.name
                    == "_default_custom_combo_kernel_horizontal_partition"
                ):
                    point = {
                        "kind": "filter"
                        if variable == "filtered_nodes"
                        else "partition_stage",
                        "variable": variable,
                        "condition": ast.get_source_segment(text, node.value),
                    }
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "grouped_nodes"
                    and call.func.attr == "extend"
                ):
                    point = {
                        "kind": "chunk",
                        "variable": "grouped_nodes",
                        "condition": ast.get_source_segment(text, node),
                    }
            if point:
                point.update(line=node.lineno, end=node.end_lineno)
                points.setdefault(node.lineno, []).append(point)
            if isinstance(node, ast.If):
                for taken, body in ((True, node.body), (False, node.orelse)):
                    if body:
                        points.setdefault(body[0].lineno, []).append(
                            {
                                "kind": "branch",
                                "taken": taken,
                                "condition": ast.get_source_segment(text, node.test),
                                "line": node.lineno,
                            }
                        )
            if isinstance(node, ast.Yield):
                points.setdefault(node.lineno, []).append(
                    {
                        "kind": "yield",
                        "line": node.lineno,
                        "variable": node.value.id
                        if isinstance(node.value, ast.Name)
                        else None,
                    }
                )
        return points

    def write(self, frame, kind, data):
        context = {
            k: self.serialize(v) for k, v in frame.f_locals.items() if k in self.KEYS
        }
        self.emit(
            {
                "event": "combo_" + kind,
                "function": frame.f_code.co_qualname,
                "file": str(Path(frame.f_code.co_filename).relative_to(self.source)),
                "line": data.get("line", frame.f_lineno),
                "parent_id": self.parent_id(frame),
                "evidence": "executed_statement",
                "context": context,
                **data,
            }
        )

    def finish_assignment(self, frame, pending):
        after = self.serialize(frame.f_locals.get(pending["variable"], []))
        self.write(frame, pending["kind"], {**pending, "after": after})

    def trace(self, frame, event, result):
        points = self.points.get((frame.f_code.co_filename, frame.f_code.co_qualname))
        if points is None:
            return None
        try:
            pending = self.pending.get(id(frame))
            if pending and (
                event == "return"
                or (
                    event == "line"
                    and not pending["line"] <= frame.f_lineno <= pending["end"]
                )
            ):
                self.finish_assignment(frame, self.pending.pop(id(frame)))
            if event == "line":
                for point in points.get(frame.f_lineno, []):
                    kind = point["kind"]
                    if kind in ("filter", "partition_stage", "chunk"):
                        if id(frame) not in self.pending:
                            variable = point["variable"]
                            before = frame.f_locals.get(
                                variable, frame.f_locals.get("nodes", [])
                            )
                            self.pending[id(frame)] = {
                                **point,
                                "before": self.serialize(before),
                            }
                    elif kind == "branch":
                        self.write(frame, kind, point)
                    elif kind == "yield":
                        self.write(
                            frame,
                            kind,
                            {
                                **point,
                                "yielded": self.serialize(
                                    frame.f_locals.get(point["variable"])
                                ),
                            },
                        )
            elif event == "exception":
                self.write(
                    frame,
                    "exception",
                    {"exception_type": result[0].__name__, "message": str(result[1])},
                )
        except Exception as exc:
            self.errors.append(
                {
                    "function": frame.f_code.co_qualname,
                    "event": "combo_trace",
                    "error": repr(exc),
                }
            )
        return self.trace

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.previous is not None:
            raise RuntimeError(
                "An existing line tracer is active; refusing to replace it"
            )
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)
