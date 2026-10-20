import json
import math
import sys
import threading
from pathlib import Path

from diagnostic_combo import ComboTrace


class DecisionObserver:
    """Observe existing calls; never replay a compiler predicate or benchmark."""

    def __init__(self, directory, source):
        from torch._inductor.scheduler import BaseSchedulerNode

        self.node_type = BaseSchedulerNode
        self.directory = Path(directory)
        self.source = Path(source)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / "decisions.jsonl").open("x")
        self.sequence = 0
        self.frames = {}
        self.errors = []
        self.previous = None
        self.round = 0
        self.combo_trace = ComboTrace(
            self.source, self.emit, self.value, self.errors, self.frame_parent_id
        )
        self.names = {
            "_init",
            "create_scheduler_node",
            "create_foreach_nodes",
            "fuse_nodes",
            "fuse_nodes_once",
            "get_possible_fusions",
            "get_possible_fusions_with_highest_priority",
            "unfusable_node",
            "can_fuse",
            "_can_fuse",
            "speedup_by_fusion",
            "fuse_two_nodes",
            "fuse_if_speedup",
            "_finish_pending_fusions",
            "_evaluate_pending_template_fusions",
            "_handle_template_overlap",
            "will_fusion_create_cycle",
            "_has_layout_conflict_for_template",
            "finalize_multi_template_buffers",
            "_replace_node",
            "create_combo_kernel_nodes",
            "combinable_nodes",
            "_topological_sort_nodes",
            "_default_group_nodes_for_combo_kernels",
            "speedup_by_combo_kernel",
            "_register_accept",
            "horizontal_partition",
            "_base_horizontal_partition",
            "_update_partition",
            "_default_custom_combo_kernel_horizontal_partition",
            "codegen_template",
            "codegen_combo_kernel",
            "_codegen_nodes",
            "codegen_node_schedule_with_kernel",
            "choice_timings",
            "get_min_choice",
            "finalize_as_triton_caller",
            "finalize_as_triton_callers",
            "do_autotuning",
            "benchmark_choices",
            "benchmark_when_ready",
            "benchmark_fused_nodes",
            "benchmark_combo_kernel",
            "lookup",
            "generate_combo_kernel_code",
            "_try_combo_with_halving",
            "_combo_fits_memory",
            "__call__",
            "log_ir_pre_fusion",
            "log_ir_post_fusion",
            "run_node",
        }

    def node(self, value):
        if not isinstance(value, self.node_type):
            return None
        members = []
        for node in value.get_nodes():
            ir = getattr(node, "node", None)
            members.append(
                {
                    "name": node.get_name(),
                    "type": type(node).__name__,
                    "ir_type": type(ir).__name__,
                    "origins": sorted(
                        n.name for n in getattr(ir, "origins", ()) if hasattr(n, "name")
                    ),
                    "layout": str(getattr(ir, "layout", None)),
                }
            )
        rw = getattr(value, "read_writes", None)
        return {
            "name": value.get_name(),
            "type": type(value).__name__,
            "members": members,
            "reads": sorted(str(dep) for dep in rw.reads) if rw else [],
            "writes": sorted(str(dep) for dep in rw.writes) if rw else [],
            "read_buffers": sorted({dep.name for dep in rw.reads}) if rw else [],
            "write_buffers": sorted({dep.name for dep in rw.writes}) if rw else [],
            "template": value.is_template(),
            "extern": value.is_extern(),
            "reduction": value.is_reduction(),
            "foreach": value.is_foreach(),
        }

    def value(self, value, depth=0):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        node = self.node(value)
        if node is not None:
            return node
        if depth > 7:
            return {"unserialized": type(value).__name__, "text": str(value)[:500]}
        if type(value).__name__ == "NodeInfo":
            return {
                k: self.value(getattr(value, k), depth + 1)
                for k in ("tiling", "numel", "rnumel", "is_persistent_reduction")
            }
        if type(value).__name__ == "PartitionState":
            return {
                k: self.value(getattr(value, k), depth + 1)
                for k in ("partitions", "cur_partition", "cur_count")
            }
        if (
            isinstance(value, (list, tuple, set))
            or type(value).__name__ == "OrderedSet"
        ):
            return [self.value(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {
                (
                    k.get_name()
                    if isinstance(k, self.node_type)
                    else str(getattr(k, "name", k))
                ): self.value(v, depth + 1)
                for k, v in value.items()
            }
        if type(value).__name__ == "FusionResult":
            return {
                "should_fuse": value.should_fuse,
                "pending": value.callable_fn is not None,
                "future_submitted": value.future is not None,
            }
        if type(value).__name__ == "PendingFusion":
            return {
                "node1": self.node(value.node1),
                "node2": self.node(value.node2),
                "future_submitted": value.future is not None,
            }
        if "Caller" in type(value).__name__:
            return {
                "type": type(value).__name__,
                "name": str(getattr(value, "name", "")),
            }
        if type(value).__name__ == "Node" and hasattr(value, "op"):
            return {"name": value.name, "op": value.op, "target": str(value.target)}
        return str(value)[:1500]

    def emit(self, data):
        self.sequence += 1
        row = {"seq": self.sequence, **data}
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.sequence % 100 == 0:
            self.stream.flush()
        return self.sequence

    def frame_parent_id(self, frame):
        while frame is not None:
            if id(frame) in self.frames:
                return self.frames[id(frame)]["seq"]
            frame = frame.f_back
        return None

    def profile(self, frame, event, result):
        if event not in ("call", "return"):
            return
        code = frame.f_code
        if (
            code.co_name not in self.names
            or "/torch/_inductor/" not in code.co_filename
        ):
            return
        own = frame.f_locals.get("self")
        if code.co_name == "_init" and type(own).__name__ != "Scheduler":
            return
        if code.co_name == "__call__" and type(own).__name__ != "WhyNoFuse":
            return
        if code.co_name == "lookup" and not code.co_filename.endswith("codecache.py"):
            return
        if code.co_name == "run_node" and not code.co_filename.endswith("graph.py"):
            return
        try:
            relative = str(Path(code.co_filename).relative_to(self.source))
            row = {
                "event": event,
                "function": code.co_qualname,
                "file": relative,
                "line": frame.f_lineno,
                "thread": threading.get_ident(),
                "round": self.round,
            }
            if event == "call":
                if code.co_name == "fuse_nodes_once":
                    self.round += 1
                    row["round"] = self.round
                parent = frame.f_back
                while parent is not None and id(parent) not in self.frames:
                    parent = parent.f_back
                parent_row = self.frames.get(id(parent), {})
                row["parent_id"] = parent_row.get("seq")
                keys = {
                    "node1",
                    "node2",
                    "nodes",
                    "node_list",
                    "subkernel_nodes",
                    "node_schedule",
                    "reason",
                    "args",
                    "accepted",
                    "is_reorder_round",
                    "multi_node",
                    "n",
                    "template_node",
                    "prologue_nodes",
                    "epilogue_nodes",
                    "possible_fusions",
                    "template_fusion_candidates",
                    "pending_fusions",
                    "node",
                    "buffer",
                    "combo_node",
                    "combo_kernel_node",
                    "node_info_map",
                    "node_info",
                    "partition_state",
                    "node_rw_count",
                    "custom_algorithm",
                    "custom_part_algorithm",
                    "enable_autotune",
                    "mixed_sizes",
                    "per_subkernel_blocks",
                    "only_gen_src_code",
                }
                row["args"] = {
                    k: self.value(v) for k, v in frame.f_locals.items() if k in keys
                }
                pair = [row["args"].get(k) for k in ("node1", "node2")]
                if all(isinstance(n, dict) and "members" in n for n in pair):
                    row["pair"] = [n["name"] for n in pair]
                    row["pair_members"] = [
                        [m["name"] for m in n["members"]] for n in pair
                    ]
                elif parent_row.get("pair"):
                    row["pair"] = parent_row["pair"]
                    row["pair_members"] = parent_row.get("pair_members")
                if code.co_name == "__call__":
                    row["pair"] = [own.name1, own.name2]
                if code.co_name in {
                    "get_min_choice",
                    "choice_timings",
                    "finalize_as_triton_caller",
                }:
                    row["buffer"] = getattr(own, "name", None)
                if code.co_name in {
                    "fuse_nodes",
                    "create_combo_kernel_nodes",
                    "finalize_multi_template_buffers",
                    "create_foreach_nodes",
                }:
                    row["schedule"] = [self.node(n) for n in own.nodes]
                row["seq"] = self.emit(row)
                self.frames[id(frame)] = row
            else:
                call = self.frames.pop(id(frame), {})
                row.update(
                    call_id=call.get("seq"),
                    parent_id=call.get("parent_id"),
                    pair=call.get("pair"),
                    pair_members=call.get("pair_members"),
                )
                row["result"] = self.value(result)
                keys = {
                    "min_ms_fused",
                    "min_ms_unfused",
                    "min_node_unfused",
                    "best_choice",
                    "choice",
                    "timings",
                    "new_timings",
                    "cache_hit",
                    "expected_layout",
                    "layout",
                    "inp_name",
                    "template_fusion_candidates",
                    "fusions_to_remove",
                    "multi_node",
                    "partition_state",
                    "node_rw_count",
                    "small_kernel",
                    "max_num_nodes",
                    "max_distance",
                    "memory_check",
                    "abs_thr_gb",
                    "pct_thr",
                    "kernel_code_list",
                }
                row["details"] = {
                    k: self.value(v)
                    for k, v in frame.f_locals.items()
                    if k.startswith("ms") or k in keys
                }
                if code.co_name in {
                    "create_combo_kernel_nodes",
                    "finalize_multi_template_buffers",
                    "create_foreach_nodes",
                    "_init",
                }:
                    row["schedule"] = [self.node(n) for n in own.nodes]
                if code.co_name == "run_node":
                    from compilation_recorder import describe_value

                    row["result"] = describe_value(result)
                self.emit(row)
        except Exception as exc:
            self.errors.append(
                {"function": code.co_qualname, "event": event, "error": repr(exc)}
            )

    def __enter__(self):
        self.combo_trace.__enter__()
        self.previous = sys.getprofile()
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.previous)
        self.combo_trace.__exit__(*exc)
        self.stream.close()
        status = {
            "events": self.sequence,
            "errors": self.errors,
            "scope": "main compilation thread; observes existing calls and returns",
            "coverage_gaps": [
                "worker thread internals",
                "unexecuted branches",
                "predicate subexpressions",
                "lowering materialization rationale",
                "library-internal GPU kernels",
            ],
            "unfinished_calls": len(self.frames),
            "combo_pending_statements": len(self.combo_trace.pending),
            "missing_combo_targets": sorted(self.combo_trace.missing_targets),
            "combo_trace": "executed branch bodies and list assignment transitions; no predicate replay",
        }
        (self.directory / "observer_status.json").write_text(
            json.dumps(status, indent=2)
        )
