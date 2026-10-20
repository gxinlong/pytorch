import json
import tempfile
from pathlib import Path

from diagnostic_combo_report import combo_analysis
from diagnostic_report import (
    analyze_events,
    build_report,
    classify,
    pair_record,
    parity,
    tree_html,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestDiagnosticReport(TestCase):
    def test_empty_directory_is_not_a_report(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "missing run_manifest"):
                build_report(folder)

    def test_process_failure_and_missing_targets_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "observations").mkdir()
            for name, value in {
                "run_manifest.json": {},
                "process_failure.json": {"returncode": -9},
                "observations/observer_status.json": {
                    "missing_combo_targets": ["example_target"]
                },
                "tlparse_result.json": {
                    "available": True,
                    "traces": [
                        {
                            "returncode": 1,
                            "index": str(run / "tlparse/trace_0/index.html"),
                        }
                    ],
                },
            }.items():
                (run / name).write_text(json.dumps(value))
            result = build_report(run)
            self.assertEqual(result["process_failure"]["returncode"], -9)
            self.assertTrue(
                any("example_target" in gap for gap in result["coverage_gaps"])
            )
            self.assertTrue(
                any(
                    "tlparse" in gap and "失败" in gap
                    for gap in result["coverage_gaps"]
                )
            )
            self.assertNotIn(
                'href="tlparse/trace_0/index.html"', (run / "report.html").read_text()
            )
            self.assertTrue((run / "kernel_dependencies.svg").is_file())
            self.assertEqual(result["kernel_dependencies"]["nodes"], [])

    def test_legacy_string_graph_identity(self):
        current = {
            "graph": {"identity": {"sha256": "abc"}},
            "scheduler": [],
            "calls": [],
            "call_schedules": {},
            "kernels": {},
        }
        historical = {**current, "graph": {"identity": "abc"}}
        self.assertTrue(parity(current, historical)["equal"]["graph_identity"])

    @parametrize("parent_id", [1, 99])
    def test_combo_direct_allow_requires_matching_branch(self, parent_id):
        events = [
            {
                "seq": 1,
                "event": "call",
                "function": "Scheduler.speedup_by_combo_kernel",
                "args": {"nodes": [{"name": "a"}, {"name": "b"}]},
            },
            {
                "seq": 2,
                "event": "combo_branch",
                "function": "Scheduler.speedup_by_combo_kernel",
                "parent_id": parent_id,
                "condition": "not config.benchmark_combo_kernel",
                "taken": True,
            },
            {
                "seq": 3,
                "event": "return",
                "function": "Scheduler.speedup_by_combo_kernel",
                "call_id": 1,
                "result": True,
                "file": "scheduler.py",
                "line": 1,
            },
        ]
        benefit = combo_analysis(events, {})["benefit"][0]
        self.assertEqual(benefit["branch_events"], [2] if parent_id == 1 else [])

    @parametrize("state,delta,percent", [("accepted", 0, 0.0), ("rejected", 128, 8.0)])
    def test_combo_memory_log_preserves_limited_evidence(self, state, delta, percent):
        lines = [
            f"ComboKernels memory-aware: {state} 27 nodes (peak delta {delta:+d} bytes = {percent:.3f}%)"
        ]
        record = combo_analysis([], {}, log_lines=lines)["native_memory_checks"][0]
        self.assertEqual(
            (record["state"], record["peak_delta_bytes"], record["peak_delta_percent"]),
            (state, delta, percent),
        )
        self.assertEqual(record["line"], 1)
        self.assertTrue(record["gap"])
        self.assertNotIn("members", record)

    @parametrize("pending", [True, False])
    def test_submission_is_not_acceptance(self, pending):
        record = pair_record(("a", "b"))
        record["candidates"] = [1]
        record["profitability"] = [
            {
                "seq": 2,
                "result": {
                    "should_fuse": None if pending else True,
                    "pending": pending,
                },
            }
        ]
        status, _, certainty, gaps = classify(record, [])
        self.assertEqual(
            status, "pending_unresolved" if pending else "candidate_uncommitted"
        )
        self.assertEqual(certainty, "unknown")
        self.assertTrue(gaps)

    def test_competition_does_not_claim_benchmark_loss(self):
        record = pair_record(("mm0", "pointwise"))
        record["profitability"] = [
            {"seq": 2, "result": {"should_fuse": None, "pending": True}}
        ]
        status, _, certainty, gaps = classify(
            record, [{"seq": 4, "pair": ["mm1", "pointwise"]}]
        )
        self.assertEqual(status, "pending_competition")
        self.assertEqual(certainty, "inference")
        self.assertEqual(record["callbacks"], [])
        self.assertTrue(gaps)

    def test_earlier_merge_is_not_pending_competition(self):
        record = pair_record(("mm0", "pointwise"))
        record["profitability"] = [
            {"seq": 5, "result": {"should_fuse": None, "pending": True}}
        ]
        self.assertEqual(
            classify(record, [{"seq": 4, "pair": ["mm1", "pointwise"]}])[0],
            "pending_unresolved",
        )

    def test_rejection_reason_belongs_to_the_actual_check(self):
        record = pair_record(("a", "b"))
        record["checks"] = [{"seq": 5, "call_id": 4, "result": False}]
        record["rejections"] = [{"seq": 2, "check_call_id": 1, "reason": "old layout"}]
        status, reason, certainty, gaps = classify(record, [])
        self.assertEqual(status, "check_rejected")
        self.assertEqual(certainty, "unknown")
        self.assertNotIn("old layout", reason)
        self.assertTrue(gaps)

    def test_choice_attribution_uses_output_identity_not_substring_or_reads(self):
        event = {
            "seq": 1,
            "event": "call",
            "function": "Scheduler._replace_node",
            "args": {
                "node": {
                    "members": [{"name": "op30"}],
                    "read_buffers": ["buf3"],
                    "write_buffers": ["buf30"],
                }
            },
        }
        events = [
            event,
            {
                "seq": 2,
                "event": "return",
                "function": "Scheduler._replace_node",
                "call_id": 1,
                "result": None,
            },
        ]
        compilation = {
            "calls": [
                {"name": "copy", "provenance_key": "copy:0"},
                {"name": "mm", "provenance_key": "mm:1"},
            ],
            "call_schedules": {
                "copy:0": [
                    {
                        "members": [
                            {"operation": "op3", "buffers": ["buf3"], "origins": []}
                        ]
                    }
                ],
                "mm:1": [
                    {
                        "members": [
                            {"operation": "op30", "buffers": ["buf30"], "origins": []}
                        ]
                    }
                ],
            },
        }
        kernels = analyze_events(events, compilation)["kernels"]
        self.assertEqual(kernels[0]["choice_events"], [])
        self.assertEqual(len(kernels[1]["choice_events"]), 1)

    def test_completed_merge_supersedes_earlier_rejection(self):
        record = pair_record(("a", "b"))
        record["checks"] = [{"seq": 1, "result": False}, {"seq": 3, "result": True}]
        record["rejections"] = [{"seq": 2, "reason": "layout"}]
        record["merges"] = [5]
        self.assertEqual(classify(record, [])[0], "merged")

    @parametrize("measured", [False, True])
    def test_cost_rejection_preserves_measurement_evidence(self, measured):
        record = pair_record(("a", "b"))
        record["profitability"] = [
            {
                "seq": 1,
                "result": {"should_fuse": False},
                "details": {"min_ms_fused": 0.2} if measured else {},
            }
        ]
        status, _, certainty, gaps = classify(record, [])
        self.assertEqual(status, "profitability_rejected")
        self.assertEqual(certainty, "direct")
        self.assertEqual(bool(gaps), not measured)

    def test_failed_backend_fusion_is_not_a_completed_merge(self):
        events = [
            {
                "seq": 1,
                "event": "call",
                "function": "Scheduler.fuse_two_nodes",
                "pair": ["a", "b"],
                "args": {},
            },
            {
                "seq": 2,
                "event": "return",
                "function": "Scheduler.fuse_two_nodes",
                "call_id": 1,
                "pair": ["a", "b"],
                "result": None,
                "file": "scheduler.py",
                "line": 1,
            },
        ]
        result = analyze_events(events, {})
        self.assertEqual(result["merge_count"], 0)
        self.assertEqual(result["opportunities"][0]["status"], "not_observed")

    def test_combo_partition_keeps_formation_parent(self):
        nodes = [
            {"name": name, "members": [{"name": name, "origins": ["add"]}]}
            for name in ("a", "b", "c")
        ]
        events = [
            {
                "seq": 1,
                "event": "call",
                "function": "Scheduler.create_combo_kernel_nodes.<locals>._register_accept",
                "args": {"accepted": nodes},
            },
            {
                "seq": 2,
                "event": "return",
                "function": "Scheduler.create_combo_kernel_nodes.<locals>._register_accept",
                "call_id": 1,
                "result": None,
            },
            {
                "seq": 3,
                "event": "return",
                "function": "ComboKernel.horizontal_partition",
                "result": [nodes[:2], nodes[2:]],
            },
        ]
        compilation = {
            "calls": [{"name": "kernel", "provenance_key": "kernel:0"}],
            "call_schedules": {
                "kernel:0": [
                    {
                        "members": [
                            {"operation": name, "origins": ["add"]}
                            for name in ("a", "b")
                        ]
                    }
                ]
            },
        }
        result = analyze_events(events, compilation)
        tree = result["kernels"][0]["formation"]
        self.assertEqual(tree["stage"], "codegen_partition")
        self.assertEqual(tree["members"], ["a", "b"])
        self.assertEqual(tree["children"][0]["members"], ["a", "b", "c"])

    def test_combo_exposes_actual_foreach_container(self):
        nodes = [
            {
                "name": name,
                "type": "SchedulerNode",
                "members": [{"name": name, "origins": ["add"]}],
            }
            for name in ("a", "b")
        ]
        container = {
            "name": "a_b",
            "type": "ForeachKernelSchedulerNode",
            "foreach": True,
            "members": [member for node in nodes for member in node["members"]],
        }
        events = [
            {
                "seq": 1,
                "event": "call",
                "function": "Scheduler.create_combo_kernel_nodes.<locals>._register_accept",
                "args": {"accepted": nodes},
            },
            {
                "seq": 2,
                "event": "return",
                "function": "Scheduler.create_combo_kernel_nodes.<locals>._register_accept",
                "call_id": 1,
                "result": None,
            },
            {
                "seq": 3,
                "event": "return",
                "function": "Scheduler.create_combo_kernel_nodes",
                "result": None,
                "schedule": [container],
            },
        ]
        compilation = {
            "calls": [{"name": "combo", "provenance_key": "combo:0"}],
            "call_schedules": {
                "combo:0": [
                    {
                        "members": [
                            {"operation": name, "origins": ["add"]}
                            for name in ("a", "b")
                        ]
                    }
                ]
            },
        }
        tree = analyze_events(events, compilation)["kernels"][0]["formation"]
        self.assertEqual(tree["node"], container)
        self.assertEqual(tree["event"], 2)
        self.assertEqual(tree["node_event"], 3)
        self.assertEqual(len(tree["children"]), 2)
        self.assertIn("ForeachKernelSchedulerNode", tree_html(tree))

    def test_initial_foreach_is_distinct_from_generic_combo(self):
        node = {
            "name": "a_b",
            "type": "ForeachKernelSchedulerNode",
            "foreach": True,
            "members": [{"name": name, "origins": ["cat"]} for name in ("a", "b")],
        }
        events = [
            {
                "seq": 1,
                "event": "return",
                "function": "Scheduler.create_foreach_nodes",
                "schedule": [node],
                "result": None,
            }
        ]
        compilation = {
            "calls": [{"name": "foreach", "provenance_key": "foreach:0"}],
            "call_schedules": {
                "foreach:0": [
                    {
                        "members": [
                            {"operation": name, "origins": ["cat"]}
                            for name in ("a", "b")
                        ],
                    }
                ]
            },
        }
        result = analyze_events(events, compilation)
        tree = result["kernels"][0]["formation"]
        self.assertEqual(tree["stage"], "initial_foreach")
        self.assertEqual(tree["node"], node)
        self.assertEqual(result["combo_count"], 0)
        self.assertEqual(result["initial_foreach_count"], 1)
        self.assertIn("普通融合前建立", tree_html(tree))


instantiate_parametrized_tests(TestDiagnosticReport)

if __name__ == "__main__":
    run_tests()
