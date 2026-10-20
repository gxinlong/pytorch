import xml.etree.ElementTree as ET

from kernel_dependencies import build_dependencies, dependency_html, dependency_svg

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestKernelDependencies(TestCase):
    @parametrize("final_outputs", [True, False])
    def test_boundary_order_views_duplicates_and_unused_input(self, final_outputs):
        view = {
            "kind": "tensor",
            "buffer": "b",
            "shape": ["s0", "2"],
            "stride": ["1", "s0"],
            "offset": "4",
            "dtype": "torch.float32",
        }
        compilation = {
            "input_names": ["x", "unused"],
            "values": {"x": {"kind": "tensor", "buffer": "x"}, "view": view},
            "calls": [
                {"name": "produce", "reads": ["x"], "writes": ["b"]},
                {"name": "sink_not_returned", "reads": ["x"], "writes": ["other"]},
            ],
            "graph": {
                "record": {
                    "nodes": [
                        {
                            "op": "output",
                            "args": {
                                "tuple": [
                                    {
                                        "tuple": [
                                            {"node": "view"},
                                            {"node": "x"},
                                            {"node": "view"},
                                            None,
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        }
        if final_outputs:
            compilation["graph_outputs"] = [
                view,
                compilation["values"]["x"],
                view,
                {"kind": "literal", "value": None},
            ]
        graph = build_dependencies(compilation)
        self.assertEqual([n["name"] for n in graph["inputs"]], ["x", "unused"])
        self.assertEqual(
            [n["path"] for n in graph["outputs"]],
            ["output[0]", "output[1]", "output[2]", "output[3]"],
        )
        self.assertEqual(graph["outputs"][0]["value"], view)
        self.assertEqual(
            {(e["source"], e["target"], e["kind"]) for e in graph["boundary_edges"]},
            {
                ("input0", "kernel0", "INPUT"),
                ("input0", "kernel1", "INPUT"),
                ("kernel0", "output0", "OUTPUT"),
                ("input0", "output1", "OUTPUT"),
                ("kernel0", "output2", "OUTPUT"),
            },
        )
        self.assertEqual(graph["outputs"][3]["gaps"], [])
        root = ET.fromstring(dependency_svg(graph, "report.html"))
        links = [n.get("href") for n in root.iter() if n.tag.endswith("}a")]
        self.assertIn("report.html#kernel-boundary-input1", links)
        self.assertIn("report.html#kernel-boundary-output3", links)
        self.assertIn("report.html#kernel-boundary-edge-0", links)

    def test_final_output_overrides_pre_realization_buffer(self):
        graph = build_dependencies(
            {
                "calls": [
                    {"name": "produce", "reads": [], "writes": ["original"]},
                    {"name": "copy", "reads": ["original"], "writes": ["final"]},
                ],
                "values": {"view": {"kind": "tensor", "buffer": "original"}},
                "graph": {
                    "record": {
                        "nodes": [
                            {"op": "output", "args": {"tuple": [{"node": "view"}]}}
                        ]
                    }
                },
                "graph_outputs": [{"kind": "tensor", "buffer": "final"}],
            }
        )
        self.assertEqual(graph["outputs"][0]["value"]["buffer"], "final")
        self.assertEqual(graph["outputs"][0]["evidence"], "final_graph_outputs")
        self.assertEqual(graph["boundary_edges"][0]["source"], "kernel1")

    def test_output_nop_is_traced_without_inventing_a_kernel(self):
        graph = build_dependencies(
            {
                "calls": [{"name": "combo", "reads": [], "writes": ["part0", "part1"]}],
                "scheduler": [
                    {
                        "type": "NopKernelSchedulerNode",
                        "members": [{"operation": "cat"}],
                        "reads": ["part0", "part1"],
                        "writes": ["cat_buffer"],
                    }
                ],
                "graph_outputs": [{"kind": "tensor", "buffer": "cat_buffer"}],
            }
        )
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(len(graph["boundary_edges"]), 1)
        self.assertEqual(
            {e["source_buffer"] for e in graph["boundary_edges"][0]["evidence"]},
            {"part0", "part1"},
        )
        self.assertEqual(graph["boundary_edges"][0]["evidence"][0]["via_nop"], ["cat"])

    @parametrize(
        "value",
        [
            {"kind": "tensor", "buffer": "missing"},
            {"kind": "scalar", "expression": "s0"},
        ],
    )
    def test_unresolved_output_is_explicit(self, value):
        graph = build_dependencies({"graph_outputs": [value]})
        self.assertEqual(graph["boundary_edges"], [])
        self.assertTrue(graph["outputs"][0]["gaps"])

    def test_absent_outputs_are_not_inferred_from_sink(self):
        graph = build_dependencies(
            {"calls": [{"name": "sink", "reads": [], "writes": ["b"]}]}
        )
        self.assertEqual(graph["outputs"], [])
        self.assertTrue(
            any("未记录图的返回值" in gap for gap in graph["coverage_gaps"])
        )

    def test_contract_nop_and_keep_independent_kernel(self):
        graph = build_dependencies(
            {
                "input_names": ["input"],
                "calls": [
                    {"name": "combo", "reads": ["input"], "writes": ["part0", "part1"]},
                    {"name": "bmm", "reads": ["cat"], "writes": ["mm"]},
                    {"name": "other", "reads": ["input"], "writes": ["other"]},
                ],
                "scheduler": [
                    {
                        "type": "NopKernelSchedulerNode",
                        "members": [{"operation": "cat_nop"}],
                        "reads": ["part0", "part1"],
                        "writes": ["cat"],
                    }
                ],
            }
        )
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 1)
        edge = graph["edges"][0]
        self.assertEqual((edge["source"], edge["target"], edge["kind"]), (0, 1, "RAW"))
        self.assertEqual(
            {e["source_buffer"] for e in edge["evidence"]}, {"part0", "part1"}
        )
        self.assertEqual(edge["evidence"][0]["via_nop"], ["cat_nop"])
        self.assertEqual(graph["unresolved"], [])

    def test_internal_intermediates_do_not_create_self_edges(self):
        graph = build_dependencies(
            {
                "input_names": ["x"],
                "removed_buffers": ["tmp"],
                "calls": [
                    {"name": "fused", "reads": ["x", "tmp"], "writes": ["tmp", "out"]},
                    {"name": "next", "reads": ["out"], "writes": ["result"]},
                ],
            }
        )
        self.assertEqual(graph["nodes"][0]["internal_buffers"], ["tmp"])
        self.assertEqual([(e["source"], e["target"]) for e in graph["edges"]], [(0, 1)])
        self.assertEqual(graph["unresolved"], [])

    def test_overwrite_versions_and_order_constraints(self):
        graph = build_dependencies(
            {
                "calls": [
                    {
                        "name": "same",
                        "reads": [],
                        "writes": ["b"],
                        "write_coverage": {"b": "full"},
                    },
                    {"name": "read", "reads": ["b"], "writes": []},
                    {
                        "name": "same",
                        "reads": [],
                        "writes": ["b"],
                        "write_coverage": {"b": "full"},
                    },
                    {"name": "read", "reads": ["b"], "writes": []},
                ],
            }
        )
        self.assertEqual(
            {(e["source"], e["target"], e["kind"]) for e in graph["edges"]},
            {(0, 1, "RAW"), (0, 2, "WAW"), (1, 2, "WAR"), (2, 3, "RAW")},
        )
        self.assertEqual([n["id"] for n in graph["nodes"]], [0, 1, 2, 3])

    @parametrize("coverage", ["full", "partial_or_atomic", "opaque"])
    def test_partial_writers_are_retained(self, coverage):
        graph = build_dependencies(
            {
                "calls": [
                    {"name": "first", "reads": [], "writes": ["b"]},
                    {
                        "name": "second",
                        "reads": [],
                        "writes": ["b"],
                        "write_coverage": {"b": coverage},
                    },
                    {"name": "consumer", "reads": ["b"], "writes": []},
                ],
            }
        )
        sources = {e["source"] for e in graph["edges"] if e["kind"] == "RAW"}
        self.assertEqual(sources, {1} if coverage == "full" else {0, 1})

    def test_mutated_input_is_not_an_internal_value(self):
        graph = build_dependencies(
            {
                "input_names": ["x"],
                "mutated_inputs": ["x"],
                "calls": [
                    {"name": "inplace", "reads": ["x"], "writes": ["x"]},
                    {"name": "consumer", "reads": ["x"], "writes": ["out"]},
                ],
            }
        )
        self.assertEqual(graph["nodes"][0]["inputs"], ["x"])
        self.assertEqual(graph["nodes"][0]["internal_buffers"], [])
        self.assertEqual(graph["edges"][0]["kind"], "RAW")
        self.assertTrue(any("mutation" in gap for gap in graph["coverage_gaps"]))

    def test_missing_mapping_and_nop_cycle_are_explicit(self):
        graph = build_dependencies(
            {
                "calls": [
                    {"name": "unmapped"},
                    {"name": "read", "reads": ["a"], "writes": []},
                ],
                "scheduler": [
                    {
                        "type": "NopKernelSchedulerNode",
                        "members": [],
                        "reads": ["b"],
                        "writes": ["a"],
                    },
                    {
                        "type": "NopKernelSchedulerNode",
                        "members": [],
                        "reads": ["a"],
                        "writes": ["b"],
                    },
                ],
            }
        )
        self.assertTrue(graph["nodes"][0]["gaps"])
        self.assertTrue(graph["nodes"][1]["gaps"])
        self.assertEqual(
            graph["unresolved"][0]["reason"], "non_emitting_dependency_cycle"
        )
        self.assertEqual(graph["edges"], [])

    def test_svg_and_html_links_escape_kernel_names(self):
        graph = build_dependencies(
            {
                "calls": [
                    {"name": 'kernel<&"script>', "reads": [], "writes": ["b"]},
                    {"name": "next", "reads": ["b"], "writes": []},
                ],
            }
        )
        root = ET.fromstring(dependency_svg(graph, "report.html"))
        links = [node.get("href") for node in root.iter() if node.tag.endswith("}a")]
        self.assertIn("report.html#kernel0", links)
        self.assertIn("report.html#kernel-dependency-edge-0", links)
        report = dependency_html(graph)
        self.assertNotIn('kernel<&"script>', report)
        self.assertIn('id="kernel-dependency-edge-0"', report)


instantiate_parametrized_tests(TestKernelDependencies)

if __name__ == "__main__":
    run_tests()
