import tempfile
from pathlib import Path

from tools.subgraph.core import (
    boundary_plan,
    coalesce_inputs,
    compare,
    digest,
    extract_graph,
    graph_record,
    load_graph,
    load_tensors,
    match_kernels,
    RegionError,
    save_graph,
    save_tensors,
)

import torch
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def make_mm_graph():
    graph = Graph()
    x = graph.placeholder("x")
    w1 = graph.placeholder("w1")
    w2 = graph.placeholder("w2")
    mm1 = graph.call_function(torch.ops.aten.mm.default, (x, w1))
    mm2 = graph.call_function(torch.ops.aten.mm.default, (x, w2))
    graph.output((mm1, mm2))
    return GraphModule(torch.nn.Module(), graph)


class TestKernelMapping(TestCase):
    @parametrize("changed_code", [False, True])
    def test_capture_fingerprint_schema(self, changed_code):
        from tools.subgraph.adapter import compare_compilations

        common = {
            "graph": {"identity": "graph"},
            "scheduler": [],
            "calls": [],
            "values": {},
        }
        baseline = {
            **common,
            "kernels": {"mm": {"sha256": "original", "normalized_sha256": "v1"}},
        }
        captured = {
            **common,
            "kernels": {
                "mm": {
                    "sha256": "changed" if changed_code else "original",
                    "normalized_sha256": "v2",
                    "fingerprint_version": 2,
                    "source": "/new/location.py",
                }
            },
        }
        self.assertEqual(
            compare_compilations(baseline, captured)["passed"], not changed_code
        )

    @parametrize(
        "names,expected",
        [(["kernel"], 2), (["kernel", "other"], 3), (["kernel", "kernel"], 2)],
    )
    def test_exact_name_and_all_calls(self, names, expected):
        calls = match_kernels(
            {
                "kernel:1": ["a"],
                "kernel:2": ["b"],
                "other:3": ["a"],
                "kernel_extra:4": ["c"],
            },
            names,
        )
        self.assertEqual(len(calls), expected)

    def test_unknown(self):
        with self.assertRaisesRegex(RegionError, "UNKNOWN_KERNEL"):
            match_kernels({"kernel:1": ["a"]}, ["kern"])

    def test_union_shared_input(self):
        gm = make_mm_graph()
        names = [node.name for node in gm.graph.nodes if node.op == "call_function"]
        subgraph, report = extract_graph(
            gm, {"mm:1": [names[0]], "mm:2": names}, ["mm"], digest(graph_record(gm))
        )
        self.assertEqual(report["nodes"], names)
        self.assertEqual(report["inputs"], ["x", "w1", "w2"])
        self.assertEqual(
            len([node for node in subgraph.graph.nodes if node.op == "placeholder"]), 3
        )

    def test_identity(self):
        with self.assertRaisesRegex(RegionError, "GRAPH_IDENTITY"):
            extract_graph(make_mm_graph(), {}, [], "different")

    def test_missing_node(self):
        gm = make_mm_graph()
        with self.assertRaisesRegex(RegionError, "MISSING_FX_NODE"):
            extract_graph(gm, {"mm:1": ["absent"]}, ["mm"], digest(graph_record(gm)))

    def test_mutation(self):
        graph = Graph()
        x = graph.placeholder("x")
        mutation = graph.call_function(torch.ops.aten.add_.Tensor, (x, x))
        graph.output(mutation)
        gm = GraphModule(torch.nn.Module(), graph)
        with self.assertRaisesRegex(RegionError, "MUTATION"):
            extract_graph(
                gm, {"add:1": [mutation.name]}, ["add"], digest(graph_record(gm))
            )

    def test_partition_cycle(self):
        graph = Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, (x,))
        b = graph.call_function(torch.ops.aten.cos.default, (a,))
        c = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        graph.output(c)
        gm = GraphModule(torch.nn.Module(), graph)
        with self.assertRaisesRegex(RegionError, "INVALID_PARTITION"):
            extract_graph(
                gm, {"region:1": [a.name, c.name]}, ["region"], digest(graph_record(gm))
            )

    def test_roundtrip_and_no_overwrite(self):
        gm = make_mm_graph()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph"
            save_graph(gm, [torch.ones(2, 3), torch.ones(3, 4), torch.ones(3, 5)], path)
            self.assertEqual(graph_record(load_graph(path)), graph_record(gm))
            original = (path / "graph.json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_graph(gm, [], path)
            self.assertEqual((path / "graph.json").read_bytes(), original)
            with (path / "metadata.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(RegionError, "GRAPH_ARTIFACT_HASH"):
                load_graph(path)

    def test_missing_boundary(self):
        report = {
            "calls": [{"provenance_key": "k:1"}],
            "inputs": ["x"],
            "outputs": [],
            "original_fusion_members": {},
        }
        with self.assertRaisesRegex(RegionError, "MISSING_BOUNDARY_MAPPING"):
            boundary_plan(report, {}, [{"provenance_key": "k:1"}])

    def test_partial_write(self):
        report = {
            "calls": [{"provenance_key": "k:1"}],
            "inputs": [],
            "outputs": [],
            "original_fusion_members": {},
        }
        call = {
            "id": "call_0",
            "provenance_key": "k:1",
            "write_coverage": {"buf0": "partial_or_atomic"},
        }
        with self.assertRaisesRegex(RegionError, "PARTIAL_WRITE"):
            boundary_plan(report, {}, [call])

    def test_unmaterialized_boundary(self):
        report = {
            "calls": [{"provenance_key": "k:1"}],
            "inputs": ["x"],
            "outputs": [],
            "original_fusion_members": {},
        }
        value = {"kind": "tensor", "buffer": "buf0"}
        call = {"id": "call_0", "provenance_key": "k:1", "reads": ["other"], "args": []}
        with self.assertRaisesRegex(RegionError, "UNMATERIALIZED_BOUNDARY"):
            boundary_plan(report, {"x": value}, [call])

    def test_shared_physical_buffer(self):
        graph = Graph()
        x, alias = graph.placeholder("x"), graph.placeholder("alias")
        output = graph.call_function(torch.ops.aten.add.Tensor, (x, alias))
        graph.output((output,))
        gm = GraphModule(torch.nn.Module(), graph)
        value = {
            "kind": "tensor",
            "buffer": "buf0",
            "shape": ["2", "3"],
            "stride": ["3", "1"],
            "offset": "0",
            "dtype": "torch.float32",
            "layout": "FixedLayout",
        }
        plan = coalesce_inputs(
            gm, {"inputs": [{"node": "x", **value}, {"node": "alias", **value}]}
        )
        self.assertEqual(len(plan["inputs"]), 1)
        self.assertEqual(plan["input_replacements"], {"alias": "x"})
        self.assertEqual(
            len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
        )

    def test_mutated_buffer(self):
        gm = make_mm_graph()
        with self.assertRaisesRegex(RegionError, "MUTATED_BOUNDARY"):
            coalesce_inputs(
                gm,
                {
                    "inputs": [
                        {
                            "node": "x",
                            "kind": "tensor",
                            "layout": "MutationLayoutSHOULDREMOVE",
                        }
                    ]
                },
            )

    def test_cross_region_mutation(self):
        report = {
            "calls": [{"provenance_key": "k:1"}, {"provenance_key": "k:2"}],
            "inputs": ["x"],
            "outputs": [],
            "original_fusion_members": {},
        }
        value = {"kind": "tensor", "buffer": "buf0"}
        calls = [
            {
                "id": "call_0",
                "ordinal": 0,
                "provenance_key": "k:1",
                "reads": ["buf0"],
                "args": [value],
            },
            {
                "id": "call_1",
                "ordinal": 1,
                "provenance_key": "mutation:3",
                "writes": ["buf0"],
            },
            {
                "id": "call_2",
                "ordinal": 2,
                "provenance_key": "k:2",
                "reads": ["buf0"],
                "args": [value],
            },
        ]
        with self.assertRaisesRegex(RegionError, "CROSS_REGION_MUTATION"):
            boundary_plan(report, {"x": value}, calls)

    def test_dynamic_constraint_missing(self):
        from tools.subgraph.verification import dynamic_entry

        gm = make_mm_graph()
        plan = {
            "inputs": [
                {"kind": "tensor", "shape": ["s0", "3"]},
                {"kind": "tensor", "shape": ["3", "4"]},
                {"kind": "tensor", "shape": ["3", "5"]},
            ]
        }
        with self.assertRaisesRegex(RegionError, "MISSING_DYNAMIC_CONSTRAINT"):
            dynamic_entry(gm, plan, {})

    @parametrize("expression,expected", [("s0", 3), ("200*s0", 600), ("2*s0 + 1", 7)])
    def test_affine_scalar_boundary(self, expression, expected):
        from tools.subgraph.verification import dynamic_entry

        graph = Graph()
        tensor, size = graph.placeholder("tensor"), graph.placeholder("size")
        graph.output((tensor, size))
        gm = GraphModule(torch.nn.Module(), graph)
        plan = {
            "inputs": [
                {"kind": "tensor", "shape": ["s0", "2"]},
                {"kind": "scalar", "expression": expression},
            ]
        }
        gm, shapes, removed, rows = dynamic_entry(
            gm, plan, {"s0": {"min": "1", "max": "256"}}
        )
        self.assertEqual(removed, [1])
        self.assertEqual(len(rows), 1)
        exported = torch.export.export(gm, (torch.empty(8, 2),), dynamic_shapes=shapes)
        self.assertEqual(exported.module()(torch.empty(3, 2))[1], expected)

    @parametrize("expression", ["s0*s0", "s0 + s1", "s1", "-s0"])
    def test_unsupported_scalar_relation(self, expression):
        from tools.subgraph.verification import dynamic_entry

        graph = Graph()
        tensor, size = graph.placeholder("tensor"), graph.placeholder("size")
        graph.output((tensor, size))
        gm = GraphModule(torch.nn.Module(), graph)
        plan = {
            "inputs": [
                {"kind": "tensor", "shape": ["s0", "2"]},
                {"kind": "scalar", "expression": expression},
            ]
        }
        with self.assertRaisesRegex(RegionError, "SCALAR_DYNAMIC_RELATION"):
            dynamic_entry(gm, plan, {"s0": {"min": "1", "max": "256"}})

    @parametrize("merged", [False, True])
    def test_duplicate_nodes_follow_output_ports(self, merged):
        from tools.subgraph.verification import match_nodes

        before = Graph()
        x = before.placeholder("x")
        first = before.create_node(
            "call_function", torch.ops.aten.sigmoid.default, (x,), name="first"
        )
        second = before.create_node(
            "call_function", torch.ops.aten.sigmoid.default, (x,), name="second"
        )
        before.output((first, second))
        after = Graph()
        value = after.placeholder("value")
        a = after.create_node(
            "call_function", torch.ops.aten.sigmoid.default, (value,), name="renamed_a"
        )
        b = (
            a
            if merged
            else after.create_node(
                "call_function",
                torch.ops.aten.sigmoid.default,
                (value,),
                name="renamed_b",
            )
        )
        after.output((b, a))
        _, equivalents, unknown = match_nodes(
            GraphModule(torch.nn.Module(), before),
            GraphModule(torch.nn.Module(), after),
        )
        self.assertEqual(unknown, [])
        self.assertEqual(
            equivalents["renamed_a"], ["first", "second"] if merged else ["second"]
        )
        if not merged:
            self.assertEqual(equivalents["renamed_b"], ["first"])

    def test_changed_node_not_matched(self):
        from tools.subgraph.verification import match_nodes

        graphs = []
        for target in (torch.ops.aten.sigmoid.default, torch.ops.aten.neg.default):
            graph = Graph()
            x = graph.placeholder("x")
            result = graph.call_function(target, (x,))
            graph.output((result,))
            graphs.append(GraphModule(torch.nn.Module(), graph))
        _, _, unknown = match_nodes(*graphs)
        self.assertEqual([row["name"] for row in unknown], ["neg_default"])

    @parametrize("external_computes_selected", [False, True])
    def test_inherited_provenance_is_not_fusion(self, external_computes_selected):
        from tools.subgraph.verification import reference_groups

        selection = {
            "nodes": ["mm", "relu"],
            "calls": [{"provenance_key": "selected:1"}],
            "original_fusion_members": {
                "selected:1": ["mm", "relu"],
                "external:2": ["relu", "mm2"],
            },
        }
        original = {
            "call_schedules": {
                "selected:1": [
                    {"members": [{"exact_origin": "mm"}, {"exact_origin": "relu"}]}
                ],
                "external:2": [
                    {
                        "members": [
                            {
                                "exact_origin": "relu"
                                if external_computes_selected
                                else "mm2"
                            }
                        ]
                    }
                ],
            }
        }
        expected, inherited = reference_groups(original, selection)
        self.assertEqual(len(expected), 2 if external_computes_selected else 1)
        self.assertEqual(len(inherited), 0 if external_computes_selected else 1)

    def test_missing_scheduler_does_not_hide_related_group(self):
        from tools.subgraph.verification import reference_groups

        selection = {
            "nodes": ["relu"],
            "calls": [{"provenance_key": "selected:1"}],
            "original_fusion_members": {
                "selected:1": ["relu"],
                "unknown:2": ["relu", "mm2"],
            },
        }
        expected, inherited = reference_groups({"call_schedules": {}}, selection)
        self.assertEqual(len(expected), 2)
        self.assertEqual(inherited, [])

    def test_unresolved_storage_alias(self):
        report = {
            "calls": [{"provenance_key": "k:1"}],
            "inputs": ["x"],
            "outputs": [],
            "original_fusion_members": {},
        }
        value = {"kind": "tensor", "buffer": "buf0", "layout": "NonOwningLayout"}
        with self.assertRaisesRegex(RegionError, "UNRESOLVED_STORAGE_ALIAS"):
            boundary_plan(report, {"x": value}, [{"provenance_key": "k:1"}])

    @parametrize("text,expected", [("1.5", 1.5), ("True", True), ("s0", 80)])
    def test_scalar_values(self, text, expected):
        from tools.subgraph.snapshots import expression

        self.assertEqual(expression(text, {"s0": 80}, integer=False), expected)

    def test_random_side_effect(self):
        graph = Graph()
        x = graph.placeholder("x")
        result = graph.call_function(torch.ops.aten.rand_like.default, (x,))
        graph.output(result)
        gm = GraphModule(torch.nn.Module(), graph)
        with self.assertRaisesRegex(RegionError, "SIDE_EFFECT"):
            extract_graph(
                gm, {"random:1": [result.name]}, ["random"], digest(graph_record(gm))
            )

    @parametrize("changed_block", [False, True])
    def test_kernel_fingerprint_names(self, changed_block):
        from tools.subgraph.core import file_hash
        from tools.subgraph.verification import kernel_fingerprint

        with tempfile.TemporaryDirectory() as temp:
            fingerprints = []
            for index, name in enumerate(("triton_mm_17", "triton_mm_0")):
                block = 64 if changed_block and index else 32
                source = Path(temp) / f"{name}.py"
                source.write_text(
                    f"async_compile.triton('{name}', '''\ndef {name}(a):\n    BLOCK_M = {block}\n''')\n"
                )
                compilation = {
                    "kernels": {
                        name: {"source": str(source), "sha256": file_hash(source)}
                    }
                }
                fingerprints.append(kernel_fingerprint(compilation, name))
            if changed_block:
                self.assertNotEqual(*fingerprints)
            else:
                self.assertEqual(*fingerprints)

    @parametrize("changed_block", [False, True])
    def test_kernel_parameter_schema(self, changed_block):
        from tools.subgraph.core import file_hash
        from tools.subgraph.verification import kernel_parameters

        with tempfile.TemporaryDirectory() as temp:
            parameters = []
            for index in range(2):
                block = 64 if changed_block and index else 32
                source = Path(temp) / f"kernel_{index}.py"
                source.write_text(
                    f"    BLOCK_M : tl.constexpr = {block}\n    GROUP_M : tl.constexpr = 8\n"
                )
                compilation = {
                    "kernels": {
                        "mm": {
                            "source": str(source),
                            "sha256": file_hash(source),
                            "configuration_lines": []
                            if index
                            else ["BLOCK_M : tl.constexpr = 32"],
                        }
                    }
                }
                parameters.append(kernel_parameters(compilation, "mm"))
            self.assertEqual(
                parameters[0],
                ["BLOCK_M : tl.constexpr = 32", "GROUP_M : tl.constexpr = 8"],
            )
            self.assertEqual(parameters[0] == parameters[1], not changed_block)


class TestBoundarySnapshots(TestCase):
    @parametrize("batch", [1, 80, 256])
    def test_mm_epilogue_noncontiguous(self, device, batch):
        graph = Graph()
        x, w = graph.placeholder("x"), graph.placeholder("w")
        mm = graph.call_function(torch.ops.aten.mm.default, (x, w))
        relu = graph.call_function(torch.ops.aten.relu.default, (mm,))
        graph.output((mm, relu))
        gm = GraphModule(torch.nn.Module(), graph)
        subgraph, _ = extract_graph(
            gm,
            {"mm_epilogue:1": [mm.name, relu.name]},
            ["mm_epilogue"],
            digest(graph_record(gm)),
        )
        left = torch.randn(16, batch, device=device).t()
        right = torch.randn(8, 16, device=device).t()
        actual = subgraph(left, right)
        self.assertEqual(actual, (left @ right, (left @ right).relu()))

    @parametrize("offset", [0, 3])
    def test_views(self, device, offset):
        backing = torch.arange(100, dtype=torch.float32, device=device)
        a = backing.as_strided((3, 4), (7, 1), offset)
        b = backing.as_strided((4, 3), (1, 7), offset)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "values"
            save_tensors([a, b, a, 80], path)
            restored = load_tensors(path, device)
            self.assertEqual(restored, [a, b, a, 80])
            self.assertEqual(restored[0].stride(), a.stride())
            self.assertEqual(restored[0].storage_offset(), offset)
            self.assertEqual(
                restored[0].untyped_storage()._cdata,
                restored[1].untyped_storage()._cdata,
            )
            self.assertEqual(
                restored[0].untyped_storage()._cdata,
                restored[2].untyped_storage()._cdata,
            )

    def test_equal_values_are_not_aliases(self, device):
        a = torch.ones(3, device=device)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "values"
            save_tensors([a, a.clone()], path)
            restored = load_tensors(path, device)
            self.assertNotEqual(
                restored[0].untyped_storage()._cdata,
                restored[1].untyped_storage()._cdata,
            )

    def test_numeric_report(self, device):
        value = torch.tensor([0.0, 2.0], device=device)
        result = compare((value,), (value.clone(),))
        self.assertTrue(result["passed"])
        self.assertTrue(result["outputs"][0]["equal"])
        self.assertFalse(compare((value + 0.01,), (value,))["passed"])
        integers = torch.tensor([2**54], device=device)
        self.assertFalse(compare((integers + 1,), (integers,))["passed"])

    def test_snapshot_corruption(self, device):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "values"
            save_tensors([torch.ones(4, device=device)], path)
            with (path / "storages.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RegionError, "SNAPSHOT_HASH"):
                load_tensors(path, device)


instantiate_parametrized_tests(TestKernelMapping)
instantiate_device_type_tests(TestBoundarySnapshots, globals())


if __name__ == "__main__":
    run_tests()
