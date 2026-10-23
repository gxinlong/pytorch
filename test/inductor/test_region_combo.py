# Owner(s): ["module: inductor"]

import ast
import json
import tempfile
from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen.triton_region_combo import export_helper, UnsupportedRegionCombo
from torch._inductor.cta_fusion import Timing
from torch._inductor.region_combo import (
    candidate_groups,
    check_group,
    check_resources,
    create_region_combos,
    profitable,
    Resources,
)
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._ordered_set import OrderedSet


@instantiate_parametrized_tests
class TestRegionComboPolicy(TestCase):
    @parametrize("ratio,expected", [(1.0, False), (1.04, False), (1.06, True)])
    def test_profit(self, ratio, expected):
        self.assertEqual(
            profitable([Timing(ratio, 100)] * 20, [Timing(1, 100)] * 20), expected
        )

    @parametrize("case", ["short", "nonfinite", "zero", "noisy"])
    def test_invalid_or_noisy_timing(self, case):
        good = [Timing(2, 100)] * 20
        bad = {
            "short": [Timing(1, 100)] * 19,
            "nonfinite": [Timing(float("inf"), 100)] * 20,
            "zero": [Timing(0, 100)] * 20,
            "noisy": [Timing(0.1, 100), Timing(4, 100)] * 10,
        }
        self.assertFalse(profitable(good, bad[case]))

    @parametrize("operation", ["program_id", "num_programs", "atomic_add", "rand"])
    def test_no_implicit_template_rewrite(self, operation):
        with self.assertRaises(UnsupportedRegionCombo):
            export_helper(f"def kernel(x):\n    x = tl.{operation}(0)\n", "member")

    def test_export_preserves_body(self):
        source = "import triton\ndef kernel(x):\n    x = combo_pid_0\n    return\n"
        _, helper = export_helper(source, "member")
        original, exported = ast.parse(source).body[1], ast.parse(helper).body[0]
        self.assertEqual([ast.dump(n) for n in original.body], [ast.dump(n) for n in exported.body])
        self.assertEqual(len(exported.args.args), 7)

    @parametrize("reason", ["different devices", "different streams", "different memory pools",
                           "aliasing", "dependent regions"])
    def test_context_and_dependency(self, reason):
        scheduler, first, second = mock.Mock(), mock.Mock(), mock.Mock()
        for node, name in ((first, "a"), (second, "b")):
            node.get_device.return_value = torch.device("cuda", 0)
            node.has_aliasing_or_mutation.return_value = False
            node.has_strict_sum.return_value = False
            node.get_operation_names.return_value = OrderedSet([name])
            node.ancestors = OrderedSet()
            node.read_writes.reads_and_writes.return_value = []
        scheduler.get_node_stream.side_effect = lambda n: int(
            n is second and reason == "different streams"
        )
        scheduler.node_to_mempool = {
            first: None, second: 1 if reason == "different memory pools" else None
        }
        if reason == "different devices":
            second.get_device.return_value = torch.device("cuda", 1)
        if reason == "aliasing":
            second.has_aliasing_or_mutation.return_value = True
        if reason == "dependent regions":
            second.ancestors.add("a")
        with self.assertRaisesRegex(UnsupportedRegionCombo, reason):
            check_group(scheduler, [first, second])

    def test_resource_outlier(self):
        light, heavy = Resources(32, 0, 0, 1024), Resources(128, 0, 0, 256)
        with self.assertRaisesRegex(UnsupportedRegionCombo, "register outlier"):
            check_resources([light, light, heavy], heavy)
        with self.assertRaisesRegex(UnsupportedRegionCombo, "occupancy drop"):
            check_resources([light, light], heavy)

    @parametrize("limit", ["shared", "spills", "unknown"])
    def test_compiled_resource_limits(self, limit):
        from torch._inductor.region_combo import resources

        launcher = mock.Mock(n_regs=32, shared=0, n_spills=0)
        if limit == "shared":
            launcher.shared = 1025
        elif limit == "spills":
            launcher.n_spills = 1
        else:
            launcher.n_regs = None
        autotuner = mock.Mock(launchers=[launcher], compile_results=[mock.Mock()])
        # Resource validation precedes any occupancy query or launch acceptance.
        device_properties = mock.Mock(shared_memory_per_block_optin=1024)
        with mock.patch("torch.cuda.get_device_properties", return_value=device_properties):
            with self.assertRaises(UnsupportedRegionCombo):
                resources(autotuner, torch.device("cuda", 0), 4)

    def test_no_benchmark_no_acceptance(self):
        from torch._inductor.region_combo import create_region_combos

        scheduler = mock.Mock()
        with (
            config.patch(benchmark_combo_kernel=False),
            mock.patch("torch._inductor.region_combo.report"),
        ):
            create_region_combos(scheduler)
        scheduler.topological_sort_schedule.assert_not_called()

    def test_ordinary_combo_partition_attribution(self):
        from torch._inductor.codegen.simd import SIMDScheduling

        backend, combo = mock.Mock(), mock.Mock()
        first, second = mock.Mock(), mock.Mock()
        first.get_nodes.return_value, second.get_nodes.return_value = [first], [second]
        kernels = [mock.Mock(), mock.Mock()]
        backend.generate_combo_kernel_code.return_value = [
            ("first", kernels[0], [first]), ("second", kernels[1], [second])
        ]
        SIMDScheduling.codegen_combo_kernel(backend, combo)
        members = [c.args[1] for c in backend.define_kernel.call_args_list]
        self.assertEqual(members, [[first], [second]])

    def test_speculative_state_restored(self):
        from torch._inductor import ir
        from torch._inductor.region_combo import speculative_codegen
        from torch._inductor.virtualized import V

        layout = ir.FixedLayout(torch.device("cpu"), torch.float32, [2], [1])
        buffer = ir.Buffer(name="buffer", layout=layout)
        graph = mock.Mock(
            name_to_buffer={"buffer": buffer}, buffer_layout_constraints={},
            removed_buffers=OrderedSet(), inplaced_to_remove=OrderedSet(),
            unaligned_buffers=OrderedSet(),
        )
        with V.set_graph_handler(graph):
            with self.assertRaisesRegex(UnsupportedRegionCombo, "layout change"):
                with speculative_codegen([]):
                    buffer.layout = ir.FixedLayout(torch.device("cpu"), torch.float32, [2], [2])
                    graph.removed_buffers.add("buffer")
                    graph.buffer_layout_constraints["buffer"] = buffer.layout
        self.assertIs(buffer.layout, layout)
        self.assertEqual(graph.buffer_layout_constraints, {})
        self.assertEqual(graph.removed_buffers, OrderedSet())


@instantiate_parametrized_tests
class TestRegionComboGrouping(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(config.patch({
            "region_combo.cta": True, "region_combo.template": True,
            "benchmark_combo_kernel": True,
        }))
        self.enterContext(mock.patch("torch._inductor.region_combo.report"))

    def make_scheduler(self, kinds, dependencies=None):
        from torch._inductor.dependencies import StarDep
        from torch._inductor.scheduler import FusedCtaSchedulerNode, Scheduler, SchedulerNode

        scheduler = object.__new__(Scheduler)
        scheduler.nodes = []
        scheduler.name_to_buf = {}
        scheduler.name_to_fused_node = {}
        scheduler.node_to_stream = {}
        scheduler.node_to_mempool = {}
        for i, kind in enumerate(kinds):
            spec = FusedCtaSchedulerNode if kind == "cta" else SchedulerNode
            node = mock.Mock(spec=spec)
            name, buffer = f"op{i}", f"buf{i}"
            node.get_name.return_value = name
            node.get_nodes.return_value = [node]
            node.get_operation_names.return_value = OrderedSet([name])
            node.get_buffer_names.return_value = OrderedSet([buffer])
            node.get_device.return_value = torch.device("cuda", 0)
            node.is_template.return_value = kind == "template"
            node.has_aliasing_or_mutation.return_value = False
            node.has_strict_sum.return_value = False
            node.read_writes = mock.Mock()
            node.read_writes.reads_and_writes.return_value = []
            node.unmet_dependencies = OrderedSet(
                StarDep(f"buf{dep}") for dep in (dependencies or {}).get(i, ())
            )
            scheduler.nodes.append(node)
            scheduler.name_to_fused_node[name] = node
            scheduler.name_to_buf[buffer] = mock.Mock()
            scheduler.name_to_buf[buffer].defining_op_name.return_value = name
            scheduler.node_to_stream[node] = 0
            scheduler.node_to_mempool[node] = None
        scheduler.compute_ancestors()
        return scheduler

    def test_interleaved_kinds(self):
        scheduler = self.make_scheduler([
            "cta", "cta", "pointwise", "template", "cta", "template", "cta",
        ])
        nodes = scheduler.nodes
        self.assertEqual(list(candidate_groups(scheduler)), [
            [nodes[i] for i in (0, 1, 4, 6)], [nodes[i] for i in (3, 5)],
        ])

    @parametrize("context", ["device", "stream", "mempool"])
    def test_interleaved_contexts(self, context):
        scheduler = self.make_scheduler(["template"] * 4)
        nodes = scheduler.nodes
        for node in nodes[1::2]:
            if context == "device":
                node.get_device.return_value = torch.device("cuda", 1)
            elif context == "stream":
                scheduler.node_to_stream[node] = 1
            else:
                scheduler.node_to_mempool[node] = (1, 2)
        self.assertEqual(list(candidate_groups(scheduler)), [nodes[::2], nodes[1::2]])

    def test_node_limit_after_filtering(self):
        scheduler = self.make_scheduler(["cta", "pointwise"] * 9)
        with config.patch(combo_kernel_max_num_nodes=8):
            self.assertEqual(list(candidate_groups(scheduler)), [
                scheduler.nodes[:16:2], [scheduler.nodes[16]],
            ])

    @parametrize("weak", [False, True])
    def test_dependency_levels(self, weak):
        from torch._inductor.dependencies import WeakDep

        scheduler = self.make_scheduler(["cta"] * 4, {1: [0], 3: [2]})
        nodes = scheduler.nodes
        if weak:
            nodes[1].unmet_dependencies = OrderedSet([WeakDep("buf0", "buf1")])
            nodes[3].unmet_dependencies = OrderedSet([WeakDep("buf2", "buf3")])
        self.assertEqual(list(candidate_groups(scheduler)), [nodes[::2], nodes[1::2]])

    def test_ineligible_member_does_not_split_peers(self):
        scheduler = self.make_scheduler(["cta"] * 3)
        scheduler.nodes[1].has_aliasing_or_mutation.return_value = True
        self.assertEqual(list(candidate_groups(scheduler)), [scheduler.nodes[::2]])

    def test_rejection_preserves_schedule(self):
        scheduler = self.make_scheduler(
            ["pointwise", "template", "pointwise", "template", "pointwise"],
            {1: [0], 3: [2], 4: [1, 3]},
        )
        nodes, mapping = scheduler.nodes[:], scheduler.name_to_fused_node.copy()
        with mock.patch("torch._inductor.region_combo.select_group", return_value=None) as select:
            create_region_combos(scheduler)
        self.assertEqual(select.call_args.args[1], [nodes[1], nodes[3]])
        self.assertEqual(scheduler.nodes, nodes)
        self.assertEqual(scheduler.name_to_fused_node, mapping)

    def test_acceptance_preserves_dependencies_and_mapping(self):
        from torch._inductor.scheduler import RegionComboSchedulerNode

        scheduler = self.make_scheduler(
            ["pointwise", "template", "pointwise", "template", "pointwise"],
            {1: [0], 3: [2], 4: [1, 3]},
        )
        nodes = scheduler.nodes[:]
        combo = mock.Mock(spec=RegionComboSchedulerNode)
        combo.members = (nodes[1], nodes[3])
        combo.get_nodes.return_value = list(combo.members)
        combo.get_name.return_value = "combo"
        combo.get_operation_names.return_value = OrderedSet(["op1", "op3"])
        combo.get_buffer_names.return_value = OrderedSet(["buf1", "buf3"])
        combo.unmet_dependencies = nodes[1].unmet_dependencies | nodes[3].unmet_dependencies
        with mock.patch("torch._inductor.region_combo.select_group", return_value=combo):
            create_region_combos(scheduler)
        self.assertEqual(scheduler.nodes, [nodes[0], nodes[2], combo, nodes[4]])
        self.assertIs(scheduler.name_to_fused_node["op1"], combo)
        self.assertIs(scheduler.name_to_fused_node["op3"], combo)
        self.assertEqual(combo.ancestors, OrderedSet(["op0", "op2"]))
        self.assertEqual(nodes[4].ancestors, OrderedSet(["op0", "op1", "op2", "op3"]))
        self.assertEqual(scheduler.node_to_stream[combo], 0)
        self.assertIsNone(scheduler.node_to_mempool[combo])


class TestRegionCombo(TestCase):
    def setUp(self):
        super().setUp()
        if not HAS_CUDA_AND_TRITON:
            self.skipTest("requires CUDA and Triton")
        torch._dynamo.reset()
        self.enterContext(config.patch({
            "max_autotune": True, "max_autotune_gemm_backends": "ATEN,TRITON",
            "region_combo.template": True, "benchmark_combo_kernel": True,
            "region_combo.max_choices_per_member": 1, "region_combo.max_configs": 1,
            "fx_graph_cache": False,
        }))

    @parametrize("batched", [False, True])
    @parametrize("epilogue", [False, True])
    def test_templates(self, device, batched, epilogue):
        shape = (2, 32, 64) if batched else (32, 64)
        x = torch.randn(shape, device=device, dtype=torch.float16)
        a = torch.randn((*shape[:-2], 64, 64), device=device, dtype=torch.float16)
        b = torch.randn((*shape[:-2], 64, 96), device=device, dtype=torch.float16)

        def fn(x, a, b):
            y, z = x @ a, x @ b
            return (y.relu(), z.relu()) if epilogue else (y, z)

        # Numerics tests exercise compiled code independently of occupancy and timing
        # decisions. Resource-policy tests and unmocked AOT measurements cover those.
        with (
            mock.patch("torch._inductor.region_combo.profitable", return_value=True),
            mock.patch("torch._inductor.region_combo.check_resources"),
        ):
            actual, code = run_and_get_code(torch.compile(fn), x, a, b)
        self.assertIn("region_combo_member_", "\n".join(code))
        self.assertEqual(actual, fn(x, a, b))

    @parametrize("failure", ["compile", "extern", "profit"])
    def test_fallback(self, device, failure):
        x = torch.randn(32, 64, device=device, dtype=torch.float16)
        a, b = (torch.randn(64, 64, device=device, dtype=torch.float16) for _ in range(2))

        def fn(x, a, b):
            return x @ a, x @ b

        if failure == "compile":
            context = mock.patch(
                "torch._inductor.region_combo.RegionComboKernel",
                side_effect=RuntimeError("injected compile failure"),
            )
        elif failure == "extern":
            context = config.patch("test_configs.force_extern_kernel_in_multi_template", True)
        else:
            context = mock.patch("torch._inductor.region_combo.profitable", return_value=False)
        with context:
            actual, code = run_and_get_code(torch.compile(fn), x, a, b)
        self.assertNotIn("region_combo_member_", "\n".join(code))
        self.assertEqual(actual, fn(x, a, b))

    def test_dynamic_stride(self, device):
        def fn(x, a, b):
            return x @ a, x @ b

        a = torch.randn(64, 64, device=device, dtype=torch.float16).t()
        b = torch.randn(64, 96, device=device, dtype=torch.float16)
        compiled = torch.compile(fn, dynamic=False)
        generated = []
        with (
            mock.patch("torch._inductor.region_combo.profitable", return_value=True),
            mock.patch("torch._inductor.region_combo.check_resources"),
        ):
            for size in (17, 33, 65):
                x = torch.randn(size, 64, device=device, dtype=torch.float16)
                torch._dynamo.mark_dynamic(x, 0, min=1, max=128)
                actual, code = run_and_get_code(compiled, x, a, b)
                generated.extend(code)
                self.assertEqual(actual, fn(x, a, b))
        self.assertIn("region_combo_member_", "\n".join(generated))

    def test_cta_regions(self, device):
        from torch._inductor.cta_fusion import CtaConfig

        def fn(a, b, c, d):
            return torch.softmax(a @ b, -1), torch.softmax(c @ d, -1)

        inputs = [torch.randn(2, 1, 64, device=device), torch.randn(2, 64, 32, device=device)] * 2
        inputs = [x.clone() for x in inputs]
        with (
            config.patch({"cta_fusion.enabled": True, "region_combo.cta": True}),
            mock.patch(
                "torch._inductor.cta_fusion.CtaFusionPlan.configs",
                return_value=[CtaConfig(1, 32, 4, 1, False)],
            ),
            mock.patch("torch._inductor.cta_fusion.profitable", return_value=True),
            mock.patch("torch._inductor.region_combo.profitable", return_value=True),
        ):
            actual, code = run_and_get_code(torch.compile(fn), *inputs)
        self.assertIn("region_combo_member_", "\n".join(code))
        self.assertEqual(actual, fn(*inputs))

    @parametrize("kind", ["cta", "template"])
    def test_nonadjacent_regions(self, device, kind):
        from torch._inductor import region_combo
        from torch._inductor.cta_fusion import CtaConfig
        from torch._inductor.scheduler import FusedCtaSchedulerNode

        def fn(x, weights, u):
            outputs = tuple(x @ weight for weight in weights)
            if kind == "cta":
                outputs = tuple(torch.softmax(y, -1) for y in outputs)
            return (*outputs, u.sin())

        if kind == "cta":
            shape, weight_shape, dtype = (2, 1, 64), (2, 64, 32), torch.float32
        else:
            shape, weight_shape, dtype = (32, 64), (64, 64), torch.float16
        inputs = (
            torch.randn(shape, device=device, dtype=dtype),
            [torch.randn(weight_shape, device=device, dtype=dtype) for _ in range(4)],
            torch.randn(37, device=device),
        )

        def interleave(scheduler):
            members = [
                n for n in scheduler.nodes
                if isinstance(n, FusedCtaSchedulerNode) or n.is_template()
            ]
            others = [n for n in scheduler.nodes if n not in members]
            self.assertEqual(len(members), 4)
            self.assertEqual(len(others), 1)
            scheduler.nodes = [*members[:2], *others, *members[2:]]
            scheduler.compute_ancestors()
            create_region_combos(scheduler)

        with (
            config.patch({"cta_fusion.enabled": kind == "cta", "region_combo.cta": True}),
            mock.patch(
                "torch._inductor.cta_fusion.CtaFusionPlan.configs",
                return_value=[CtaConfig(1, 32, 4, 1, False)],
            ),
            mock.patch("torch._inductor.cta_fusion.profitable", return_value=True),
            mock.patch.object(region_combo, "create_region_combos", side_effect=interleave),
            mock.patch.object(region_combo, "profitable", return_value=True),
            mock.patch.object(region_combo, "check_resources"),
            mock.patch.object(region_combo, "report", wraps=region_combo.report) as report,
        ):
            actual, code = run_and_get_code(torch.compile(fn), *inputs)
        self.assertEqual(actual, fn(*inputs))
        self.assertIn("region_combo_member_3", "\n".join(code))
        emitted = [call.args[0] for call in report.call_args_list if call.args[1] == "emitted"]
        self.assertEqual([len(nodes) for nodes in emitted], [4])

    def test_aot_and_graph(self, device):
        class Model(torch.nn.Module):
            def forward(self, x, a, b):
                return (x @ a).relu(), (x @ b).relu()

        inputs = (torch.randn(32, 64, device=device, dtype=torch.float16),
                  torch.randn(64, 64, device=device, dtype=torch.float16),
                  torch.randn(64, 96, device=device, dtype=torch.float16))
        model = Model()
        from torch._inductor import region_combo

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(region_combo, "profitable", return_value=True),
            mock.patch.object(region_combo, "check_resources"),
            mock.patch.object(region_combo, "report", wraps=region_combo.report) as report,
        ):
            package = torch._inductor.aoti_compile_and_package(
                torch.export.export(model, inputs), package_path=str(Path(directory) / "combo.pt2")
            )
            loaded = torch._inductor.aoti_load_package(package, run_single_threaded=True)
            self.assertTrue(any(call.args[1] == "emitted" for call in report.call_args_list))
            expected = model(*inputs)
            self.assertEqual(loaded(*inputs), expected)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    loaded(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = loaded(*inputs)
            graph.replay()
            self.assertEqual(actual, expected)

    def test_resource_split_and_attribution(self, device):
        from torch._inductor import region_combo

        def fn(x, a, b, c):
            return x @ a, x @ b, x @ c

        inputs = [torch.randn(32, 64, device=device, dtype=torch.float16)] + [
            torch.randn(64, 64, device=device, dtype=torch.float16) for _ in range(3)
        ]
        original = region_combo.check_resources

        def limit(members, combined):
            if len(members) > 2:
                raise UnsupportedRegionCombo("injected shared-memory limit")
            return original(members, combined)

        with (
            mock.patch.object(region_combo, "check_resources", side_effect=limit),
            mock.patch.object(region_combo, "profitable", return_value=True),
            mock.patch.object(region_combo, "report", wraps=region_combo.report) as report,
        ):
            actual, code = run_and_get_code(torch.compile(fn), *inputs)
        self.assertEqual(actual, fn(*inputs))
        self.assertIn("region_combo_member_", "\n".join(code))
        emitted = [call.args[0] for call in report.call_args_list if call.args[1] == "emitted"]
        self.assertEqual([len(nodes) for nodes in emitted], [2])
        reasons = [call.args[1] for call in report.call_args_list]
        self.assertIn("partition sequence measured", reasons)

    def test_diagnostic_members(self, device):
        def fn(x, a, b):
            return x @ a, x @ b

        inputs = [torch.randn(32, 64, device=device, dtype=torch.float16)] + [
            torch.randn(64, 64, device=device, dtype=torch.float16) for _ in range(2)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            config.patch({"trace.enabled": True, "trace.debug_dir": directory}),
            mock.patch("torch._inductor.region_combo.profitable", return_value=True),
            mock.patch("torch._inductor.region_combo.check_resources"),
        ):
            actual = torch.compile(fn)(*inputs)
            records = [
                json.loads(line)
                for path in Path(directory).rglob("region_combo.jsonl")
                for line in path.read_text().splitlines()
            ]
        self.assertEqual(actual, fn(*inputs))
        emitted = [r for r in records if r["reason"] == "emitted"]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]["nodes"]), 2)


instantiate_device_type_tests(TestRegionCombo, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
