# Owner(s): ["module: inductor"]

import json
import tempfile
from collections import defaultdict
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.cta_fusion import (
    CtaConfig,
    CtaFusionPlan,
    fuse_small_kernels,
    make_plan,
    measure,
    profitable,
    Timing,
    TimingCache,
    UnsupportedCtaFusion,
)
from torch._inductor.dependencies import ReadWrites, StarDep, WeakDep
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    NodeUser,
    OutputNode,
    Scheduler,
    SchedulerBuffer,
)
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
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
class TestCtaFusionPolicy(TestCase):
    @parametrize("us,eligible", [(49.9, True), (50.0, False), (50.1, False)])
    def test_duration(self, us, eligible):
        sample = Timing(us / 10, 100)
        self.assertEqual(
            sample.avg_us < config.cta_fusion.max_kernel_duration_us, eligible
        )

    @parametrize(
        "ratio,expected", [(1.0, False), (1.04, False), (1.05, True), (1.2, True)]
    )
    def test_profitability(self, ratio, expected):
        with config.patch("cta_fusion.min_speedup", 1.05):
            self.assertEqual(
                profitable([Timing(ratio, 100)] * 5, [Timing(1.0, 100)] * 5),
                expected,
            )

    def test_invalid_samples(self):
        baseline = [Timing(2.0, 100)] * 5
        for sample in (Timing(float("inf"), 100), Timing(0, 100), Timing(1, 0)):
            self.assertFalse(profitable(baseline, [sample] * 5))

    def test_round_consistency(self):
        baseline = [Timing(2.0, 100)] * 5
        self.assertFalse(
            profitable(baseline, [Timing(0.1, 100)] * 3 + [Timing(2.1, 100)] * 2)
        )

    def test_weighted_average(self):
        self.assertEqual(Timing(1.0, 200).avg_us, 5.0)

    def test_timing_cache(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "torch._inductor.runtime.runtime_utils.cache_dir",
                return_value=directory,
            ),
        ):
            cache = TimingCache("test")
            baseline, fused = [Timing(2, 100)] * 5, [Timing(1, 100)] * 5
            self.assertIsNone(cache.read())
            cache.write(baseline, fused)
            self.assertEqual(cache.read(), (baseline, fused))
            data = json.loads(cache.path.read_text())
            data["protocol"] = "obsolete"
            cache.path.write_text(json.dumps(data))
            self.assertIsNone(cache.read())

    def test_repeated_inputs(self):
        from torch._inductor.codegen.triton_cta import BenchmarkCall

        value = torch.ones(3)

        def factory(tensors):
            return lambda: tensors["x"].add_(1)

        call = BenchmarkCall((factory,), {"x": value, "alias": value}, ("x",))
        call()
        copies, reset = call.prepare(4)
        for _ in range(2):
            reset()
            for replica in copies:
                replica()
                self.assertIs(replica.tensors["x"], replica.tensors["alias"])
                self.assertEqual(replica.tensors["x"], torch.full((3,), 2.0))
        call.reset()
        self.assertEqual(value, torch.ones(3))

    @parametrize(
        "reason",
        [
            "different streams",
            "different memory pools",
            "aliasing or mutation",
            "strict sum",
        ],
    )
    def test_legality_before_timing(self, reason):
        first, second, scheduler = mock.Mock(), mock.Mock(), mock.Mock()
        for node in (first, second):
            node.get_device.return_value = torch.device("cuda", 0)
            node.has_aliasing_or_mutation.return_value = (
                reason == "aliasing or mutation"
            )
            node.has_strict_sum.return_value = reason == "strict sum"
        scheduler.get_node_stream.side_effect = [
            0,
            1 if reason == "different streams" else 0,
        ]
        scheduler.node_to_mempool = {
            first: 1,
            second: 2 if reason == "different memory pools" else 1,
        }
        with self.assertRaisesRegex(UnsupportedCtaFusion, reason):
            make_plan(scheduler, first, second)


class TestCtaFusionCandidates(TestCase):
    def setUp(self):
        super().setUp()
        self.plan = self.enterContext(
            mock.patch(
                "torch._inductor.cta_fusion.make_plan",
                side_effect=UnsupportedCtaFusion("candidate enumeration only"),
            )
        )
        self.enterContext(mock.patch("torch._inductor.cta_fusion.report"))
        self.enterContext(config.patch(use_dce=True))

    def make_scheduler(self):
        scheduler = object.__new__(Scheduler)
        scheduler.nodes, scheduler.name_to_buf = [], {}
        scheduler.name_to_donated_buffer = {}
        scheduler.mutation_renames, scheduler.mutation_real_name = {}, {}
        specs = [
            ("alias", "part", [], ["concat"]),
            ("concat", "concat", ["part"], []),
            ("dead", "unused", ["concat"], []),
            ("live", "result", ["concat"], []),
        ]
        for name, output, reads, aliases in specs:
            node = mock.Mock(spec=BaseSchedulerNode)
            node.node = mock.Mock()
            node.node.get_unbacked_symbol_defs.return_value = []
            node.get_name.return_value = name
            node.get_first_name.return_value = name
            node.get_buffer_names.return_value = OrderedSet([output])
            node.has_side_effects.return_value = False
            node.can_inplace.return_value = False
            node.read_writes = ReadWrites(
                OrderedSet(StarDep(n) for n in reads),
                OrderedSet([StarDep(output)]),
                OrderedSet(),
            )
            node.unmet_dependencies = node.read_writes.reads
            buffer_ir = mock.Mock()
            buffer_ir.layout = ir.FixedLayout(torch.device("cpu"), torch.float32, [4], [1])
            buffer_ir.get_name.return_value = output
            buffer_ir.get_inputs_that_alias_output.return_value = aliases
            buffer_ir.get_mutation_names.return_value = []
            buffer = SchedulerBuffer(scheduler, buffer_ir, node)
            node.get_outputs.return_value = [buffer]
            scheduler.nodes.append(node)
            scheduler.name_to_buf[output] = buffer
        scheduler.name_to_node = {n.get_name(): n for n in scheduler.nodes}
        scheduler.name_to_fused_node = scheduler.name_to_node.copy()
        graph = mock.Mock(
            graph_inputs={}, graph_outputs=[], constants={}, mutated_inputs=OrderedSet(),
            removed_operations=OrderedSet(), removed_buffers=OrderedSet(),
            additional_buffer_deps=defaultdict(list), additional_star_deps=defaultdict(list),
        )
        graph.get_output_names.return_value = ["result"]
        self.enterContext(V.set_graph_handler(graph))
        scheduler.compute_dependencies()
        scheduler.dead_node_elimination()
        scheduler.name_to_fused_node = {n.get_name(): n for n in scheduler.nodes}
        return scheduler

    def test_stale_alias_user_after_dce(self):
        scheduler = self.make_scheduler()
        part, concat = (scheduler.name_to_buf[n] for n in ("part", "concat"))
        self.assertIn("dead", V.graph.removed_operations)
        self.assertNotIn("dead", scheduler.name_to_fused_node)
        self.assertIn("dead", [u.get_name() for u in part.users])
        self.assertNotIn("dead", [u.get_name() for u in concat.users])
        self.assertIsInstance(scheduler.name_to_buf["result"].users[0].node, OutputNode)
        original_nodes = scheduler.nodes[:]
        original_users = {name: buf.users[:] for name, buf in scheduler.name_to_buf.items()}
        fuse_small_kernels(scheduler)
        pairs = {(c.args[1].get_name(), c.args[2].get_name()) for c in self.plan.call_args_list}
        self.assertEqual(pairs, {("alias", "concat"), ("concat", "live")})
        self.assertEqual(scheduler.nodes, original_nodes)
        self.assertEqual(
            {name: buf.users for name, buf in scheduler.name_to_buf.items()}, original_users
        )

    def test_fused_consumer_first_name(self):
        scheduler = self.make_scheduler()
        reference = mock.Mock(spec=BaseSchedulerNode)
        reference.get_name.return_value = "live_other"
        reference.get_first_name.return_value = "live"
        scheduler.name_to_buf["concat"].users = [NodeUser(reference)]
        fuse_small_kernels(scheduler)
        self.assertTrue(any(
            call.args[1:] == (scheduler.name_to_node["concat"], scheduler.name_to_node["live"])
            for call in self.plan.call_args_list
        ))

    def test_weak_only_edge(self):
        scheduler = self.make_scheduler()
        live = scheduler.name_to_node["live"]
        live.read_writes.reads = OrderedSet([WeakDep("concat", "result")])
        live.unmet_dependencies = live.read_writes.reads
        scheduler.name_to_buf["concat"].users = [NodeUser(live, is_weak=True)]
        fuse_small_kernels(scheduler)
        self.assertTrue(self.plan.called)
        self.assertTrue(all(call.args[2] is not live for call in self.plan.call_args_list))

    def test_consumer_outside_remaining(self):
        scheduler = self.make_scheduler()
        live = scheduler.name_to_node["live"]
        scheduler.nodes.remove(live)
        fuse_small_kernels(scheduler)
        self.assertTrue(self.plan.called)
        self.assertTrue(all(call.args[2] is not live for call in self.plan.call_args_list))

    def test_missing_live_producer_raises(self):
        scheduler = self.make_scheduler()
        del scheduler.name_to_fused_node["concat"]
        with self.assertRaisesRegex(KeyError, "concat"):
            fuse_small_kernels(scheduler)


class TestCtaFusion(TestCase):
    def setUp(self):
        super().setUp()
        if not HAS_CUDA_AND_TRITON:
            self.skipTest("requires CUDA and Triton")
        self.enterContext(
            config.patch(
                {
                    "cta_fusion.enabled": True,
                    "cta_fusion.max_configs_per_pair": 2,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN,TRITON",
                    "fx_graph_cache": False,
                    "force_disable_caches": True,
                }
            )
        )
        # Numerics and capability tests are independent of noisy GPU timing.
        self.enterContext(
            mock.patch(
                "torch._inductor.cta_fusion.measure", return_value=Timing(0.001, 1)
            )
        )
        self.enterContext(
            mock.patch("torch._inductor.cta_fusion.profitable", return_value=True)
        )

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("use_dot", [False, True])
    def test_bmm_softmax(self, device, dtype, use_dot):
        def fn(a, b):
            value = torch.bmm(a, b)
            return torch.softmax(value, dim=-1)

        a = torch.randn(2, 1, 32, device=device, dtype=dtype)
        b = torch.randn(2, 32, 27, device=device, dtype=dtype)
        with mock.patch.object(
            CtaFusionPlan,
            "configs",
            return_value=[CtaConfig(1, 32, 4, 1, use_dot)],
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)
        self.assertEqual(actual, fn(a, b))
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("kind", ["rhs_gemm", "reduce_rows"])
    def test_complete_matrix(self, device, kind):
        def fn(a, b, c):
            value = a @ b
            return c @ value if kind == "rhs_gemm" else value.sum(0)

        a = torch.randn(17, 19, device=device) * 0.1
        b = torch.randn(19, 13, device=device) * 0.1
        c = torch.randn(5, 17, device=device) * 0.1
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b, c)
        self.assertEqual(actual, fn(a, b, c))
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("length", [2, 3, 4])
    def test_mm_chain(self, device, length):
        def fn(x, weights):
            for w in weights:
                x = (x @ w).relu()
            return x

        x = torch.randn(17, 16, device=device)
        weights = [torch.randn(16, 16, device=device) * 0.1 for _ in range(length)]
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x, weights)
        self.assertEqual(actual, fn(x, weights))
        self.assertIn("cta_unit", "\n".join(code))
        self.assertEqual("\n".join(code).count("for cta_loop_"), length)
        self.assertEqual("\n".join(code).count("async_compile.triton("), 1)

    def test_external_intermediate_user(self, device):
        def fn(a, b, c):
            x = a @ b
            return x, x @ c

        inputs = [torch.randn(16, 16, device=device) for _ in range(3)]
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *inputs)
        self.assertEqual(actual, fn(*inputs))
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("batch", [2, 80, 256])
    def test_bmm_reduction_branches(self, device, batch):
        def fn(a, b, c):
            scores = torch.bmm(a, b)
            invalid = torch.isinf(scores).any(-1, keepdim=True)
            maximum = scores.amax(-1, keepdim=True)
            shifted = torch.where(invalid, scores, scores - maximum)
            exps = shifted.exp()
            total = exps.sum(-1, keepdim=True)
            normalized = exps / total
            return torch.bmm(normalized, c), maximum, total

        a = torch.randn(batch, 1, 128, device=device) * 0.1
        b = torch.randn(batch, 128, 27, device=device) * 0.1
        c = torch.randn(batch, 27, 16, device=device)
        b[0, 0, 0] = float("nan")
        b[1, 0, 0] = float("inf")
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b, c)
        self.assertEqual(actual, fn(a, b, c))
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("transpose", [False, True])
    def test_strided_view(self, device, transpose):
        def fn(a, b, c):
            value = torch.bmm(a, b)
            return value.reshape(-1, 27) @ c

        a = torch.randn(2, 3, 35, device=device)
        b = torch.randn(2, 27, 35, device=device).transpose(-1, -2)
        c = torch.randn(27, 1, device=device)
        if transpose:
            a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b, c)
        self.assertEqual(actual, fn(a, b, c), atol=1e-4, rtol=1e-4)
        self.assertIn("cta_unit", "\n".join(code))

    def test_three_stage_bmm(self, device):
        def fn(a, b, c):
            return torch.bmm(torch.softmax(torch.bmm(a, b), -1), c)

        args = (
            torch.randn(2, 1, 32, device=device),
            torch.randn(2, 32, 27, device=device),
            torch.randn(2, 27, 16, device=device),
        )
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, fn(*args))
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("triple_us,kernels", [(85.0, 1), (110.0, 2)])
    def test_current_baseline_and_original_eligibility(
        self, device, triple_us, kernels
    ):
        def fn(a, b, c, d):
            return torch.bmm(torch.bmm(torch.bmm(a, b), c), d)

        def cost(key):
            count = "".join(key).count("for cta_loop_")
            return {0: 40.0, 1: 40.0, 2: 60.0, 3: triple_us}[count]

        def timing(call):
            us = (
                sum(cost(key) for key in call.key)
                if len(call.factories) > 1
                else cost(call.key)
            )
            return Timing(us / 1000, 1)

        inputs = [torch.randn(2, 16, 16, device=device) * 0.1 for _ in range(4)]
        with (
            mock.patch("torch._inductor.cta_fusion.measure", side_effect=timing),
            mock.patch("torch._inductor.cta_fusion.profitable", side_effect=profitable),
            mock.patch.object(
                CtaFusionPlan, "configs", return_value=[CtaConfig(1, 32, 4, 1, True)]
            ),
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *inputs)
        self.assertEqual(actual, fn(*inputs))
        source = "\n".join(code)
        self.assertEqual(
            source.count("async_compile.triton(") + source.count("extern_kernels.bmm("),
            kernels,
        )

    @parametrize("failure", ["compile", "resource", "duration"])
    def test_fallback(self, device, failure):
        def fn(a, b):
            return torch.bmm(a, b).sum(-1)

        a = torch.randn(2, 1, 32, device=device)
        b = torch.randn(2, 32, 27, device=device)
        if failure == "compile":
            context = mock.patch(
                "torch._inductor.codegen.triton_cta.CtaBenchmark.compile",
                side_effect=RuntimeError("test compilation failure"),
            )
        elif failure == "resource":
            context = config.patch("cta_fusion.max_intermediate_tile_elements", 1)
        else:
            context = mock.patch(
                "torch._inductor.cta_fusion.measure", return_value=Timing(0.05, 1)
            )
        with context:
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)
        self.assertEqual(actual, fn(a, b), atol=1e-4, rtol=1e-4)
        self.assertNotIn("cta_unit", "\n".join(code))

    def test_external_dependency(self, device):
        def fn(a, b):
            value = a @ b
            ordered = value.sort(dim=-1).values
            return value + ordered

        a, b = [torch.randn(16, 16, device=device) for _ in range(2)]
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)
        self.assertEqual(actual, fn(a, b))
        self.assertNotIn("cta_unit", "\n".join(code))

    @parametrize("dims", [(1, 1, 1), (1, 19, 1), (17, 1, 13)])
    def test_unit_dimensions(self, device, dims):
        m, k, n = dims

        def fn(a, b, c):
            return (a @ b) @ c

        args = [
            torch.randn(*shape, device=device) for shape in ((m, k), (k, n), (n, 3))
        ]
        actual = torch.compile(fn, fullgraph=True)(*args)
        self.assertEqual(actual, fn(*args), atol=1e-4, rtol=1e-4)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_intermediate_cast(self, device, dtype):
        def fn(a, b, c):
            value = (a @ b).to(dtype).float()
            return value @ c

        args = [torch.randn(16, 16, device=device) for _ in range(3)]
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, fn(*args), atol=1e-4, rtol=1e-4)
        self.assertIn("cta_unit", "\n".join(code))

    @parametrize("batched", [False, True])
    def test_prologue_epilogue(self, device, batched):
        def fn(a, bias, b, c, residual):
            return (((a + bias) @ b).relu() @ c).relu() + residual

        shapes = (
            ((2, 1, 16), (16,), (2, 16, 16), (2, 16, 13), ())
            if batched
            else ((17, 16), (16,), (16, 16), (16, 13), ())
        )
        args = [torch.randn(shape, device=device) * 0.1 for shape in shapes]
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, fn(*args))
        self.assertIn("cta_unit", "\n".join(code))

    def test_unbounded_intermediate(self, device):
        def fn(a, b):
            return (a @ b).sum(-1)

        a, b = [torch.randn(17, 17, device=device) for _ in range(2)]
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=True), a, b
        )
        self.assertEqual(actual, fn(a, b), atol=1e-4, rtol=1e-4)
        self.assertNotIn("cta_unit", "\n".join(code))

    def test_multirow_tile(self, device):
        def fn(a, b):
            return (torch.bmm(a, b) + 0.25).softmax(-1)

        a = torch.randn(2, 32, 35, device=device, dtype=torch.float16)
        b = torch.randn(2, 35, 64, device=device, dtype=torch.float16)
        with config.patch("cta_fusion.enabled", False):
            reference = torch.compile(fn, fullgraph=True)(a, b)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)
        self.assertEqual(actual, fn(a, b), atol=1e-4, rtol=1e-3)
        self.assertEqual(actual, reference, atol=1e-4, rtol=1e-3)
        self.assertIn("cta_unit", "\n".join(code))

    def test_reduction_neutral_values(self, device):
        def fn(a, b):
            scores = torch.bmm(a, b)
            return (
                scores.amin(-1),
                scores.mean(-1),
                (scores > 0).all(-1),
                (scores > 0).any(-1),
            )

        a = torch.ones(2, 1, 32, device=device)
        b = torch.ones(2, 32, 27, device=device)
        b[0, 0, 0] = float("nan")
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)
        self.assertEqual(actual, fn(a, b))
        self.assertIn("cta_unit", "\n".join(code))

    def test_measure_restores_inputs(self, device):
        from torch._inductor.codegen.triton_cta import BenchmarkCall

        value = torch.ones(3, device=device)

        def factory(tensors):
            return lambda: tensors["x"].add_(1)

        call = BenchmarkCall((factory,), {"x": value}, ("x",))
        timing = measure(call, iterations=7)
        self.assertTrue(timing.valid)
        self.assertEqual(value, torch.ones(3, device=device))


instantiate_device_type_tests(TestCtaFusion, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
