import json
import tempfile
import zipfile
from argparse import Namespace
from pathlib import Path

from diagnose import resolve_config
from model_io import clone_inputs, compare, input_metadata, inspect_model, load_program

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestArchiveAndConfig(TestCase):
    def test_native_exported_program_load(self):
        class Example(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "exported.pt2"
            ep = torch.export.export(Example(), (torch.ones(2),))
            torch.export.save(ep, path)
            loaded = load_program(inspect_model(path))
            self.assertEqual(loaded.graph_signature, ep.graph_signature)
            self.assertEqual(loaded.example_inputs, ep.example_inputs)

    def test_environment_only_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"environment": {"TEST_VALUE": None}}))
            result = resolve_config(
                Namespace(
                    profile_file=None,
                    profile=None,
                    config=config,
                    set=[],
                    tf32="inherit",
                )
            )
            self.assertEqual(result["environment"], {"TEST_VALUE": None})
            self.assertEqual(result["inductor_configs"], {})

    def test_aot_only_is_not_an_exported_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "compiled.pt2"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("package/data/aotinductor/model/model.so", b"")
            with self.assertRaisesRegex(ValueError, "AOT-only"):
                inspect_model(path)

    def test_multiple_programs_require_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multi.pt2"
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("left", "right"):
                    archive.writestr(
                        f"package/models/{name}.json", json.dumps({"graph_module": {}})
                    )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                inspect_model(path)
            self.assertEqual(inspect_model(path, "right")["model_name"], "right")

    def test_explicit_config_wins_and_environment_null_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            profile, config = (
                Path(folder) / "profile.json",
                Path(folder) / "config.json",
            )
            profile.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "default": {
                                "inductor_configs": {"combo_kernel_max_num_nodes": 8},
                                "environment": {"TEST_VALUE": "1"},
                            }
                        }
                    }
                )
            )
            config.write_text(
                json.dumps(
                    {
                        "inductor_configs": {"combo_kernel_max_num_nodes": 16},
                        "environment": {"TEST_VALUE": None},
                    }
                )
            )
            result = resolve_config(
                Namespace(
                    profile_file=profile,
                    profile=None,
                    config=config,
                    set=["combo_kernel_max_num_nodes=30"],
                    tf32="off",
                )
            )
            self.assertEqual(
                result["inductor_configs"]["combo_kernel_max_num_nodes"], 30
            )
            self.assertIsNone(result["environment"]["TEST_VALUE"])
            self.assertFalse(result["torch_backends"]["cuda.matmul.allow_tf32"])


class TestInputContract(TestCase):
    def test_clone_keeps_alias_stride_and_offset(self, device):
        base = torch.arange(48, device=device, dtype=torch.float32).reshape(6, 8)
        inputs = ((base[:, 1::2], base[1:, :].transpose(0, 1)), {"extra": base})
        copied = clone_inputs(inputs)
        self.assertEqual(input_metadata(*inputs), input_metadata(*copied))
        self.assertEqual(inputs, copied)
        self.assertNotEqual(
            base.untyped_storage()._cdata, copied[1]["extra"].untyped_storage()._cdata
        )
        copied[1]["extra"][1, 1] = 123
        self.assertEqual(copied[0][0][1, 0].item(), 123)

    @parametrize("changed", [False, True])
    def test_nonfinite_comparison(self, device, changed):
        expected = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0], device=device
        )
        actual = expected.clone()
        if changed:
            actual[1] = -float("inf")
        self.assertEqual(
            compare((actual,), (expected,), 1e-4, 1e-5)["passed"], not changed
        )


instantiate_parametrized_tests(TestArchiveAndConfig)
instantiate_device_type_tests(TestInputContract, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
