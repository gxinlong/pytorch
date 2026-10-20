"""Inspect archive structure before loading Torch; preserve saved tensor storage."""

import hashlib
import json
import zipfile
from pathlib import Path


SIDECARS = (
    "serialized_exported_program.json",
    "serialized_state_dict.pt",
    "serialized_constants.pt",
    "serialized_example_inputs.pt",
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inspect_model(path, model_name=None, graph_source=None):
    path = Path(path).resolve()
    source = Path(graph_source).resolve() if graph_source else path
    if not path.is_file():
        raise ValueError(f"Input PT2 does not exist: {path}")
    if source.is_dir():
        missing = [name for name in SIDECARS if not (source / name).is_file()]
        if missing:
            raise ValueError(
                f"Incomplete ExportedProgram sidecars in {source}: {missing}"
            )
        graph = json.loads((source / SIDECARS[0]).read_text())
        if not isinstance(graph, dict) or "graph_module" not in graph:
            raise ValueError(
                f"{source / SIDECARS[0]} is not a serialized ExportedProgram"
            )
        return {
            "input": str(path),
            "source": str(source),
            "format": "sidecars",
            "model_name": model_name or "model",
            "graph": graph,
            "inspection_scope": "container_and_graph_entry_only; tensor payloads and executable compatibility unchecked",
            "files": {str(source / n): sha(source / n) for n in SIDECARS},
            "input_sha256": sha(path),
        }
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Expected a PT2 ZIP archive: {source}")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        modern = [n for n in names if "/models/" in n and n.endswith(".json")]
        legacy = [n for n in names if n == SIDECARS[0]]
        candidates = modern or legacy
        if not candidates:
            raise ValueError(
                "No ExportedProgram found. An AOT-only PT2 cannot reconstruct its FX graph. "
                "Supply --graph-source with the original exported .pt2 or a directory containing "
                "serialized_exported_program.json, serialized_state_dict.pt, "
                "serialized_constants.pt and serialized_example_inputs.pt."
            )
        if model_name:
            candidates = [
                n for n in candidates if Path(n).stem == model_name or n in legacy
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"Select exactly one ExportedProgram with --model-name; candidates: {candidates}"
            )
        member = candidates[0]
        graph = json.loads(archive.read(member))
        if not isinstance(graph, dict) or "graph_module" not in graph:
            raise ValueError(f"{member} is not a serialized ExportedProgram")
        selected = Path(member).stem if modern else "model"
        return {
            "input": str(path),
            "source": str(source),
            "format": "modern" if modern else "legacy",
            "model_name": selected,
            "graph_member": member,
            "graph": graph,
            "inspection_scope": "container_and_graph_entry_only; tensor payloads and executable compatibility unchecked",
            "files": {str(source): sha(source)},
            "input_sha256": sha(path),
        }


def graph_devices(graph):
    devices = set()

    def visit(value):
        if isinstance(value, dict):
            device = value.get("device")
            if isinstance(device, dict) and device.get("type"):
                devices.add(device["type"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(graph)
    return sorted(devices)


def load_program(info):
    import torch

    if info["format"] == "sidecars":
        from torch._export.serde.serialize import deserialize, SerializedArtifact

        source = Path(info["source"])
        return deserialize(
            SerializedArtifact(
                **{
                    key: (source / name).read_bytes()
                    for key, name in zip(
                        (
                            "exported_program",
                            "state_dict",
                            "constants",
                            "example_inputs",
                        ),
                        SIDECARS,
                    )
                }
            )
        )
    if info["format"] == "legacy":
        return torch.export.load(info["source"])
    from torch.export.pt2_archive._package import (
        _load_exported_programs,
        PT2ArchiveReader,
    )

    with PT2ArchiveReader(info["source"]) as reader:
        selected = f"models/{info['model_name']}.json"
        return _load_exported_programs(reader, [selected], None)[info["model_name"]]


def input_metadata(args, kwargs):
    import torch
    from torch.utils._pytree import tree_flatten

    values, spec = tree_flatten((args, kwargs))
    storages, rows = {}, []
    for value in values:
        if isinstance(value, torch.Tensor):
            key = (str(value.device), value.untyped_storage()._cdata)
            storage_id = storages.setdefault(key, len(storages))
            rows.append(
                {
                    "storage": storage_id,
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "offset": value.storage_offset(),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
        else:
            rows.append({"type": type(value).__name__, "value": repr(value)})
    return {"tree_spec": str(spec), "values": rows}


def load_inputs(ep, path=None):
    import torch

    value = torch.load(path, weights_only=False) if path else ep.example_inputs
    if value is None:
        raise ValueError(
            "No example inputs; provide --inputs saved by torch.save((args, kwargs), path)"
        )
    if not (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], (tuple, list))
        and isinstance(value[1], dict)
    ):
        raise ValueError("Inputs must have the structure (args, kwargs)")
    return tuple(value[0]), value[1]


def clone_inputs(value):
    import io

    import torch

    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def compare(actual, expected, rtol, atol):
    import torch
    from torch.utils._pytree import tree_flatten

    actual, actual_spec = tree_flatten(actual)
    expected, expected_spec = tree_flatten(expected)
    if actual_spec != expected_spec:
        return {"passed": False, "reason": "output_pytree_mismatch"}
    rows = []
    for lhs, rhs in zip(actual, expected):
        try:
            torch.testing.assert_close(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True)
            passed, error = True, None
        except (AssertionError, ValueError, TypeError) as exc:
            passed, error = False, str(exc)
        row = {"passed": passed, "error": error}
        if (
            isinstance(lhs, torch.Tensor)
            and isinstance(rhs, torch.Tensor)
            and lhs.shape == rhs.shape
        ):
            finite = torch.isfinite(lhs) & torch.isfinite(rhs)
            dtype = (
                torch.complex128
                if lhs.is_complex() or rhs.is_complex()
                else torch.float64
            )
            difference = (lhs[finite].to(dtype) - rhs[finite].to(dtype)).abs()
            row["max_abs_finite"] = (
                difference.max().item() if difference.numel() else 0.0
            )
        rows.append(row)
    return {
        "passed": all(r["passed"] for r in rows),
        "outputs": rows,
        "rtol": rtol,
        "atol": atol,
    }
