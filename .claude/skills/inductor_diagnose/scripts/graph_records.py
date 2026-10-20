from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.fx import Node


class RegionError(RuntimeError):
    def __init__(self, code, detail, **evidence):
        super().__init__(f"{code}: {detail}")
        self.report = {"code": code, "detail": detail, **evidence}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def target_name(target):
    if isinstance(target, str):
        return target
    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return f"torch.ops.{target}"
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None)
    if module and name and "<" not in name:
        return f"{module}:{name}"
    raise RegionError("UNSERIALIZABLE_TARGET", repr(target))


def encode(value):
    if isinstance(value, Node):
        return {"node": value.name}
    if isinstance(value, tuple):
        return {"tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return {"list": [encode(item) for item in value]}
    if isinstance(value, dict):
        return {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, slice):
        return {"slice": [encode(value.start), encode(value.stop), encode(value.step)]}
    if isinstance(
        value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
    ):
        return {"torch_literal": str(value)}
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return {"symbol": str(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RegionError("UNSERIALIZABLE_ARGUMENT", repr(value))


def tensor_meta(value):
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": [str(v) for v in value.shape],
            "stride": [str(v) for v in value.stride()],
            "offset": str(value.storage_offset()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return {"kind": "sequence", "items": [tensor_meta(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {key: tensor_meta(item) for key, item in value.items()},
        }
    return {"kind": type(value).__name__, "value": str(value)}


def graph_record(gm):
    nodes = []
    for node in gm.graph.nodes:
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target_name(node.target),
                "args": encode(node.args),
                "kwargs": encode(node.kwargs),
            }
        )
    return {"nodes": nodes}


def save_graph(gm, inputs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = graph_record(gm)
    write_json(directory / "graph.json", record)
    write_json(
        directory / "metadata.json",
        {
            node.name: {
                "value": tensor_meta(node.meta.get("val")),
                "stack_trace": node.meta.get("stack_trace", ""),
            }
            for node in gm.graph.nodes
        },
    )
    (directory / "graph_code.py").write_text(gm.code)
    identity = {"sha256": digest(record), "roundtrip": "not_checked"}
    write_json(directory / "identity.json", identity)
    return identity
