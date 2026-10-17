from __future__ import annotations

import hashlib
import importlib
import json
import math
import operator
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule, validate_partition


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


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


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


def resolve_target(name):
    if name.startswith("torch.ops."):
        value = torch.ops
        for part in name.removeprefix("torch.ops.").split("."):
            value = getattr(value, part)
        return value
    module, qualname = name.split(":", 1)
    value = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


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


def decode(value, nodes):
    if not isinstance(value, dict):
        return value
    if "node" in value:
        return nodes[value["node"]]
    if "tuple" in value:
        return tuple(decode(item, nodes) for item in value["tuple"])
    if "list" in value:
        return [decode(item, nodes) for item in value["list"]]
    if "dict" in value:
        return {decode(key, nodes): decode(item, nodes) for key, item in value["dict"]}
    if "slice" in value:
        return slice(*(decode(item, nodes) for item in value["slice"]))
    if "float" in value:
        return float(value["float"])
    if "torch_literal" in value:
        name = value["torch_literal"]
        return (
            getattr(torch, name[6:])
            if name.startswith("torch.")
            else torch.device(name)
        )
    raise RegionError("UNRESTORABLE_ARGUMENT", repr(value))


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
    from torch._dynamo.repro.after_aot import save_graph_repro

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = graph_record(gm)
    attributes = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            raise RegionError("UNSUPPORTED_MODULE", node.name)
        if node.op == "get_attr":
            value = gm
            for part in node.target.split("."):
                value = getattr(value, part)
            if isinstance(value, torch.Tensor):
                if hasattr(value, "fake_mode"):
                    raise RegionError("FAKE_CONSTANT", node.name)
                attributes[node.target] = value.detach().cpu()
            else:
                raise RegionError("UNSUPPORTED_ATTRIBUTE", node.target)
    if attributes:
        torch.save(attributes, directory / "attributes.pt")
    write_json(directory / "graph.json", record)
    write_json(
        directory / "metadata.json",
        {
            node.name: {
                "value": tensor_meta(node.meta.get("val")),
                "from_node": [
                    item.to_dict() for item in node.meta.get("from_node", [])
                ],
                "stack_trace": node.meta.get("stack_trace", ""),
            }
            for node in gm.graph.nodes
        },
    )
    with (directory / "repro.py").open("x") as stream:
        save_graph_repro(stream, gm, inputs, "inductor", save_dir=None)
    restored = load_graph(directory)
    if graph_record(restored) != record:
        raise RegionError(
            "GRAPH_ROUNDTRIP", "Restored graph differs from the saved graph"
        )
    write_json(
        directory / "identity.json",
        {
            "sha256": digest(record),
            "roundtrip": True,
            "files": {
                path.name: file_hash(path)
                for path in directory.iterdir()
                if path.is_file()
            },
        },
    )
    return digest(record)


def load_graph(directory):
    importlib.import_module("torch._inductor.inductor_prims")
    directory = Path(directory)
    record = json.loads((directory / "graph.json").read_text())
    identity_file = directory / "identity.json"
    if identity_file.exists():
        identity = json.loads(identity_file.read_text())
        if digest(record) != identity["sha256"]:
            raise RegionError(
                "GRAPH_IDENTITY", "Restored graph does not match its saved identity"
            )
        for name, expected in identity.get("files", {}).items():
            if file_hash(directory / name) != expected:
                raise RegionError("GRAPH_ARTIFACT_HASH", name)
    root = torch.nn.Module()
    attribute_file = directory / "attributes.pt"
    if attribute_file.exists():
        for name, value in torch.load(attribute_file, weights_only=True).items():
            parent = root
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.register_buffer(parts[-1], value)
    graph = Graph()
    nodes = {}
    metadata_file = directory / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    for row in record["nodes"]:
        target = (
            resolve_target(row["target"])
            if row["op"] == "call_function"
            else row["target"]
        )
        node = graph.create_node(
            row["op"],
            target,
            decode(row["args"], nodes),
            decode(row["kwargs"], nodes),
            name=row["name"],
        )
        node.meta["region_source"] = row["name"]
        node.meta["saved_metadata"] = metadata.get(row["name"], {})
        nodes[row["name"]] = node
    gm = GraphModule(root, graph)
    graph.lint()
    if graph_record(gm) != record:
        raise RegionError("GRAPH_ROUNDTRIP", "Structural manifest did not roundtrip")
    return gm


def match_kernels(mapping, names):
    calls = []
    for name in dict.fromkeys(names):
        matches = []
        for key, nodes in mapping.items():
            base, separator, suffix = key.rpartition(":")
            kernel = base if separator and suffix.isdecimal() else key
            if kernel == name:
                matches.append({"name": name, "provenance_key": key, "nodes": nodes})
        if not matches:
            raise RegionError("UNKNOWN_KERNEL", name, available=sorted(mapping))
        calls.extend(matches)
    return calls


def extract_graph(gm, mapping, names, expected_identity):
    identity = digest(graph_record(gm))
    if identity != expected_identity:
        raise RegionError(
            "GRAPH_IDENTITY",
            "Graph and compilation bundle do not match",
            expected=expected_identity,
            actual=identity,
        )
    calls = match_kernels(mapping, names)
    selected = {name for call in calls for name in call["nodes"]}
    all_nodes = {node.name: node for node in gm.graph.nodes}
    missing = selected - all_nodes.keys()
    if missing:
        raise RegionError(
            "MISSING_FX_NODE",
            "Provenance refers to absent nodes",
            nodes=sorted(missing),
        )
    nodes = [node for node in gm.graph.nodes if node.name in selected]
    for node in nodes:
        if node.op != "call_function":
            raise RegionError("UNSUPPORTED_NODE", node.name, op=node.op)
        schema = getattr(node.target, "_schema", None)
        pure_python = {
            operator.getitem,
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.eq,
            operator.ne,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
        }
        if schema is None and node.target not in pure_python:
            raise RegionError("UNPROVEN_PURITY", node.name, target=str(node.target))
        if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
            raise RegionError("SIDE_EFFECT", node.name, target=str(node.target))
        if schema and any(
            arg.alias_info and arg.alias_info.is_write for arg in schema.arguments
        ):
            raise RegionError("MUTATION", node.name, target=str(node.target))
        if node.is_impure():
            raise RegionError("SIDE_EFFECT", node.name, target=str(node.target))
    if not validate_partition(nodes):
        adjacent = {
            other.name
            for node in nodes
            for other in (*node.all_input_nodes, *node.users)
            if other.name not in selected
        }
        raise RegionError(
            "INVALID_PARTITION",
            "Region has a dependency path through external nodes",
            nodes=[node.name for node in nodes],
            adjacent_nodes=sorted(adjacent),
            candidate_kernels=[
                key
                for key, members in mapping.items()
                if adjacent.intersection(members)
            ],
        )
    subgraph, inputs, outputs = fuse_as_graphmodule(
        gm, nodes, "KernelRegion", always_return_tuple=True
    )
    if not outputs:
        raise RegionError("EMPTY_OUTPUT", "Region has no externally observable output")
    related = {
        key: values for key, values in mapping.items() if selected.intersection(values)
    }
    report = {
        "graph_identity": identity,
        "selected_names": list(dict.fromkeys(names)),
        "calls": calls,
        "nodes": [node.name for node in nodes],
        "inputs": [node.name for node in inputs],
        "outputs": [node.name for node in outputs],
        "original_fusion_members": related,
        "additional_kernels_for_selected_nodes": [
            key for key in related if key not in {c["provenance_key"] for c in calls}
        ],
    }
    return subgraph, report


def boundary_plan(report, values, calls):
    selected_keys = {call["provenance_key"] for call in report["calls"]}
    selected_calls = [
        call for call in calls if call.get("provenance_key") in selected_keys
    ]
    if len(selected_calls) != len(selected_keys):
        raise RegionError(
            "CALL_MAPPING",
            "Provenance and wrapper calls differ",
            requested=sorted(selected_keys),
            mapped=selected_calls,
        )
    for call in selected_calls:
        unresolved = {
            name: status
            for name, status in call.get("write_coverage", {}).items()
            if status.startswith("partial")
        }
        if unresolved:
            raise RegionError("PARTIAL_WRITE", call["id"], buffers=unresolved)
    result = {"calls": selected_calls, "inputs": [], "outputs": []}
    for role in ("inputs", "outputs"):
        for name in report[role]:
            value = values.get(name)
            if not value or value.get("kind") not in ("tensor", "scalar"):
                raise RegionError(
                    "MISSING_BOUNDARY_MAPPING",
                    name,
                    value=value,
                    related_kernels=report["original_fusion_members"],
                )
            if value["kind"] == "scalar":
                result[role].append({"node": name, **value})
                continue
            if "Mutation" in value.get("layout", ""):
                raise RegionError("MUTATED_BOUNDARY", name, boundary=value)
            if value.get("layout") == "NonOwningLayout":
                raise RegionError(
                    "UNRESOLVED_STORAGE_ALIAS",
                    name,
                    detail_hint="A non-owning buffer needs an explicit backing-storage mapping",
                    boundary=value,
                )
            candidates = []
            for call in selected_calls:
                direction = "reads" if role == "inputs" else "writes"
                if value["buffer"] not in call.get(direction, []):
                    continue
                for index, arg in enumerate(call["args"]):
                    if arg.get("buffer") == value["buffer"]:
                        if arg.get("layout") == "NonOwningLayout":
                            raise RegionError(
                                "UNRESOLVED_STORAGE_ALIAS", name, argument=arg
                            )
                        candidates.append((call, index, arg))
            if not candidates:
                raise RegionError(
                    "UNMATERIALIZED_BOUNDARY",
                    name,
                    value=value,
                    related_kernels=report["original_fusion_members"],
                )
            ordered = sorted(candidates, key=lambda item: item[0]["ordinal"])
            first, last = ordered[0][0]["ordinal"], ordered[-1][0]["ordinal"]
            intervening = [
                other
                for other in calls
                if first <= other["ordinal"] <= last
                and other.get("provenance_key") not in selected_keys
                and value["buffer"] in other.get("writes", [])
            ]
            if intervening:
                raise RegionError("CROSS_REGION_MUTATION", name, calls=intervening)
            call, index, arg = ordered[0] if role == "inputs" else ordered[-1]
            result[role].append(
                {
                    "node": name,
                    **value,
                    "call_id": call["id"],
                    "argument_index": index,
                    "wrapper_arg": arg["expression"],
                    "argument_layout": arg,
                    "phase": "before" if role == "inputs" else "after",
                }
            )
    return result


def coalesce_inputs(gm, plan):
    canonical = {}
    replacements = {}
    inputs = []
    nodes = {node.name: node for node in gm.graph.nodes}
    for row in plan["inputs"]:
        if row["kind"] != "tensor":
            inputs.append(row)
            continue
        if "Mutation" in row.get("layout", ""):
            raise RegionError("MUTATED_BOUNDARY", row["node"], boundary=row)
        key = (
            row["buffer"],
            tuple(row["shape"]),
            tuple(row["stride"]),
            row["offset"],
            row["dtype"],
        )
        if key in canonical:
            old, new = nodes[row["node"]], nodes[canonical[key]["node"]]
            old.replace_all_uses_with(new)
            gm.graph.erase_node(old)
            replacements[row["node"]] = new.name
        else:
            canonical[key] = row
            inputs.append(row)
    gm.graph.lint()
    gm.recompile()
    return {**plan, "inputs": inputs, "input_replacements": replacements}


def save_tensors(values, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    storages = {}
    payloads = {}
    records = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            records.append({"scalar": value})
            continue
        key = (str(value.device), value.untyped_storage()._cdata)
        if key not in storages:
            storage_id = f"storage_{len(storages)}"
            storages[key] = storage_id
            storage = value.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            payloads[storage_id] = raw.cpu().clone()
        records.append(
            {
                "storage": storages[key],
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "offset": value.storage_offset(),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    torch.save(payloads, directory / "storages.pt")
    write_json(
        directory / "manifest.json",
        {"values": records, "sha256": file_hash(directory / "storages.pt")},
    )


def load_tensors(directory, device):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "storages.pt") != manifest["sha256"]:
        raise RegionError("SNAPSHOT_HASH", str(directory))
    payloads = torch.load(
        directory / "storages.pt", map_location="cpu", weights_only=True
    )
    payloads = {key: value.to(device) for key, value in payloads.items()}
    result = []
    for row in manifest["values"]:
        if "scalar" in row:
            result.append(row["scalar"])
        else:
            dtype = getattr(torch, row["dtype"].removeprefix("torch."))
            value = torch.empty(0, dtype=dtype, device=device).set_(
                payloads[row["storage"]].untyped_storage(),
                row["offset"],
                row["shape"],
                row["stride"],
            )
            result.append(value)
    return result


def compare(actual, reference):
    from torch.utils._pytree import tree_flatten

    actual, actual_spec = tree_flatten(actual)
    reference, reference_spec = tree_flatten(reference)
    if actual_spec != reference_spec or len(actual) != len(reference):
        raise RegionError("OUTPUT_STRUCTURE", "Output pytrees differ")
    rows = []
    for index, (value, expected) in enumerate(zip(actual, reference)):
        if not isinstance(value, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            rows.append(
                {
                    "index": index,
                    "passed": value == expected,
                    "equal": value == expected,
                }
            )
            continue
        if value.shape != expected.shape or value.dtype != expected.dtype:
            rows.append({"index": index, "passed": False, "reason": "shape_or_dtype"})
            continue
        rtol, atol = (1e-3, 1e-3) if expected.dtype == torch.float16 else (1e-4, 1e-5)
        if not expected.is_floating_point():
            rtol, atol = 0, 0
        lhs, rhs = value.detach().double(), expected.detach().double()
        delta = (lhs - rhs).abs()
        relative = torch.where(
            rhs.abs() > 0, delta / rhs.abs(), torch.where(delta == 0, 0.0, float("inf"))
        )
        max_abs = delta.max().item() if delta.numel() else 0.0
        max_rel = relative.max().item() if relative.numel() else 0.0
        rows.append(
            {
                "index": index,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "rtol": rtol,
                "atol": atol,
                "passed": bool(
                    torch.isfinite(lhs).all()
                    and torch.isfinite(rhs).all()
                    and (
                        torch.allclose(lhs, rhs, rtol=rtol, atol=atol)
                        if expected.is_floating_point()
                        else torch.equal(value, expected)
                    )
                ),
                "equal": bool(torch.equal(value, expected)),
                "equal_elements": int((value == expected).sum()),
                "elements": value.numel(),
                "max_abs": max_abs if math.isfinite(max_abs) else str(max_abs),
                "max_rel": max_rel if math.isfinite(max_rel) else str(max_rel),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "outputs": rows}
