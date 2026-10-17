import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import sympy

import torch

from .core import file_hash, RegionError


@contextmanager
def capture_stdout(path):
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with Path(path).open("xb") as stream:
            os.dup2(stream.fileno(), 1)
            yield
            sys.stdout.flush()
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def expression(text, bindings, integer=True):
    symbols = {name: sympy.Integer(value) for name, value in bindings.items()}
    value = sympy.sympify(text, locals=symbols)
    if not integer and value in (sympy.true, sympy.false):
        return bool(value)
    if value.free_symbols or (integer and value.is_integer is not True):
        raise RegionError("UNBOUND_SYMBOL", text, bindings=bindings)
    return int(value) if value.is_integer is True else float(value)


def restore_capture(plan, directory):
    directory = Path(directory)
    snapshots = {}
    bindings = {}
    sources = []
    call_by_id = {row["id"]: row for row in plan["calls"]}
    scalars = []
    log = directory / "calls.log"
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.startswith("FX_REGION_SCALAR "):
                continue
            _, call_id, index, text = line.split()
            index = int(index)
            value = float(text) if any(c in text for c in ".eE") else int(text)
            symbolic = call_by_id[call_id]["args"][index]["expression"]
            scalars.append(
                {
                    "call_id": call_id,
                    "argument": index,
                    "expression": symbolic,
                    "value": value,
                }
            )
            if re.fullmatch(r"[su]\d+", symbolic):
                if symbolic in bindings and bindings[symbolic] != value:
                    raise RegionError("SYMBOL_CONFLICT", symbolic)
                bindings[symbolic] = value
    for role in ("inputs", "outputs"):
        for row in plan[role]:
            if row["kind"] != "tensor":
                continue
            key = (row["call_id"], row["argument_index"], row["phase"])
            if key in snapshots:
                continue
            matches = list(
                (directory / "tmp/aoti_torch").glob(
                    f"{key[2]}_{key[0]}_arg_{key[1]}_*.pt"
                )
            )
            if len(matches) != 1:
                raise RegionError(
                    "MISSING_SNAPSHOT", repr(key), files=[str(path) for path in matches]
                )
            path = matches[0]
            value = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(value, torch.Tensor):
                raise RegionError(
                    "SNAPSHOT_TYPE", str(path), actual=type(value).__name__
                )
            snapshots[key] = value
            layout = row["argument_layout"]
            for symbolic, actual in zip(layout["shape"], value.shape):
                if re.fullmatch(r"[su]\d+", symbolic):
                    if symbolic in bindings and bindings[symbolic] != actual:
                        raise RegionError("SYMBOL_CONFLICT", symbolic)
                    bindings[symbolic] = int(actual)
            sources.append(
                {
                    "path": str(path),
                    "sha256": file_hash(path),
                    "call_id": key[0],
                    "argument": key[1],
                    "phase": key[2],
                }
            )
    canonical = {}
    results = {}
    for role in ("inputs", "outputs"):
        values = []
        for row in plan[role]:
            if row["kind"] == "scalar":
                values.append(expression(row["expression"], bindings, integer=False))
                continue
            key = (row["call_id"], row["argument_index"], row["phase"])
            snapshot = snapshots[key]
            shape = [expression(v, bindings) for v in row["shape"]]
            stride = [expression(v, bindings) for v in row["stride"]]
            offset = expression(row["offset"], bindings)
            layout = row["argument_layout"]
            if list(snapshot.shape) != [
                expression(v, bindings) for v in layout["shape"]
            ] or list(snapshot.stride()) != [
                expression(v, bindings) for v in layout["stride"]
            ]:
                raise RegionError("SNAPSHOT_LAYOUT", row["node"])
            if str(snapshot.dtype) != row["dtype"]:
                raise RegionError("SNAPSHOT_DTYPE", row["node"])
            storage_key = (role, row["buffer"])
            if storage_key not in canonical:
                canonical[storage_key] = (
                    snapshot,
                    expression(layout["offset"], bindings),
                )
            backing, argument_offset = canonical[storage_key]
            absolute_offset = backing.storage_offset() + offset - argument_offset
            if absolute_offset < 0:
                raise RegionError("SNAPSHOT_STORAGE_RANGE", row["node"])
            try:
                view = backing.as_strided(shape, stride, absolute_offset)
            except RuntimeError as error:
                raise RegionError(
                    "SNAPSHOT_STORAGE_RANGE", row["node"], reason=str(error)
                ) from error
            values.append(view)
        results[role] = values
    return (
        results["inputs"],
        results["outputs"],
        {
            "symbols": bindings,
            "scalars": scalars,
            "files": sources,
            "alias_policy": "IR buffer identity within immutable region boundary; allocator addresses are never compared",
        },
    )
