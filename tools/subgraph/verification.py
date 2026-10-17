import importlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import TypedDict

import sympy


class FusionGroup(TypedDict):
    key: str
    members: list[str]


def dynamic_entry(gm, plan, constraints):
    import operator

    import torch

    from .core import RegionError

    rows = list(plan["inputs"])
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    symbols: dict[str, tuple[torch.fx.Node, int]] = {}
    for index, row in enumerate(rows):
        if row["kind"] == "tensor":
            for axis, size in enumerate(row["shape"]):
                if re.fullmatch(r"[su]\d+", size):
                    symbols.setdefault(size, (placeholders[index], axis))
    removed = []
    first_compute = next(node for node in gm.graph.nodes if node.op != "placeholder")
    for index, row in enumerate(rows):
        if row["kind"] != "scalar":
            continue
        expr = sympy.sympify(row["expression"])
        if not expr.free_symbols:
            continue
        if len(expr.free_symbols) != 1:
            raise RegionError("SCALAR_DYNAMIC_RELATION", row["expression"])
        symbol = next(iter(expr.free_symbols))
        coefficient, constant = expr.diff(symbol), expr.subs(symbol, 0)
        if (
            str(symbol) not in symbols
            or not coefficient.is_Integer
            or not constant.is_Integer
            or coefficient <= 0
        ):
            raise RegionError("SCALAR_DYNAMIC_RELATION", row["expression"])
        tensor, axis = symbols[str(symbol)]
        with gm.graph.inserting_before(first_compute):
            replacement = gm.graph.call_function(
                torch.ops.aten.sym_size.int, (tensor, axis)
            )
            if coefficient != 1:
                replacement = gm.graph.call_function(
                    operator.mul, (replacement, int(coefficient))
                )
            if constant != 0:
                replacement = gm.graph.call_function(
                    operator.add, (replacement, int(constant))
                )
        placeholders[index].replace_all_uses_with(replacement)
        gm.graph.erase_node(placeholders[index])
        removed.append(index)
    gm.graph.lint()
    gm.recompile()
    rows = [row for index, row in enumerate(rows) if index not in removed]
    dimensions = {}
    shapes = []
    for row in rows:
        shape = {}
        if row["kind"] == "tensor":
            for axis, text in enumerate(row["shape"]):
                expr = sympy.sympify(text)
                if not expr.free_symbols:
                    continue
                if len(expr.free_symbols) != 1:
                    raise RegionError("DYNAMIC_EXPRESSION", text)
                symbol = next(iter(expr.free_symbols))
                name = str(symbol)
                if name not in dimensions:
                    bounds = constraints.get(name)
                    if bounds is None:
                        raise RegionError("MISSING_DYNAMIC_CONSTRAINT", name)
                    minimum, maximum = (
                        sympy.sympify(bounds["min"]),
                        sympy.sympify(bounds["max"]),
                    )
                    if not minimum.is_Integer or not maximum.is_Integer:
                        raise RegionError(
                            "UNBOUNDED_DYNAMIC_CONSTRAINT", name, bounds=bounds
                        )
                    dimensions[name] = torch.export.Dim(
                        name, min=int(minimum), max=int(maximum)
                    )
                coefficient = expr.diff(symbol)
                constant = expr.subs(symbol, 0)
                if (
                    not coefficient.is_Integer
                    or not constant.is_Integer
                    or coefficient <= 0
                ):
                    raise RegionError("DYNAMIC_EXPRESSION", text)
                shape[axis] = int(coefficient) * dimensions[name] + int(constant)
        shapes.append(shape if row["kind"] == "tensor" else None)
    return gm, tuple(shapes), removed, rows


def node_signatures(gm):
    import torch

    from .core import digest, encode, target_name

    signatures = {}
    inputs = 0

    def argument(value):
        if isinstance(value, torch.fx.Node):
            return {"value": signatures[value.name]}
        if isinstance(value, (tuple, list)):
            return [argument(item) for item in value]
        if isinstance(value, dict):
            return {key: argument(item) for key, item in value.items()}
        return encode(value)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            value = {"input": inputs}
            inputs += 1
        else:
            value = {
                "op": node.op,
                "target": target_name(node.target),
                "args": argument(node.args),
                "kwargs": argument(node.kwargs),
            }
        signatures[node.name] = digest(value)
    return signatures


def kernel_source(compilation, name):
    import hashlib

    from torch._inductor.utils import IndentedBuffer

    from .core import RegionError

    record = compilation["kernels"].get(name)
    if not record:
        return None
    if "source" in record:
        body = Path(record["source"]).read_text()
        if hashlib.sha256(body.encode()).hexdigest() != record["sha256"]:
            raise RegionError("KERNEL_SOURCE_HASH", name)
    else:
        body = None
        cache = Path(compilation["wrapper"]).parent.parent / "cache"
        for path in cache.rglob("*.py"):
            source = path.read_text()
            if f"def {name}(" not in source:
                continue
            wrapper = IndentedBuffer()
            wrapper.writeline(f"async_compile.triton({name!r}, '''")
            wrapper.splice(source, strip=True)
            wrapper.writeline("''', device_str='cuda')")
            candidate = wrapper.getvalue()
            if hashlib.sha256(candidate.encode()).hexdigest() == record["sha256"]:
                body = candidate
                break
        if body is None:
            raise RegionError("KERNEL_SOURCE_UNAVAILABLE", name)
    return body


def kernel_fingerprint(compilation, name):
    import hashlib

    body = kernel_source(compilation, name)
    if body is None:
        return None
    normalized = re.sub(rf"\b{re.escape(name)}\b", "KERNEL", body)
    return hashlib.sha256(normalized.encode()).hexdigest()


def kernel_parameters(compilation, name):
    body = kernel_source(compilation, name)
    if body is None:
        return []
    lines = [line.strip() for line in body.splitlines()]
    return [
        line
        for line in lines
        if re.match(r"(?:num_warps|num_stages)\s*=", line) or "tl.constexpr =" in line
    ]


def match_nodes(reference, current):
    import torch

    from .core import target_name

    expected = node_signatures(reference)
    actual = node_signatures(current)
    by_signature: dict[str, list[str]] = {}
    for name, signature in expected.items():
        by_signature.setdefault(signature, []).append(name)
    anchors: dict[str, set[str]] = {}
    visited = set()

    def align(left, right):
        if isinstance(left, torch.fx.Node) and isinstance(right, torch.fx.Node):
            pair = (left.name, right.name)
            if pair in visited or expected[left.name] != actual[right.name]:
                return
            visited.add(pair)
            anchors.setdefault(right.name, set()).add(left.name)
            align(left.args, right.args)
            align(left.kwargs, right.kwargs)
        elif (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
        ):
            for old, new in zip(left, right):
                align(old, new)
        elif (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
        ):
            for key in left:
                align(left[key], right[key])

    align(
        next(node for node in reference.graph.nodes if node.op == "output"),
        next(node for node in current.graph.nodes if node.op == "output"),
    )
    mapping, equivalents, unknown = {}, {}, []
    for node in current.graph.nodes:
        matches = by_signature.get(actual[node.name], [])
        resolved = matches if len(matches) == 1 else sorted(anchors.get(node.name, []))
        if resolved:
            mapping[node.name] = resolved[0]
            equivalents[node.name] = resolved
        elif node.op == "call_function":
            unknown.append(
                {
                    "name": node.name,
                    "target": target_name(node.target),
                    "signature": actual[node.name],
                }
            )
    return mapping, equivalents, unknown


def reference_groups(original, selection):
    selected = set(selection["nodes"])
    called = {row["provenance_key"] for row in selection["calls"]}
    expected, inherited = [], []
    for key, nodes in selection["original_fusion_members"].items():
        members = sorted(selected.intersection(nodes))
        if not members:
            continue
        exact = {
            member["exact_origin"]
            for schedule in original["call_schedules"].get(key, [])
            for member in schedule["members"]
        }
        row = {
            "key": key,
            "members": members,
            "outside_selection": sorted(set(nodes) - selected),
        }
        if key in called or not exact or selected.intersection(exact):
            expected.append(row)
        else:
            inherited.append({**row, "exact_origins": sorted(exact)})
    return expected, inherited


def fusion_report(reference_gm, original, standalone, selection):
    from .core import load_graph

    expected_signatures = node_signatures(reference_gm)
    current_gm = load_graph(standalone["graph"]["directory"])
    mapping, equivalents, unmatched = match_nodes(reference_gm, current_gm)
    kernel_nodes = {
        name
        for nodes in standalone["provenance"]["cppCodeToPost"].values()
        for name in nodes
    }
    unknown = [row for row in unmatched if row["name"] in kernel_nodes]
    selected = set(selection["nodes"])
    expected, inherited = reference_groups(original, selection)
    actual: list[FusionGroup] = []
    for key, nodes in standalone["provenance"]["cppCodeToPost"].items():
        members = sorted(
            {
                origin
                for name in nodes
                for origin in equivalents.get(name, [f"UNMATCHED:{name}"])
            }
        )
        actual.append({"key": key, "members": members})
    normalized_expected = sorted(tuple(row["members"]) for row in expected)
    normalized_actual = sorted(tuple(row["members"]) for row in actual)
    same = (
        normalized_expected == normalized_actual
        and not unknown
        and not any(row["outside_selection"] for row in expected)
    )
    configurations = []
    relations = []
    symbol_mapping: dict[str, str] = {}
    for current, previous in mapping.items():
        before = original["values"].get(previous, {})
        after = standalone["values"].get(current, {})
        for old, new in zip(before.get("shape", []), after.get("shape", [])):
            if re.fullmatch(r"[su]\d+", old) and re.fullmatch(r"[su]\d+", new):
                if new in symbol_mapping and symbol_mapping[new] != old:
                    from .core import RegionError

                    raise RegionError("SYMBOL_MAPPING_CONFLICT", new)
                symbol_mapping[new] = old

    def buffer_labels(compilation, names):
        labels: dict[str, set[str]] = {}
        for name, value in compilation["values"].items():
            if name in names and value.get("buffer"):
                labels.setdefault(value["buffer"], set()).add(names[name])
        return {key: sorted(value) for key, value in labels.items()}

    baseline_labels = buffer_labels(
        original, {name: name for name in expected_signatures}
    )
    standalone_labels = buffer_labels(standalone, mapping)

    def dependencies(compilation, key, labels):
        schedules = compilation["call_schedules"].get(key, [])
        return {
            direction: sorted(
                [
                    labels.get(buffer, ["UNMAPPED:" + buffer])
                    for buffer in {
                        buffer for row in schedules for buffer in row[direction]
                    }
                ]
            )
            for direction in ("reads", "writes")
        }

    def launch_arguments(compilation, key, kernel, renames):
        call = next(
            (
                call
                for call in compilation["calls"]
                if call.get("provenance_key") == key
            ),
            None,
        )
        if call is None:
            return None
        return [
            re.sub(
                r"\b[su]\d+\b",
                lambda match: renames.get(match[0], match[0]),
                arg["expression"].replace(kernel, "KERNEL"),
            )
            for arg in call["args"]
            if arg["kind"] == "scalar"
        ]

    for row in expected:
        candidates = [other for other in actual if other["members"] == row["members"]]
        if len(candidates) != 1:
            continue
        baseline_name = row["key"].rsplit(":", 1)[0]
        actual_name = candidates[0]["key"].rsplit(":", 1)[0]
        baseline_kernel = original["kernels"].get(baseline_name, {})
        current_kernel = standalone["kernels"].get(actual_name, {})
        old_parameters = kernel_parameters(original, baseline_name)
        new_parameters = kernel_parameters(standalone, actual_name)
        old_launch = launch_arguments(original, row["key"], baseline_name, {})
        new_launch = launch_arguments(
            standalone, candidates[0]["key"], actual_name, symbol_mapping
        )
        configurations.append(
            {
                "members": row["members"],
                "baseline": old_parameters,
                "standalone": new_parameters,
                "baseline_launch_arguments": old_launch,
                "standalone_launch_arguments": new_launch,
                "same_launch_arguments": old_launch is not None
                and old_launch == new_launch,
                "same_parameters": old_parameters == new_parameters,
                "same_code_fingerprint": bool(baseline_kernel and current_kernel)
                and kernel_fingerprint(original, baseline_name)
                == kernel_fingerprint(standalone, actual_name),
                "fingerprint_normalization": "Exact generated kernel identifier in definition, metadata and async_compile registration",
            }
        )
        baseline_dependencies = dependencies(original, row["key"], baseline_labels)
        current_dependencies = dependencies(
            standalone, candidates[0]["key"], standalone_labels
        )
        relations.append(
            {
                "members": row["members"],
                "baseline": baseline_dependencies,
                "standalone": current_dependencies,
                "match": baseline_dependencies == current_dependencies,
            }
        )
    relations_match = len(relations) == len(expected) and all(
        row["match"] for row in relations
    )
    materialization = []
    for current, previous in mapping.items():
        if previous not in selected:
            continue
        before = original["values"].get(previous, {})
        after = standalone["values"].get(current, {})

        def normalized(value):
            if isinstance(value, list):
                return [normalized(item) for item in value]
            if isinstance(value, str):
                return re.sub(
                    r"\b[su]\d+\b",
                    lambda match: symbol_mapping.get(match[0], match[0]),
                    value,
                )
            return value

        keys = ("kind", "shape", "stride", "offset", "dtype")
        if any(before.get(key) != normalized(after.get(key)) for key in keys):
            materialization.append(
                {"node": previous, "baseline": before, "standalone": after}
            )
    reverse: dict[str, list[str]] = {}
    for current, previous in equivalents.items():
        for name in previous:
            reverse.setdefault(name, []).append(current)
    outputs = selection["outputs"]
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            old_buffers = [
                original["values"].get(name, {}).get("buffer") for name in (left, right)
            ]
            new_buffers = [
                {
                    standalone["values"].get(node, {}).get("buffer")
                    for node in reverse.get(name, [])
                }
                for name in (left, right)
            ]
            if all(old_buffers) and all(
                len(group) == 1 and None not in group for group in new_buffers
            ):
                if (old_buffers[0] == old_buffers[1]) != (
                    new_buffers[0] == new_buffers[1]
                ):
                    materialization.append(
                        {
                            "kind": "output_storage_alias",
                            "nodes": [left, right],
                            "baseline_shared": old_buffers[0] == old_buffers[1],
                            "standalone_shared": new_buffers[0] == new_buffers[1],
                        }
                    )
    implementation = (
        same
        and relations_match
        and not materialization
        and len(configurations) == len(expected)
        and all(
            row["same_parameters"]
            and row["same_code_fingerprint"]
            and row["same_launch_arguments"]
            for row in configurations
        )
    )
    return {
        "fusion_groups_match": same,
        "implementation_matches": implementation,
        "read_write_relations_match": relations_match,
        "read_write_relations": relations,
        "symbol_mapping": symbol_mapping,
        "materialization_differences": materialization,
        "expected": expected,
        "actual": actual,
        "node_mapping": mapping,
        "node_equivalences": equivalents,
        "inherited_external_provenance": inherited,
        "unmatched_host_nodes": [
            row for row in unmatched if row["name"] not in kernel_nodes
        ],
        "new_or_changed_nodes": unknown,
        "configurations": configurations,
        "removed_groups": [
            list(group)
            for group in normalized_expected
            if group not in normalized_actual
        ],
        "added_groups": [
            list(group)
            for group in normalized_actual
            if group not in normalized_expected
        ],
    }


def benchmark(model, inputs):
    import torch

    from .core import compare, RegionError

    for _ in range(30):
        model(*inputs)
    torch.cuda.synchronize()
    ordinary = tuple(value.clone() for value in model(*inputs))
    rows = []
    for index in range(20):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        wall = time.perf_counter()
        start.record()
        for _ in range(10):
            model(*inputs)
        end.record()
        end.synchronize()
        rows.append(
            {
                "mode": "ordinary",
                "index": index,
                "event_ms": start.elapsed_time(end) / 10,
                "wall_ms": (time.perf_counter() - wall) * 1000 / 10,
            }
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            model(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            graph_output = model(*inputs)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    parity = compare(tuple(graph_output), ordinary)
    if not parity["passed"]:
        raise RegionError(
            "CUDA_GRAPH_NUMERICS",
            "Graph replay differs from ordinary",
            comparison=parity,
        )
    for index in range(20):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        wall = time.perf_counter()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        rows.append(
            {
                "mode": "cuda_graph",
                "index": index,
                "event_ms": start.elapsed_time(end) / 10,
                "wall_ms": (time.perf_counter() - wall) * 1000 / 10,
            }
        )
    summary = {
        mode: {
            metric: statistics.median(
                row[metric] for row in rows if row["mode"] == mode
            )
            for metric in ("event_ms", "wall_ms")
        }
        for mode in ("ordinary", "cuda_graph")
    }
    return {
        "protocol": {
            "warmup": 30,
            "samples": 20,
            "calls_per_sample": 10,
            "side_stream_warmup": 5,
            "graph_warmup_replays": 3,
            "graph_warmup_calls": 30,
        },
        "samples": rows,
        "median_ms": summary,
        "cuda_graph_numerics": parity,
    }


def verify_case(args):
    from .runner import configure_process, read_bundle, read_json, source_identity

    case = read_json(args.case)
    bundle_path = case["bundle"]
    attempt = getattr(args, "attempt", None)
    if (
        attempt is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", attempt) is None
    ):
        raise ValueError("Invalid verification attempt name")
    directory = "verify" if attempt is None else f"verify_{attempt}"
    run, build = Path(case["run"]) / directory, Path(case["build"]) / directory
    torch = configure_process(run, build)
    from .adapter import CompilationRecorder
    from .core import (
        compare,
        file_hash,
        load_graph,
        load_tensors,
        RegionError,
        save_graph,
        write_json,
    )

    bundle = read_bundle(bundle_path)
    write_json(
        run / "source_identity.json", source_identity(bundle["identity"]["commit"])
    )
    plan = read_json(Path(case["run"]) / "boundary.json")
    selection = read_json(Path(case["run"]) / "mapping.json")
    captured = read_json(Path(case["run"]) / "capture/capture.json")
    inputs_report = read_json(bundle["input_report"])
    original = read_json(bundle["compilation"])
    gm = load_graph(case["graph"]).cuda()
    gm, shapes, removed, input_rows = dynamic_entry(
        gm, plan, inputs_report["range_constraints"]
    )
    write_json(
        run / "entry.json",
        {
            "scalar_inputs_rebound_to_shapes": removed,
            "inputs": input_rows,
            "range_constraints": inputs_report["range_constraints"],
        },
    )
    datasets = []
    for row in captured["samples"]:
        inputs = load_tensors(Path(row["directory"]) / "inputs", "cuda")
        inputs = tuple(
            value for index, value in enumerate(inputs) if index not in removed
        )
        outputs = tuple(load_tensors(Path(row["directory"]) / "outputs", "cuda"))
        datasets.append((row["batch"], inputs, outputs))
    primary = next((row for row in datasets if row[0] == 80), datasets[0])
    configs = inputs_report["configs"]
    torch.backends.cuda.matmul.allow_tf32 = inputs_report["model_config"]["use_tf32"]
    torch.backends.cudnn.allow_tf32 = inputs_report["model_config"]["use_tf32"]
    fourier_root = Path(bundle["torchfourier_root"])
    sys.path[:0] = [str(fourier_root), str(fourier_root / "fourier")]
    optimization_scope = importlib.import_module(
        "fourier.aot_optimization_policy"
    ).optimization_scope

    model_config = inputs_report["model_config"]
    with (
        torch.no_grad(),
        optimization_scope(
            model_config.get("export_optimizations", {}),
            use_tf32=model_config["use_tf32"],
            unbacked_safe=model_config.get("unbacked_safe", False),
        ),
    ):
        ep = torch.export.export(gm, primary[1], dynamic_shapes=shapes)
        with CompilationRecorder(build / "diagnostics"):
            package = torch._inductor.aoti_compile_and_package(
                ep, package_path=str(build / "subgraph.pt2"), inductor_configs=configs
            )
    standalone = read_json(build / "diagnostics/compilation.json")
    save_graph(gm, [], build / "entry_graph")
    write_json(
        run / "experiment.json",
        {
            "source_commit": bundle["identity"]["commit"],
            "package": package,
            "package_sha256": file_hash(package),
            "graph": str(build / "entry_graph"),
            "samples": captured["samples"],
            "removed_scalar_inputs": removed,
            "use_tf32": model_config["use_tf32"],
        },
    )
    tool_root = str(Path(__file__).resolve().parents[2])
    with (build / "run_experiment.py").open("x") as stream:
        stream.write(
            f"import sys\nsys.path.insert(0, {tool_root!r})\n"
            "from tools.subgraph.experiment import main\n"
            f"main({str(run / 'experiment.json')!r})\n"
        )
    grouping = fusion_report(gm, original, standalone, selection)
    write_json(run / "fusion.json", grouping)
    model = torch._inductor.aoti_load_package(package, run_single_threaded=True)
    results = []
    with torch.inference_mode():
        for batch, inputs, reference in datasets:
            eager = compare(tuple(gm(*inputs)), reference)
            aot = compare(tuple(model(*inputs)), reference)
            row = {"batch": batch, "eager": eager, "aot": aot, "same_package": package}
            if args.benchmark and eager["passed"] and aot["passed"]:
                row["benchmark"] = benchmark(model, inputs)
            write_json(run / f"b{batch}.json", row)
            results.append(row)
    result = {
        "independently_runnable": True,
        "numerics_pass": all(
            row["eager"]["passed"] and row["aot"]["passed"] for row in results
        ),
        "fusion_groups_match": grouping["fusion_groups_match"],
        "implementation_matches": grouping["implementation_matches"],
        "read_write_relations_match": grouping["read_write_relations_match"],
        "experiment": str(build / "run_experiment.py"),
        "package": package,
        "batches": results,
    }
    write_json(run / "verification.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "batches"}),
        flush=True,
    )
    if not result["numerics_pass"] or not result["fusion_groups_match"]:
        raise RegionError(
            "VERIFICATION_FAILED",
            "See independent numeric and fusion reports",
            report=str(run / "verification.json"),
        )
