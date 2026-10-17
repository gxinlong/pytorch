from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def read_bundle(path):
    from .core import file_hash, RegionError

    bundle = read_json(path)
    for key in ("compilation", "input_report", "package"):
        if file_hash(bundle[key]) != bundle["hashes"][key]:
            raise RegionError("BUNDLE_HASH", key)
    return bundle


def configure_process(run, build):
    expected_gpu = "GPU-693e3eac-1d0e-2b9f-c31b-1a8efbcce283"
    actual_gpu = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if actual_gpu != expected_gpu:
        raise RuntimeError(f"Physical GPU3 identity changed: {actual_gpu}")
    occupants = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if occupants:
        raise RuntimeError(f"Physical GPU3 is occupied by PIDs: {occupants}")
    os.environ["CUDA_VISIBLE_DEVICES"] = expected_gpu
    run, build = Path(run), Path(build)
    run.mkdir(parents=True, exist_ok=False)
    build.mkdir(parents=True, exist_ok=False)
    for name in ("cache", "triton", "unpack", "debug"):
        (build / name).mkdir()
    (run / "trace").mkdir()
    os.environ.update(
        {
            "INDUCTOR_PROVENANCE": "1",
            "TORCH_TRACE": str(run / "trace"),
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": str(build / "debug"),
            "TORCHINDUCTOR_CACHE_DIR": str(build / "cache"),
            "TRITON_CACHE_DIR": str(build / "triton"),
            "TMPDIR": str(build / "unpack"),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        }
    )
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only physical GPU3 through CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    return torch


def repository_identity(source, expected):
    from .core import RegionError

    git = ["git", "-C", str(source)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    reference = subprocess.check_output(
        [*git, "rev-parse", f"{expected}^{{commit}}"], text=True
    ).strip()
    changed = set()
    for args in (
        ["diff", "--name-only", "--no-renames", "-z", reference, "HEAD", "--"],
        ["diff", "--name-only", "--no-renames", "-z", reference, "--"],
        ["diff", "--cached", "--name-only", "--no-renames", "-z", reference, "--"],
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ):
        changed.update(
            filter(None, subprocess.check_output([*git, *args], text=True).split("\0"))
        )
    outside_tool = sorted(
        path for path in changed if not path.startswith("tools/subgraph/")
    )
    if outside_tool:
        raise RegionError(
            "SOURCE_IDENTITY",
            "Source differs from the compiler reference outside tools/subgraph",
            expected=reference,
            actual=commit,
            paths=outside_tool,
        )
    return {
        "commit": commit,
        "compiler_reference": reference,
        "compiler_matches_reference": True,
        "tool_changes": sorted(changed),
    }


def source_identity(expected):
    import triton

    import torch

    from .core import file_hash, RegionError

    source = Path(torch.__file__).resolve().parent.parent
    identity = repository_identity(source, expected)
    manifest = source / "source_identity.json"
    if manifest.exists():
        identity["source_manifest"] = read_json(manifest)
    tool = Path(__file__).resolve().parent
    identity["tool_files"] = {
        str(path.relative_to(tool)): file_hash(path)
        for path in sorted(tool.rglob("*.py"))
    }
    if not torch.__version__.startswith("2.14.0"):
        raise RegionError("RUNTIME_VERSION", torch.__version__)
    return {
        **identity,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "wheel_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": sys.executable,
    }


def load_full_model(model_path, fourier_root, batch=80):
    import torch

    from .core import file_hash, RegionError

    sys.path[:0] = [str(fourier_root), str(fourier_root / "fourier")]
    aot_runtime = importlib.import_module("fourier.aot_runtime")

    runtime_config = vars(aot_runtime)
    required_triton = runtime_config["AOT_TRITON_VERSION"]
    runtime_config["AOT_TRITON_VERSION"] = aot_runtime.installed_triton_version()
    aot_compile = importlib.import_module("fourier.aot_compile")
    aot_ops = importlib.import_module("fourier.aot_ops")
    validation = importlib.import_module("fourier.aot_validation")
    const2placeholder = importlib.import_module(
        "constant2placeholder"
    ).const2placeholder

    assets = model_path.parent
    config = read_json(assets / "aotconfig.json")
    if batch < 1 or batch > config["max_batch_size"]:
        raise RegionError(
            "BATCH_CONSTRAINT", str(batch), maximum=config["max_batch_size"]
        )
    registration = aot_ops.register_required_ops(
        config, artifact_path=model_path, strict=True
    )
    ep = torch.export.load(model_path)
    template = ep.example_inputs[0][0]
    slices, normalization = validation._load_integer_float_slices(assets, template)
    provider = validation.SyntheticValidationInputProvider(
        batch_sizes=[batch], seeds=[], integer_float_slices=slices
    )
    captured = provider.build_cases(template, config)[0]
    business_inputs = {
        key: provider._canonicalize_integer_float_slices(
            provider._values(value, key, batch), key
        )
        for key, value in captured.inputs.items()
    }
    ep._example_inputs = ((business_inputs,), {})
    with torch.no_grad():
        exported, weight_names, weights, shapes, name_map = const2placeholder(
            ep, config["dynamic"], config["max_batch_size"]
        )
        cuda_ep, examples = aot_compile.export_program_to_cuda(
            exported, config["dynamic"], shapes
        )
    configs, ignored = aot_compile.build_aot_inductor_configs(
        config.get("unbacked_safe", False), config.get("export_optimizations", {})
    )
    configs.update(
        {
            "aot_inductor.use_runtime_constant_folding": False,
            "max_autotune_gemm_backends": "ATEN,CUTLASS,TRITON",
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "trace.enabled": True,
        }
    )
    tf32 = aot_compile.configure_tf32(config["use_tf32"])
    lowerings = aot_ops.register_compile_lowerings(cuda_ep)
    inputs = examples
    import hashlib

    from .core import tensor_meta

    input_hashes = {
        key: hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for key, value in business_inputs.items()
    }
    report = {
        "model": str(model_path),
        "model_sha256": file_hash(model_path),
        "batch": batch,
        "sample": "captured-b80" if batch == 80 else f"captured-resized-b{batch}",
        "input_hashes": input_hashes,
        "normalization": normalization,
        "weight_names": weight_names,
        "name_map": name_map,
        "model_config": config,
        "configs": configs,
        "tf32": tf32,
        "registration": registration,
        "lowerings": lowerings,
        "range_constraints": {
            str(key): {"min": str(value.lower), "max": str(value.upper)}
            for key, value in cuda_ep.range_constraints.items()
        },
        "input_metadata": tensor_meta(inputs),
        "triton_compatibility": {
            "required": required_triton,
            "used": aot_runtime.AOT_TRITON_VERSION,
        },
        "helper_sha256": file_hash(fourier_root / "fourier/aot_validation.py"),
    }
    print(
        json.dumps(
            {
                "stage": "model_loaded",
                "batch": batch,
                "input_hashes": input_hashes,
                "normalization": normalization,
                "configs": configs,
            }
        ),
        flush=True,
    )
    del ep, exported, weights, business_inputs
    gc.collect()
    return cuda_ep, inputs, report


def compile_full(model_path, fourier_root, run, build, capture_plan=None):
    import torch

    from .adapter import CompilationRecorder
    from .core import write_json

    optimization_scope = importlib.import_module(
        "fourier.aot_optimization_policy"
    ).optimization_scope
    ep, inputs, report = load_full_model(model_path, fourier_root)
    write_json(run / "inputs.json", report)
    from .core import save_graph

    save_graph(
        ep.graph_module,
        [node.meta.get("val") for node in ep.graph.nodes if node.op == "placeholder"],
        build / "input_graph",
    )
    package = build / "full.pt2"
    model_config = report["model_config"]
    print(json.dumps({"stage": "compile_full", "package": str(package)}), flush=True)
    with (
        torch.no_grad(),
        optimization_scope(
            model_config.get("export_optimizations", {}),
            use_tf32=model_config["use_tf32"],
            unbacked_safe=model_config.get("unbacked_safe", False),
        ),
    ):
        with CompilationRecorder(build / "diagnostics", capture_plan):
            torch._inductor.aoti_compile_and_package(
                ep, package_path=str(package), inductor_configs=dict(report["configs"])
            )
    write_json(
        run / "compilation.json", read_json(build / "diagnostics/compilation.json")
    )
    print(json.dumps({"stage": "compiled", "package": str(package)}), flush=True)
    del ep
    gc.collect()
    return package, inputs, report


def prepare(args):
    run = args.run_dir.resolve()
    build = (
        args.build_dir.resolve()
        if args.build_dir
        else Path.cwd()
        / "build/inductor_test/kernel-fx-subgraph"
        / (run.name + "_" + uuid.uuid4().hex[:8])
    )
    torch = configure_process(run, build)
    from .core import file_hash, save_tensors, write_json

    identity = source_identity(args.source_commit)
    sys.path[:0] = [
        str(args.torchfourier_root),
        str(args.torchfourier_root / "fourier"),
    ]
    package, inputs, report = compile_full(
        args.exported_program.resolve(), args.torchfourier_root, run, build
    )
    model = torch._inductor.aoti_load_package(str(package), run_single_threaded=True)
    with torch.inference_mode():
        output = model(*inputs)
        torch.cuda.synchronize()
    save_tensors(list(output), run / "baseline_output_b80")
    write_json(
        run / "bundle.json",
        {
            "schema_version": 1,
            "run": str(run),
            "build": str(build),
            "identity": identity,
            "source_root": os.environ.get("FX_SUBGRAPH_SOURCE_ROOT"),
            "model": str(args.exported_program.resolve()),
            "torchfourier_root": str(args.torchfourier_root),
            "package": str(package),
            "input_report": str(run / "inputs.json"),
            "compilation": str(run / "compilation.json"),
            "status": "PREPARED",
            "hashes": {
                "compilation": file_hash(run / "compilation.json"),
                "input_report": file_hash(run / "inputs.json"),
                "package": file_hash(package),
            },
            "artifacts": {
                "scheduler_ir": [str(path) for path in build.rglob("ir_*fusion.txt")],
                "debug_root": str(build / "debug"),
                "trace_root": str(run / "trace"),
            },
        },
    )
    print(
        json.dumps({"status": "PREPARED", "bundle": str(run / "bundle.json")}),
        flush=True,
    )


def extract(args):
    from .core import (
        boundary_plan,
        coalesce_inputs,
        extract_graph,
        load_graph,
        save_graph,
        write_json,
    )

    bundle_path = args.bundle.resolve()
    bundle = read_bundle(bundle_path)
    compilation = read_json(bundle["compilation"])
    gm = load_graph(compilation["graph"]["directory"])
    subgraph, report = extract_graph(
        gm,
        compilation["provenance"]["cppCodeToPost"],
        args.kernel,
        compilation["graph"]["identity"],
    )
    plan = boundary_plan(report, compilation["values"], compilation["calls"])
    if set(compilation["mutated_inputs"]).intersection(
        row.get("buffer") for row in plan["inputs"]
    ):
        from .core import RegionError

        raise RegionError(
            "CROSS_REGION_MUTATION", "A boundary input is mutated by the full graph"
        )
    plan = coalesce_inputs(subgraph, plan)
    case_id = uuid.uuid4().hex[:12]
    case_dir = Path(bundle["run"]) / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    case_build = Path(bundle["build"]) / "cases" / case_id
    case_build.mkdir(parents=True, exist_ok=False)
    save_graph(subgraph, [], case_build / "extracted_graph")
    write_json(case_dir / "mapping.json", report)
    write_json(case_dir / "boundary.json", plan)
    write_json(
        case_dir / "case.json",
        {
            "bundle": str(bundle_path),
            "run": str(case_dir),
            "build": str(case_build),
            "graph": str(case_build / "extracted_graph"),
            "status": "EXTRACTED",
        },
    )
    print(
        json.dumps(
            {"status": "EXTRACTED", "case": str(case_dir / "case.json"), **report},
            indent=2,
        )
    )


def capture(args):
    case = read_json(args.case)
    run, build = Path(case["run"]) / "capture", Path(case["build"]) / "capture"
    torch = configure_process(run, build)
    bundle = read_bundle(case["bundle"])
    from .adapter import compare_compilations
    from .core import compare, RegionError, save_tensors, write_json
    from .snapshots import capture_stdout, restore_capture

    write_json(
        run / "source_identity.json", source_identity(bundle["identity"]["commit"])
    )
    baseline = read_json(bundle["compilation"])
    plan = read_json(Path(case["run"]) / "boundary.json")
    cache = Path(bundle["build"]) / "cache/cache"
    seeded = []
    if cache.exists():
        shutil.copytree(cache, build / "cache/cache")
        seeded.extend(str(path) for path in cache.rglob("*") if path.is_file())
    cache_root = Path(bundle["build"]) / "cache"
    for source in cache_root.rglob("*"):
        if source.is_file() and source.suffix in (".kernel_perf", ".best_config"):
            destination = build / "cache" / source.relative_to(cache_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            seeded.append(str(source))
    from .core import file_hash

    write_json(
        run / "autotune_source.json",
        {
            "source": str(cache_root),
            "files": [{"path": path, "sha256": file_hash(path)} for path in seeded],
            "purpose": "Reuse baseline algorithm, fusion benchmark and pointwise measurements for capture-copy equivalence only; no graph or package cache",
        },
    )
    fourier_root = Path(bundle["torchfourier_root"])
    sys.path[:0] = [str(fourier_root), str(fourier_root / "fourier")]
    package, inputs, report = compile_full(
        Path(bundle["model"]), fourier_root, run, build, plan
    )
    original_inputs = read_json(bundle["input_report"])
    if (
        report["model_sha256"] != original_inputs["model_sha256"]
        or report["input_hashes"] != original_inputs["input_hashes"]
    ):
        raise RegionError(
            "CAPTURE_INPUT_MISMATCH", "Model or full-model inputs changed since prepare"
        )
    changed = [
        key
        for key in (
            "configs",
            "model_config",
            "tf32",
            "helper_sha256",
            "triton_compatibility",
        )
        if report[key] != original_inputs[key]
    ]
    if changed:
        raise RegionError(
            "CAPTURE_CONFIG_MISMATCH",
            "Compilation configuration changed since prepare",
            fields=changed,
        )
    parity = compare_compilations(baseline, read_json(run / "compilation.json"))
    write_json(run / "compiler_parity.json", parity)
    if not parity["passed"]:
        raise RegionError(
            "CAPTURE_COMPILER_MISMATCH",
            "Capture copy differs from baseline",
            checks=parity["checks"],
        )
    original = torch._inductor.aoti_load_package(
        bundle["package"], run_single_threaded=True
    )
    instrumented = torch._inductor.aoti_load_package(
        str(package), run_single_threaded=True
    )
    validation = importlib.import_module("fourier.aot_validation")

    rows = []
    batches = list(dict.fromkeys(args.batch or [80, 1, 256]))
    original_cwd = Path.cwd()
    for batch in batches:
        if batch < 1 or batch > report["model_config"]["max_batch_size"]:
            raise RegionError("BATCH_CONSTRAINT", str(batch))
        sample = dict(inputs[0])
        for source, target in report["name_map"].items():
            sample[target] = validation.SyntheticValidationInputProvider._values(
                inputs[0][target], source, batch
            ).cuda()
        directory = run / f"b{batch}"
        directory.mkdir()
        import hashlib

        actual_hashes = {
            source: hashlib.sha256(
                sample[target].cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for source, target in report["name_map"].items()
        }
        with torch.inference_mode():
            expected = original(sample)
            torch.cuda.synchronize()
            try:
                os.chdir(directory)
                with capture_stdout(directory / "calls.log"):
                    actual = instrumented(sample)
                    torch.cuda.synchronize()
            finally:
                os.chdir(original_cwd)
        full_parity = compare(actual, expected)
        write_json(directory / "full_output_parity.json", full_parity)
        if not full_parity["passed"]:
            raise RegionError(
                "CAPTURE_OUTPUT_MISMATCH", str(batch), comparison=full_parity
            )
        boundary_inputs, boundary_outputs, metadata = restore_capture(plan, directory)
        save_tensors(boundary_inputs, directory / "inputs")
        save_tensors(boundary_outputs, directory / "outputs")
        write_json(
            directory / "source.json",
            {
                "batch": batch,
                "input_hashes": actual_hashes,
                "capture_package_sha256": file_hash(package),
                "metadata": metadata,
                "forward_count": 1,
                "compiler_parity": True,
                "full_output_parity": True,
            },
        )
        rows.append({"batch": batch, "directory": str(directory), "passed": True})
    write_json(run / "capture.json", {"status": "CAPTURED", "samples": rows})
    print(json.dumps({"status": "CAPTURED", "samples": rows}), flush=True)


def verify(args):
    from .verification import verify_case

    verify_case(args)
