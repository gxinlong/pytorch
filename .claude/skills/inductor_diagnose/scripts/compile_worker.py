"""Fresh-process worker. All configuration environment is applied by diagnose.py."""

import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, nullcontext
from pathlib import Path

from diagnose import check_gpu, write
from model_io import (
    clone_inputs,
    compare,
    input_metadata,
    load_inputs,
    load_program,
    sha,
)


def git(source, *args):
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def setup_context(plan):
    if not plan.get("setup"):
        return nullcontext()
    spec = importlib.util.spec_from_file_location("diagnose_user_setup", plan["setup"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compilation_context(plan)


def verify_files(files):
    for name, expected in files.items():
        if sha(name) != expected:
            raise RuntimeError(f"File changed since run preparation: {name}")


def compile_once(run, plan):
    verify_files(plan["input_files"])
    verify_files(
        {
            str(run / "scripts_snapshot" / name): value
            for name, value in plan["scripts"].items()
        }
    )
    if plan["gpu"]:
        current = check_gpu(plan["gpu"]["uuid"])
        write(run / "gpu_before.json", current)
    from compilation_recorder import CompilationRecorder
    from diagnostic_observer import DecisionObserver
    from graph_records import graph_record

    import torch
    import torch._inductor.scheduler as scheduler

    source = Path(torch.__file__).resolve().parent.parent
    if plan["source_root"] and source != Path(plan["source_root"]):
        raise RuntimeError(
            f"Imported Torch source mismatch: {source} != {plan['source_root']}"
        )
    if Path(scheduler.__file__).resolve() != source / "torch/_inductor/scheduler.py":
        raise RuntimeError("Scheduler source differs from the imported Torch source")
    source_files = {
        str(p.relative_to(source)): sha(p)
        for p in (source / "torch/_inductor").rglob("*.py")
    }
    source_files["torch/export/__init__.py"] = sha(source / "torch/export/__init__.py")
    identity = {
        "commit": git(source, "rev-parse", "HEAD"),
        "branch": git(source, "branch", "--show-current"),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "wheel_commit": torch.version.git_version,
        "native_extension": torch._C.__file__,
        "python": sys.executable,
        "source_root": str(source),
        "git_status": git(source, "status", "--porcelain"),
        "gpu": plan["gpu"],
        "compiler_source_hashes": source_files,
    }
    if plan["device"] == "cuda":
        import triton

        identity["triton"] = triton.__version__
        torch.cuda.set_device(0)
    patch = git(source, "diff", "--binary", "HEAD", "--", "torch")
    (run / "source.patch").write_text(patch or "")
    write(run / "identity.json", identity)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for name, value in plan["torch_backends"].items():
        owner = torch.backends
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], value)
    write(
        run / "backend_settings.json",
        {
            "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    )
    with ExitStack() as stack:
        stack.enter_context(setup_context(plan))
        ep = load_program(plan["model"])
        args, kwargs = load_inputs(ep, plan["inputs"])
        ep._example_inputs = (args, kwargs)
        write(run / "input_fx.json", graph_record(ep.graph_module))
        (run / "input_fx.py").write_text(ep.graph_module.code)
        write(run / "input_metadata.json", input_metadata(args, kwargs))
        write(
            run / "export_contract.json",
            {
                "signature": str(ep.graph_signature),
                "call_spec": str(ep.call_spec),
                "range_constraints": {
                    str(k): str(v) for k, v in ep.range_constraints.items()
                },
            },
        )
        torch.save((args, kwargs), run / "inputs.pt")
        saved_inputs = torch.load(run / "inputs.pt", weights_only=False)
        if input_metadata(*saved_inputs) != input_metadata(args, kwargs):
            raise RuntimeError(
                "Input serialization changed shape, stride, offset or storage aliases"
            )
        if not plan["skip_validation"] and (
            ep.graph_signature.buffers_to_mutate
            or ep.graph_signature.user_inputs_to_mutate
        ):
            raise ValueError(
                "Automatic comparison requires a non-mutating graph. Supply an appropriate validation workflow; --skip-validation is explicit."
            )
        reference = None
        stack.enter_context(torch.no_grad())
        if not plan["skip_validation"]:
            ref_args, ref_kwargs = clone_inputs((args, kwargs))
            reference = ep.module()(*ref_args, **ref_kwargs)
            torch.save(reference, run / "eager_reference.pt")
        configs = plan["compile_configs"]
        stack.enter_context(torch._inductor.config.patch(configs))
        write(
            run / "effective_config.json",
            {
                k: v
                for k, v in torch._inductor.config.get_config_copy().items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        )
        write(run / "compile_configs.json", configs)
        started = time.monotonic()
        package = run / "subgraph.pt2"
        observer = DecisionObserver(run / "observations", source)
        try:
            with CompilationRecorder(run / "diagnostics"), observer:
                write(run / "compile_started.json", {"time": time.time()})
                torch._inductor.aoti_compile_and_package(
                    ep, package_path=str(package), inductor_configs=configs
                )
        finally:
            names = {
                "torch/_inductor/scheduler.py",
                "torch/_inductor/codegen/triton_combo_kernel.py",
            }
            if (run / "observations/decisions.jsonl").exists():
                for line in (
                    (run / "observations/decisions.jsonl").read_text().splitlines()
                ):
                    names.add(json.loads(line)["file"])
            hashes = {}
            for name in sorted(names):
                path = source / name
                if path.is_file():
                    target = run / "source_snapshot" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    hashes[name] = sha(path)
            write(run / "source_snapshot_hashes.json", hashes)
        write(
            run / "compile_result.json",
            {
                "status": "compiled",
                "compile_invocations": 1,
                "seconds": time.monotonic() - started,
                "package": str(package),
                "sha256": sha(package),
            },
        )
        verify_files(plan["input_files"])
        verify_files(
            {str(source / name): digest for name, digest in source_files.items()}
        )
        model = torch._inductor.aoti_load_package(
            str(package), run_single_threaded=True
        )
        if plan["skip_validation"]:
            write(
                run / "validation.json",
                {
                    "passed": None,
                    "status": "skipped_by_request",
                    "package_load": "passed",
                },
            )
        else:
            test_args, test_kwargs = clone_inputs((args, kwargs))
            actual = model(*test_args, **test_kwargs)
            checks = {
                "vs_exported_program": compare(
                    actual, reference, plan["rtol"], plan["atol"]
                )
            }
            if plan["reference_outputs"]:
                saved = torch.load(plan["reference_outputs"], weights_only=False)
                checks["vs_saved_reference"] = compare(
                    actual, saved, plan["rtol"], plan["atol"]
                )
            if plan["reference_aot"]:
                old = torch._inductor.aoti_load_package(
                    plan["reference_aot"], run_single_threaded=True
                )
                old_args, old_kwargs = clone_inputs((args, kwargs))
                checks["vs_reference_aot"] = compare(
                    actual, old(*old_args, **old_kwargs), plan["rtol"], plan["atol"]
                )
            if plan["device"] == "cuda":
                torch.cuda.synchronize()
            passed = all(c["passed"] for c in checks.values())
            write(
                run / "validation.json",
                {
                    "passed": passed,
                    "checks": checks,
                    "package_load": "passed",
                    "scope": "one saved input case; no exhaustive input or performance validation",
                },
            )
            if not passed:
                raise RuntimeError(
                    "Compiled outputs failed correctness comparison; see validation.json"
                )


def main(run):
    run = Path(run).resolve()
    with (run / "worker_started.json").open("x") as stream:
        json.dump({"time": time.time()}, stream)
    plan = json.loads((run / "run_manifest.json").read_text())
    failure, report_error = None, None
    try:
        compile_once(run, plan)
    except Exception:
        failure = traceback.format_exc()
        write(run / "failure.json", {"traceback": failure})
    try:
        from diagnostic_report import build_report

        analysis = build_report(run)
        if not failure:
            from torch._logging import trace_structured

            trace_structured(
                "artifact",
                metadata_fn=lambda: {"name": "combo_decisions", "encoding": "json"},
                payload_fn=lambda: json.dumps(
                    analysis["combo_decisions"], ensure_ascii=False
                ),
            )
        parses = []
        parser = plan["tlparse"]
        if parser and Path(parser).is_file():
            for index, log in enumerate(sorted((run / "trace").glob("*.log"))):
                output = run / "tlparse" / f"trace_{index}"
                command = [
                    parser,
                    str(log),
                    "--inductor-provenance",
                    "--no-browser",
                    "-o",
                    str(output),
                ]
                with (run / f"tlparse_{index}.log").open("w") as stream:
                    result = subprocess.run(
                        command, stdout=stream, stderr=subprocess.STDOUT
                    )
                parses.append(
                    {
                        "command": command,
                        "returncode": result.returncode,
                        "index": str(output / "index.html"),
                    }
                )
        write(
            run / "tlparse_result.json", {"available": bool(parser), "traces": parses}
        )
        analysis = build_report(run)
    except Exception:
        report_error = traceback.format_exc()
        write(run / "report_failure.json", {"traceback": report_error})
    incomplete = bool(report_error)
    if not report_error:
        observer = analysis["observer"]
        incomplete = bool(
            observer.get("errors")
            or observer.get("unfinished_calls")
            or observer.get("combo_pending_statements")
            or observer.get("missing_combo_targets")
            or not parses
            or any(p["returncode"] or not Path(p["index"]).is_file() for p in parses)
            or not analysis["kernels"]
            or any(not k["members"] for k in analysis["kernels"])
        )
    status = (
        "failed" if failure else ("complete_with_gaps" if incomplete else "complete")
    )
    write(
        run / "status.json",
        {
            "status": status,
            "compile_invocations": int((run / "compile_started.json").exists()),
            "report": str(run / "report.html"),
            "failure": bool(failure),
            "report_failure": bool(report_error),
        },
    )
    if not report_error:
        try:
            build_report(run)
        except Exception:
            write(run / "report_failure.json", {"traceback": traceback.format_exc()})
            write(run / "status.json", {"status": "failed", "report_failure": True})
            raise SystemExit(1) from None
    print(json.dumps({"status": status, "run": str(run)}), flush=True)
    if failure or report_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
