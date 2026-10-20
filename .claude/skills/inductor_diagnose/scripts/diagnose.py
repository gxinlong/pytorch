#!/usr/bin/env python3
"""Launch one fresh diagnostic process, or inspect a PT2 without importing Torch."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from model_io import graph_devices, inspect_model, sha


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def resolve_config(args):
    result = {"inductor_configs": {}, "environment": {}, "torch_backends": {}}
    if args.profile_file:
        profiles = json.loads(args.profile_file.read_text())
        selected = args.profile or profiles.get("default_profile", "default")
        result.update(profiles["profiles"][selected])
    elif args.profile:
        raise ValueError("--profile requires --profile-file")
    if args.config:
        config = json.loads(args.config.read_text())
        if not any(
            key in config
            for key in ("inductor_configs", "environment", "torch_backends")
        ):
            config = {"inductor_configs": config}
        for section in result:
            result[section] = {**result[section], **config.get(section, {})}
    for override in args.set:
        key, value = override.split("=", 1)
        result["inductor_configs"][key] = json.loads(value)
    if args.tf32 != "inherit":
        result["torch_backends"].update(
            {
                "cuda.matmul.allow_tf32": args.tf32 == "on",
                "cudnn.allow_tf32": args.tf32 == "on",
            }
        )
    return result


def check_gpu(selected):
    fields = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                selected,
                "--query-gpu=uuid,name,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(fields) != 1:
        raise ValueError("Select exactly one GPU")
    uuid = fields[0].split(",")[0]
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(
            f"GPU {uuid} is occupied; no compilation started. PIDs: {processes}"
        )
    return {"uuid": uuid, "snapshot": fields[0], "compute_pids": processes}


def prepare(args):
    info = inspect_model(args.pt2, args.model_name, args.graph_source)
    devices = graph_devices(info.pop("graph"))
    if args.inspect:
        print(
            json.dumps({**info, "graph_devices": devices}, ensure_ascii=False, indent=2)
        )
        return None
    config = resolve_config(args)
    if any(d not in ("cpu", "cuda") for d in devices):
        raise ValueError(f"This adapter supports CPU/CUDA; graph devices: {devices}")
    device = "cuda" if "cuda" in devices else "cpu"
    gpu = None
    if device == "cuda":
        selected = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES")
        if not selected or "," in selected or selected == "-1":
            raise ValueError(
                "CUDA graph requires --gpu or a single selected CUDA_VISIBLE_DEVICES"
            )
        gpu = check_gpu(selected)
    elif args.gpu:
        raise ValueError(
            "The saved graph is CPU; --gpu does not move an ExportedProgram"
        )
    model = re.sub(r"[^A-Za-z0-9_.-]", "_", args.pt2.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = (
        args.output_dir
        or Path.cwd() / "agent_space/model_subgraph" / model / f"diagnose_{stamp}"
    ).resolve()
    run.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    snapshot = run / "scripts_snapshot"
    snapshot.mkdir()
    for path in scripts.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    files = {**info["files"], info["input"]: info["input_sha256"]}
    for path in (
        args.inputs,
        args.reference_aot,
        args.reference_outputs,
        args.config,
        args.profile_file,
        args.baseline_compilation,
    ):
        if path:
            files[str(path.resolve())] = sha(path)
    if args.setup:
        files[str(args.setup.resolve())] = sha(args.setup)
    env = os.environ.copy()
    inherited = {
        k: v
        for k, v in env.items()
        if k.startswith(
            ("TORCH", "TRITON", "CUDA", "INDUCTOR", "AOT_", "FIXED_", "CONCAT_")
        )
    }
    for key, value in config["environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value).replace("<work_dir>", str(run))
    for key, folder in {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "xdg",
        "CUDA_CACHE_PATH": "cuda_cache",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "TORCHINDUCTOR_CACHE_DIR": "cache",
        "TRITON_CACHE_DIR": "triton",
        "TORCH_COMPILE_DEBUG_DIR": "debug",
        "TORCH_TRACE": "trace",
    }.items():
        (run / folder).mkdir()
        env[key] = str(run / folder)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "INDUCTOR_PROVENANCE": "1",
            "TORCH_COMPILE_DEBUG": "1",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCH_LOGS": "+inductor,fusion,schedule,ir_pre_fusion,ir_post_fusion,output_code",
            "TORCH_LOGS_OUT": str(run / "compiler.log"),
            "LOG_AUTOTUNE_RESULTS": "1",
            "INDUCTOR_LOG_URL_FOR_GRAPH_XFORM": str(run / "graph_transforms"),
            "CUDA_VISIBLE_DEVICES": gpu["uuid"] if gpu else "-1",
        }
    )
    (run / "graph_transforms").mkdir()
    if args.source_root:
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(args.source_root.resolve()), env.get("PYTHONPATH")))
        )
    config["inductor_configs"].update(
        {
            "trace.enabled": True,
            "trace.debug_log": True,
            "trace.debug_dir": str(run / "debug"),
            "trace.log_autotuning_results": True,
            "trace.log_url_for_graph_xform": str(run / "graph_transforms"),
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    plan = {
        "schema_version": 1,
        "model_name": args.model_name or args.pt2.stem,
        "model": info,
        "input_files": files,
        "inputs": str(args.inputs.resolve()) if args.inputs else None,
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "setup": str(args.setup.resolve()) if args.setup else None,
        "reference_aot": str(args.reference_aot.resolve())
        if args.reference_aot
        else None,
        "reference_outputs": str(args.reference_outputs.resolve())
        if args.reference_outputs
        else None,
        "skip_validation": args.skip_validation,
        "rtol": args.rtol,
        "atol": args.atol,
        "device": device,
        "gpu": gpu,
        "compile_configs": config["inductor_configs"],
        "torch_backends": config["torch_backends"],
        "profile_environment": config["environment"],
        "inherited_environment": inherited,
        "diagnostic_environment": {
            k: env[k]
            for k in env
            if k in inherited
            or k.startswith(
                (
                    "TORCH",
                    "TRITON",
                    "CUDA",
                    "INDUCTOR",
                    "PYTHON",
                    "AOT_",
                    "FIXED_",
                    "CONCAT_",
                )
            )
        },
        "scripts": {p.name: sha(p) for p in snapshot.glob("*.py")},
        "tlparse": str(args.tlparse.resolve())
        if args.tlparse
        else shutil.which("tlparse"),
        "argv": sys.argv,
        "python": sys.executable,
    }
    write(run / "run_manifest.json", plan)
    if args.baseline_compilation:
        shutil.copyfile(args.baseline_compilation, run / "historical_compilation.json")
    write(run / "status.json", {"status": "prepared", "compile_invocations": 0})
    (run / "README.txt").write_text(
        "Inductor 单次编译诊断。report.html 是中文分析；tlparse/ 是原生报告。\n"
        "observations/decisions.jsonl 保存补充事件；diagnostics/compilation.json 保存节点到调用映射。\n"
        "scripts_snapshot/ 保存实际执行脚本；run_manifest.json 保存配置及输入身份。\n"
        "源码及运行环境见 identity.json 和 source_snapshot/。\n"
        "报告解释本次编译，调优计时不是端到端性能结果。\n"
    )
    write(
        run / "postprocessing.json",
        {"scripts": "scripts_snapshot", "model_recompiled_for_report": False},
    )
    write(
        run / "command.json",
        {
            "argv": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "new_output_required": True,
        },
    )
    return run, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt2", type=Path)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect archive only; no Torch import or compile",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--graph-source",
        type=Path,
        help="Original EP PT2 or explicit serialized sidecar directory",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="torch.save((args, kwargs), file), preserving storage aliases",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument(
        "--gpu", help="Physical GPU index or UUID; never switch automatically"
    )
    parser.add_argument("--source-root", type=Path, help="Expected PyTorch source root")
    parser.add_argument("--tf32", choices=("inherit", "on", "off"), default="inherit")
    parser.add_argument(
        "--setup",
        type=Path,
        help="Optional user-selected module with compilation_context(plan)",
    )
    parser.add_argument("--reference-aot", type=Path)
    parser.add_argument("--reference-outputs", type=Path)
    parser.add_argument("--baseline-compilation", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--tlparse", type=Path)
    args = parser.parse_args()
    prepared = prepare(args)
    if prepared is None:
        return
    run, env = prepared
    command = [
        sys.executable,
        "-B",
        "-u",
        str(run / "scripts_snapshot/compile_worker.py"),
        str(run),
    ]
    print(
        json.dumps({"run": str(run), "command": command}, ensure_ascii=False),
        flush=True,
    )
    with (run / "execution.log").open("x") as log:
        process = subprocess.run(
            command, env=env, cwd=run, stdout=log, stderr=subprocess.STDOUT
        )
    if process.returncode:
        write(
            run / "process_failure.json",
            {"returncode": process.returncode, "log": "execution.log"},
        )
        write(
            run / "status.json",
            {
                "status": "failed",
                "returncode": process.returncode,
                "compile_invocations": int((run / "compile_started.json").exists()),
            },
        )
        with (run / "failure_report.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(run / "scripts_snapshot/diagnostic_report.py"),
                    str(run),
                ],
                env=env,
                cwd=run,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    print(
        json.dumps(
            {"run": str(run), "returncode": process.returncode}, ensure_ascii=False
        )
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
