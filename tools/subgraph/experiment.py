import argparse
import json
from pathlib import Path

from .runner import configure_process, read_json, source_identity


def main(manifest_path):
    parser = argparse.ArgumentParser(
        description="Run an independently compiled FX region with captured data"
    )
    parser.add_argument("--batch", type=int, default=80)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("aot", "eager"), default="aot")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch = configure_process(args.run_dir.resolve(), args.build_dir.resolve())
    from .core import (
        compare,
        file_hash,
        load_graph,
        load_tensors,
        RegionError,
        write_json,
    )
    from .verification import benchmark

    manifest = read_json(manifest_path)
    write_json(
        args.run_dir / "source_identity.json",
        source_identity(manifest["source_commit"]),
    )
    if file_hash(manifest["package"]) != manifest["package_sha256"]:
        raise RegionError("PACKAGE_HASH", manifest["package"])
    sample = next(
        (row for row in manifest["samples"] if row["batch"] == args.batch), None
    )
    if sample is None:
        raise RegionError("MISSING_CAPTURE_BATCH", str(args.batch))
    inputs = tuple(
        value
        for index, value in enumerate(
            load_tensors(Path(sample["directory"]) / "inputs", "cuda")
        )
        if index not in manifest["removed_scalar_inputs"]
    )
    outputs = tuple(load_tensors(Path(sample["directory"]) / "outputs", "cuda"))
    torch.backends.cuda.matmul.allow_tf32 = manifest["use_tf32"]
    torch.backends.cudnn.allow_tf32 = manifest["use_tf32"]
    if args.mode == "aot":
        model = torch._inductor.aoti_load_package(
            manifest["package"], run_single_threaded=True
        )
    else:
        model = load_graph(manifest["graph"]).cuda()
    with torch.inference_mode():
        result = {
            "batch": args.batch,
            "mode": args.mode,
            "numerics": compare(tuple(model(*inputs)), outputs),
        }
        if result["numerics"]["passed"] and args.benchmark:
            result["benchmark"] = benchmark(model, inputs)
    write_json(args.run_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    if not result["numerics"]["passed"]:
        raise RegionError("NUMERICS_FAILED", str(args.run_dir / "result.json"))
