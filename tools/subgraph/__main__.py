import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract an AOT FX region by exact kernel names"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--exported-program", required=True, type=Path)
    prepare.add_argument("--run-dir", required=True, type=Path)
    prepare.add_argument("--build-dir", type=Path)
    prepare.add_argument(
        "--torchfourier-root",
        type=Path,
        default=Path("/root/gaoxinlong/Desktop/repo/torchfourier"),
    )
    prepare.add_argument(
        "--source-commit", default="2b3ec34829036a65cd9d1398ea72a0167dc37470"
    )
    extract = commands.add_parser("extract")
    extract.add_argument("--bundle", required=True, type=Path)
    extract.add_argument("--kernel", action="append", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--case", required=True, type=Path)
    capture.add_argument("--batch", action="append", type=int)
    verify = commands.add_parser("verify")
    verify.add_argument("--case", required=True, type=Path)
    verify.add_argument("--benchmark", action="store_true")
    verify.add_argument(
        "--attempt",
        help="Create a new verification directory using the existing captured data",
    )
    args = parser.parse_args()
    source = os.environ.get("FX_SUBGRAPH_SOURCE_ROOT")
    if source:
        sys.path.insert(0, source)
    from . import runner

    try:
        getattr(runner, args.command)(args)
    except Exception as error:
        if not hasattr(error, "report"):
            raise
        import json

        print(
            json.dumps({"status": "BLOCKED", **error.report}, indent=2), file=sys.stderr
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
