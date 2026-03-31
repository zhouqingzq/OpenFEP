from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .benchmark_registry import (
    benchmark_status,
    benchmark_root,
    import_benchmark_bundle,
    list_benchmark_bundles,
    validate_benchmark_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Segmentum benchmark bundles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered benchmark bundles.")
    list_parser.add_argument("--root", type=Path, default=None, help="Optional benchmark registry root override.")

    validate_parser = subparsers.add_parser("validate", help="Validate a benchmark bundle.")
    validate_parser.add_argument("benchmark_id", help="Benchmark identifier to validate.")
    validate_parser.add_argument("--root", type=Path, default=None, help="Optional benchmark registry root override.")

    import_parser = subparsers.add_parser("import", help="Import an external benchmark bundle directory or zip.")
    import_parser.add_argument("source", type=Path, help="Path to a bundle directory or zip archive.")
    import_parser.add_argument("--root", type=Path, default=None, help="Optional benchmark registry root override.")
    import_parser.add_argument("--benchmark-id", default=None, help="Optional benchmark id override.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "list":
            payload = {
                "root": str((args.root or benchmark_root()).resolve()),
                "bundles": [
                    bundle.to_dict() | {"benchmark_status": benchmark_status(bundle.benchmark_id, root=args.root).to_dict()}
                    for bundle in list_benchmark_bundles(root=args.root)
                ],
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "validate":
            payload = validate_benchmark_bundle(args.benchmark_id, root=args.root).to_dict()
            payload["benchmark_status"] = benchmark_status(args.benchmark_id, root=args.root).to_dict()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if payload["ok"] else 2

        if args.command == "import":
            bundle = import_benchmark_bundle(args.source, benchmark_id=args.benchmark_id, destination_root=args.root)
            payload = {
                "imported": bundle.to_dict(),
                "benchmark_status": benchmark_status(bundle.benchmark_id, root=args.root).to_dict(),
                "registry_root": str((args.root or benchmark_root()).resolve()),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
