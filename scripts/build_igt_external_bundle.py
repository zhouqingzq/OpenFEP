from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.benchmark_registry import benchmark_status, validate_benchmark_bundle
from segmentum.igt_external_bundle import build_igt_external_bundle


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an iowa_gambling_task external benchmark bundle from local IGT participant folders.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "igt dataset",
        help="Raw IGT dataset directory.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=ROOT / "external_benchmark_registry",
        help="Benchmark registry root where the generated bundle will be written.",
    )
    args = parser.parse_args()

    report = build_igt_external_bundle(args.source, args.destination_root)
    validation = validate_benchmark_bundle("iowa_gambling_task", root=args.destination_root)
    status = benchmark_status("iowa_gambling_task", root=args.destination_root)
    print(
        json.dumps(
            {
                "build_report": report.to_dict(),
                "validation": validation.to_dict(),
                "benchmark_status": status.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
