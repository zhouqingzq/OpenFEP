from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from segmentum.m47_strict_audit import write_m47_strict_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict M4.7 code acceptance audit.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for strict audit outputs.",
    )
    parser.add_argument(
        "--skip-self-tests",
        action="store_true",
        help="Skip the three M4.7 self-test commands.",
    )
    parser.add_argument(
        "--include-live-regressions",
        action="store_true",
        help="Run the full live M4.1-M4.6 regression suite for strict G9.",
    )
    parser.add_argument(
        "--self-test-timeout-seconds",
        type=int,
        default=900,
        help="Timeout for each M4.7 self-test command.",
    )
    parser.add_argument(
        "--regression-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout for the live M4.1-M4.6 regression command.",
    )
    args = parser.parse_args()

    round_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    paths = write_m47_strict_audit(
        round_started_at=round_started_at,
        output_root=args.output_root,
        run_m47_self_tests=not args.skip_self_tests,
        run_live_regressions=args.include_live_regressions,
        self_test_timeout_seconds=args.self_test_timeout_seconds,
        regression_timeout_seconds=args.regression_timeout_seconds,
    )
    print(json.dumps({"paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
