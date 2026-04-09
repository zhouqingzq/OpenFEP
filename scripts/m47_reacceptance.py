from __future__ import annotations

import argparse
import json

from segmentum.m47_reacceptance import write_m47_reacceptance_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild independent M4.7 acceptance evidence.")
    parser.add_argument(
        "--include-regressions",
        action="store_true",
        help="Also run the M4.1-M4.6 regression prerequisite for G9.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Optional output directory for the evidence JSON and summary.",
    )
    args = parser.parse_args()

    paths = write_m47_reacceptance_artifacts(
        include_regressions=args.include_regressions,
        reports_dir=args.reports_dir,
    )
    print(
        json.dumps(
            {
                "include_regressions": args.include_regressions,
                "paths": paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
