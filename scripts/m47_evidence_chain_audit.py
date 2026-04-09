from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from segmentum.m47_evidence_chain_audit import write_m47_evidence_chain_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the M4.7 evidence chain without running legacy regressions.")
    parser.add_argument(
        "--include-regressions",
        action="store_true",
        help="Also run the M4.1-M4.6 regression prerequisite. Disabled by default for this audit.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional alternate root directory containing reports/ and artifacts/ outputs.",
    )
    args = parser.parse_args()

    round_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    paths = write_m47_evidence_chain_audit(
        round_started_at=round_started_at,
        output_root=args.output_root,
        include_regressions=args.include_regressions,
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
