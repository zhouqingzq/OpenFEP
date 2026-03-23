from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m230_audit import (
    M230_REPORT_PATH,
    write_m230_acceptance_artifacts,
)


if __name__ == "__main__":
    round_started_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
    paths = write_m230_acceptance_artifacts(round_started_at=round_started_at)
    report = json.loads(M230_REPORT_PATH.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "mode": "write",
                "status": report["status"],
                "recommendation": report["recommendation"],
                "milestone_tests": len(report["tests"]["milestone"]),
                "regression_tests": len(report["tests"]["regressions"]),
                "paths": paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
