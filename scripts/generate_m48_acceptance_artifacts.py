from __future__ import annotations

import json
from datetime import datetime, timezone

from segmentum.m48_audit import M48_REPORT_PATH, write_m48_acceptance_artifacts


if __name__ == "__main__":
    round_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    paths = write_m48_acceptance_artifacts(round_started_at=round_started_at)
    report = json.loads(M48_REPORT_PATH.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "formal_acceptance_conclusion": report["formal_acceptance_conclusion"],
                "paths": paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
