from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m229_audit import M229_REPORT_PATH, write_m229_acceptance_artifacts


if __name__ == "__main__":
    paths = write_m229_acceptance_artifacts()
    report = json.loads(M229_REPORT_PATH.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "mode": "write",
                "status": report["status"],
                "recommendation": report["recommendation"],
                "paths": paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
