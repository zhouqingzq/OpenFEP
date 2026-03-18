from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m225_benchmarks import SEED_SET, run_m225_open_world_transfer


if __name__ == "__main__":
    payload = run_m225_open_world_transfer(seed_set=list(SEED_SET))
    preview = {
        "mode": "preview",
        "status": payload["acceptance_report"]["status"],
        "recommendation": payload["acceptance_report"]["recommendation"],
        "acceptance_report": payload["acceptance_report"],
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))
