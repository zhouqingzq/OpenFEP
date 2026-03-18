from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m226_maturity_audit import SEED_SET, run_m226_full_replay


if __name__ == "__main__":
    payload = run_m226_full_replay(seed_set=list(SEED_SET))
    print(
        json.dumps(
            {
                "mode": "preview",
                "status": payload["final_report"]["status"],
                "final_status": payload["final_report"]["final_status"],
                "default_mature": payload["final_report"]["default_mature"],
                "blocking_reasons": payload["final_report"]["blocking_reasons"],
                "current_round_replay_coverage": payload["final_report"]["current_round_replay_coverage"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
