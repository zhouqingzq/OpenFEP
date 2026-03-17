from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m222_benchmarks import run_m222_long_horizon_trial


if __name__ == "__main__":
    payload = run_m222_long_horizon_trial()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
