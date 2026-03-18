from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark


if __name__ == "__main__":
    payload = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
