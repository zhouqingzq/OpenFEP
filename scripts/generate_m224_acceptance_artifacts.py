from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m224_benchmarks import SEED_SET, write_m224_acceptance_artifacts


if __name__ == "__main__":
    paths = write_m224_acceptance_artifacts(seed_set=list(SEED_SET))
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, ensure_ascii=False))
