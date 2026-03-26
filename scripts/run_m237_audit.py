from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m237_audit import write_m237_total_acceptance_artifacts


if __name__ == "__main__":
    payload = write_m237_total_acceptance_artifacts()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
