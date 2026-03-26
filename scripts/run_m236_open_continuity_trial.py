from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m236_open_continuity_trial import write_m236_acceptance_artifacts


if __name__ == "__main__":
    payload = write_m236_acceptance_artifacts(strict=True, execute_test_suites=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
