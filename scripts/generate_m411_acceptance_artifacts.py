from __future__ import annotations

import json

from segmentum.m411_phenomenology import write_m411_acceptance_artifacts


if __name__ == "__main__":
    paths = write_m411_acceptance_artifacts()
    print(json.dumps(paths, indent=2, ensure_ascii=False))
