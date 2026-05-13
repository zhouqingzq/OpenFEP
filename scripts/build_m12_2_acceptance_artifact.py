from __future__ import annotations

import json
from pathlib import Path

from segmentum.reciprocal_role.acceptance import build_m12_2_acceptance_artifact


def main() -> None:
    artifact = build_m12_2_acceptance_artifact()
    path = Path("artifacts/m12_2_acceptance_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
