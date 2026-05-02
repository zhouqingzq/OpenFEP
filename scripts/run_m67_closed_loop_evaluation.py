from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentum.dialogue.m67_scenarios import run_m67_evaluation


def main() -> int:
    result = run_m67_evaluation(artifacts_dir=PROJECT_ROOT / "artifacts")
    artifact_path = result.get("artifact_path", "")
    passed = bool(result.get("all_passed"))
    print(f"M6.7 closed-loop evaluation artifact: {artifact_path}")
    print(
        "M6.7 scenarios passed: "
        f"{result.get('passed_scenario_count')}/{result.get('scenario_count')}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

