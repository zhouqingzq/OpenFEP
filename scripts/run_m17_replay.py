from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from segmentum.dialogue.runtime.m17_replay import run_m17_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline M17 replay, calibration, and ablation.")
    parser.add_argument("--session", action="append", default=[], help="Session directory or explicit file. Repeatable.")
    parser.add_argument("--glob", dest="glob_pattern", default="", help="Optional glob for batch session selection.")
    parser.add_argument("--out", required=True, help="Output directory for artifacts.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for random baseline.")
    parser.add_argument("--max-sessions", type=int, default=0, help="Optional max sessions to include.")
    parser.add_argument("--fail-on-low-sample", action="store_true", help="Exit non-zero when low sample warnings are present.")
    parser.add_argument("--fixture-mode", action="store_true", help="Allow tiny deterministic fixtures with honest warnings.")
    args = parser.parse_args()

    session_paths = [Path(value).resolve() for value in args.session]
    if args.glob_pattern:
        session_paths.extend(sorted(Path().resolve().glob(args.glob_pattern)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in session_paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    if args.max_sessions and args.max_sessions > 0:
        deduped = deduped[: args.max_sessions]
    result = run_m17_replay(session_paths=deduped, out_dir=Path(args.out).resolve(), seed=args.seed)
    summary = {
        "sessions": [str(path) for path in deduped],
        "coverage": result["coverage"],
        "calibration": {
            "overall_brier_score": result["calibration"].get("overall_brier_score"),
            "expected_calibration_error": result["calibration"].get("expected_calibration_error"),
            "low_sample_warning": result["calibration"].get("low_sample_warning"),
        },
        "ablation": {
            "bundle_required_decision_rate": result["ablation"].get("bundle_required_decision_rate"),
            "policy_delta_vs_best_single": result["ablation"].get("policy_delta_vs_best_single"),
            "top_k_equivalent_behavior_rate": result["ablation"].get("top_k_equivalent_behavior_rate"),
            "low_sample_warning": result["ablation"].get("low_sample_warning"),
        },
        "warnings": result.get("warnings", []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_low_sample and (
        result["calibration"].get("low_sample_warning")
        or result["ablation"].get("low_sample_warning")
    ) and not args.fixture_mode:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
