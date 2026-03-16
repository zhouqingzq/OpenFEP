from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.m210_validation_evaluation import (
    build_m210_audit_payload,
    write_m210_audit_outputs,
)
from segmentum.m210_benchmarks import (
    ARTIFACTS_DIR,
    run_longitudinal_stability,
    run_personality_validation,
    summarize_profile_behaviors,
    write_json,
)


def generate_artifacts(
    *,
    validation_seed: int = 44,
    stability_seed: int = 91,
    validation_cycles_per_world: int = 36,
    stability_cycles_per_world: int = 60,
    validation_repeats: int = 4,
    stability_repeats: int = 3,
) -> dict[str, Path]:
    personality_validation = run_personality_validation(
        seed=validation_seed,
        cycles_per_world=validation_cycles_per_world,
        repeats=validation_repeats,
    )
    longitudinal_stability = run_longitudinal_stability(
        seed=stability_seed,
        cycles_per_world=stability_cycles_per_world,
        repeats=stability_repeats,
    )
    profile_summary = summarize_profile_behaviors(
        personality_validation,
        longitudinal_stability,
    )
    audit_payload = build_m210_audit_payload(
        personality_validation=personality_validation,
        longitudinal_stability=longitudinal_stability,
        profile_summary=profile_summary,
    )

    written = {
        "m210_personality_anova": write_json(
            ARTIFACTS_DIR / "m210_personality_anova.json",
            personality_validation,
        ),
        "m210_longitudinal_stability": write_json(
            ARTIFACTS_DIR / "m210_longitudinal_stability.json",
            longitudinal_stability,
        ),
        "m210_profile_behavior_summary": write_json(
            ARTIFACTS_DIR / "m210_profile_behavior_summary.json",
            profile_summary,
        ),
        "m210_audit_summary": write_json(
            ARTIFACTS_DIR / "m210_audit_summary.json",
            audit_payload,
        ),
    }
    report_path = Path(__file__).resolve().parent.parent / "reports" / "m210_strict_audit_report.md"
    _, written["m210_audit_report"] = write_m210_audit_outputs(
        audit_payload=audit_payload,
        summary_path=ARTIFACTS_DIR / "m210_audit_summary.json",
        report_path=report_path,
    )
    return written


def main() -> None:
    generate_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
