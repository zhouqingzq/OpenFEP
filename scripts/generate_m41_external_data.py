from __future__ import annotations

import json
from pathlib import Path

from segmentum.m41_external_generator import EXTERNAL_PROFILE_REGISTRY, run_external_trial


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "m41_external"
OUTPUT_PATH = OUTPUT_DIR / "sample_external_behavior.jsonl"


SEEDS = [101, 102, 103, 104]
SOURCE_NAMES = ["ethnography_lab", "ops_audit", "field_observation"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for profile_name, parameters in EXTERNAL_PROFILE_REGISTRY.items():
        for seed in SEEDS:
            trial = run_external_trial(
                parameters=parameters,
                seed=seed,
                tick_count=50,
                scenario_family="external_holdout",
            )
            session_id = f"{profile_name}-{seed}"
            subject_id = f"{profile_name}-subject-{seed % 2}"
            source_name = SOURCE_NAMES[(len(profile_name) + seed) % len(SOURCE_NAMES)]
            for record in trial["logs"]:
                rows.append(
                    {
                        "schema_version": "m41.external.event.v2",
                        "generator_id": "external_v1",
                        "source_name": source_name,
                        "task_name": record["task_context"]["phase"],
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "seed": seed,
                        "tick": record["tick"],
                        "timestamp": record["timestamp"],
                        "split": "heldout",
                        "ground_truth_profile": profile_name,
                        "ground_truth_parameters": trial["ground_truth_parameters"],
                        "percept_summary": record["percept_summary"],
                        "observation_evidence": record["observation_evidence"],
                        "prediction_error_vector": record["prediction_error_vector"],
                        "attention_allocation": record["attention_allocation"],
                        "candidate_actions": record["candidate_actions"],
                        "resource_state": record["resource_state"],
                        "internal_confidence": record["internal_confidence"],
                        "selected_action": record["selected_action"],
                        "result_feedback": record["result_feedback"],
                        "model_update": record["model_update"],
                        "prediction_error": record["prediction_error"],
                        "update_magnitude": record["update_magnitude"],
                    }
                )
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "profiles": sorted(EXTERNAL_PROFILE_REGISTRY), "output": str(OUTPUT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
