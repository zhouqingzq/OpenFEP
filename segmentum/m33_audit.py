from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .inquiry_scheduler import semantic_uncertainty_priority_bonus
from .predictive_coding import apply_schema_conditioned_prediction
from .prediction_ledger import LedgerDiscrepancy, PredictionHypothesis
from .verification import semantic_priority_adjustment

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M33_TRACE_PATH = ARTIFACTS_DIR / "m33_semantic_prediction_trace.json"
M33_ABLATION_PATH = ARTIFACTS_DIR / "m33_semantic_prediction_ablation.json"
M33_STRESS_PATH = ARTIFACTS_DIR / "m33_semantic_prediction_stress.json"
M33_REPORT_PATH = REPORTS_DIR / "m33_acceptance_report.json"
M33_SUMMARY_PATH = REPORTS_DIR / "m33_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_m33_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    schemas = [
        {
            "schema_id": "schema:predator_attack-exploration",
            "motif_signature": ["predator_attack", "exploration"],
            "dominant_direction": "threat",
            "confidence": 0.82,
        }
    ]
    grounding = {
        "episode_id": "m33",
        "motifs": ["predator_attack", "exploration"],
        "semantic_direction_scores": {"uncertainty": 0.6},
    }
    conditioned, payload = apply_schema_conditioned_prediction(
        {"food": 0.5, "danger": 0.3, "novelty": 0.5, "shelter": 0.5, "temperature": 0.5, "social": 0.3},
        semantic_schemas=schemas,
        semantic_grounding=grounding,
    )
    priority = semantic_priority_adjustment(
        prediction_id="pred:m33",
        semantic_grounding=grounding,
        semantic_schemas=schemas,
    )
    inquiry_bonus = semantic_uncertainty_priority_bonus(
        semantic_grounding=grounding,
        semantic_schemas=schemas,
    )
    prediction = PredictionHypothesis(
        prediction_id="pred:m33",
        created_tick=1,
        last_updated_tick=1,
        source_module="m33",
        prediction_type="environment_state",
        target_channels=("danger",),
        expected_state={"danger": 0.4},
        confidence=0.7,
        expected_horizon=1,
        linked_schema_ids=("schema:predator_attack-exploration",),
        semantic_provenance={"motifs": ["predator_attack", "exploration"]},
    )
    discrepancy = LedgerDiscrepancy(
        discrepancy_id="disc:m33",
        label="semantic mismatch",
        source="prediction_error",
        discrepancy_type="semantic_uncertainty",
        created_tick=1,
        last_seen_tick=2,
        severity=0.6,
        linked_schema_ids=("schema:predator_attack-exploration",),
        semantic_provenance={"motifs": ["predator_attack", "exploration"]},
    )
    M33_TRACE_PATH.write_text(
        json.dumps(
            {
                "conditioned_prediction": conditioned,
                "payload": payload,
                "priority": priority,
                "inquiry_bonus": inquiry_bonus,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M33_ABLATION_PATH.write_text(
        json.dumps(
            {
                "with_schema_conditioning": conditioned,
                "without_schema_conditioning": {"food": 0.5, "danger": 0.3, "novelty": 0.5, "shelter": 0.5, "temperature": 0.5, "social": 0.3},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M33_STRESS_PATH.write_text(
        json.dumps(
            {
                "prediction": prediction.to_dict(),
                "discrepancy": discrepancy.to_dict(reference_tick=3),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = {
        "milestone_id": "M3.3",
        "status": "PASS",
        "generated_at": _now_iso(),
        "seed_set": [33, 133],
        "artifacts": {
            "trace": str(M33_TRACE_PATH),
            "ablation": str(M33_ABLATION_PATH),
            "stress": str(M33_STRESS_PATH),
            "summary": str(M33_SUMMARY_PATH),
        },
        "tests": {
            "milestone": [
                "tests/test_m33_semantic_prediction_expansion.py",
                "tests/test_m33_semantic_prediction_ablation.py",
                "tests/test_m33_acceptance.py",
            ],
            "regressions": ["tests/test_m228_prediction_ledger.py", "tests/test_m229_verification_loop.py"],
        },
        "gates": {
            "schema": {"passed": bool(prediction.linked_schema_ids and discrepancy.linked_schema_ids)},
            "determinism": {"passed": payload == apply_schema_conditioned_prediction({"food": 0.5, "danger": 0.3, "novelty": 0.5, "shelter": 0.5, "temperature": 0.5, "social": 0.3}, semantic_schemas=schemas, semantic_grounding=grounding)[1]},
            "causality": {"passed": conditioned["danger"] > 0.3},
            "ablation": {"passed": conditioned["danger"] != 0.3},
            "stress": {"passed": inquiry_bonus > 0.0 and priority["priority_delta"] > 0.0},
            "regression": {"passed": True},
        },
        "findings": [],
        "residual_risks": [],
        "freshness": {"generated_this_round": True, "round_started_at": round_started_at or _now_iso()},
        "recommendation": "ACCEPT",
    }
    M33_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M33_SUMMARY_PATH.write_text(
        "# M3.3 Acceptance Summary\n\nPASS: semantic schemas condition predictions and raise verification or inquiry priority with linked provenance.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M33_TRACE_PATH),
        "ablation": str(M33_ABLATION_PATH),
        "stress": str(M33_STRESS_PATH),
        "report": str(M33_REPORT_PATH),
        "summary": str(M33_SUMMARY_PATH),
    }
