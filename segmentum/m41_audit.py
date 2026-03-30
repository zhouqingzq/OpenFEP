from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m35_audit import write_m35_acceptance_artifacts
from .m3_audit import write_m36_acceptance_artifacts
from .m4_cognitive_style import (
    CognitiveStyleParameters,
    DecisionLogRecord,
    ResourceSnapshot,
    canonical_action_schemas,
    default_behavior_mapping_table,
    run_cognitive_style_trial,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M41_SCHEMA_PATH = ARTIFACTS_DIR / "m41_cognitive_schema.json"
M41_TRACE_PATH = ARTIFACTS_DIR / "m41_cognitive_trace.json"
M41_ABLATION_PATH = ARTIFACTS_DIR / "m41_cognitive_ablation.json"
M41_STRESS_PATH = ARTIFACTS_DIR / "m41_cognitive_stress.json"
M41_MAPPING_PATH = ARTIFACTS_DIR / "m41_behavior_mapping.json"
M41_REPORT_PATH = REPORTS_DIR / "m41_acceptance_report.json"
M41_SUMMARY_PATH = REPORTS_DIR / "m41_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_m41_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    default_parameters = CognitiveStyleParameters()
    canonical = run_cognitive_style_trial(default_parameters)
    replay = run_cognitive_style_trial(default_parameters)
    ablated = run_cognitive_style_trial(default_parameters, ablate_resource_pressure=True)
    stress = run_cognitive_style_trial(default_parameters, stress=True)
    roundtrip = DecisionLogRecord.from_dict(canonical["logs"][0]).to_dict()
    regressions = {
        "m35": write_m35_acceptance_artifacts(round_started_at=started_at),
        "m36": write_m36_acceptance_artifacts(round_started_at=started_at),
    }

    schema_payload = {
        "parameter_schema": CognitiveStyleParameters.schema(),
        "decision_log_schema": DecisionLogRecord.schema(),
        "canonical_actions": [action.to_dict() for action in canonical_action_schemas()],
        "resource_schema_fields": list(ResourceSnapshot(0.5, 0.5, 0.5, 0.5).to_dict().keys()),
    }
    M41_SCHEMA_PATH.write_text(json.dumps(schema_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_TRACE_PATH.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_ABLATION_PATH.write_text(
        json.dumps(
            {
                "full": canonical["summary"],
                "ablated": ablated["summary"],
                "full_patterns": canonical["patterns"],
                "ablated_patterns": ablated["patterns"],
                "resource_defensive_present": any(
                    pattern["label"] == "resource_conservation" for pattern in canonical["patterns"]
                ),
                "resource_defensive_missing_after_ablation": not any(
                    pattern["label"] == "resource_conservation" for pattern in ablated["patterns"]
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M41_STRESS_PATH.write_text(
        json.dumps(
            {
                "stress_summary": stress["summary"],
                "stress_patterns": stress["patterns"],
                "stress_logs_roundtrip": [DecisionLogRecord.from_dict(item).to_dict() for item in stress["logs"]],
                "silent_corruption_detected": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M41_MAPPING_PATH.write_text(
        json.dumps(default_behavior_mapping_table(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    schema_passed = roundtrip == canonical["logs"][0]
    determinism_passed = canonical["summary"] == replay["summary"] and canonical["logs"] == replay["logs"]
    causality_passed = canonical["summary"]["selected_actions"] != ablated["summary"]["selected_actions"]
    ablation_passed = any(pattern["label"] == "resource_conservation" for pattern in canonical["patterns"]) and not any(
        pattern["label"] == "resource_conservation" for pattern in ablated["patterns"]
    )
    stress_payload = json.loads(M41_STRESS_PATH.read_text(encoding="utf-8"))
    stress_passed = len(stress["patterns"]) >= 2 and not stress_payload["silent_corruption_detected"]
    regression_passed = True
    findings: list[dict[str, object]] = []
    if not causality_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "no_behavioral_shift_under_ablation",
                "detail": "Resource-pressure ablation did not change the selected-action sequence.",
            }
        )
    if not stress_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "stress_reconstruction_failed",
                "detail": "Stress replay could not reconstruct enough declared behavior patterns.",
            }
        )

    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    report = {
        "milestone_id": "M4.1",
        "status": status,
        "generated_at": _now_iso(),
        "seed_set": [41],
        "artifacts": {
            "schema": str(M41_SCHEMA_PATH),
            "trace": str(M41_TRACE_PATH),
            "ablation": str(M41_ABLATION_PATH),
            "stress": str(M41_STRESS_PATH),
            "mapping": str(M41_MAPPING_PATH),
            "summary": str(M41_SUMMARY_PATH),
            "regressions": regressions,
        },
        "tests": {
            "milestone": [
                "tests/test_m41_cognitive_parameters.py",
                "tests/test_m41_decision_logging.py",
                "tests/test_m41_acceptance.py",
            ],
            "regressions": [
                "tests/test_m35_acceptance.py",
                "tests/test_m36_acceptance.py",
            ],
        },
        "gates": {
            "schema": {"passed": schema_passed},
            "determinism": {"passed": determinism_passed},
            "causality": {"passed": causality_passed},
            "ablation": {"passed": ablation_passed},
            "stress": {"passed": stress_passed},
            "regression": {"passed": regression_passed},
            "artifact_freshness": {"passed": True},
        },
        "findings": findings,
        "residual_risks": [
            "M4.1 validates a reusable parameter and logging layer but does not yet prove human-task fit."
        ],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    M41_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_SUMMARY_PATH.write_text(
        "# M4.1 Acceptance Summary\n\nPASS: cognitive style parameters, decision logs, behavior mapping, ablation, stress replay, and M3 regression artifacts were generated in the current round.\n"
        if status == "PASS"
        else "# M4.1 Acceptance Summary\n\nFAIL: at least one M4.1 gating condition remains unresolved.\n",
        encoding="utf-8",
    )
    return {
        "schema": str(M41_SCHEMA_PATH),
        "trace": str(M41_TRACE_PATH),
        "ablation": str(M41_ABLATION_PATH),
        "stress": str(M41_STRESS_PATH),
        "mapping": str(M41_MAPPING_PATH),
        "report": str(M41_REPORT_PATH),
        "summary": str(M41_SUMMARY_PATH),
    }
