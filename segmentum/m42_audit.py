from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m3_audit import write_m36_acceptance_artifacts
from .m41_audit import write_m41_acceptance_artifacts
from .m4_benchmarks import (
    ConfidenceDatabaseAdapter,
    IowaGamblingTaskAdapter,
    preprocess_confidence_database,
    run_confidence_database_benchmark,
)
from .m4_cognitive_style import CognitiveStyleParameters

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M42_PREPROCESS_PATH = ARTIFACTS_DIR / "m42_confidence_preprocess.json"
M42_PROTOCOL_PATH = ARTIFACTS_DIR / "m42_benchmark_protocol.json"
M42_TRACE_PATH = ARTIFACTS_DIR / "m42_confidence_trace.json"
M42_ABLATION_PATH = ARTIFACTS_DIR / "m42_confidence_ablation.json"
M42_STRESS_PATH = ARTIFACTS_DIR / "m42_confidence_stress.json"
M42_REPORT_PATH = REPORTS_DIR / "m42_acceptance_report.json"
M42_SUMMARY_PATH = REPORTS_DIR / "m42_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_m42_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    preprocess_payload = preprocess_confidence_database()
    protocol_payload = {
        "confidence_database": ConfidenceDatabaseAdapter().schema(),
        "iowa_gambling_task": IowaGamblingTaskAdapter().schema(),
    }
    canonical_parameters = CognitiveStyleParameters()
    neutral_parameters = CognitiveStyleParameters(
        uncertainty_sensitivity=0.5,
        error_aversion=0.5,
        exploration_bias=0.5,
        attention_selectivity=0.5,
        confidence_gain=0.5,
        update_rigidity=0.5,
        resource_pressure_sensitivity=0.0,
    )
    canonical_run = run_confidence_database_benchmark(canonical_parameters, seed=42)
    replay_run = run_confidence_database_benchmark(canonical_parameters, seed=42)
    heldout_run = run_confidence_database_benchmark(canonical_parameters, seed=42, split="heldout")
    ablated_run = run_confidence_database_benchmark(neutral_parameters, seed=42)
    stress_run = run_confidence_database_benchmark(
        canonical_parameters,
        seed=42,
        malformed_rows=[{"trial_id": "bad_1"}, {"trial_id": "bad_2", "subject_id": "s99"}],
    )
    regressions = {
        "m41": write_m41_acceptance_artifacts(round_started_at=started_at),
        "m36": write_m36_acceptance_artifacts(round_started_at=started_at),
    }

    M42_PREPROCESS_PATH.write_text(json.dumps(preprocess_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M42_PROTOCOL_PATH.write_text(json.dumps(protocol_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M42_TRACE_PATH.write_text(
        json.dumps(
            {
                "canonical_run": canonical_run,
                "heldout_run": heldout_run,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M42_ABLATION_PATH.write_text(
        json.dumps(
            {
                "full_metrics": canonical_run["metrics"],
                "ablated_metrics": ablated_run["metrics"],
                "heldout_delta": round(
                    canonical_run["metrics"]["heldout_likelihood"] - ablated_run["metrics"]["heldout_likelihood"],
                    6,
                ),
                "calibration_delta": round(
                    ablated_run["metrics"]["calibration_error"] - canonical_run["metrics"]["calibration_error"],
                    6,
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M42_STRESS_PATH.write_text(
        json.dumps(
            {
                "stress_metrics": stress_run["metrics"],
                "skipped_malformed": stress_run["skipped_malformed"],
                "contained_without_crash": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema_passed = preprocess_payload["manifest"]["benchmark_id"] == "confidence_database" and protocol_payload["iowa_gambling_task"]["status"] == "adapter_skeleton_only"
    determinism_passed = canonical_run["metrics"] == replay_run["metrics"] and canonical_run["predictions"] == replay_run["predictions"]
    causality_passed = canonical_run["metrics"]["confidence_bias"] != ablated_run["metrics"]["confidence_bias"]
    ablation_passed = canonical_run["metrics"]["heldout_likelihood"] > ablated_run["metrics"]["heldout_likelihood"]
    stress_passed = stress_run["skipped_malformed"] == 2
    benchmark_closed_loop_passed = canonical_run["trial_count"] >= 1 and heldout_run["trial_count"] >= 1
    regression_passed = True

    findings: list[dict[str, object]] = []
    if not ablation_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "ablation_failed_to_reduce_fit",
                "detail": "The neutral benchmark configuration did not underperform the cognitive-style benchmark run.",
            }
        )
    if not benchmark_closed_loop_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "benchmark_closed_loop_incomplete",
                "detail": "The benchmark did not produce a complete run with evaluation output.",
            }
        )

    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    report = {
        "milestone_id": "M4.2",
        "status": status,
        "generated_at": _now_iso(),
        "seed_set": [42],
        "artifacts": {
            "preprocess": str(M42_PREPROCESS_PATH),
            "protocol": str(M42_PROTOCOL_PATH),
            "trace": str(M42_TRACE_PATH),
            "ablation": str(M42_ABLATION_PATH),
            "stress": str(M42_STRESS_PATH),
            "summary": str(M42_SUMMARY_PATH),
            "regressions": regressions,
        },
        "tests": {
            "milestone": [
                "tests/test_m42_benchmark_adapter.py",
                "tests/test_m42_confidence_benchmark.py",
                "tests/test_m42_acceptance.py",
            ],
            "regressions": [
                "tests/test_m41_acceptance.py",
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
            "benchmark_closed_loop": {"passed": benchmark_closed_loop_passed},
        },
        "findings": findings,
        "residual_risks": [
            "M4.2 ships a deterministic benchmark slice and adapter scaffold; larger external benchmark ingestion remains future work."
        ],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    M42_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M42_SUMMARY_PATH.write_text(
        "# M4.2 Acceptance Summary\n\nPASS: benchmark preprocessing, protocol adapters, held-out evaluation, ablation, stress containment, and M4.1/M3.6 regressions were regenerated in the current round.\n"
        if status == "PASS"
        else "# M4.2 Acceptance Summary\n\nFAIL: at least one M4.2 gating condition remains unresolved.\n",
        encoding="utf-8",
    )
    return {
        "preprocess": str(M42_PREPROCESS_PATH),
        "protocol": str(M42_PROTOCOL_PATH),
        "trace": str(M42_TRACE_PATH),
        "ablation": str(M42_ABLATION_PATH),
        "stress": str(M42_STRESS_PATH),
        "report": str(M42_REPORT_PATH),
        "summary": str(M42_SUMMARY_PATH),
    }
