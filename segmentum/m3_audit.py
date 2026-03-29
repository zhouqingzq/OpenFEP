from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m31_audit import write_m31_acceptance_artifacts
from .m32_audit import write_m32_acceptance_artifacts
from .m33_audit import write_m33_acceptance_artifacts
from .m34_audit import write_m34_acceptance_artifacts
from .m35_audit import write_m35_acceptance_artifacts
from .m345_process_benchmarks import run_m345_process_benchmark, write_m345_process_benchmark_artifacts
from .m3_open_world_trial import run_open_world_growth_trial

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M36_TRACE_PATH = ARTIFACTS_DIR / "m36_open_world_growth_trace.json"
M36_ABLATION_PATH = ARTIFACTS_DIR / "m36_open_world_growth_ablation.json"
M36_STRESS_PATH = ARTIFACTS_DIR / "m36_open_world_growth_stress.json"
M36_SNAPSHOT_PATH = ARTIFACTS_DIR / "m36_open_world_growth_snapshots.json"
M36_FAILURE_AUDIT_PATH = ARTIFACTS_DIR / "m36_open_world_growth_failure_audit.json"
M36_REPORT_PATH = REPORTS_DIR / "m36_acceptance_report.json"
M36_SUMMARY_PATH = REPORTS_DIR / "m36_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float) -> float:
    return round(float(value), 6)


def write_m3_prior_acceptance_round(*, round_started_at: str | None = None) -> dict[str, object]:
    started_at = round_started_at or _now_iso()
    prior_paths = {
        "m31": write_m31_acceptance_artifacts(round_started_at=started_at),
        "m32": write_m32_acceptance_artifacts(round_started_at=started_at),
        "m33": write_m33_acceptance_artifacts(round_started_at=started_at),
        "m34": write_m34_acceptance_artifacts(round_started_at=started_at),
        "m35": write_m35_acceptance_artifacts(round_started_at=started_at),
        "m345": write_m345_process_benchmark_artifacts(),
    }
    return {"round_started_at": started_at, "artifacts": prior_paths}


def write_m36_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    prior_round = write_m3_prior_acceptance_round(round_started_at=started_at)
    benchmark = run_m345_process_benchmark()
    trial = run_open_world_growth_trial()
    ablation = run_open_world_growth_trial(ablation_mode="flattened")
    stress = run_open_world_growth_trial(stress=True)

    M36_TRACE_PATH.write_text(
        json.dumps(
            {
                "summary": trial["summary"],
                "catalog": trial["catalog"],
                "subjects": {
                    subject_id: {"trace": payload["trace"], "metrics": payload["metrics"]}
                    for subject_id, payload in trial["subjects"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M36_ABLATION_PATH.write_text(
        json.dumps(
            {
                "full_trial": trial["summary"],
                "flattened_trial": ablation["summary"],
                "semantic_schema_delta": _round(
                    float(trial["summary"]["schema_count_mean"]) - float(ablation["summary"]["schema_count_mean"])
                ),
                "style_diversity_delta": int(trial["summary"]["style_label_diversity"])
                - int(ablation["summary"]["style_label_diversity"]),
                "process_observability_drop": bool(trial["summary"]["process_observability"])
                and not bool(ablation["summary"]["process_observability"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M36_STRESS_PATH.write_text(
        json.dumps(
            {
                "stress_summary": stress["summary"],
                "stress_subjects": {subject_id: payload["metrics"] for subject_id, payload in stress["subjects"].items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M36_SNAPSHOT_PATH.write_text(
        json.dumps(
            {
                "subjects": {
                    subject_id: {
                        "final_snapshot": payload["final_snapshot"],
                        "restart_snapshot": payload["restart_snapshot"],
                    }
                    for subject_id, payload in trial["subjects"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stress_subject_metrics = {
        subject_id: payload["metrics"]
        for subject_id, payload in stress["subjects"].items()
    }
    stress_style_continuities = [
        float(metrics["style_continuity"])
        for metrics in stress_subject_metrics.values()
    ]
    stress_style_continuity_min = min(stress_style_continuities)
    semantic_growth_passed = (
        trial["summary"]["schema_count_mean"] >= 3.0
        and trial["summary"]["schema_count_max"] <= 5
        and ablation["summary"]["schema_count_mean"] < trial["summary"]["schema_count_mean"]
    )
    process_motivation_passed = bool(trial["summary"]["process_observability"]) and bool(
        benchmark["gates"]["process_reorientation"]["passed"]
    )
    style_stability_passed = (
        trial["summary"]["style_label_diversity"] >= 3
        and trial["summary"]["style_continuity_mean"] >= 0.5
        and ablation["summary"]["style_label_diversity"] < trial["summary"]["style_label_diversity"]
    )
    continuity_passed = bool(trial["summary"]["restart_continuity"]) and all(
        float(payload["metrics"]["style_continuity"]) >= 0.5 for payload in trial["subjects"].values()
    )
    compositional_diversity_passed = trial["summary"]["narrative_diversity"] >= 4 and trial["summary"]["task_diversity"] >= 8
    bounded_growth_passed = (
        stress["summary"]["schema_count_max"] <= 5
        and stress["summary"]["style_continuity_mean"] >= 0.5
        and stress_style_continuity_min >= 0.5
    )

    findings: list[dict[str, object]] = []
    if not compositional_diversity_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "insufficient_compositional_diversity",
                "detail": "The trial did not span enough narratives or tasks to support the compositional diversity claim.",
            }
        )
    if not bounded_growth_passed:
        findings.append(
            {
                "severity": "S1",
                "label": "unbounded_growth_under_stress",
                "detail": "Stress replay produced unstable schema growth or subject-level style continuity loss.",
            }
        )

    failure_audit = {
        "milestone_id": "M3.6",
        "artifact_family": "failure_audit",
        "generated_at": _now_iso(),
        "status": "CLEAR" if not findings else "ISSUES_FOUND",
        "blocking_findings": findings,
        "evaluated_gates": {
            "semantic_growth_controlled": semantic_growth_passed,
            "process_motivation_observable": process_motivation_passed,
            "style_diversity_and_stability": style_stability_passed,
            "restart_continuity": continuity_passed,
            "open_world_compositional_diversity": compositional_diversity_passed,
            "bounded_growth_under_stress": bounded_growth_passed,
        },
        "stress_subject_metrics": stress_subject_metrics,
        "ablation_deltas": {
            "semantic_schema_delta": _round(
                float(trial["summary"]["schema_count_mean"]) - float(ablation["summary"]["schema_count_mean"])
            ),
            "style_diversity_delta": int(trial["summary"]["style_label_diversity"])
            - int(ablation["summary"]["style_label_diversity"]),
            "process_observability_drop": bool(trial["summary"]["process_observability"])
            and not bool(ablation["summary"]["process_observability"]),
        },
        "residual_risks": [
            "The trial remains deterministic and hand-authored; it demonstrates compositional diversity, not unscripted open-world autonomy."
        ],
    }
    M36_FAILURE_AUDIT_PATH.write_text(json.dumps(failure_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    status = "PASS"
    recommendation = "ACCEPT"
    if findings:
        status = "FAIL"
        recommendation = "BLOCK"

    report = {
        "milestone_id": "M3.6",
        "status": status,
        "generated_at": _now_iso(),
        "seed_set": [31, 32, 33, 34, 35, 36, 136],
        "artifacts": {
            "trace": str(M36_TRACE_PATH),
            "ablation": str(M36_ABLATION_PATH),
            "stress": str(M36_STRESS_PATH),
            "snapshots": str(M36_SNAPSHOT_PATH),
            "failure_audit": str(M36_FAILURE_AUDIT_PATH),
            "summary": str(M36_SUMMARY_PATH),
            "prior_round": prior_round["artifacts"],
        },
        "tests": {
            "milestone": [
                "tests/test_m36_open_world_growth.py",
                "tests/test_m36_acceptance.py",
            ],
            "regressions": [
                "tests/test_m31_acceptance.py",
                "tests/test_m32_acceptance.py",
                "tests/test_m33_acceptance.py",
                "tests/test_m34_acceptance.py",
                "tests/test_m35_acceptance.py",
                "tests/test_m345_process_benchmark.py",
            ],
        },
        "gates": {
            "schema": {"passed": semantic_growth_passed},
            "determinism": {"passed": continuity_passed},
            "causality": {"passed": process_motivation_passed},
            "ablation": {"passed": ablation["summary"]["schema_count_mean"] < trial["summary"]["schema_count_mean"]},
            "stress": {"passed": bounded_growth_passed},
            "regression": {"passed": benchmark["status"] == "PASS"},
            "semantic_growth_controlled": {
                "passed": semantic_growth_passed,
                "schema_count_mean": trial["summary"]["schema_count_mean"],
                "schema_count_max": trial["summary"]["schema_count_max"],
            },
            "process_motivation_observable": {
                "passed": process_motivation_passed,
                "benchmark_focus_bonus_gain": benchmark["metrics"]["focus_bonus_gain"],
            },
            "style_diversity_and_stability": {
                "passed": style_stability_passed,
                "style_label_diversity": trial["summary"]["style_label_diversity"],
                "style_continuity_mean": trial["summary"]["style_continuity_mean"],
            },
            "restart_continuity": {
                "passed": continuity_passed,
                "restart_continuity": trial["summary"]["restart_continuity"],
            },
            "open_world_compositional_diversity": {
                "passed": compositional_diversity_passed,
                "narrative_diversity": trial["summary"]["narrative_diversity"],
                "task_diversity": trial["summary"]["task_diversity"],
            },
            "bounded_growth_under_stress": {
                "passed": bounded_growth_passed,
                "stress_schema_count_max": stress["summary"]["schema_count_max"],
                "stress_style_continuity_mean": stress["summary"]["style_continuity_mean"],
                "stress_style_continuity_min": _round(stress_style_continuity_min),
            },
        },
        "findings": findings,
        "residual_risks": list(failure_audit["residual_risks"])
        if status == "PASS"
        else [
            "Open-world growth trial must be rerun after resolving the blocking findings.",
            *failure_audit["residual_risks"],
        ],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }

    M36_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M36_SUMMARY_PATH.write_text(
        "# M3.6 Acceptance Summary\n\n"
        f"- Status: {status}\n"
        f"- Recommendation: {recommendation}\n"
        f"- Schema count mean: {trial['summary']['schema_count_mean']}\n"
        f"- Style diversity: {trial['summary']['style_label_diversity']}\n"
        f"- Restart continuity: {trial['summary']['restart_continuity']}\n"
        f"- Stress schema max: {stress['summary']['schema_count_max']}\n"
        f"- Stress style continuity min: {_round(stress_style_continuity_min)}\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M36_TRACE_PATH),
        "ablation": str(M36_ABLATION_PATH),
        "stress": str(M36_STRESS_PATH),
        "snapshots": str(M36_SNAPSHOT_PATH),
        "failure_audit": str(M36_FAILURE_AUDIT_PATH),
        "report": str(M36_REPORT_PATH),
        "summary": str(M36_SUMMARY_PATH),
    }
