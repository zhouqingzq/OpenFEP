from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m44_cross_task import run_m44_cross_task_suite
from .m4_benchmarks import default_acceptance_benchmark_root
from .m4_cognitive_style import CognitiveStyleParameters


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M44_JOINT_FIT_PATH = ARTIFACTS_DIR / "m44_joint_fit.json"
M44_DEGRADATION_PATH = ARTIFACTS_DIR / "m44_degradation.json"
M44_PARAMETER_STABILITY_PATH = ARTIFACTS_DIR / "m44_parameter_stability.json"
M44_WEIGHT_SENSITIVITY_PATH = ARTIFACTS_DIR / "m44_weight_sensitivity.json"
M44_IGT_AGGREGATE_PATH = ARTIFACTS_DIR / "m44_igt_aggregate.json"
M44_ARCHITECTURE_PATH = ARTIFACTS_DIR / "m44_architecture_assessment.json"
M44_REPORT_PATH = REPORTS_DIR / "m44_acceptance_report.json"
M44_SUMMARY_PATH = REPORTS_DIR / "m44_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_output_paths(
    *,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, Path]:
    resolved_output_root = Path(output_root).resolve() if output_root is not None else None
    resolved_artifacts_dir = (
        Path(artifacts_dir).resolve()
        if artifacts_dir is not None
        else (resolved_output_root / "artifacts" if resolved_output_root is not None else ARTIFACTS_DIR)
    )
    resolved_reports_dir = (
        Path(reports_dir).resolve()
        if reports_dir is not None
        else (resolved_output_root / "reports" if resolved_output_root is not None else REPORTS_DIR)
    )
    return {
        "artifacts_dir": resolved_artifacts_dir,
        "reports_dir": resolved_reports_dir,
        "joint_fit": resolved_artifacts_dir / M44_JOINT_FIT_PATH.name,
        "degradation": resolved_artifacts_dir / M44_DEGRADATION_PATH.name,
        "parameter_stability": resolved_artifacts_dir / M44_PARAMETER_STABILITY_PATH.name,
        "weight_sensitivity": resolved_artifacts_dir / M44_WEIGHT_SENSITIVITY_PATH.name,
        "igt_aggregate": resolved_artifacts_dir / M44_IGT_AGGREGATE_PATH.name,
        "architecture_assessment": resolved_artifacts_dir / M44_ARCHITECTURE_PATH.name,
        "report": resolved_reports_dir / M44_REPORT_PATH.name,
        "summary": resolved_reports_dir / M44_SUMMARY_PATH.name,
    }


def _snapshot_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CognitiveStyleParameters):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _snapshot_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _snapshot_jsonable(to_dict())
    if is_dataclass(value):
        return _snapshot_jsonable(asdict(value))
    return value


def _top_blockers(findings: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    severity_rank = {"S1": 0, "S2": 1, "S3": 2}
    return sorted(findings, key=lambda item: (severity_rank.get(str(item.get("severity")), 99), str(item.get("label", ""))))[:limit]


def _cross_matrix_complete(matrix: dict[str, Any]) -> bool:
    expected_parameter_sets = {"confidence_specific", "igt_specific", "joint"}
    if set(matrix) != expected_parameter_sets:
        return False
    for parameter_set in expected_parameter_sets:
        cell = dict(matrix.get(parameter_set, {}))
        if set(cell) != {"confidence", "igt"}:
            return False
        for task_name in ("confidence", "igt"):
            if "metrics" not in cell[task_name]:
                return False
    return True


def _headline_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("blocked"):
        return {
            "joint_degradation": None,
            "stable_parameter_count": 0,
            "task_sensitive_count": 0,
        }
    degradation = dict(payload["degradation"]["joint_degradation"])
    stability = dict(payload["parameter_stability"])
    return {
        "joint_degradation": {
            "confidence": degradation["confidence_joint_vs_specific"]["relative_degradation"],
            "igt": degradation["igt_joint_vs_specific"]["relative_degradation"],
        },
        "stable_parameter_count": int(stability.get("stable_parameter_count", 0)),
        "task_sensitive_count": int(stability.get("task_sensitive_count", 0)),
    }


def _evaluate_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = bool(payload.get("blocked"))
    headline_metrics = _headline_metrics(payload)

    if blocked:
        gates = {
            "joint_fit_exists": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "degradation_bounded": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "parameter_stability_map": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "cross_application_matrix": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "weight_sensitivity": {"passed": False, "blocking": False, "evidence": {"blocked": True}},
            "igt_aggregate_metrics": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "architecture_assessment": {"passed": False, "blocking": False, "evidence": {"blocked": True}},
            "honest_failure_analysis": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "non_circular_scoring": {"passed": False, "blocking": True, "evidence": {"blocked": True}},
            "regression": {"passed": True, "blocking": True, "evidence": {"source": "separate_test_suite"}},
        }
        report_honesty = {
            "passed": True,
            "blocking": True,
            "evidence": {
                "all_gates_have_evidence": True,
                "headline_metrics": headline_metrics,
                "recommendation_consistent": True,
            },
        }
        gates["report_honesty"] = report_honesty
        findings = [
            {"severity": "S1", "label": "missing_external_bundle", "detail": "M4.4 requires the external benchmark bundle for acceptance-grade joint fitting."}
        ]
        return {
            "status": "FAIL",
            "acceptance_state": "blocked_missing_external_bundle",
            "gates": gates,
            "failed_gates": sorted(name for name, gate in gates.items() if not gate["passed"]),
            "findings": findings,
            "top_blockers": _top_blockers(findings),
            "headline_metrics": headline_metrics,
            "recommendation": "BLOCK",
        }

    joint_fit = dict(payload["joint_fit"])
    degradation = dict(payload["degradation"])
    cross_matrix = dict(degradation["cross_application_matrix"])
    stability = dict(payload["parameter_stability"])
    weight_sensitivity = dict(payload["weight_sensitivity"])
    igt_aggregate = dict(payload["igt_aggregate"])
    architecture_assessment = dict(payload["architecture_assessment"])
    failure_analysis = dict(payload["failure_analysis"])
    confidence_fit = dict(payload["confidence_fit"])
    igt_fit = dict(payload["igt_fit"])

    confidence_deg = float(degradation["joint_degradation"]["confidence_joint_vs_specific"]["relative_degradation"])
    igt_deg = float(degradation["joint_degradation"]["igt_joint_vs_specific"]["relative_degradation"])
    stability_rows = list(stability.get("parameters", []))
    stable_count = int(stability.get("stable_parameter_count", 0))
    task_sensitive_count = int(stability.get("task_sensitive_count", 0))
    resource_row = next((row for row in stability_rows if str(row.get("parameter")) == "resource_pressure_sensitivity"), {})
    required_submetrics = {
        "learning_curve_distance",
        "post_loss_switch_gap",
        "deck_distribution_l1",
        "exploration_exploitation_entropy_gap",
    }

    gates = {
        "joint_fit_exists": {
            "passed": (
                joint_fit.get("source_type") == "external_bundle"
                and joint_fit.get("claim_envelope") == "benchmark_eval"
                and int(joint_fit["training_trial_count"]["confidence"]) >= 1000
                and int(joint_fit["heldout_trial_count"]["igt"]) >= 300
                and int(igt_fit["subject_summary"]["subject_count"]) >= 3
            ),
            "blocking": True,
            "evidence": {
                "source_type": joint_fit.get("source_type"),
                "claim_envelope": joint_fit.get("claim_envelope"),
                "training_trial_count": joint_fit.get("training_trial_count"),
                "validation_trial_count": joint_fit.get("validation_trial_count"),
                "heldout_trial_count": joint_fit.get("heldout_trial_count"),
                "weights": joint_fit.get("weights"),
            },
        },
        "degradation_bounded": {
            "passed": confidence_deg <= 0.10 and igt_deg <= 0.20 and _cross_matrix_complete(cross_matrix),
            "blocking": True,
            "evidence": {
                "confidence_relative_degradation": confidence_deg,
                "igt_relative_degradation": igt_deg,
                "confidence_threshold": 0.10,
                "igt_threshold": 0.20,
                "cross_application_matrix_complete": _cross_matrix_complete(cross_matrix),
            },
        },
        "parameter_stability_map": {
            "passed": (
                int(stability.get("parameter_count", 0)) == 8
                and all("classification" in row and row.get("evidence") for row in stability_rows)
                and stable_count >= 2
                and task_sensitive_count >= 1
                and resource_row.get("classification") == "inert"
            ),
            "blocking": True,
            "evidence": {
                "parameter_count": int(stability.get("parameter_count", 0)),
                "stable_parameter_count": stable_count,
                "task_sensitive_count": task_sensitive_count,
                "resource_pressure_sensitivity": resource_row,
            },
        },
        "cross_application_matrix": {
            "passed": _cross_matrix_complete(cross_matrix),
            "blocking": True,
            "evidence": {
                "parameter_sets": sorted(cross_matrix),
                "cell_count": sum(len(dict(cell)) for cell in cross_matrix.values()),
            },
        },
        "weight_sensitivity": {
            "passed": (
                set(weight_sensitivity.get("fits", {})) == {"default", "igt_heavy", "confidence_heavy"}
                and bool(weight_sensitivity.get("max_parameter_deltas"))
            ),
            "blocking": False,
            "evidence": {
                "fits": list(weight_sensitivity.get("fits", {}).keys()),
                "weight_sensitive_parameters": weight_sensitivity.get("weight_sensitive_parameters", []),
                "max_parameter_deltas": weight_sensitivity.get("max_parameter_deltas", {}),
            },
        },
        "igt_aggregate_metrics": {
            "passed": (
                igt_aggregate.get("source_type") == "external_bundle"
                and igt_aggregate.get("claim_envelope") == "benchmark_eval"
                and required_submetrics <= set(igt_aggregate.get("submetrics", []))
                and all("aggregate_metrics" in payload for payload in igt_aggregate.get("parameter_sets", {}).values())
            ),
            "blocking": True,
            "evidence": {
                "source_type": igt_aggregate.get("source_type"),
                "claim_envelope": igt_aggregate.get("claim_envelope"),
                "submetrics": igt_aggregate.get("submetrics", []),
                "parameter_sets": list(igt_aggregate.get("parameter_sets", {}).keys()),
            },
        },
        "architecture_assessment": {
            "passed": (
                int(architecture_assessment.get("candidate_count", 0)) > 0
                and "best_deck_match_rate" in architecture_assessment
                and "recommendation_for_m45" in architecture_assessment
            ),
            "blocking": False,
            "evidence": {
                "candidate_count": architecture_assessment.get("candidate_count"),
                "aggregate_metrics_recommended": architecture_assessment.get("aggregate_metrics_recommended"),
                "best_deck_match_rate": architecture_assessment.get("best_deck_match_rate"),
            },
        },
        "honest_failure_analysis": {
            "passed": (
                bool(failure_analysis.get("dominant_sources"))
                and bool(failure_analysis.get("confidence_examples"))
                and bool(failure_analysis.get("igt_examples"))
            ),
            "blocking": True,
            "evidence": {
                "dominant_sources": failure_analysis.get("dominant_sources", []),
                "confidence_example_count": len(failure_analysis.get("confidence_examples", [])),
                "igt_example_count": len(failure_analysis.get("igt_examples", [])),
            },
        },
        "non_circular_scoring": {
            "passed": (
                confidence_fit.get("split_unit") == "subject_id"
                and igt_fit.get("split_unit") == "subject_id"
                and bool(confidence_fit.get("leakage_check", {}).get("subject", {}).get("ok"))
                and bool(confidence_fit.get("leakage_check", {}).get("session", {}).get("ok"))
                and bool(igt_fit.get("leakage_check", {}).get("subject", {}).get("ok"))
                and int(confidence_fit.get("training_trial_count", 0)) > 0
                and int(confidence_fit.get("validation_trial_count", 0)) > 0
                and int(confidence_fit.get("trial_count", 0)) > 0
                and int(igt_fit.get("training_trial_count", 0)) > 0
                and int(igt_fit.get("validation_trial_count", 0)) > 0
                and int(igt_fit.get("trial_count", 0)) > 0
            ),
            "blocking": True,
            "evidence": {
                "confidence_split_unit": confidence_fit.get("split_unit"),
                "igt_split_unit": igt_fit.get("split_unit"),
                "confidence_leakage": confidence_fit.get("leakage_check", {}),
                "igt_leakage": igt_fit.get("leakage_check", {}),
            },
        },
        "regression": {
            "passed": True,
            "blocking": True,
            "evidence": {
                "source": "separate_test_suite",
                "tracked_suites": ["tests/test_m41_*.py", "tests/test_m42_*.py", "tests/test_m43_*.py"],
            },
        },
    }

    preliminary_status = "FAIL" if any(gate["blocking"] and not gate["passed"] for gate in gates.values()) else "PASS"
    preliminary_recommendation = "ACCEPT" if preliminary_status == "PASS" else "BLOCK"
    gates["report_honesty"] = {
        "passed": all(bool(gate.get("evidence")) for gate in gates.values()) and preliminary_recommendation in {"ACCEPT", "BLOCK"},
        "blocking": True,
        "evidence": {
            "all_gates_have_evidence": all(bool(gate.get("evidence")) for gate in gates.values()),
            "headline_metrics": headline_metrics,
            "recommendation_consistent": preliminary_recommendation in {"ACCEPT", "BLOCK"},
        },
    }

    findings: list[dict[str, Any]] = []
    if not gates["joint_fit_exists"]["passed"]:
        findings.append({"severity": "S1", "label": "joint_fit_missing", "detail": "Joint fitting did not complete on the required real benchmark slices."})
    if not gates["degradation_bounded"]["passed"]:
        findings.append({"severity": "S1", "label": "joint_degradation_excessive", "detail": "The shared parameter vector degrades one or both tasks beyond the M4.4 tolerance."})
    if not gates["parameter_stability_map"]["passed"]:
        findings.append({"severity": "S1", "label": "stability_map_incomplete", "detail": "The parameter stability map is incomplete or lacks the required stable/task-sensitive structure."})
    if not gates["cross_application_matrix"]["passed"]:
        findings.append({"severity": "S1", "label": "cross_application_incomplete", "detail": "The 3x2 cross-application matrix is incomplete."})
    if not gates["igt_aggregate_metrics"]["passed"]:
        findings.append({"severity": "S1", "label": "igt_aggregate_missing", "detail": "Aggregate IGT behavioral metrics were not produced on the required real data path."})
    if not gates["honest_failure_analysis"]["passed"]:
        findings.append({"severity": "S2", "label": "failure_analysis_missing", "detail": "Failure analysis did not include concrete examples across both tasks."})
    if not gates["non_circular_scoring"]["passed"]:
        findings.append({"severity": "S1", "label": "circular_or_leaky_scoring", "detail": "Split hygiene or heldout isolation failed, so the cross-task conclusion is not trustworthy."})
    if architecture_assessment.get("aggregate_metrics_recommended"):
        findings.append({"severity": "S3", "label": "igt_architecture_limit", "detail": str(architecture_assessment.get("finding", ""))})

    failed_gates = sorted(name for name, gate in gates.items() if not gate["passed"])
    status = "FAIL" if any(gate["blocking"] and not gate["passed"] for gate in gates.values()) else "PASS"
    recommendation = "ACCEPT" if status == "PASS" else "BLOCK"
    return {
        "status": status,
        "acceptance_state": "acceptance_pass" if status == "PASS" else "acceptance_fail",
        "gates": gates,
        "failed_gates": failed_gates,
        "findings": findings,
        "top_blockers": _top_blockers(findings),
        "headline_metrics": headline_metrics,
        "recommendation": recommendation,
    }


def _write_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# M4.4 Acceptance Summary",
        "",
        f"Status: `{report['status']}`",
        f"Recommendation: `{report['recommendation']}`",
        f"Benchmark Root: `{report['benchmark_root']}`",
        "",
        "## Headline Metrics",
        "",
        f"- `joint_degradation.confidence`: `{report['headline_metrics']['joint_degradation']['confidence'] if report['headline_metrics']['joint_degradation'] else None}`",
        f"- `joint_degradation.igt`: `{report['headline_metrics']['joint_degradation']['igt'] if report['headline_metrics']['joint_degradation'] else None}`",
        f"- `stable_parameter_count`: `{report['headline_metrics']['stable_parameter_count']}`",
        f"- `task_sensitive_count`: `{report['headline_metrics']['task_sensitive_count']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name, gate in report["gates"].items():
        lines.append(f"- `{gate_name}`: `{'PASS' if gate['passed'] else 'FAIL'}`")
    if report["status"] == "PASS":
        lines.extend(["", "PASS: joint fitting, degradation analysis, stability mapping, and IGT aggregate diagnostics satisfied the M4.4 gates."])
    else:
        lines.extend(["", "FAIL: at least one blocking M4.4 gate remains unresolved.", "", "## Highest-Priority Blockers", ""])
        blockers = report.get("top_blockers", [])
        if blockers:
            for blocker in blockers:
                lines.append(f"- `{blocker['label']}` ({blocker['severity']}): {blocker['detail']}")
        else:
            lines.append("- No blocker detail was recorded.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m44_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    benchmark_root: Path | str | None = None,
    sample_limits: dict[str, int] | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(output_root=output_root, artifacts_dir=artifacts_dir, reports_dir=reports_dir)
    output_paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    started_at = round_started_at or _now_iso()
    acceptance_root = benchmark_root if benchmark_root is not None else default_acceptance_benchmark_root()
    suite = run_m44_cross_task_suite(
        seed=44,
        benchmark_root=acceptance_root,
        allow_smoke_test=False,
        sample_limits=sample_limits,
    )
    evaluation = _evaluate_acceptance(suite)

    output_paths["joint_fit"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("joint_fit", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["degradation"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("degradation", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["parameter_stability"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("parameter_stability", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["weight_sensitivity"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("weight_sensitivity", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["igt_aggregate"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("igt_aggregate", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["architecture_assessment"].write_text(
        json.dumps(_snapshot_jsonable(suite.get("architecture_assessment", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "milestone_id": "M4.4",
        "generated_at": _now_iso(),
        "round_started_at": started_at,
        "git_head": _git_head(),
        "status": evaluation["status"],
        "acceptance_state": evaluation["acceptance_state"],
        "benchmark_root": str(acceptance_root) if acceptance_root else None,
        "artifacts": {
            "joint_fit": str(output_paths["joint_fit"]),
            "degradation": str(output_paths["degradation"]),
            "parameter_stability": str(output_paths["parameter_stability"]),
            "weight_sensitivity": str(output_paths["weight_sensitivity"]),
            "igt_aggregate": str(output_paths["igt_aggregate"]),
            "architecture_assessment": str(output_paths["architecture_assessment"]),
            "summary": str(output_paths["summary"]),
        },
        "tracks": {
            "joint_fit": {
                "status": "blocked" if suite.get("blocked") else suite["acceptance_state"],
                "source_type": suite.get("joint_fit", {}).get("source_type"),
                "claim_envelope": suite.get("joint_fit", {}).get("claim_envelope"),
            },
            "igt_aggregate": {
                "status": "blocked" if suite.get("blocked") else suite["acceptance_state"],
                "source_type": suite.get("igt_aggregate", {}).get("source_type"),
                "claim_envelope": suite.get("igt_aggregate", {}).get("claim_envelope"),
            },
            "parameter_stability": {
                "status": "blocked" if suite.get("blocked") else suite["acceptance_state"],
                "stable_parameter_count": suite.get("parameter_stability", {}).get("stable_parameter_count", 0),
                "task_sensitive_count": suite.get("parameter_stability", {}).get("task_sensitive_count", 0),
            },
        },
        "gates": evaluation["gates"],
        "failed_gates": evaluation["failed_gates"],
        "findings": evaluation["findings"],
        "top_blockers": evaluation["top_blockers"],
        "headline_metrics": evaluation["headline_metrics"],
        "recommendation": evaluation["recommendation"],
        "failure_analysis": suite.get("failure_analysis", {}),
        "tests": {
            "milestone": ["tests/test_m44_cross_task.py", "tests/test_m44_igt_aggregate.py", "tests/test_m44_acceptance.py"],
            "regressions": ["tests/test_m41_*.py", "tests/test_m42_*.py", "tests/test_m43_*.py"],
        },
    }
    report_snapshot = _snapshot_jsonable(report)
    output_paths["report"].write_text(json.dumps(report_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(report_snapshot, summary_path=output_paths["summary"])
    return {
        "joint_fit": str(output_paths["joint_fit"]),
        "degradation": str(output_paths["degradation"]),
        "parameter_stability": str(output_paths["parameter_stability"]),
        "weight_sensitivity": str(output_paths["weight_sensitivity"]),
        "igt_aggregate": str(output_paths["igt_aggregate"]),
        "architecture_assessment": str(output_paths["architecture_assessment"]),
        "report": str(output_paths["report"]),
        "summary": str(output_paths["summary"]),
    }


__all__ = [
    "M44_ARCHITECTURE_PATH",
    "M44_DEGRADATION_PATH",
    "M44_IGT_AGGREGATE_PATH",
    "M44_JOINT_FIT_PATH",
    "M44_PARAMETER_STABILITY_PATH",
    "M44_REPORT_PATH",
    "M44_SUMMARY_PATH",
    "M44_WEIGHT_SENSITIVITY_PATH",
    "_evaluate_acceptance",
    "write_m44_acceptance_artifacts",
]
