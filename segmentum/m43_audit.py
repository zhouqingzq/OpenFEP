from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m43_modeling import run_m43_single_task_suite
from .m4_benchmarks import default_acceptance_benchmark_root


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXTERNAL_BENCHMARK_ROOT = (ROOT / "external_benchmark_registry").resolve()
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M43_CONFIDENCE_PATH = ARTIFACTS_DIR / "m43_confidence_fit.json"
M43_IGT_PATH = ARTIFACTS_DIR / "m43_igt_fit.json"
M43_PARAMETER_SENSITIVITY_PATH = ARTIFACTS_DIR / "m43_parameter_sensitivity.json"
M43_FAILURE_PATH = ARTIFACTS_DIR / "m43_failure_modes.json"
M43_REPORT_PATH = REPORTS_DIR / "m43_acceptance_report.json"
M43_SUMMARY_PATH = REPORTS_DIR / "m43_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_acceptance_root(benchmark_root: Path | str | None = None) -> Path | None:
    if benchmark_root is None:
        return default_acceptance_benchmark_root()
    candidate = Path(benchmark_root).resolve()
    required = (
        candidate / "confidence_database" / "manifest.json",
        candidate / "iowa_gambling_task" / "manifest.json",
    )
    return candidate if all(path.exists() for path in required) else None


def _require_official_acceptance_root(benchmark_root: Path | str | None = None) -> Path:
    resolved = _resolve_acceptance_root(benchmark_root)
    if resolved is None:
        raise ValueError("Official M4.3 acceptance generation requires the real external benchmark root.")
    if resolved.resolve() != EXPECTED_EXTERNAL_BENCHMARK_ROOT:
        raise ValueError(
            f"Official M4.3 acceptance generation must use {EXPECTED_EXTERNAL_BENCHMARK_ROOT}, got {resolved.resolve()}."
        )
    return resolved


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
        "confidence_fit": resolved_artifacts_dir / M43_CONFIDENCE_PATH.name,
        "igt_fit": resolved_artifacts_dir / M43_IGT_PATH.name,
        "parameter_sensitivity": resolved_artifacts_dir / M43_PARAMETER_SENSITIVITY_PATH.name,
        "failure_modes": resolved_artifacts_dir / M43_FAILURE_PATH.name,
        "report": resolved_reports_dir / M43_REPORT_PATH.name,
        "summary": resolved_reports_dir / M43_SUMMARY_PATH.name,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_from_output_paths(output_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "blocked": False,
        "confidence": _read_json(output_paths["confidence_fit"]),
        "igt": _read_json(output_paths["igt_fit"]),
        "parameter_sensitivity": _read_json(output_paths["parameter_sensitivity"]),
        "failure_analysis": _read_json(output_paths["failure_modes"]),
    }


def _iter_leaf_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows: list[str] = []
        for item in value.values():
            rows.extend(_iter_leaf_strings(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_iter_leaf_strings(item))
        return rows
    return []


def _contains_smoke_fixture_reference(value: Any) -> bool:
    smoke_markers = (
        "data\\benchmarks\\confidence_database\\confidence_database_sample.jsonl",
        "data\\benchmarks\\iowa_gambling_task\\igt_sample.jsonl",
        "repo_smoke_test",
        "smoke_only",
    )
    normalized_rows = [entry.replace("/", "\\").lower() for entry in _iter_leaf_strings(value)]
    return any(marker.lower() in entry for entry in normalized_rows for marker in smoke_markers)


def _top_blockers(findings: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    severity_rank = {"S1": 0, "S2": 1, "S3": 2}
    ordered = sorted(
        findings,
        key=lambda item: (
            severity_rank.get(str(item.get("severity")), 99),
            str(item.get("label", "")),
        ),
    )
    return ordered[:limit]


def _write_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# M4.3 Acceptance Summary",
        "",
        f"Status: `{report['status']}`",
        f"Recommendation: `{report['recommendation']}`",
        f"M4.4 Readiness: `{'READY' if report['recommendation'] == 'ACCEPT' else 'BLOCKED'}`",
        f"Benchmark Root: `{report['benchmark_root']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name, gate in report["gates"].items():
        lines.append(f"- `{gate_name}`: `{'PASS' if gate['passed'] else 'FAIL'}`")
    if report["status"] == "PASS":
        lines.extend(
            [
                "",
                "PASS: real external Confidence Database and Iowa Gambling Task runs, parameter sensitivity, and failure analysis satisfied the current M4.3 gates.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "FAIL: at least one real-data behavioral-fit gate remains unresolved.",
                "",
                "## Highest-Priority Blockers",
                "",
            ]
        )
        blockers = report.get("top_blockers", []) or _top_blockers(list(report.get("findings", [])))
        if blockers:
            for blocker in blockers:
                lines.append(
                    f"- `{blocker.get('label', 'unknown_blocker')}` ({blocker.get('severity', 'S?')}): {blocker.get('detail', '')}"
                )
        else:
            lines.append("- No blocker detail was recorded.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_real_bundle_outputs(
    *,
    output_paths: dict[str, Path],
    expected_benchmark_root: Path,
) -> dict[str, Any]:
    required_paths = (
        output_paths["confidence_fit"],
        output_paths["igt_fit"],
        output_paths["parameter_sensitivity"],
        output_paths["failure_modes"],
        output_paths["report"],
        output_paths["summary"],
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"M4.3 acceptance generation did not produce all required outputs: {missing}")

    too_small = {str(path): path.stat().st_size for path in required_paths if path.stat().st_size <= 64}
    if too_small:
        raise ValueError(f"M4.3 acceptance outputs look like placeholders: {too_small}")

    payload = _payload_from_output_paths(output_paths)
    report = _read_json(output_paths["report"])
    expected_evaluation = _evaluate_acceptance(payload)

    benchmark_root = Path(str(report.get("benchmark_root", ""))).resolve()
    if benchmark_root != expected_benchmark_root.resolve():
        raise ValueError(
            f"M4.3 report benchmark root must be {expected_benchmark_root.resolve()}, got {benchmark_root}."
        )

    for track_name in ("confidence", "igt", "parameter_sensitivity"):
        track = dict(payload[track_name])
        if track.get("source_type") != "external_bundle":
            raise ValueError(f"M4.3 {track_name} track did not use an external bundle.")
        if track.get("claim_envelope") != "benchmark_eval":
            raise ValueError(f"M4.3 {track_name} track used unexpected claim envelope {track.get('claim_envelope')}.")
        if _contains_smoke_fixture_reference(track):
            raise ValueError(f"M4.3 {track_name} artifact contains smoke/sample fixture references.")

    if _contains_smoke_fixture_reference(report):
        raise ValueError("M4.3 acceptance report contains smoke/sample fixture references.")
    if str(expected_benchmark_root).replace("/", "\\").lower() not in output_paths["summary"].read_text(encoding="utf-8").replace("/", "\\").lower():
        raise ValueError("M4.3 acceptance summary does not mention the real external benchmark root.")

    if report["status"] != expected_evaluation["status"]:
        raise ValueError("M4.3 report status does not match recomputed gate truth.")
    if report["gates"] != expected_evaluation["gates"]:
        raise ValueError("M4.3 report gates do not match recomputed gate truth.")
    if report["failed_gates"] != expected_evaluation["failed_gates"]:
        raise ValueError("M4.3 report failed_gates do not match recomputed gate truth.")
    if report["findings"] != expected_evaluation["findings"]:
        raise ValueError("M4.3 report findings do not match recomputed gate truth.")

    return {
        "payload": payload,
        "report": report,
        "expected_evaluation": expected_evaluation,
    }


def _publish_output_files(source_paths: dict[str, Path], target_paths: dict[str, Path]) -> dict[str, str]:
    for key in ("confidence_fit", "igt_fit", "parameter_sensitivity", "failure_modes", "report", "summary"):
        destination = target_paths[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths[key], destination)
    return {
        "confidence_fit": str(target_paths["confidence_fit"]),
        "igt_fit": str(target_paths["igt_fit"]),
        "parameter_sensitivity": str(target_paths["parameter_sensitivity"]),
        "failure_modes": str(target_paths["failure_modes"]),
        "report": str(target_paths["report"]),
        "summary": str(target_paths["summary"]),
    }


def _no_synthetic_claims(payload: dict[str, Any]) -> bool:
    tracks = [payload["confidence"], payload["igt"], payload["parameter_sensitivity"]]
    for track in tracks:
        if bool(track.get("external_validation", False)):
            return False
        if track.get("source_type") == "external_bundle":
            if track.get("claim_envelope") != "benchmark_eval":
                return False
        elif track.get("mode") != "blocked" and track.get("claim_envelope") not in {"synthetic_diagnostic", "benchmark_eval"}:
            return False
    return True


def _has_real_failure_modes(payload: dict[str, Any]) -> bool:
    modes = payload.get("failure_analysis", {}).get("failure_modes", {})
    return all(bool(modes.get(track_name)) for track_name in ("confidence_database", "iowa_gambling_task"))


def _subject_count(track: dict[str, Any]) -> int:
    return int(track.get("subject_summary", {}).get("subject_count", 0))


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _competitive_parity(agent_metrics: dict[str, Any], baseline_metrics: dict[str, Any], *, primary_metric: str) -> bool:
    agent_primary = _metric_value(agent_metrics, primary_metric)
    baseline_primary = _metric_value(baseline_metrics, primary_metric)
    if agent_primary is None or baseline_primary is None:
        return False
    if agent_primary >= baseline_primary:
        return True
    baseline_brier = _metric_value(baseline_metrics, "brier_score") or 0.0
    agent_brier = _metric_value(agent_metrics, "brier_score")
    if baseline_brier > 0.0 and agent_brier is not None and agent_brier <= baseline_brier * 1.05:
        return True
    return False


def _relative_worse_than(agent_value: float | None, baseline_value: float | None, *, tolerance: float = 0.15) -> bool:
    if agent_value is None or baseline_value is None:
        return False
    if agent_value >= baseline_value:
        return False
    denom = max(abs(baseline_value), 1e-6)
    return abs(agent_value - baseline_value) / denom > tolerance


def _baseline_evidence(
    track: dict[str, Any],
    *,
    primary_metric: str,
    lower_baselines: tuple[str, ...],
    competitive_baselines: tuple[str, ...],
) -> dict[str, Any]:
    baselines = dict(track.get("baselines", {}))
    agent_metrics = dict(track.get("metrics", {}))
    agent_primary = _metric_value(agent_metrics, primary_metric)
    lower_rows: list[dict[str, Any]] = []
    for baseline_name in lower_baselines:
        baseline_metrics = dict(baselines.get(baseline_name, {}).get("metrics", {}))
        baseline_primary = _metric_value(baseline_metrics, primary_metric)
        lower_rows.append(
            {
                "name": baseline_name,
                "present": baseline_name in baselines,
                "baseline_primary_metric": baseline_primary,
                "agent_primary_metric": agent_primary,
                "agent_beats_baseline": (
                    None if agent_primary is None or baseline_primary is None else agent_primary > baseline_primary
                ),
            }
        )
    competitive_rows: list[dict[str, Any]] = []
    for baseline_name in competitive_baselines:
        baseline_metrics = dict(baselines.get(baseline_name, {}).get("metrics", {}))
        baseline_primary = _metric_value(baseline_metrics, primary_metric)
        competitive_rows.append(
            {
                "name": baseline_name,
                "present": baseline_name in baselines,
                "baseline_primary_metric": baseline_primary,
                "agent_primary_metric": agent_primary,
                "parity_matched": _competitive_parity(agent_metrics, baseline_metrics, primary_metric=primary_metric)
                if baseline_name in baselines
                else False,
                "relative_gap_blocking": _relative_worse_than(agent_primary, baseline_primary) if baseline_name in baselines else False,
            }
        )
    stored_ladder = dict(track.get("baseline_ladder", {}))
    return {
        "primary_metric": primary_metric,
        "agent_primary_metric": agent_primary,
        "lower_baselines": lower_rows,
        "competitive_baselines": competitive_rows,
        "computed_lower_baselines_beaten": all(row["agent_beats_baseline"] is True for row in lower_rows) if lower_rows else False,
        "reported_lower_baselines_beaten": bool(stored_ladder.get("lower_baselines_beaten", False)),
        "computed_competitive_baseline_matched": any(row["parity_matched"] for row in competitive_rows),
        "reported_competitive_baseline_matched": bool(stored_ladder.get("competitive_baseline_matched", False)),
        "computed_competitive_review_block": all(row["relative_gap_blocking"] for row in competitive_rows) if competitive_rows else False,
        "reported_competitive_review_block": bool(stored_ladder.get("competitive_review_block", False)),
    }


def _failure_modes_have_examples(entries: Any) -> bool:
    if not isinstance(entries, list) or not entries:
        return False
    for row in entries:
        if not isinstance(row, dict):
            return False
        examples = row.get("examples")
        if not isinstance(examples, list) or not examples:
            return False
    return True


def _build_headline_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    confidence = payload.get("confidence", {})
    igt = payload.get("igt", {})
    sensitivity = payload.get("parameter_sensitivity", {})
    return {
        "confidence_database": {
            "trial_count": int(confidence.get("trial_count", 0)),
            "subject_count": _subject_count(confidence),
            "claim_envelope": confidence.get("claim_envelope"),
            "external_bundle": confidence.get("source_type") == "external_bundle",
        },
        "iowa_gambling_task": {
            "trial_count": int(igt.get("trial_count", 0)),
            "subject_count": _subject_count(igt),
            "claim_envelope": igt.get("claim_envelope"),
            "external_bundle": igt.get("source_type") == "external_bundle",
        },
        "parameter_sensitivity": {
            "trial_count": None,
            "subject_count": None,
            "claim_envelope": sensitivity.get("claim_envelope"),
            "external_bundle": sensitivity.get("source_type") == "external_bundle",
        },
    }


def _evaluate_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = bool(payload.get("blocked"))
    confidence = payload.get("confidence", {})
    igt = payload.get("igt", {})
    sensitivity = payload.get("parameter_sensitivity", {})
    failure_modes = dict(payload.get("failure_analysis", {}).get("failure_modes", {}))
    confidence_subject_count = _subject_count(confidence)
    igt_subject_count = _subject_count(igt)
    headline_metrics = _build_headline_metrics(payload)
    confidence_baseline = _baseline_evidence(
        confidence,
        primary_metric="heldout_likelihood",
        lower_baselines=("random", "stimulus_only"),
        competitive_baselines=("statistical_logistic", "human_match_ceiling"),
    )
    igt_baseline = _baseline_evidence(
        igt,
        primary_metric="deck_match_rate",
        lower_baselines=("random", "frequency_matching"),
        competitive_baselines=("human_behavior",),
    )

    if blocked:
        gates = {
            "fit_confidence_db": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "mode": confidence.get("mode"),
                    "source_type": confidence.get("source_type"),
                    "required_external_bundle": True,
                },
            },
            "fit_igt": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "mode": igt.get("mode"),
                    "source_type": igt.get("source_type"),
                    "required_external_bundle": True,
                    "protocol_mode": igt.get("protocol_mode"),
                },
            },
            "baseline_ladder": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "confidence_database": confidence_baseline,
                    "iowa_gambling_task": igt_baseline,
                },
            },
            "parameter_sensitivity": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "source_type": sensitivity.get("source_type"),
                    "claim_envelope": sensitivity.get("claim_envelope"),
                    "parameter_count": len(list(sensitivity.get("parameters", []))),
                    "active_parameter_count": int(sensitivity.get("active_parameter_count", 0)),
                    "required_active_parameter_count": int(sensitivity.get("required_active_parameter_count", 4)),
                },
            },
            "honest_failure_analysis": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "failure_mode_tracks": sorted(failure_modes),
                    "confidence_database_examples": False,
                    "iowa_gambling_task_examples": False,
                },
            },
            "non_circular_scoring": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "confidence_split_unit": confidence.get("split_unit"),
                    "igt_split_unit": igt.get("split_unit"),
                },
            },
            "no_synthetic_claims": {
                "passed": True,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "confidence_claim_envelope": confidence.get("claim_envelope"),
                    "igt_claim_envelope": igt.get("claim_envelope"),
                    "parameter_sensitivity_claim_envelope": sensitivity.get("claim_envelope"),
                    "external_validation_flags": {
                        "confidence_database": bool(confidence.get("external_validation", False)),
                        "iowa_gambling_task": bool(igt.get("external_validation", False)),
                        "parameter_sensitivity": bool(sensitivity.get("external_validation", False)),
                    },
                },
            },
            "sample_size_sufficient": {
                "passed": False,
                "blocking": True,
                "evidence": {
                    "blocked": True,
                    "confidence_trial_count": int(confidence.get("trial_count", 0)),
                    "confidence_subject_count": confidence_subject_count,
                    "igt_subject_count": igt_subject_count,
                    "required_confidence_trial_count": 1000,
                    "required_confidence_subject_count": 10,
                    "required_igt_subject_count": 3,
                },
            },
            "regression": {
                "passed": True,
                "blocking": True,
                "evidence": {
                    "source": "separate_test_suite",
                    "tracked_suites": ["tests/test_m41_*.py", "tests/test_m42_*.py"],
                    "interface_constraints": [
                        "CognitiveStyleParameters unchanged",
                        "BenchmarkAdapter protocol unchanged",
                    ],
                },
            },
        }
        findings = [
            {
                "severity": "S1",
                "label": "external_bundle_missing",
                "detail": "Acceptance-grade external bundles were not found, so M4.3 remains blocked instead of claiming benchmark fit.",
            }
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

    confidence_is_external = (
        confidence.get("mode") == "benchmark_eval"
        and confidence.get("source_type") == "external_bundle"
        and confidence.get("claim_envelope") == "benchmark_eval"
        and not bool(confidence.get("external_validation", False))
    )
    igt_is_external = (
        igt.get("mode") == "benchmark_eval"
        and igt.get("protocol_mode") == "standard_100"
        and igt.get("source_type") == "external_bundle"
        and igt.get("claim_envelope") == "benchmark_eval"
        and not bool(igt.get("external_validation", False))
    )
    parameter_sensitivity_is_external = (
        sensitivity.get("source_type") == "external_bundle"
        and sensitivity.get("claim_envelope") == "benchmark_eval"
        and not bool(sensitivity.get("external_validation", False))
    )
    confidence_competitive_present = any(row["present"] for row in confidence_baseline["competitive_baselines"])
    igt_competitive_present = any(row["present"] for row in igt_baseline["competitive_baselines"])
    sample_size_passed = (
        int(confidence.get("trial_count", 0)) >= 1000
        and confidence_subject_count >= 10
        and igt_subject_count >= 3
    )
    parameter_count = len(list(sensitivity.get("parameters", [])))
    parameter_sensitivity_passed = (
        parameter_sensitivity_is_external
        and parameter_count == 8
        and int(sensitivity.get("active_parameter_count", 0)) >= int(sensitivity.get("required_active_parameter_count", 4))
    )
    confidence_failure_modes = failure_modes.get("confidence_database", [])
    igt_failure_modes = failure_modes.get("iowa_gambling_task", [])
    honest_failure_analysis_passed = (
        _failure_modes_have_examples(confidence_failure_modes)
        and _failure_modes_have_examples(igt_failure_modes)
    )
    non_circular_passed = (
        confidence.get("split_unit") == "subject_id"
        and igt.get("split_unit") == "subject_id"
        and bool(confidence.get("leakage_check", {}).get("subject", {}).get("ok"))
        and bool(confidence.get("leakage_check", {}).get("session", {}).get("ok"))
        and bool(igt.get("leakage_check", {}).get("subject", {}).get("ok"))
        and int(confidence.get("training_trial_count", 0)) > 0
        and int(confidence.get("validation_trial_count", 0)) > 0
        and int(confidence.get("trial_count", 0)) > 0
        and int(igt.get("training_trial_count", 0)) > 0
        and int(igt.get("validation_trial_count", 0)) > 0
        and int(igt.get("trial_count", 0)) > 0
        and confidence_baseline["primary_metric"] == "heldout_likelihood"
        and igt_baseline["primary_metric"] == "deck_match_rate"
    )
    no_synthetic_claims_passed = _no_synthetic_claims(payload)
    baseline_ladder_passed = (
        all(row["present"] for row in confidence_baseline["lower_baselines"])
        and all(row["present"] for row in confidence_baseline["competitive_baselines"])
        and all(row["present"] for row in igt_baseline["lower_baselines"])
        and all(row["present"] for row in igt_baseline["competitive_baselines"])
        and confidence_baseline["computed_lower_baselines_beaten"] == confidence_baseline["reported_lower_baselines_beaten"]
        and igt_baseline["computed_lower_baselines_beaten"] == igt_baseline["reported_lower_baselines_beaten"]
        and confidence_baseline["computed_competitive_baseline_matched"] == confidence_baseline["reported_competitive_baseline_matched"]
        and igt_baseline["computed_competitive_baseline_matched"] == igt_baseline["reported_competitive_baseline_matched"]
        and confidence_baseline["computed_competitive_review_block"] == confidence_baseline["reported_competitive_review_block"]
        and igt_baseline["computed_competitive_review_block"] == igt_baseline["reported_competitive_review_block"]
    )

    gates = {
        "fit_confidence_db": {
            "passed": (
                confidence_is_external
                and int(confidence.get("trial_count", 0)) >= 1000
                and confidence_subject_count >= 10
                and confidence_baseline["computed_lower_baselines_beaten"]
                and confidence_competitive_present
            ),
            "blocking": True,
            "evidence": {
                "mode": confidence.get("mode"),
                "source_type": confidence.get("source_type"),
                "claim_envelope": confidence.get("claim_envelope"),
                "external_validation": bool(confidence.get("external_validation", False)),
                "trial_count": int(confidence.get("trial_count", 0)),
                "subject_count": confidence_subject_count,
                "required_trial_count": 1000,
                "required_subject_count": 10,
                "baseline_comparisons": confidence_baseline,
                "competitive_comparison_present": confidence_competitive_present,
            },
        },
        "fit_igt": {
            "passed": (
                igt_is_external
                and igt_subject_count >= 3
                and igt_baseline["computed_lower_baselines_beaten"]
                and igt_competitive_present
            ),
            "blocking": True,
            "evidence": {
                "mode": igt.get("mode"),
                "protocol_mode": igt.get("protocol_mode"),
                "source_type": igt.get("source_type"),
                "claim_envelope": igt.get("claim_envelope"),
                "external_validation": bool(igt.get("external_validation", False)),
                "trial_count": int(igt.get("trial_count", 0)),
                "subject_count": igt_subject_count,
                "required_subject_count": 3,
                "baseline_comparisons": igt_baseline,
                "competitive_comparison_present": igt_competitive_present,
            },
        },
        "baseline_ladder": {
            "passed": baseline_ladder_passed,
            "blocking": True,
            "evidence": {
                "confidence_database": confidence_baseline,
                "iowa_gambling_task": igt_baseline,
                "requires_honest_competitive_reporting": True,
            },
        },
        "parameter_sensitivity": {
            "passed": parameter_sensitivity_passed,
            "blocking": True,
            "evidence": {
                "source_type": sensitivity.get("source_type"),
                "claim_envelope": sensitivity.get("claim_envelope"),
                "external_validation": bool(sensitivity.get("external_validation", False)),
                "parameter_count": parameter_count,
                "required_parameter_count": 8,
                "active_parameter_count": int(sensitivity.get("active_parameter_count", 0)),
                "required_active_parameter_count": int(sensitivity.get("required_active_parameter_count", 4)),
            },
        },
        "honest_failure_analysis": {
            "passed": honest_failure_analysis_passed,
            "blocking": True,
            "evidence": {
                "confidence_database_mode_count": len(confidence_failure_modes) if isinstance(confidence_failure_modes, list) else 0,
                "iowa_gambling_task_mode_count": len(igt_failure_modes) if isinstance(igt_failure_modes, list) else 0,
                "confidence_database_examples_present": _failure_modes_have_examples(confidence_failure_modes),
                "iowa_gambling_task_examples_present": _failure_modes_have_examples(igt_failure_modes),
            },
        },
        "non_circular_scoring": {
            "passed": non_circular_passed,
            "blocking": True,
            "evidence": {
                "confidence_split_unit": confidence.get("split_unit"),
                "igt_split_unit": igt.get("split_unit"),
                "confidence_training_trial_count": int(confidence.get("training_trial_count", 0)),
                "confidence_validation_trial_count": int(confidence.get("validation_trial_count", 0)),
                "confidence_eval_trial_count": int(confidence.get("trial_count", 0)),
                "igt_training_trial_count": int(igt.get("training_trial_count", 0)),
                "igt_validation_trial_count": int(igt.get("validation_trial_count", 0)),
                "igt_eval_trial_count": int(igt.get("trial_count", 0)),
                "confidence_subject_leakage_ok": bool(confidence.get("leakage_check", {}).get("subject", {}).get("ok")),
                "confidence_session_leakage_ok": bool(confidence.get("leakage_check", {}).get("session", {}).get("ok")),
                "igt_subject_leakage_ok": bool(igt.get("leakage_check", {}).get("subject", {}).get("ok")),
                "primary_metrics": {
                    "confidence_database": confidence_baseline["primary_metric"],
                    "iowa_gambling_task": igt_baseline["primary_metric"],
                },
            },
        },
        "no_synthetic_claims": {
            "passed": no_synthetic_claims_passed,
            "blocking": True,
            "evidence": {
                "confidence_database": {
                    "source_type": confidence.get("source_type"),
                    "claim_envelope": confidence.get("claim_envelope"),
                    "external_validation": bool(confidence.get("external_validation", False)),
                },
                "iowa_gambling_task": {
                    "source_type": igt.get("source_type"),
                    "claim_envelope": igt.get("claim_envelope"),
                    "external_validation": bool(igt.get("external_validation", False)),
                },
                "parameter_sensitivity": {
                    "source_type": sensitivity.get("source_type"),
                    "claim_envelope": sensitivity.get("claim_envelope"),
                    "external_validation": bool(sensitivity.get("external_validation", False)),
                },
            },
        },
        "sample_size_sufficient": {
            "passed": sample_size_passed,
            "blocking": True,
            "evidence": {
                "confidence_trial_count": int(confidence.get("trial_count", 0)),
                "confidence_subject_count": confidence_subject_count,
                "igt_subject_count": igt_subject_count,
                "required_confidence_trial_count": 1000,
                "required_confidence_subject_count": 10,
                "required_igt_subject_count": 3,
            },
        },
        "regression": {
            "passed": True,
            "blocking": True,
            "evidence": {
                "source": "separate_test_suite",
                "tracked_suites": ["tests/test_m41_*.py", "tests/test_m42_*.py"],
                "interface_constraints": [
                    "CognitiveStyleParameters unchanged",
                    "BenchmarkAdapter protocol unchanged",
                ],
            },
        },
    }
    findings: list[dict[str, Any]] = []
    if not gates["fit_confidence_db"]["passed"]:
        findings.append({"severity": "S1", "label": "confidence_fit_requirements_unmet", "detail": "Confidence Database acceptance requires external benchmark evaluation, sufficient sample size, lower-baseline wins, and at least one competitive comparison."})
    if not gates["fit_igt"]["passed"]:
        findings.append({"severity": "S1", "label": "igt_fit_requirements_unmet", "detail": "IGT acceptance requires standard_100 external evaluation, sufficient subject coverage, lower-baseline wins, and a competitive comparison."})
    if not gates["baseline_ladder"]["passed"]:
        findings.append({"severity": "S1", "label": "baseline_ladder_inconsistent", "detail": "Baseline ladder reporting is incomplete or inconsistent with the underlying primary-metric comparisons."})
    if not gates["parameter_sensitivity"]["passed"]:
        findings.append({"severity": "S2", "label": "parameter_inertia", "detail": "Fewer than four cognitive parameters produced measurable effects under the real-data sensitivity sweep."})
    if not gates["honest_failure_analysis"]["passed"]:
        findings.append({"severity": "S2", "label": "failure_analysis_missing", "detail": "Failure modes were not documented with concrete real-trial examples for both tasks."})
    if not gates["non_circular_scoring"]["passed"]:
        findings.append({"severity": "S1", "label": "split_leakage_or_circularity", "detail": "Leakage or split hygiene failed, so benchmark-fit claims are not trustworthy."})
    if not gates["no_synthetic_claims"]["passed"]:
        findings.append({"severity": "S1", "label": "claim_labeling_invalid", "detail": "At least one artifact used an invalid claim envelope or external validation label."})
    if not gates["sample_size_sufficient"]["passed"]:
        findings.append({"severity": "S2", "label": "sample_size_insufficient", "detail": "The evaluation slice is too small for M4.3 acceptance."})
    failed_gates = sorted(name for name, gate in gates.items() if not gate["passed"])
    status = "FAIL" if any(gate["blocking"] and not gate["passed"] for gate in gates.values()) else "PASS"
    recommendation = (
        "ACCEPT"
        if all(gates[name]["passed"] for name in (
            "fit_confidence_db",
            "fit_igt",
            "baseline_ladder",
            "parameter_sensitivity",
            "honest_failure_analysis",
            "non_circular_scoring",
            "regression",
        ))
        and gates["sample_size_sufficient"]["passed"]
        and gates["no_synthetic_claims"]["passed"]
        else "BLOCK"
    )
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


def write_m43_acceptance_artifacts(
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
    acceptance_root = _resolve_acceptance_root(benchmark_root)
    suite_root = acceptance_root if acceptance_root is not None else ROOT / "_missing_external_bundle_"
    payload = run_m43_single_task_suite(seed=43, benchmark_root=suite_root, allow_smoke_test=False, sample_limits=sample_limits)
    evaluation = _evaluate_acceptance(payload)

    output_paths["confidence_fit"].write_text(json.dumps(payload.get("confidence", {}), indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["igt_fit"].write_text(json.dumps(payload.get("igt", {}), indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["parameter_sensitivity"].write_text(
        json.dumps(payload.get("parameter_sensitivity", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["failure_modes"].write_text(
        json.dumps(payload.get("failure_analysis", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "milestone_id": "M4.3",
        "generated_at": _now_iso(),
        "round_started_at": started_at,
        "git_head": _git_head(),
        "status": evaluation["status"],
        "acceptance_state": evaluation["acceptance_state"],
        "benchmark_root": str(acceptance_root) if acceptance_root else None,
        "artifacts": {
            "confidence_fit": str(output_paths["confidence_fit"]),
            "igt_fit": str(output_paths["igt_fit"]),
            "parameter_sensitivity": str(output_paths["parameter_sensitivity"]),
            "failure_modes": str(output_paths["failure_modes"]),
            "summary": str(output_paths["summary"]),
        },
        "tracks": {
            "confidence_database": {
                "status": "blocked" if payload.get("blocked") else payload["confidence"]["mode"],
                "trial_count": payload.get("confidence", {}).get("trial_count", 0),
                "subject_count": payload.get("confidence", {}).get("subject_summary", {}).get("subject_count", 0),
                "source_type": payload.get("confidence", {}).get("source_type"),
                "claim_envelope": payload.get("confidence", {}).get("claim_envelope"),
                "primary_metric": payload.get("confidence", {}).get("metrics", {}).get("primary_metric"),
                "lower_baselines_beaten": payload.get("confidence", {}).get("baseline_ladder", {}).get("lower_baselines_beaten"),
                "competitive_baseline_matched": payload.get("confidence", {}).get("baseline_ladder", {}).get("competitive_baseline_matched"),
            },
            "iowa_gambling_task": {
                "status": "blocked" if payload.get("blocked") else payload["igt"]["mode"],
                "trial_count": payload.get("igt", {}).get("trial_count", 0),
                "subject_count": payload.get("igt", {}).get("subject_summary", {}).get("subject_count", 0),
                "source_type": payload.get("igt", {}).get("source_type"),
                "claim_envelope": payload.get("igt", {}).get("claim_envelope"),
                "protocol_mode": payload.get("igt", {}).get("protocol_mode"),
                "primary_metric": payload.get("igt", {}).get("metrics", {}).get("primary_metric"),
                "lower_baselines_beaten": payload.get("igt", {}).get("baseline_ladder", {}).get("lower_baselines_beaten"),
                "competitive_baseline_matched": payload.get("igt", {}).get("baseline_ladder", {}).get("competitive_baseline_matched"),
            },
            "parameter_sensitivity": {
                "status": "blocked" if payload.get("blocked") else "benchmark_eval",
                "active_parameter_count": payload.get("parameter_sensitivity", {}).get("active_parameter_count", 0),
                "required_active_parameter_count": payload.get("parameter_sensitivity", {}).get("required_active_parameter_count", 4),
                "claim_envelope": payload.get("parameter_sensitivity", {}).get("claim_envelope"),
                "source_type": payload.get("parameter_sensitivity", {}).get("source_type"),
            },
        },
        "gates": evaluation["gates"],
        "failed_gates": evaluation["failed_gates"],
        "findings": evaluation["findings"],
        "top_blockers": evaluation["top_blockers"],
        "headline_metrics": evaluation["headline_metrics"],
        "recommendation": evaluation["recommendation"],
        "failure_modes": payload.get("failure_analysis", {}).get("failure_modes", {}),
        "tests": {
            "milestone": ["tests/test_m43_single_task_fit.py", "tests/test_m43_baselines.py", "tests/test_m43_acceptance.py"],
            "regressions": ["tests/test_m41_*.py", "tests/test_m42_*.py"],
        },
        "readiness": {
            "deployment_readiness": "READY" if evaluation["status"] == "PASS" else "NOT_READY",
            "blocked": evaluation["status"] != "PASS",
        },
    }
    output_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(report, summary_path=output_paths["summary"])
    return {
        "confidence_fit": str(output_paths["confidence_fit"]),
        "igt_fit": str(output_paths["igt_fit"]),
        "parameter_sensitivity": str(output_paths["parameter_sensitivity"]),
        "failure_modes": str(output_paths["failure_modes"]),
        "report": str(output_paths["report"]),
        "summary": str(output_paths["summary"]),
    }


def publish_m43_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    benchmark_root: Path | str | None = None,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, str]:
    official_root = _require_official_acceptance_root(benchmark_root)
    official_output_paths = _resolve_output_paths()
    started_at = round_started_at or _now_iso()
    with tempfile.TemporaryDirectory(prefix="m43_acceptance_", dir=str(ROOT)) as tmpdir:
        temp_output_paths = _resolve_output_paths(output_root=tmpdir)
        write_m43_acceptance_artifacts(
            round_started_at=started_at,
            benchmark_root=official_root,
            sample_limits=sample_limits,
            output_root=tmpdir,
        )
        _validate_real_bundle_outputs(output_paths=temp_output_paths, expected_benchmark_root=official_root)
        return _publish_output_files(temp_output_paths, official_output_paths)
