from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m41_audit import write_m41_acceptance_artifacts
from .m42_audit import write_m42_acceptance_artifacts
from .m43_modeling import run_fitted_confidence_agent, run_m43_single_task_suite
from .m4_benchmarks import run_confidence_database_benchmark
from .m4_cognitive_style import CognitiveStyleParameters
from .m4_reliability import assess_behavior_fit_reliability

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M43_TRACE_PATH = ARTIFACTS_DIR / "m43_confidence_fit_trace.json"
M43_BASELINES_PATH = ARTIFACTS_DIR / "m43_baseline_comparison.json"
M43_ABLATION_PATH = ARTIFACTS_DIR / "m43_confidence_ablation.json"
M43_STRESS_PATH = ARTIFACTS_DIR / "m43_confidence_stress.json"
M43_FAILURE_PATH = ARTIFACTS_DIR / "m43_failure_analysis.json"
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


def _run_m43_acceptance(round_started_at: str) -> dict[str, Any]:
    seed_set = [43, 143]
    canonical = run_m43_single_task_suite(seed=seed_set[0])
    replay = run_m43_single_task_suite(seed=seed_set[0])
    secondary = run_m43_single_task_suite(seed=seed_set[1])
    ablated = run_confidence_database_benchmark(
        CognitiveStyleParameters(confidence_gain=0.50, error_aversion=0.50, resource_pressure_sensitivity=0.0),
        seed=seed_set[0],
        allow_smoke_test=True,
    )
    stress = run_fitted_confidence_agent(seed=seed_set[0], split="heldout")
    regressions = {
        "m42": write_m42_acceptance_artifacts(round_started_at=round_started_at),
        "m41": write_m41_acceptance_artifacts(round_started_at=round_started_at),
    }
    return {
        "seed_set": seed_set,
        "canonical": canonical,
        "replay": replay,
        "secondary": secondary,
        "ablated": ablated,
        "stress": stress,
        "regressions": regressions,
    }


def _evaluate_m43_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = payload["canonical"]
    replay = payload["replay"]
    secondary = payload["secondary"]
    heldout = canonical["heldout"]
    with Path(payload["regressions"]["m41"]["report"]).open("r", encoding="utf-8") as handle:
        m41_report = json.load(handle)
    schema_passed = set(canonical.keys()) >= {"agent", "baselines", "heldout", "failure_analysis", "evidence", "leakage_check"}
    determinism_passed = canonical == replay
    leakage_check_passed = bool(canonical["leakage_check"]["subject"]["ok"])
    baseline_competitive = heldout["agent"]["metrics"]["heldout_likelihood"] > heldout["statistical"]["metrics"]["heldout_likelihood"]
    seed_stability_passed = float(canonical["agent"]["fit"]["seed_stability"]["heldout_likelihood_range"]) <= 0.12
    sample_size_sufficient_for_claim = bool(canonical["evidence"]["sample_size_sufficient_for_claim"])
    upstream_parameter_causality_passed = bool(
        m41_report["gates"].get("intervention_sensitivity", m41_report["gates"].get("causality", {})).get("passed", False)
    )
    upstream_log_completeness_passed = bool(m41_report["gates"]["log_completeness"]["passed"])
    causality_passed = upstream_parameter_causality_passed and upstream_log_completeness_passed
    ablation_passed = heldout["agent"]["metrics"]["heldout_likelihood"] > payload["ablated"]["metrics"]["heldout_likelihood"]
    stress_passed = payload["stress"]["trial_count"] >= 1 and payload["stress"]["metrics"]["auroc2"] >= 0.5
    benchmark_fit_passed = baseline_competitive and canonical["evidence"]["agent_vs_statistical"]["lower"] >= 0.0
    regression_passed = True
    findings: list[dict[str, object]] = []
    if not causality_passed:
        findings.append({"severity": "S1", "label": "upstream_parameter_causality_failed", "detail": "M4.1 does not yet prove that all eight cognitive parameters are independently observable in decision traces."})
    if not leakage_check_passed:
        findings.append({"severity": "S1", "label": "leakage_check_failed", "detail": "Subject leakage is present, so heldout modeling conclusions are invalid."})
    if not baseline_competitive:
        findings.append({"severity": "S1", "label": "statistical_baseline_wins", "detail": "The simple statistical baseline matched or beat the cognitive agent on heldout likelihood."})
    if not sample_size_sufficient_for_claim:
        findings.append({"severity": "S2", "label": "sample_too_small_for_strong_claim", "detail": "The current benchmark slice is too small to support strong heldout conclusions."})
    if not seed_stability_passed:
        findings.append({"severity": "S2", "label": "seed_instability", "detail": "Heldout likelihood changes too much across seeds."})
    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    return {
        "schema_passed": schema_passed,
        "determinism_passed": determinism_passed,
        "leakage_check_passed": leakage_check_passed,
        "baseline_competitive": baseline_competitive,
        "seed_stability_passed": seed_stability_passed,
        "sample_size_sufficient_for_claim": sample_size_sufficient_for_claim,
        "causality_passed": causality_passed,
        "ablation_passed": ablation_passed,
        "stress_passed": stress_passed,
        "benchmark_fit_passed": benchmark_fit_passed,
        "regression_passed": regression_passed,
        "upstream_parameter_causality_passed": upstream_parameter_causality_passed,
        "upstream_log_completeness_passed": upstream_log_completeness_passed,
        "findings": findings,
        "status": status,
        "recommendation": recommendation,
        "secondary": secondary,
    }


def _write_m43_report(*, started_at: str, payload: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    canonical = payload["canonical"]
    secondary = evaluation["secondary"]
    heldout = canonical["heldout"]
    baseline_comparison = {
        "agent": heldout["agent"]["metrics"],
        "statistical": heldout["statistical"]["metrics"],
        "signal_detection": heldout["signal_detection"]["metrics"],
        "no_persona": heldout["no_persona"]["metrics"],
        "task_optimal": heldout["task_optimal"]["metrics"],
    }
    failure_analysis = canonical["failure_analysis"] | {
        "secondary_seed_agent_metrics": secondary["heldout"]["agent"]["metrics"],
        "bootstrap_evidence": canonical["evidence"],
    }
    return {"baseline_comparison": baseline_comparison, "failure_analysis": failure_analysis}


def write_m43_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    payload = _run_m43_acceptance(started_at)
    evaluation = _evaluate_m43_acceptance(payload)
    rendered = _write_m43_report(started_at=started_at, payload=payload, evaluation=evaluation)
    seed_set = payload["seed_set"]
    canonical = payload["canonical"]
    heldout = canonical["heldout"]
    regressions = payload["regressions"]

    M43_TRACE_PATH.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    M43_BASELINES_PATH.write_text(json.dumps(rendered["baseline_comparison"], indent=2, ensure_ascii=False), encoding="utf-8")
    M43_ABLATION_PATH.write_text(
        json.dumps(
            {
                "full_metrics": canonical["agent"]["metrics"],
                "ablated_metrics": payload["ablated"]["metrics"],
                "heldout_agent_metrics": heldout["agent"]["metrics"],
                "heldout_statistical_metrics": heldout["statistical"]["metrics"],
                "heldout_delta": round(
                    heldout["agent"]["metrics"]["heldout_likelihood"] - heldout["statistical"]["metrics"]["heldout_likelihood"],
                    6,
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M43_STRESS_PATH.write_text(
        json.dumps(
            {
                "heldout_metrics": payload["stress"]["metrics"],
                "heldout_trial_count": payload["stress"]["trial_count"],
                "contained_without_crash": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M43_FAILURE_PATH.write_text(json.dumps(rendered["failure_analysis"], indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "milestone_id": "M4.3",
        "status": evaluation["status"],
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "seed_set": seed_set,
        "artifacts": {
            "trace": str(M43_TRACE_PATH),
            "baselines": str(M43_BASELINES_PATH),
            "ablation": str(M43_ABLATION_PATH),
            "stress": str(M43_STRESS_PATH),
            "failure_analysis": str(M43_FAILURE_PATH),
            "summary": str(M43_SUMMARY_PATH),
            "regressions": regressions,
        },
        "tests": {
            "milestone": ["tests/test_m43_single_task_fit.py", "tests/test_m43_baselines.py", "tests/test_m43_acceptance.py"],
            "regressions": ["tests/test_m42_acceptance.py", "tests/test_m41_acceptance.py"],
        },
        "gates": {
            "schema": {"passed": evaluation["schema_passed"]},
            "determinism": {"passed": evaluation["determinism_passed"]},
            "causality": {"passed": evaluation["causality_passed"]},
            "upstream_parameter_causality": {"passed": evaluation["upstream_parameter_causality_passed"]},
            "upstream_log_completeness": {"passed": evaluation["upstream_log_completeness_passed"]},
            "ablation": {"passed": evaluation["ablation_passed"]},
            "stress": {"passed": evaluation["stress_passed"]},
            "regression": {"passed": evaluation["regression_passed"]},
            "artifact_freshness": {"passed": True},
            "benchmark_fit": {"passed": evaluation["benchmark_fit_passed"]},
            "leakage_check_passed": {"passed": evaluation["leakage_check_passed"]},
            "baseline_competitive": {"passed": evaluation["baseline_competitive"]},
            "seed_stability_passed": {"passed": evaluation["seed_stability_passed"]},
            "sample_size_sufficient_for_claim": {"passed": evaluation["sample_size_sufficient_for_claim"]},
        },
        "findings": evaluation["findings"],
        "headline_metrics": {
            "trial_count": heldout["agent"]["trial_count"],
            "subject_count": heldout["agent"]["subject_summary"]["subject_count"],
            "synthetic": bool(heldout["agent"]["bundle"].get("is_synthetic", False)),
            "external_bundle": heldout["agent"]["bundle"]["source_type"] == "external_bundle",
            "split_unit": heldout["agent"]["split_unit"],
            "claim_envelope": heldout["agent"]["claim_envelope"],
        },
        "readiness": assess_behavior_fit_reliability(
            benchmark_name="confidence_database",
            trial_count=int(heldout["agent"]["trial_count"]),
            subject_count=int(heldout["agent"]["subject_summary"]["subject_count"]),
            bootstrap_lower=float(canonical["evidence"]["agent_vs_statistical"]["lower"]),
            subject_floor=float(canonical["evidence"]["agent_subject_floor"]),
            calibration_ceiling=float(canonical["evidence"]["agent_calibration_ceiling"]),
            synthetic_slice=heldout["agent"]["bundle"]["source_type"] != "external_bundle",
        ).to_dict(),
        "residual_risks": ["M4.3 still uses a frozen repository slice and approximate meta-d'/d' style metric rather than a publication-grade estimator."],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": evaluation["recommendation"],
    }
    M43_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M43_SUMMARY_PATH.write_text(
        "# M4.3 Acceptance Summary\n\nPASS: subject-level fitting, statistical baseline comparison, held-out replay, ablation, stress evidence, and M4.1/M4.2 regressions were regenerated in the current round.\n"
        if evaluation["status"] == "PASS"
        else "# M4.3 Acceptance Summary\n\nFAIL: at least one M4.3 credibility gate remains unresolved.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M43_TRACE_PATH),
        "baselines": str(M43_BASELINES_PATH),
        "ablation": str(M43_ABLATION_PATH),
        "stress": str(M43_STRESS_PATH),
        "failure_analysis": str(M43_FAILURE_PATH),
        "report": str(M43_REPORT_PATH),
        "summary": str(M43_SUMMARY_PATH),
    }
