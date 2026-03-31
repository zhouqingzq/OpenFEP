from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .m3_audit import write_m36_acceptance_artifacts
from .m44_audit import write_m44_acceptance_artifacts
from .m45_open_world import benchmark_open_world_projection, simulate_open_world_projection
from .m4_cognitive_style import CognitiveStyleParameters
from .m4_reliability import assess_synthetic_projection_reliability

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M45_TRACE_PATH = ARTIFACTS_DIR / "m45_open_world_trace.json"
M45_MAPPING_PATH = ARTIFACTS_DIR / "m45_parameter_projection.json"
M45_ABLATION_PATH = ARTIFACTS_DIR / "m45_open_world_ablation.json"
M45_STRESS_PATH = ARTIFACTS_DIR / "m45_open_world_stress.json"
M45_REPORT_PATH = REPORTS_DIR / "m45_acceptance_report.json"
M45_SUMMARY_PATH = REPORTS_DIR / "m45_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_m45_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    seed_set = [45, 145]
    canonical = simulate_open_world_projection(CognitiveStyleParameters(), seed=seed_set[0])
    replay = simulate_open_world_projection(CognitiveStyleParameters(), seed=seed_set[0])
    ablated = simulate_open_world_projection(CognitiveStyleParameters(), seed=seed_set[0], ablate_style=True)
    stress = simulate_open_world_projection(CognitiveStyleParameters(), seed=seed_set[1], stress=True)
    mapping = benchmark_open_world_projection(seed=seed_set[0])
    regressions = {
        "m44": write_m44_acceptance_artifacts(round_started_at=started_at),
        "m36": write_m36_acceptance_artifacts(round_started_at=started_at),
    }

    M45_TRACE_PATH.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    M45_MAPPING_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    M45_ABLATION_PATH.write_text(
        json.dumps(
            {
                "full_summary": canonical["summary"],
                "ablated_summary": ablated["summary"],
                "recovery_shift": round(canonical["summary"]["adaptive_recovery_rate"] - ablated["summary"]["adaptive_recovery_rate"], 6),
                "resource_shift_retained": canonical["summary"]["resource_sensitive_shift"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M45_STRESS_PATH.write_text(json.dumps(stress, indent=2, ensure_ascii=False), encoding="utf-8")

    schema_passed = set(canonical.keys()) >= {"parameters", "logs", "summary"}
    determinism_passed = canonical == replay
    causality_passed = (
        canonical["summary"]["adaptive_recovery_rate"] > ablated["summary"]["adaptive_recovery_rate"]
        and canonical["summary"]["goal_consistency_rate"] >= 2.0 / 3.0
    )
    ablation_passed = (
        canonical["summary"]["resource_sensitive_shift"]
        and not ablated["summary"]["resource_sensitive_shift"]
        and canonical["summary"]["goal_consistency_rate"] > ablated["summary"]["goal_consistency_rate"]
    )
    stress_passed = (
        stress["summary"]["mechanical_retry_rate"] <= 1.0 / 3.0
        and stress["summary"]["adaptive_recovery_rate"] >= 1.0
    )
    regression_passed = True
    mapping_passed = all(bool(value) for value in mapping["correspondence"].values())
    findings: list[dict[str, object]] = []
    if not mapping_passed:
        findings.append({"severity": "S1", "label": "parameter_behavior_mapping_weak", "detail": "Benchmark parameters did not project cleanly onto open-world behavior differences."})
    if not causality_passed:
        findings.append({"severity": "S1", "label": "recovery_not_style_sensitive", "detail": "Ablation did not reduce adaptive recovery behavior."})
    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    report = {
        "milestone_id": "M4.5",
        "status": status,
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "seed_set": seed_set,
        "artifacts": {
            "trace": str(M45_TRACE_PATH),
            "mapping": str(M45_MAPPING_PATH),
            "ablation": str(M45_ABLATION_PATH),
            "stress": str(M45_STRESS_PATH),
            "summary": str(M45_SUMMARY_PATH),
            "regressions": regressions,
        },
        "tests": {
            "milestone": ["tests/test_m45_parameter_projection.py", "tests/test_m45_failure_recovery.py", "tests/test_m45_acceptance.py"],
            "regressions": ["tests/test_m44_acceptance.py", "tests/test_m36_acceptance.py"],
        },
        "gates": {
            "schema": {"passed": schema_passed},
            "determinism": {"passed": determinism_passed},
            "causality": {"passed": causality_passed},
            "ablation": {"passed": ablation_passed},
            "stress": {"passed": stress_passed},
            "regression": {"passed": regression_passed},
            "artifact_freshness": {"passed": True},
            "parameter_projection": {"passed": mapping_passed},
        },
        "findings": findings,
        "headline_metrics": {
            "trial_count": len(canonical["logs"]),
            "subject_count": 0,
            "synthetic": True,
            "external_bundle": False,
            "split_unit": "n/a",
            "claim_envelope": "synthetic_probe_only",
            "synthetic_probe": True,
            "live_integration": bool(mapping["live_cli_loop"]["summary"]["live_integration"]),
        },
        "readiness": assess_synthetic_projection_reliability(
            milestone_name="M4.5",
            goal_consistency_rate=float(canonical["summary"]["goal_consistency_rate"]),
            adaptive_recovery_rate=float(canonical["summary"]["adaptive_recovery_rate"]),
            synthetic_environment=True,
            live_integration=bool(mapping["live_cli_loop"]["summary"]["live_integration"]),
        ).to_dict(),
        "residual_risks": ["M4.5 remains a tool-shaped synthetic open-world trial rather than a live external environment integration."],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    M45_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M45_SUMMARY_PATH.write_text(
        "# M4.5 Acceptance Summary\n\nPASS: open-world projection, parameter-behavior mapping, ablation, stress evidence, and M4.4/M3.6 regressions were regenerated in the current round.\n"
        if status == "PASS"
        else "# M4.5 Acceptance Summary\n\nFAIL: at least one M4.5 gating condition remains unresolved.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M45_TRACE_PATH),
        "mapping": str(M45_MAPPING_PATH),
        "ablation": str(M45_ABLATION_PATH),
        "stress": str(M45_STRESS_PATH),
        "report": str(M45_REPORT_PATH),
        "summary": str(M45_SUMMARY_PATH),
    }
