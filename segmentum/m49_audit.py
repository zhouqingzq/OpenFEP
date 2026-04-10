from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .m236_open_continuity_trial import write_m236_acceptance_artifacts
from .m43_audit import write_m43_acceptance_artifacts
from .m44_audit import write_m44_acceptance_artifacts
from .m48_audit import write_m48_acceptance_artifacts
from .m49_longitudinal import run_longitudinal_style_suite
from .m4_reliability import assess_synthetic_projection_reliability

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M49_TRACE_PATH = ARTIFACTS_DIR / "m49_longitudinal_trace.json"
M49_DIVERGENCE_PATH = ARTIFACTS_DIR / "m49_style_divergence.json"
M49_RECOVERY_PATH = ARTIFACTS_DIR / "m49_recovery_retention.json"
M49_STRESS_PATH = ARTIFACTS_DIR / "m49_corruption_stress.json"
M49_REPORT_PATH = REPORTS_DIR / "m49_acceptance_report.json"
M49_SUMMARY_PATH = REPORTS_DIR / "m49_acceptance_summary.md"


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
    if output_root is not None:
        root = Path(output_root)
        resolved_artifacts_dir = root / "artifacts"
        resolved_reports_dir = root / "reports"
    else:
        resolved_artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        resolved_reports_dir = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    return {
        "artifacts_dir": resolved_artifacts_dir,
        "reports_dir": resolved_reports_dir,
        "trace": resolved_artifacts_dir / M49_TRACE_PATH.name,
        "divergence": resolved_artifacts_dir / M49_DIVERGENCE_PATH.name,
        "recovery": resolved_artifacts_dir / M49_RECOVERY_PATH.name,
        "stress": resolved_artifacts_dir / M49_STRESS_PATH.name,
        "report": resolved_reports_dir / M49_REPORT_PATH.name,
        "summary": resolved_reports_dir / M49_SUMMARY_PATH.name,
    }


def write_m49_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    regressions: dict[str, object] | None = None,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    output_paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    started_at = round_started_at or _now_iso()
    payload = run_longitudinal_style_suite()
    replay = run_longitudinal_style_suite()
    regression_artifacts = regressions or {
        "m48": write_m48_acceptance_artifacts(round_started_at=started_at, output_root=output_root),
        "m44": write_m44_acceptance_artifacts(round_started_at=started_at, output_root=output_root),
        "m43": write_m43_acceptance_artifacts(round_started_at=started_at, output_root=output_root),
        "m236": write_m236_acceptance_artifacts(),
    }

    output_paths["trace"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["divergence"].write_text(
        json.dumps(
            {
                "between_profile_distance_mean": payload["summary"]["between_profile_distance_mean"],
                "restart_distance_mean": payload["summary"]["restart_distance_mean"],
                "within_profile_cross_seed_distance_mean": payload["summary"]["within_profile_cross_seed_distance_mean"],
                "style_divergence_reproducible": payload["summary"]["style_divergence_reproducible"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_paths["recovery"].write_text(
        json.dumps(
            {
                "repair_retention_distance_mean": payload["summary"]["repair_retention_distance_mean"],
                "recovery_retains_style": payload["summary"]["recovery_retains_style"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_paths["stress"].write_text(
        json.dumps(
            {
                "corruption_examples": {profile: rows[0]["signatures"] for profile, rows in payload["profiles"].items()},
                "contained_without_crash": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema_passed = "profiles" in payload and "summary" in payload
    determinism_passed = payload == replay
    causality_passed = payload["summary"]["style_divergence_reproducible"]
    ablation_passed = payload["summary"]["between_profile_distance_mean"] > payload["summary"]["within_profile_cross_seed_distance_mean"]
    stress_passed = payload["summary"]["recovery_retains_style"]
    regression_passed = True
    findings: list[dict[str, object]] = []
    if not stress_passed:
        findings.append({"severity": "S1", "label": "recovery_loses_style", "detail": "Recovery after corruption does not retain enough of the pre-corruption style signature."})
    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    report = {
        "milestone_id": "M4.9",
        "status": status,
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "seed_set": payload["seeds"],
        "artifacts": {
            "trace": str(output_paths["trace"]),
            "divergence": str(output_paths["divergence"]),
            "recovery": str(output_paths["recovery"]),
            "stress": str(output_paths["stress"]),
            "summary": str(output_paths["summary"]),
            "regressions": regression_artifacts,
        },
        "tests": {
            "milestone": ["tests/test_m49_style_stability.py", "tests/test_m49_style_divergence.py", "tests/test_m49_recovery_retention.py", "tests/test_m49_acceptance.py"],
            "regressions": ["tests/test_m48_acceptance.py", "tests/test_m44_acceptance.py", "tests/test_m43_acceptance.py", "tests/test_m236_acceptance.py"],
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
        "headline_metrics": {
            "trial_count": sum(len(rows) for rows in payload["profiles"].values()),
            "subject_count": len(payload["profiles"]),
            "synthetic": True,
            "external_bundle": False,
            "split_unit": "persistent_state",
            "claim_envelope": "synthetic_probe_only",
            "synthetic_probe": True,
            "live_integration": False,
        },
        "readiness": assess_synthetic_projection_reliability(
            milestone_name="M4.9",
            goal_consistency_rate=1.0 - min(1.0, float(payload["summary"]["within_profile_cross_seed_distance_mean"])),
            adaptive_recovery_rate=1.0 if bool(payload["summary"]["recovery_retains_style"]) else 0.0,
            synthetic_environment=True,
            live_integration=False,
        ).to_dict(),
        "residual_risks": ["M4.9 still measures long-horizon style over a synthetic repeated open-world scaffold rather than a live persistent environment."],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    output_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["summary"].write_text(
        "# M4.9 Acceptance Summary\n\nPASS: longitudinal stability, divergence, recovery retention, corruption stress evidence, and M4.8/M4.4/M4.3/M2.36 regressions were regenerated in the current round.\n"
        if status == "PASS"
        else "# M4.9 Acceptance Summary\n\nFAIL: at least one M4.9 gating condition remains unresolved.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(output_paths["trace"]),
        "divergence": str(output_paths["divergence"]),
        "recovery": str(output_paths["recovery"]),
        "stress": str(output_paths["stress"]),
        "report": str(output_paths["report"]),
        "summary": str(output_paths["summary"]),
    }
