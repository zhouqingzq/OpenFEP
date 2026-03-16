from __future__ import annotations

import json
from pathlib import Path


def build_m210_audit_payload(
    *,
    personality_validation: dict[str, object],
    longitudinal_stability: dict[str, object],
    profile_summary: dict[str, object],
) -> dict[str, object]:
    validation_acceptance = dict(personality_validation.get("acceptance", {}))
    stability_acceptance = dict(longitudinal_stability.get("acceptance", {}))
    profile_payload = dict(profile_summary.get("profiles", {}))
    stale_or_missing = []
    if not personality_validation:
        stale_or_missing.append("personality_validation")
    if not longitudinal_stability:
        stale_or_missing.append("longitudinal_stability")
    if not profile_summary:
        stale_or_missing.append("profile_summary")
    failing_profiles = [
        name
        for name, payload in profile_payload.items()
        if not bool(payload.get("final_passed", False))
    ]
    gate_results = {
        "statistical_support": bool(validation_acceptance.get("passed", False)),
        "longitudinal_stability": bool(stability_acceptance.get("passed", False)),
        "profile_level_consistency": not failing_profiles,
        "artifact_freshness": not stale_or_missing,
    }
    overall_passed = all(gate_results.values())
    return {
        "milestone": "M2.10",
        "validation_acceptance": validation_acceptance,
        "stability_acceptance": stability_acceptance,
        "gate_results": gate_results,
        "failing_profiles": failing_profiles,
        "stale_or_missing_inputs": stale_or_missing,
        "final_recommendation": {
            "status": "ACCEPT_M210" if overall_passed else "REJECT_M210",
            "passed": overall_passed,
        },
    }


def render_m210_audit_report(audit_payload: dict[str, object]) -> str:
    gate_results = dict(audit_payload.get("gate_results", {}))
    validation = dict(audit_payload.get("validation_acceptance", {}))
    stability = dict(audit_payload.get("stability_acceptance", {}))
    status = str(dict(audit_payload.get("final_recommendation", {})).get("status", "UNKNOWN"))
    lines = [
        "# M2.10 Strict Audit Report",
        "",
        f"Final status: `{status}`",
        "",
        "## Gates",
        f"- statistical_support: {gate_results.get('statistical_support', False)}",
        f"- longitudinal_stability: {gate_results.get('longitudinal_stability', False)}",
        f"- profile_level_consistency: {gate_results.get('profile_level_consistency', False)}",
        f"- artifact_freshness: {gate_results.get('artifact_freshness', False)}",
        "",
        "## Statistical Evidence",
        f"- significant_metrics: {validation.get('significant_metrics', [])}",
        f"- effect_metrics: {validation.get('effect_metrics', [])}",
        "",
        "## Stability Evidence",
        f"- profiles_passing: {stability.get('profiles_passing', 0)} / {stability.get('required_profiles', 0)}",
        f"- passed_profiles: {stability.get('passed_profiles', [])}",
    ]
    failing_profiles = list(audit_payload.get("failing_profiles", []))
    if failing_profiles:
        lines.extend(["", "## Failing Profiles", *(f"- {name}" for name in failing_profiles)])
    stale = list(audit_payload.get("stale_or_missing_inputs", []))
    if stale:
        lines.extend(["", "## Missing Inputs", *(f"- {name}" for name in stale)])
    return "\n".join(lines) + "\n"


def write_m210_audit_outputs(
    *,
    audit_payload: dict[str, object],
    summary_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    summary_target = Path(summary_path)
    report_target = Path(report_path)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_target.write_text(render_m210_audit_report(audit_payload), encoding="utf-8")
    return summary_target, report_target
