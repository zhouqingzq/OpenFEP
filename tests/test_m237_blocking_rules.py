from __future__ import annotations

from segmentum.m237_audit import DEPENDENCY_REPORTS, build_m237_total_acceptance


def _report(*, status: str = "PASS", recommendation: str = "ACCEPT", fresh: bool = True) -> dict[str, object]:
    return {
        "status": status,
        "recommendation": recommendation,
        "freshness": {"current_round": fresh},
        "generated_at": "2026-03-26T07:30:00+00:00",
    }


def test_failing_dependency_blocks_total_acceptance() -> None:
    overrides = {milestone_id: _report() for milestone_id in DEPENDENCY_REPORTS}
    overrides["M2.22"] = _report(status="FAIL", recommendation="REJECT", fresh=True)

    payload = build_m237_total_acceptance(report_overrides=overrides)

    assert payload["status"] == "BLOCKED"
    assert "M2.22" in payload["summary"]["blocking_milestones"]
    assert any(
        pillar["pillar_id"] == "autonomous_homeostasis_and_continuity" and not pillar["passed"]
        for pillar in payload["pillars"]
    )


def test_stale_dependency_blocks_total_acceptance_even_if_report_passed() -> None:
    overrides = {milestone_id: _report() for milestone_id in DEPENDENCY_REPORTS}
    overrides["M2.25"] = _report(status="PASS", recommendation="ACCEPT", fresh=False)

    payload = build_m237_total_acceptance(report_overrides=overrides)

    assert payload["status"] == "BLOCKED"
    assert "M2.25" in payload["summary"]["blocking_milestones"]
    assert any("M2.25" in item for item in payload["residual_risks"])
