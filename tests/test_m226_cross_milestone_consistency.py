from __future__ import annotations

from segmentum.m226_maturity_audit import SEED_SET, build_m226_maturity_audit

from tests._m226_test_utils import base_standardized_replays


def test_cross_milestone_contradictions_are_surfaced() -> None:
    replays = base_standardized_replays()
    replays["M2.25"]["score"] = 0.52
    replays["M2.25"]["spurious_transfer"] = True

    payload = build_m226_maturity_audit(standardized_replays=replays, seed_set=list(SEED_SET), codebase_version="test-sha")
    cross = payload["cross_milestone_consistency"]

    assert cross["high_severity_conflict_count"] >= 1
    assert any(check["check_id"] == "cross_001_narrative_vs_transfer" and not check["pass"] for check in cross["checks"])


def test_m223_protocol_alignment_prevents_false_cross_conflicts() -> None:
    replays = base_standardized_replays()
    replays["M2.23"]["seed_set"] = list(SEED_SET)
    replays["M2.23"]["milestone_status"] = True
    replays["M2.23"]["false_consistency"] = False
    replays["M2.23"]["gating_metrics_summary"]["protocol_integrity"] = True

    payload = build_m226_maturity_audit(standardized_replays=replays, seed_set=list(SEED_SET), codebase_version="test-sha")
    cross = payload["cross_milestone_consistency"]
    failed_checks = {
        check["check_id"]
        for check in cross["checks"]
        if not check["pass"]
    }
    residual_risk_ids = {
        item["risk_id"]
        for item in payload["final_report"]["residual_risk_summary"]["risks"]
    }

    assert "cross_002_identity_vs_autonomy" not in failed_checks
    assert "cross_003_workspace_vs_self_report" not in failed_checks
    assert "cross_002_identity_vs_autonomy" not in residual_risk_ids
    assert "cross_003_workspace_vs_self_report" not in residual_risk_ids
