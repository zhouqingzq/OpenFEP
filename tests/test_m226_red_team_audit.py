from __future__ import annotations

from segmentum.m226_maturity_audit import SEED_SET, build_m226_maturity_audit

from tests._m226_test_utils import base_standardized_replays


def test_red_team_failures_downgrade_final_maturity_status() -> None:
    replays = base_standardized_replays()
    replays["M2.24"]["report_leakage"] = True
    replays["M2.24"]["gating_metrics_summary"]["semantic_report_leakage"] = False

    payload = build_m226_maturity_audit(standardized_replays=replays, seed_set=list(SEED_SET), codebase_version="test-sha")
    report = payload["final_report"]

    assert payload["red_team_audit"]["high_severity_red_team_failures"] >= 1
    assert report["final_status"] != "DEFAULT_MATURE"
    assert report["default_mature"] is False
