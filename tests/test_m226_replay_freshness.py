from __future__ import annotations

from segmentum.m226_maturity_audit import SEED_SET, build_m226_maturity_audit

from tests._m226_test_utils import base_standardized_replays


def test_all_required_milestone_replays_run_this_round() -> None:
    payload = build_m226_maturity_audit(standardized_replays=base_standardized_replays(), seed_set=list(SEED_SET), codebase_version="test-sha")
    freshness = payload["replay_freshness"]

    assert freshness["current_round_replay_coverage"] == 1.0
    assert freshness["inherited_only_critical_metric_count"] == 0
    assert freshness["stale_artifact_misuse_rate"] == 0.0
    assert freshness["generated_this_round"] is True
    assert len(freshness["milestone_replays"]) == 5
    assert all(item["current_round_replay_status"] for item in freshness["milestone_replays"])
