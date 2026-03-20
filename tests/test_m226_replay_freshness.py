from __future__ import annotations

from segmentum.m224_benchmarks import run_m224_workspace_benchmark
from segmentum.m226_maturity_audit import SEED_SET, _standardize_m224, build_m226_maturity_audit

from tests._m226_test_utils import base_standardized_replays


def test_all_required_milestone_replays_run_this_round() -> None:
    payload = build_m226_maturity_audit(standardized_replays=base_standardized_replays(), seed_set=list(SEED_SET), codebase_version="test-sha")
    freshness = payload["replay_freshness"]
    report = payload["final_report"]

    assert freshness["current_round_replay_coverage"] == 1.0
    assert freshness["inherited_only_critical_metric_count"] == 0
    assert freshness["stale_artifact_misuse_rate"] == 0.0
    assert freshness["generated_this_round"] is True
    assert len(freshness["milestone_replays"]) == 5
    assert all(item["current_round_replay_status"] for item in freshness["milestone_replays"])
    assert report["codebase_provenance"]["git_commit"] == "test-sha"
    assert report["codebase_provenance"]["binding_scope"] == "provided"


def test_m224_direct_runtime_payload_is_current_round_for_m226() -> None:
    standardized = _standardize_m224(run_m224_workspace_benchmark())

    assert standardized["freshness_status"] is True
    assert standardized["current_round_replay_status"] is True
    assert standardized["evidence_origin"] == "current_round_replay"
