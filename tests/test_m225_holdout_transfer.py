from __future__ import annotations

from segmentum.m225_benchmarks import SEED_SET, run_m225_open_world_transfer


def test_run_preview_is_blocked_until_freshness_is_verified() -> None:
    payload = run_m225_open_world_transfer(seed_set=list(SEED_SET))
    report = payload["acceptance_report"]
    assert report["status"] == "BLOCKED"
    assert report["recommendation"] == "BLOCK"
    assert report["freshness"]["generated_this_round"] is False
    assert report["gates"]["freshness_generated_this_round"] is False
    assert next(item for item in report["tests"] if item["name"] == "artifact_freshness")["status"] == "blocked"


def test_holdout_transfer_metric_is_derived_from_trace_evidence() -> None:
    payload = run_m225_open_world_transfer(seed_set=list(SEED_SET))
    report = payload["acceptance_report"]
    transfer_graph = payload["artifacts"]["transfer_graph"]
    holdouts = set(report["holdout_worlds"])
    tuning_worlds = set(transfer_graph["tuning_worlds"])
    assert holdouts
    assert holdouts.isdisjoint(tuning_worlds)
    assert report["holdout_breakdown"]["holdout_transfer_success_rate"] >= 0.70
    assert report["holdout_breakdown"]["unseen_world_survival_ratio"] >= 0.85

    full_row = next(
        row
        for row in payload["rows"]
        if row["variant"] == "full_system" and row["protocol"] == "holdout_transfer_protocol" and row["seed"] == SEED_SET[0]
    )
    shuffled_row = next(
        row
        for row in payload["rows"]
        if row["variant"] == "shuffled_world_label" and row["protocol"] == "holdout_transfer_protocol" and row["seed"] == SEED_SET[0]
    )
    transfer_events = [
        action["transfer_outcome"]
        for action in full_row["action_trace"]
        if action["transfer_outcome"] in {"success", "failure"}
    ]
    assert len(transfer_events) == int(full_row["raw_metrics"]["transfer_opportunities"])
    expected_retention = full_row["raw_metrics"]["transfer_successes"] / full_row["raw_metrics"]["transfer_opportunities"]
    assert full_row["metrics"]["transfer_retention_score"] == expected_retention
    assert full_row["metrics"]["holdout_transfer_success_rate"] == 1.0
    assert shuffled_row["metrics"]["holdout_transfer_success_rate"] == 0.0
    assert full_row["raw_metrics"]["transfer_successes"] > shuffled_row["raw_metrics"]["transfer_successes"]
