from __future__ import annotations

from segmentum.m236_open_continuity_trial import build_m236_runtime_evidence


def test_inquiry_stability_metrics_react_to_survival_only_ablation() -> None:
    full = build_m236_runtime_evidence(seed_set=(236,), variant="full")
    survival_only = build_m236_runtime_evidence(seed_set=(236,), variant="survival_only")

    full_inquiry = full["aggregate_metrics"]["inquiry_stability"]
    survival_inquiry = survival_only["aggregate_metrics"]["inquiry_stability"]

    assert full_inquiry["mean_active_targets"] > survival_inquiry["mean_active_targets"]
    assert full_inquiry["low_value_suppression_rate"] >= 0.30
    assert survival_only["aggregate_acceptance"]["gates"]["active_bounded_inquiry"]["passed"] is False
    assert survival_inquiry["inquiry_collapse_detected"] is True or survival_inquiry["mean_active_targets"] < 0.55

