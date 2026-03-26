from __future__ import annotations

from segmentum.m236_open_continuity_trial import build_m236_runtime_evidence


def test_continuity_metrics_are_recorded_and_fractured_identity_breaks_bounds() -> None:
    full = build_m236_runtime_evidence(seed_set=(236,), variant="full")
    fractured = build_m236_runtime_evidence(seed_set=(236,), variant="fractured_identity")

    full_identity = full["aggregate_metrics"]["identity_retention"]
    fractured_identity = fractured["aggregate_metrics"]["identity_retention"]

    assert full_identity["continuity_mean"] >= 0.70
    assert full_identity["restart_consistency"] >= 0.62
    assert full_identity["anchor_stability"] >= 0.44
    assert fractured_identity["restart_consistency"] < full_identity["restart_consistency"]
    assert fractured["aggregate_acceptance"]["gates"]["bounded_continuity"]["passed"] is False

