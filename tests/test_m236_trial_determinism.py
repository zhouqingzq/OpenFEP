from __future__ import annotations

from segmentum.m236_open_continuity_trial import build_m236_runtime_evidence


def test_trial_replay_is_deterministic_for_same_seed_set() -> None:
    first = build_m236_runtime_evidence(seed_set=(236,), variant="full")
    second = build_m236_runtime_evidence(seed_set=(236,), variant="full")

    assert first["determinism"]["stable_replay"] is True
    assert second["determinism"]["stable_replay"] is True
    assert first["aggregate_metrics"] == second["aggregate_metrics"]
    assert first["aggregate_acceptance"] == second["aggregate_acceptance"]
    assert first["audit_records"] == second["audit_records"]

