from __future__ import annotations

from segmentum.m236_open_continuity_trial import OpenContinuityTrial


def test_trial_has_explicit_multi_phase_schedule_and_transition_records() -> None:
    trial = OpenContinuityTrial()
    schedule = [phase.phase_id for phase in trial.phases]

    assert "baseline" in schedule
    assert "ambiguity" in schedule
    assert "social_rupture" in schedule
    assert "maintenance" in schedule
    assert "delayed_verification" in schedule
    assert "open_inquiry" in schedule
    assert "restart_shock" in schedule
    assert "conflict_reopen" in schedule
    assert "reconciliation" in schedule
    assert "trace_reactivation" in schedule
    assert "recovery" in schedule

    record = trial.run_seed(seed=236)
    assert len(record.phase_summaries) == len(trial.phases)
    assert len(record.transitions) == len(trial.phases) - 1
    assert any(item.to_phase == "restart_shock" for item in record.transitions)
    assert any(item.to_phase == "recovery" for item in record.transitions)
    assert all(item.transition_coherence >= 0.0 for item in record.transitions)

