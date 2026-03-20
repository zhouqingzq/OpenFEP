from __future__ import annotations

from segmentum.m225_benchmarks import run_m225_open_world_transfer


def test_rule_shifts_trigger_bounded_runtime_adaptation() -> None:
    payload = run_m225_open_world_transfer()
    report = payload["acceptance_report"]
    rows = payload["artifacts"]["rule_shift_recovery"]["rule_shift_rows"]
    assert report["goal_details"]["rule_shift_recovery_rate"] >= 0.70
    assert report["goal_details"]["mean_rule_shift_recovery_ticks"] <= 8.5
    assert report["goal_details"]["bounded_policy_reconfiguration_score"] >= 0.70

    first = rows[0]
    assert first["raw_metrics"]["rule_shift_recovery_tick"] is not None
    assert first["raw_metrics"]["rule_shift_recovery_tick"] <= first["protocol_definition"]["recovery_window"]
    assert first["event_trace"][5]["event_kind"] == "rule_change"
    assert first["action_trace"][5]["action"] in {"hide", "scan", "exploit_shelter", "rest", "thermoregulate", "seek_contact", "forage"}
    assert first["action_trace"][5]["response"] in {"rule_model_updated", "rule_shift_observed"}
    assert first["policy_adaptation_trace"][5]["policy_shift"] in {"bounded_reconfiguration", "rule_shift_detected"}
    assert all(float(row["catastrophic_collapse"]) == 0.0 for row in rows)
