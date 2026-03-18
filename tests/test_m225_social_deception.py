from __future__ import annotations

from segmentum.m225_benchmarks import run_m225_open_world_transfer


def test_deceptive_salience_metrics_are_trace_aggregated() -> None:
    payload = run_m225_open_world_transfer()
    report = payload["acceptance_report"]
    social_artifact = payload["artifacts"]["social_deception"]
    assert report["deception_breakdown"]["adversarial_resistance_score"] >= 0.70
    assert report["deception_breakdown"]["social_deception_resistance"] >= 0.70
    assert report["deception_breakdown"]["deceptive_salience_error_rate"] <= 0.20

    events = social_artifact["adversarial_event_log"]
    hijack_rate = sum(1 for event in events if bool(event.get("policy_hijacked"))) / max(1, len(events))
    assert hijack_rate <= 0.20

    row = next(
        row
        for row in payload["rows"]
        if row["variant"] == "full_system" and row["protocol"] == "misleading_salience_protocol"
    )
    derived_error_rate = row["raw_metrics"]["salience_errors"] / max(1, row["raw_metrics"]["salience_decoys"])
    assert row["metrics"]["deceptive_salience_error_rate"] == derived_error_rate
    assert all(event["response"] != "decoy_followed" for event in row["adversarial_event_log"])
