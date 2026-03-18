from __future__ import annotations

from segmentum.m225_benchmarks import run_m225_open_world_transfer


def test_adapter_ablation_changes_runtime_path_and_recovery_outcome() -> None:
    payload = run_m225_open_world_transfer()
    report = payload["acceptance_report"]
    artifact = payload["artifacts"]["adapter_degradation"]
    assert report["adapter_breakdown"]["adapter_failure_recovery_rate"] >= 0.75
    assert report["adapter_breakdown"]["error_attribution_accuracy"] >= 0.80
    assert report["adapter_breakdown"]["catastrophic_collapse_ratio"] <= 0.10

    full_row = next(
        row
        for row in payload["rows"]
        if row["variant"] == "full_system" and row["protocol"] == "adapter_degradation_protocol"
    )
    ablated_row = next(
        row
        for row in payload["rows"]
        if row["variant"] == "adapter_degraded" and row["protocol"] == "adapter_degradation_protocol"
    )
    assert full_row["variant_configuration"]["fallback_depth"] > ablated_row["variant_configuration"]["fallback_depth"]
    assert full_row["variant_configuration"]["adapter_resilience"] > ablated_row["variant_configuration"]["adapter_resilience"]
    assert full_row["action_trace"][0]["action"] in {"hide", "scan", "rest", "exploit_shelter", "seek_contact", "forage", "thermoregulate"}
    assert ablated_row["action_trace"][0]["action"] in {"hide", "scan", "rest", "exploit_shelter", "seek_contact", "forage", "thermoregulate"}
    assert full_row["action_trace"][0]["decision_explanation"]
    assert full_row["event_trace"][0]["observation"]
    assert full_row["raw_metrics"]["adapter_recoveries"] > ablated_row["raw_metrics"]["adapter_recoveries"]

    anomalies = artifact["adapter_anomaly_log"]
    assert any(event["expected_domain"] == "adapter_error" for event in anomalies)
    assert any(event["predicted_domain"] == "adapter_error" for event in anomalies)
    assert all(bool(event["trace_retained"]) for event in anomalies)
    assert all(bool(event["attribution_retained"]) for event in anomalies)
