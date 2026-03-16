from __future__ import annotations

from segmentum.m211_readiness import (
    build_attention_summary,
    build_personality_summary,
    build_pre_m3_readiness_report,
    build_snapshot_compatibility_payload,
    build_transfer_summary,
    run_soak_regression,
)


def test_readiness_gates_reflect_actual_metrics() -> None:
    readiness = build_pre_m3_readiness_report(
        attention_summary=build_attention_summary(),
        transfer_summary=build_transfer_summary(),
        personality_summary=build_personality_summary(),
        soak_regression=run_soak_regression(),
        snapshot_compatibility=build_snapshot_compatibility_payload(),
    ).payload
    gates = readiness["acceptance"]["gate_results"]
    assert gates["attention_main_loop_established"] is True
    assert gates["transfer_benchmark_established"] is True
    assert gates["personality_narrative_evidence_established"] is True


def test_no_stale_artifact_can_falsely_mark_ready() -> None:
    stale_attention = build_attention_summary()
    stale_attention["acceptance"]["passed"] = False
    readiness = build_pre_m3_readiness_report(
        attention_summary=stale_attention,
        transfer_summary=build_transfer_summary(),
        personality_summary=build_personality_summary(),
        soak_regression=run_soak_regression(),
        snapshot_compatibility=build_snapshot_compatibility_payload(),
    ).payload
    assert readiness["final_recommendation"]["passed"] is False
    assert readiness["acceptance"]["gate_results"]["attention_main_loop_established"] is False
