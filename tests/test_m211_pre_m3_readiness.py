from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from segmentum.m211_readiness import (
    build_pre_m3_readiness_report,
)


@lru_cache(maxsize=1)
def _baseline_inputs() -> dict[str, object]:
    return {
        "attention_summary": {
            "acceptance": {"passed": True},
            "generated_at": "2026-03-20T00:00:00+00:00",
        },
        "transfer_summary": {
            "acceptance": {
                "passed": True,
                "verified_world_count": 3,
            },
            "generated_at": "2026-03-20T00:00:01+00:00",
            "seed_set": [42, 43, 44, 142, 143],
            "summary": {
                "protocol": {
                    "seed_set": [42, 43, 44, 142, 143],
                    "seed_protocol_origin": "tests:test_m211_pre_m3_readiness",
                }
            },
            "freshness": {
                "current_round_replay": True,
                "execution_origin": "tests:test_m211_pre_m3_readiness",
                "seed_protocol_origin": "tests:test_m211_pre_m3_readiness",
            },
        },
        "personality_summary": {
            "acceptance": {"passed": True},
            "generated_at": "2026-03-20T00:00:02+00:00",
        },
        "soak_regression": {
            "passed": True,
            "seed": 17,
            "cycles": 256,
        },
        "snapshot_compatibility": {
            "passed": True,
        },
    }


def _fresh_baseline_inputs() -> dict[str, object]:
    return deepcopy(_baseline_inputs())


def test_readiness_gates_reflect_actual_metrics() -> None:
    readiness = build_pre_m3_readiness_report(**_fresh_baseline_inputs()).payload
    gates = readiness["acceptance"]["gate_results"]
    assert gates["attention_main_loop_established"] is True
    assert gates["transfer_benchmark_established"] is True
    assert gates["personality_narrative_evidence_established"] is True
    assert readiness["milestone_id"] == "Pre-M3"
    assert "recommendation" in readiness
    assert "codebase_provenance" in readiness
    assert "workspace_fingerprint" in readiness["codebase_provenance"]


def test_no_stale_artifact_can_falsely_mark_ready() -> None:
    baseline = _fresh_baseline_inputs()
    stale_attention = baseline["attention_summary"]
    stale_attention["acceptance"]["passed"] = False
    readiness = build_pre_m3_readiness_report(
        attention_summary=stale_attention,
        transfer_summary=baseline["transfer_summary"],
        personality_summary=baseline["personality_summary"],
        soak_regression=baseline["soak_regression"],
        snapshot_compatibility=baseline["snapshot_compatibility"],
    ).payload
    assert readiness["recommendation"]["passed"] is False
    assert readiness["acceptance"]["gate_results"]["attention_main_loop_established"] is False


def test_pre_m3_report_blocks_when_family_coverage_is_not_runtime_replay() -> None:
    baseline = _fresh_baseline_inputs()
    readiness = build_pre_m3_readiness_report(
        attention_summary=baseline["attention_summary"],
        transfer_summary=baseline["transfer_summary"],
        personality_summary=baseline["personality_summary"],
        soak_regression=baseline["soak_regression"],
        snapshot_compatibility=baseline["snapshot_compatibility"],
        runtime_family_coverage={
            "verification_status": "FRAMEWORK_SCHEMA_PROBE_ONLY",
            "evidence_origin": "evals/m3_readiness_evaluation.py:framework_family_schema_probe",
            "revalidated_this_round": False,
            "evidence_kind": "framework_schema_probe",
            "family_schema_count": 4,
            "runtime_validated_family_count": 0,
            "fully_graduated": False,
            "missing_graduation_families": [
                "danger_avoidance",
                "resource_risk",
                "retreat_vs_explore",
                "integrity_preservation",
            ],
            "family_coverage_status": "FRAMEWORK_ONLY_NOT_RUNTIME_VALIDATED",
        },
    ).payload
    assert readiness["status"] == "BLOCKED"
    assert readiness["recommendation"]["status"] == "NOT_READY_FOR_M3"
    assert any(
        finding["title"] == "Family coverage is only a framework/schema probe"
        for finding in readiness["findings"]
    )


def test_pre_m3_report_blocks_when_resource_risk_has_no_real_graduation() -> None:
    baseline = _fresh_baseline_inputs()
    transfer = baseline["transfer_summary"]
    readiness = build_pre_m3_readiness_report(
        attention_summary=baseline["attention_summary"],
        transfer_summary=transfer,
        personality_summary=baseline["personality_summary"],
        soak_regression=baseline["soak_regression"],
        snapshot_compatibility=baseline["snapshot_compatibility"],
        runtime_family_coverage={
            "verification_status": "LIMITED_RUNTIME_REPLAY",
            "evidence_origin": "evals/m3_readiness_evaluation.py:runtime_counterfactual_replay",
            "revalidated_this_round": True,
            "evidence_kind": "runtime_replay",
            "family_schema_count": 4,
            "runtime_validated_family_count": 3,
            "fully_graduated": False,
            "missing_graduation_families": ["resource_risk"],
            "family_coverage_status": "PARTIAL_RUNTIME_VALIDATED",
            "limitations": [
                "Runtime replay is current-round evidence, but the following families still lack real graduation: resource_risk."
            ],
        },
    ).payload
    assert readiness["status"] == "BLOCKED"
    assert readiness["gates"]["runtime_family_coverage_revalidated"] is False
    assert readiness["recommendation"]["status"] == "NOT_READY_FOR_M3"
    assert any("resource_risk" in risk for risk in readiness["residual_risks"])
    assert any(
        finding["title"] == "Runtime family replay still lacks full graduation"
        for finding in readiness["findings"]
    )


def test_transfer_seed_protocol_is_machine_readable_and_propagated() -> None:
    baseline = _fresh_baseline_inputs()
    transfer = baseline["transfer_summary"]
    readiness = build_pre_m3_readiness_report(
        attention_summary=baseline["attention_summary"],
        transfer_summary=transfer,
        personality_summary=baseline["personality_summary"],
        soak_regression=baseline["soak_regression"],
        snapshot_compatibility=baseline["snapshot_compatibility"],
    ).payload
    assert transfer["seed_set"]
    assert transfer["summary"]["protocol"]["seed_set"] == transfer["seed_set"]
    transfer_test = next(
        item for item in readiness["tests"]["milestone_specific_benchmarks"] if item["name"] == "transfer"
    )
    transfer_artifact = next(
        item for item in readiness["artifacts"] if item["path"] == "artifacts/pre_m3_transfer_summary.json"
    )
    assert transfer_test["seed_set"] == transfer["seed_set"]
    assert transfer_test["current_round_replay"] is True
    assert transfer_artifact["seed_set"] == transfer["seed_set"]
    assert readiness["freshness"]["transfer_summary"]["seed_set"] == transfer["seed_set"]
    assert readiness["freshness"]["transfer_summary"]["current_round_replay"] is True
    assert not any("canonical seed_set" in risk for risk in readiness["residual_risks"])
