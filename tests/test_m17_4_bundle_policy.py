from __future__ import annotations

from segmentum.dialogue.runtime.m13_memory_efe import evaluate_memory_efe
from segmentum.dialogue.runtime.m17_bundle_policy import (
    assemble_memory_evidence_bundles,
    bundle_decision_event,
    evaluate_bundle_decision,
)


NOW = 1_700_000_000


def _row(
    memory_id: str,
    *,
    support: float,
    expectation_id: str = "exp_1",
    prediction_id: str = "pred:p1",
    evidence_ref: str,
    contradiction_risk: float = 0.0,
) -> dict[str, object]:
    return {
        "id": memory_id,
        "memory_id": memory_id,
        "item_support": support,
        "expectation_ids": [expectation_id],
        "prediction_ids": [prediction_id],
        "episode_ids": [f"ep:{memory_id}"],
        "evidence_refs": [evidence_ref],
        "contradiction_risk": contradiction_risk,
    }


def test_bundle_requires_multiple_unique_memories() -> None:
    bundles, diagnostics = assemble_memory_evidence_bundles(
        [
            _row("m1", support=0.53, evidence_ref="e1"),
            _row("m2", support=0.49, evidence_ref="e2"),
        ],
        allowed_expectation_ids=["exp_1"],
    )

    assert diagnostics.to_dict() == {
        "retrieval_eligible_count": 2,
        "bundle_linkable_count": 2,
        "unlinked_count": 0,
    }
    assert bundles
    assert bundles[0].bundle_required is True
    assert bundles[0].max_single_support < 0.60
    assert bundles[0].unique_memory_count == 2
    assert bundles[0].unique_evidence_ref_count == 2


def test_bundle_rejects_duplicate_evidence_inflation() -> None:
    bundles, _ = assemble_memory_evidence_bundles(
        [
            _row("m1", support=0.53, evidence_ref="e1"),
            _row("m2", support=0.49, evidence_ref="e1"),
        ],
        allowed_expectation_ids=["exp_1"],
    )

    assert bundles
    assert bundles[0].bundle_required is False
    assert bundles[0].redundancy_penalty >= 0.10


def test_bundle_rejects_when_best_single_would_trigger() -> None:
    bundles, _ = assemble_memory_evidence_bundles(
        [
            _row("m1", support=0.61, evidence_ref="e1"),
            _row("m2", support=0.25, evidence_ref="e2"),
        ],
        allowed_expectation_ids=["exp_1"],
    )

    decision = evaluate_bundle_decision(bundles[0], consumer_kind="reply_policy_bias")
    assert decision.commit is False
    assert decision.best_single_counterfactual_would_trigger is True
    assert "best_single_would_trigger" in decision.violation_codes


def test_bundle_contradiction_penalty_suppresses_conflicting_members() -> None:
    bundles, _ = assemble_memory_evidence_bundles(
        [
            _row("m1", support=0.55, evidence_ref="e1"),
            _row("m2", support=0.52, evidence_ref="e2", contradiction_risk=1.0),
        ],
        allowed_expectation_ids=["exp_1"],
    )

    assert bundles
    assert bundles[0].bundle_required is False
    assert bundles[0].contradiction_penalty > 0.0


def test_bundle_decision_emits_counterfactual_audit_fields() -> None:
    bundles, _ = assemble_memory_evidence_bundles(
        [
            _row("m1", support=0.53, evidence_ref="e1"),
            _row("m2", support=0.49, evidence_ref="e2"),
        ],
        allowed_expectation_ids=["exp_1"],
    )

    decision = evaluate_bundle_decision(bundles[0], consumer_kind="reply_policy_bias")
    event = bundle_decision_event(bundle=bundles[0], decision=decision, turn_index=3, now=NOW)

    assert event["type"] == "BundleDecisionEvent"
    assert event["best_single_counterfactual_would_trigger"] is False
    assert event["aggregated_support"] >= 0.74
    assert event["max_single_support"] < 0.60


def test_reply_policy_bias_can_be_bundle_required() -> None:
    state = {
        "temporal_state": {"last_user_turn_at": NOW},
        "pending_expectations": [
            {
                "id": "exp_1",
                "status": "pending",
                "content": "clarify the benchmark expectation",
                "due_at_epoch": NOW - 30,
                "bound_memory_ids": ["m1", "m2"],
                "evidence_refs": ["anchor_e1"],
                "confidence": 0.9,
            }
        ],
        "short_term_memory": [
            {
                "id": "m1",
                "content": "clarify the benchmark expectation",
                "evidence_refs": ["anchor_e1"],
                "salience": 1.0,
                "precision": 1.0,
                "value_proxy": 1.0,
            },
            {
                "id": "m2",
                "content": "clarify the benchmark expectation",
                "evidence_refs": ["anchor_e2"],
                "salience": 1.0,
                "precision": 1.0,
                "value_proxy": 1.0,
            },
        ],
    }
    retrieved = [
        _row("m1", support=0.53, evidence_ref="anchor_e1"),
        _row("m2", support=0.49, evidence_ref="anchor_e2"),
    ]

    result = evaluate_memory_efe(
        state,
        phase="in_turn",
        now=NOW,
        turn_index=2,
        user_active=True,
        retrieved_memories=retrieved,
        memory_dynamics={"control_guidance": {}},
        conscious_plan={},
    )

    assert result.reply_angle_bias == "repair_expectation"
    assert result.bundle_linkage_diagnostics["bundle_linkable_count"] == 2
    assert any(event["type"] == "BundleDecisionEvent" for event in result.events)
