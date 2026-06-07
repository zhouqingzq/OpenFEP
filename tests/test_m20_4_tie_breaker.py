"""Tests for M20.4 v1 §5 gated M18.5 tie-breaker feedback.

The feedback row is written to
`state["m18_5_attribution_feedback"]` after the M20.2
dispatch. M18.5 reads the row on subsequent turns as a
*supplementary* input, strictly after its structural
decision.

Engagement conditions (C1 fix):
- `decision_level == "microadjust"`
- `outcome == "confirmed"`
- `ambiguity_band == "high"`
- `confiance > 0.85` (strict)
- `addressed_participant_ids` empty
- `mentioned_participant_ids` empty
- `reply_to_turn_id` empty
- `m18_5_structural_decision ∈ {clarify_addressee, no_reply}`
"""

from __future__ import annotations

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
    SettledValue,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    M20_4_TIE_BREAKER_CONFIDENCE_MIN,
    build_m18_5_attribution_feedback_row,
    emit_m20_4_tie_breaker_feedback,
    record_m18_5_attribution_feedback,
    write_addressee_graph_microadjust,
)


def _commitment(*, observable: str = "addressee_target_match") -> ActiveCommitment:
    return ActiveCommitment(
        commit_id="cid_tb",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_tb",
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload={
            "hypothesis": {
                "addressed_to_assistant": True,
                "confidence": 0.9,
            } if observable == "addressee_target_match" else {
                "is_about_assistant_claim": True,
                "reaction_to_turn_id": "turn_0",
                "confidence": 0.7,
            },
            "hypothesis_commit_id": "abcd" * 10,
            "current_turn_id": "0",
            "inbound_bounded_excerpt": "hi",
            "ambiguity_band": "high",
            "group_turn_binding_snapshot": {
                "ambiguity_band": "high",
                "addressed_participant_ids": [],
                "mentioned_participant_ids": [],
                "reply_to_turn_id": "",
            },
        },
        target={"m18_7_commit_id": "abcd" * 10},
        due_at={"kind": "next_turn"},
        priority=0.9,
        confidence=0.9,
        evidence_refs=("turn_0_user_utterance",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )


def _decision(*, level: str = "microadjust") -> GradedCorrectionDecision:
    return GradedCorrectionDecision(
        commit_id="cid_tb",
        correction_level=level,
        routed_owner_id="group_addressee_graph",
        reason_codes=("graded_correction_routed",),
        evidence_refs=(),
        magnitude_before=1.0,
        magnitude_after=1.0,
        outcome="confirmed",
        at="2026-06-06T00:00:00Z",
        turn_index=0,
        engineering_proxy_label="mvp_local_group_attribution",
    )


def _settled_value(*, outcome: str = "confirmed", magnitude: float = 1.0) -> SettledValue:
    return SettledValue(
        commit_id="cid_tb",
        outcome=outcome,
        magnitude=magnitude,
        evidence_refs=("turn_0_user_utterance",),
        reason_codes=("settler_llm_judge",),
        at="2026-06-06T00:00:00Z",
        turn_index=0,
        settler_type="llm_judge",
        engineering_proxy_label="mvp_local_group_attribution",
    )


# === Engagement logic ================================================


def test_tie_breaker_constant_is_frozen() -> None:
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN == 0.85


def test_tie_breaker_engages_when_all_conditions_hold() -> None:
    """microadjust + confirmed + ambiguity=high + confiance>0.85 +
    no structural addressee + m18_5 in {clarify, no_reply} → engaged.
    """
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True
    assert row["patched_decision"] == "reply_to_current_speaker"
    assert row["patched_reason"] == "tie_breaker_engaged"
    assert state["m18_5_attribution_feedback"]


def test_tie_breaker_engages_when_m18_5_decision_is_no_reply() -> None:
    """M20.4 v1 covers BOTH clarify_addressee AND no_reply
    (M20.4 DECIDED 1)."""
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(),
        m18_5_structural_decision="no_reply",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


# === Rejection conditions (C1 fix uses AND not OR) ===================


def test_tie_breaker_does_not_engage_when_ambiguity_band_low() -> None:
    state: dict = {}
    # Patch observable_payload to have low ambiguity_band.
    commitment = _commitment()
    commitment.observable_payload["ambiguity_band"] = "low"
    commitment.observable_payload["group_turn_binding_snapshot"]["ambiguity_band"] = "low"
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_decision"] is None
    assert row["patched_reason"] == "ambiguity_band_not_high"


def test_tie_breaker_does_not_engage_when_confidence_below_threshold() -> None:
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["hypothesis"]["confidence"] = 0.5
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "confidence_below_threshold"


def test_tie_breaker_does_not_engage_when_confidence_at_threshold() -> None:
    """Strict `> 0.85` inequality, not `>= 0.85`."""
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["hypothesis"]["confidence"] = 0.85
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "confidence_below_threshold"


def test_tie_breaker_does_not_engage_when_structural_explicit_addressee() -> None:
    """C1 fix: addressed_participant_ids non-empty → reject
    (C1 was OR; should be AND).
    """
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["group_turn_binding_snapshot"]["addressed_participant_ids"] = ["alice"]
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "structural_explicit_addressee"


def test_tie_breaker_does_not_engage_when_explicit_mention_of_other() -> None:
    """C1 fix: mentioned_participant_ids non-empty → reject."""
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["group_turn_binding_snapshot"]["mentioned_participant_ids"] = ["bob"]
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "explicit_mention_of_other"


def test_tie_breaker_does_not_engage_when_explicit_reply_to_set() -> None:
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["group_turn_binding_snapshot"]["reply_to_turn_id"] = "turn_5"
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "explicit_reply_to_set"


def test_tie_breaker_does_not_engage_when_m18_5_decision_is_reply() -> None:
    """When M18.5 already returned reply_to_current_speaker,
    no flip is needed.
    """
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(),
        m18_5_structural_decision="reply_to_current_speaker",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "m18_5_decision_not_flippable"


def test_tie_breaker_does_not_engage_when_outcome_is_violated() -> None:
    """Only `confirmed` outcome engages the tie-breaker."""
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(outcome="violated"),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "outcome_not_confirmed"


def test_tie_breaker_does_not_engage_when_level_is_revoke() -> None:
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(level="revoke"),
        commitment=_commitment(),
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "level_not_microadjust"


def test_tie_breaker_does_not_engage_when_outcome_is_ambiguous() -> None:
    state: dict = {}
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(outcome="ambiguous"),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is False
    assert row["patched_reason"] == "outcome_not_confirmed"


# === Diagnostic surface ==============================================


def test_attribution_diagnostics_records_tie_breaker_engaged_total() -> None:
    state: dict = {}
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=_commitment(),
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("tie_breaker_engaged_total") == 1


def test_attribution_diagnostics_records_tie_breaker_rejected_total() -> None:
    state: dict = {}
    commitment = _commitment()
    commitment.observable_payload["ambiguity_band"] = "low"
    commitment.observable_payload["group_turn_binding_snapshot"]["ambiguity_band"] = "low"
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state["m20_4_attribution_diagnostics"]
    assert diag.get("tie_breaker_rejected_total") == 1


def test_attribution_diagnostics_records_rejected_by_reason_histogram() -> None:
    state: dict = {}
    # Two distinct rejection reasons
    commitment_low = _commitment()
    commitment_low.observable_payload["ambiguity_band"] = "low"
    commitment_low.observable_payload["group_turn_binding_snapshot"]["ambiguity_band"] = "low"
    emit_m20_4_tie_breaker_feedback(
        state=state, decision=_decision(), commitment=commitment_low,
        settled_value=_settled_value(), m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    commitment_violated = _commitment()
    emit_m20_4_tie_breaker_feedback(
        state=state, decision=_decision(), commitment=commitment_violated,
        settled_value=_settled_value(outcome="violated"),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    bucket = state["m20_4_attribution_diagnostics"]["tie_breaker_rejected_by_reason"]
    assert bucket.get("ambiguity_band_not_high") == 1
    assert bucket.get("outcome_not_confirmed") == 1


# === State surface ==================================================


def test_record_m18_5_attribution_feedback_writes_to_state() -> None:
    state: dict = {}
    row = record_m18_5_attribution_feedback(
        state=state,
        feedback_id="fb_test_1",
        current_turn_id=0,
        m18_5_structural_decision="no_reply",
        hypothesis={"confidence": 0.9},
        ambiguity_band="high",
        engaged=True,
        patched_decision="reply_to_current_speaker",
        patched_reason="tie_breaker_engaged",
        at="2026-06-06T00:00:00Z",
    )
    assert state["m18_5_attribution_feedback"]["fb_test_1"] == row
    assert state["m18_5_attribution_feedback"]["fb_test_1"]["tie_breaker_engaged"] is True


def test_record_m18_5_attribution_feedback_handles_non_dict_state() -> None:
    """Defensive: non-dict state is a no-op."""
    row = record_m18_5_attribution_feedback(
        state=None,  # type: ignore[arg-type]
        feedback_id="fb_x",
        current_turn_id=0,
        m18_5_structural_decision="no_reply",
        hypothesis={},
        ambiguity_band="high",
        engaged=False,
        patched_decision=None,
        patched_reason="x",
        at="2026-06-06T00:00:00Z",
    )
    assert row == {}


def test_build_m18_5_attribution_feedback_row_shape() -> None:
    row = build_m18_5_attribution_feedback_row(
        feedback_id="fb_shape",
        current_turn_id=42,
        m18_5_structural_decision="clarify_addressee",
        hypothesis={"confidence": 0.9},
        ambiguity_band="high",
        engaged=True,
        patched_decision="reply_to_current_speaker",
        patched_reason="tie_breaker_engaged",
        at="2026-06-06T00:00:00Z",
    )
    assert row["feedback_id"] == "fb_shape"
    assert row["current_turn_id"] == 42
    assert row["m18_5_structural_decision"] == "clarify_addressee"
    assert row["hypothesis"] == {"confidence": 0.9}
    assert row["ambiguity_band"] == "high"
    assert row["tie_breaker_engaged"] is True
    assert row["patched_decision"] == "reply_to_current_speaker"
    assert row["patched_reason"] == "tie_breaker_engaged"
    assert row["at"] == "2026-06-06T00:00:00Z"
    assert row["engineering_proxy_label"] == "mvp_local_group_attribution"
