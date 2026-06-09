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

P0-3 (2026-06-08): the confidence threshold is now per-field.
`addressee` engages at `> 0.9`; `reaction` engages at `> 0.7`;
unknown kinds fall back to the v1 `> 0.85` default. The
per-field values come from M18.7.1 v3 real-LLM calibration
(`reports/m18_7_2_implementation_summary.md`).
"""

from __future__ import annotations

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
    SettledValue,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    M20_4_TIE_BREAKER_CONFIDENCE_MIN,
    M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED,
    M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND,
    _kind_from_observable,
    _tie_breaker_min_for,
    build_m18_5_attribution_feedback_row,
    emit_m20_4_tie_breaker_feedback,
    record_m18_5_attribution_feedback,
    write_addressee_graph_microadjust,
)


def _commitment(
    *,
    observable: str = "addressee_target_match",
    addressed_to_assistant: bool = True,
    confidence: float | None = None,
) -> ActiveCommitment:
    # P0-3 (2026-06-08): per-field thresholds. addressee
    # engages at conf > 0.9 (fixture uses 0.91); reaction
    # engages at conf > 0.7 (fixture uses 0.71). The v1 0.9
    # and 0.7 confidences from the prior fixture no longer
    # engage under strict inequality.
    #
    # P0-6 (2026-06-09): the `addressee` kind is further
    # split by sub-class. The default fixture uses
    # `addressed_to_assistant=True` with conf=0.96
    # (just above the new 0.95 P0-6 bar) so the
    # v1-pinned "engages" tests continue to engage.
    # The 0.95 boundary is exclusively pinned by the
    # new P0-6 tests; `test_tie_breaker_per_field_addressee_threshold_engages_at_0_91`
    # has been renamed to
    # `test_p0_6_tie_breaker_addressee_directed_engages_at_0_96`
    # and now pins the new bar.
    if confidence is None:
        if observable == "addressee_target_match":
            confidence = 0.96
        elif observable == "reaction_attribution_match":
            confidence = 0.71
        else:
            confidence = 0.86  # unknown observable → v1 0.85 default engages
    return ActiveCommitment(
        commit_id="cid_tb",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_tb",
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload={
            "hypothesis": {
                "addressed_to_assistant": addressed_to_assistant,
                "confidence": confidence,
            } if observable == "addressee_target_match" else {
                "is_about_assistant_claim": True,
                "reaction_to_turn_id": "turn_0",
                "confidence": confidence,
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
    """P0-3 (2026-06-08): the v1 single threshold (0.85) is
    preserved as the `M20_4_TIE_BREAKER_CONFIDENCE_MIN` alias
    (= `_M20_4_TIE_BREAKER_DEFAULT`) for backward compat. The
    new per-field dispatch reads
    `M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND` directly.
    """
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN == 0.85
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND == {
        "addressee": 0.9,
        "reaction": 0.7,
    }
    # All three values are strict inequalities; check the
    # exact v3 calibration candidates. Drift from these
    # values is intentional and requires a calibration
    # re-run (M18.7.1) plus a documented decision (M20.4).
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND["addressee"] > 0.5
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND["reaction"] > 0.5


# === P0-3 per-field dispatch ==========================================


def test_tie_breaker_kind_from_observable_dispatch() -> None:
    """`_kind_from_observable` is the per-field dispatch
    entry point. addressee_target_match -> "addressee",
    reaction_attribution_match -> "reaction", unknown -> "".
    """
    assert _kind_from_observable("addressee_target_match") == "addressee"
    assert _kind_from_observable("reaction_attribution_match") == "reaction"
    assert _kind_from_observable("") == ""
    assert _kind_from_observable("unknown_observable") == ""
    # Whitespace + case insensitive
    assert _kind_from_observable("  Addressee_Target_Match  ") == "addressee"
    # Non-string fallback
    assert _kind_from_observable(None) == ""
    assert _kind_from_observable(123) == ""


def test_tie_breaker_min_for_dispatch() -> None:
    """`_tie_breaker_min_for(kind)` returns the per-field
    threshold; unknown kinds fall back to the v1 0.85.
    """
    assert _tie_breaker_min_for("addressee") == 0.9
    assert _tie_breaker_min_for("reaction") == 0.7
    assert _tie_breaker_min_for("unknown_kind") == 0.85
    assert _tie_breaker_min_for("") == 0.85
    assert _tie_breaker_min_for(None) == 0.85
    # Whitespace + case insensitive
    assert _tie_breaker_min_for("  Addressee  ") == 0.9


def test_tie_breaker_per_field_addressee_threshold_engages_at_0_91() -> None:
    """P0-3 (now superseded by P0-6): addressee tie-breaker
    engages at conf > 0.9. P0-6 (2026-06-09) further splits
    the `addressee` kind by sub-class: at conf=0.91 the
    `addressed_to_assistant=True` sub-class REJECTS (0.91 <
    0.95 P0-6 bar); the `addressed_to_assistant=False`
    sub-class still engages (0.91 > 0.9 v1 default). This
    test now pins the "not addressed" sub-class engagement
    at conf=0.91. The "addressed" sub-class engagement at
    conf=0.91 is pinned by
    `test_p0_6_tie_breaker_addressee_directed_rejects_at_0_91`.
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=False,  # "not addressed" sub-class
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.91
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


def test_tie_breaker_per_field_addressee_threshold_rejects_at_0_89() -> None:
    """P0-3: addressee tie-breaker rejects at conf=0.89,
    which would have engaged under the v1 0.85 threshold.
    This is the strict-inequality boundary: 0.89 < 0.9.
    """
    state: dict = {}
    commitment = _commitment(observable="addressee_target_match")
    commitment.observable_payload["hypothesis"]["confidence"] = 0.89
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


def test_tie_breaker_per_field_reaction_threshold_engages_at_0_71() -> None:
    """P0-3: reaction tie-breaker engages at conf > 0.7
    (v3 surfaced 0.7 from reaction calibration)."""
    state: dict = {}
    commitment = _commitment(observable="reaction_attribution_match")
    commitment.observable_payload["hypothesis"]["confidence"] = 0.71
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


def test_tie_breaker_per_field_reaction_threshold_engages_at_0_85() -> None:
    """P0-3 clarification: M18.7.1 surfaced a calibration
    candidate of 0.7 for reaction. The candidate is the
    threshold that minimizes the bin-level gap (ECE/Brier),
    NOT a behavioral tightening. Lowering the threshold from
    0.85 to 0.7 makes engagement MORE permissive (engages
    when conf > 0.7, including 0.85).

    At conf=0.85 with the v3 candidate threshold of 0.7,
    the tie-breaker STILL engages because 0.85 > 0.7. The
    v3 data has 1 reaction prediction at conf=0.85 with
    accuracy 0.0 (overconfidence_at_high_band drift). The
    candidate does NOT fix that specific case; it just
    matches the calibration data's Brier-minimizing boundary.
    A future M20.4 follow-up could raise the reaction
    threshold to 0.9+ to specifically reject the 0.85 case;
    that would be a separate decision, not P0-3.
    """
    state: dict = {}
    commitment = _commitment(observable="reaction_attribution_match")
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
    assert row["tie_breaker_engaged"] is True


def test_tie_breaker_per_field_reaction_threshold_rejects_at_0_69() -> None:
    """P0-3 strict-inequality boundary for reaction: 0.69
    rejects (0.69 < 0.7), 0.71 engages (0.71 > 0.7)."""
    state: dict = {}
    commitment = _commitment(observable="reaction_attribution_match")
    commitment.observable_payload["hypothesis"]["confidence"] = 0.69
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


def test_tie_breaker_unknown_observable_falls_back_to_v1_default() -> None:
    """A future observable (M20.4.x) without a per-field
    threshold falls back to the v1 0.85 default. The
    fallback is intentionally conservative: no regression
    from v1 for any new field."""
    state: dict = {}
    commitment = _commitment(observable="future_field_match")
    commitment.observable_payload["hypothesis"]["confidence"] = 0.86
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


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


# === P0-6: tie-breaker sub-class split =================================
#
# P0-6 (2026-06-09) further splits the `addressee` kind by
# sub-class. The `addressed_to_assistant=True` sub-class
# engages at conf > 0.95 (P1: `recall_on_addressed = 0.0`).
# The `addressed_to_assistant=False` sub-class keeps the
# v1 0.9 default (P1: `precision_on_not_addressed = 1.0`).
# Other kinds and the strict-inequality style are unchanged.


def test_p0_6_tie_breaker_addressee_directed_constant_is_frozen() -> None:
    """The 0.95 / 0.9 split is the v1 → P0-6 contract."""
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED == 0.95
    # P0-3 per-field values preserved.
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND["addressee"] == 0.9
    assert M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND["reaction"] == 0.7


def test_p0_6_tie_breaker_min_for_addressee_subclass_dispatch() -> None:
    """P0-6 dispatch: `addressee` kind with the
    `addressed_to_assistant` flag returns the per-sub-class
    threshold. Other kinds and the unknown-kinds fallback
    are unchanged.
    """
    # Addressee + addressed=True → 0.95 (P0-6 raise).
    assert (
        _tie_breaker_min_for(
            "addressee", addressed_to_assistant=True
        )
        == 0.95
    )
    # Addressee + addressed=False → 0.9 (v1 default).
    assert (
        _tie_breaker_min_for(
            "addressee", addressed_to_assistant=False
        )
        == 0.9
    )
    # Addressee + addressed=None → 0.9 (v1 default,
    # preserves back-compat for callers that don't
    # supply the sub-class flag).
    assert (
        _tie_breaker_min_for("addressee", addressed_to_assistant=None)
        == 0.9
    )
    # Reaction is unaffected by the sub-class flag.
    assert _tie_breaker_min_for("reaction") == 0.7
    assert (
        _tie_breaker_min_for(
            "reaction", addressed_to_assistant=True
        )
        == 0.7
    )
    # Unknown kind → v1 0.85 default, regardless of flag.
    assert (
        _tie_breaker_min_for(
            "unknown_kind", addressed_to_assistant=True
        )
        == 0.85
    )


def test_p0_6_tie_breaker_addressee_directed_engages_at_0_96() -> None:
    """P0-6: `addressed_to_assistant=True` engages at
    conf=0.96 (just above the new 0.95 P0-6 bar).
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.96
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


def test_p0_6_tie_breaker_addressee_directed_rejects_at_0_91() -> None:
    """P0-6: `addressed_to_assistant=True` rejects at
    conf=0.91 (would have engaged under v1 0.9 / P0-3
    0.9 default). The new 0.95 bar prevents bad flips
    on false-positive "addressed" admits. The
    "not addressed" sub-class still engages at 0.91
    (pinned by
    `test_tie_breaker_per_field_addressee_threshold_engages_at_0_91`).
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.91
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


def test_p0_6_tie_breaker_addressee_directed_rejects_at_0_95_boundary() -> None:
    """P0-6: `addressed_to_assistant=True` at conf=0.95
    REJECTS (strict `>`). The 0.95 boundary is the
    P0-6 test surface.
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.95
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


def test_p0_6_tie_breaker_addressee_not_directed_keeps_v1_threshold() -> None:
    """P0-6: `addressed_to_assistant=False` keeps the
    v1 0.9 default. At conf=0.91 (just above 0.9), the
    tie-breaker engages. This pins the "not addressed"
    sub-class as the precision-1.0 path that doesn't
    need a tighter bar.
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=False,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.91
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


def test_p0_6_tie_breaker_reaction_observable_unaffected_by_subclass_flag() -> None:
    """P0-6: the `reaction` kind ignores the
    `addressed_to_assistant` flag (the flag is only
    meaningful for the `addressee` kind). At conf=0.71,
    the reaction tie-breaker engages.
    """
    state: dict = {}
    commitment = _commitment(observable="reaction_attribution_match")
    commitment.observable_payload["hypothesis"]["confidence"] = 0.71
    row = emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    assert row is not None
    assert row["tie_breaker_engaged"] is True


def test_p0_6_tie_breaker_diagnostics_addressee_directed_engaged_counter() -> None:
    """P0-6: when the `addressed_to_assistant=True`
    sub-class engages, the new
    `tie_breaker_engaged_addressee_directed_total`
    counter is bumped. The aggregate
    `tie_breaker_engaged_total` is preserved.
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.96
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("tie_breaker_engaged_addressee_directed_total") == 1
    )
    # v1 aggregate is preserved.
    assert diag.get("tie_breaker_engaged_total") == 1


def test_p0_6_tie_breaker_diagnostics_addressee_not_directed_engaged_counter() -> None:
    """P0-6: when the `addressed_to_assistant=False`
    sub-class engages, the new
    `tie_breaker_engaged_addressee_not_directed_total`
    counter is bumped. The aggregate
    `tie_breaker_engaged_total` is preserved.
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=False,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.91
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("tie_breaker_engaged_addressee_not_directed_total")
        == 1
    )
    # v1 aggregate is preserved.
    assert diag.get("tie_breaker_engaged_total") == 1


def test_p0_6_tie_breaker_diagnostics_addressee_directed_low_confidence_reject_counter() -> None:
    """P0-6: when the `addressed_to_assistant=True`
    sub-class rejects on confidence, the new
    `tie_breaker_rejected_confidence_low_addressee_directed_total`
    counter is bumped (in addition to the v1
    `tie_breaker_rejected_total` /
    `tie_breaker_rejected_by_reason`).
    """
    state: dict = {}
    commitment = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    commitment.observable_payload["hypothesis"]["confidence"] = 0.91
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=commitment,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get(
            "tie_breaker_rejected_confidence_low_addressee_directed_total"
        )
        == 1
    )
    # v1 aggregate is preserved.
    assert diag.get("tie_breaker_rejected_total") == 1
    assert (
        diag.get("tie_breaker_rejected_by_reason", {}).get(
            "confidence_below_threshold"
        )
        == 1
    )


def test_p0_6_tie_breaker_mixed_batch_subclass_split() -> None:
    """P0-6: mixed batch — 1 'addressed' @ 0.96 (engages),
    1 'addressed' @ 0.91 (rejects on confidence), 1
    'not addressed' @ 0.91 (engages). The v1 aggregate
    `tie_breaker_engaged_total` is 2; the per-sub-class
    counters are 1 + 1; the per-sub-class reject counter
    is 1.
    """
    state: dict = {}
    # 1. addressed @ 0.96 — engage.
    c1 = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    c1.observable_payload["hypothesis"]["confidence"] = 0.96
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=c1,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    # 2. addressed @ 0.91 — reject on confidence.
    c2 = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=True,
    )
    c2.observable_payload["hypothesis"]["confidence"] = 0.91
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=c2,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    # 3. not addressed @ 0.91 — engage.
    c3 = _commitment(
        observable="addressee_target_match",
        addressed_to_assistant=False,
    )
    c3.observable_payload["hypothesis"]["confidence"] = 0.91
    emit_m20_4_tie_breaker_feedback(
        state=state,
        decision=_decision(),
        commitment=c3,
        settled_value=_settled_value(),
        m18_5_structural_decision="clarify_addressee",
        at="2026-06-06T00:00:00Z",
    )
    diag = state.get("m20_4_attribution_diagnostics", {})
    # v1 aggregate.
    assert diag.get("tie_breaker_engaged_total") == 2
    assert diag.get("tie_breaker_rejected_total") == 1
    # P0-6 per-sub-class.
    assert (
        diag.get("tie_breaker_engaged_addressee_directed_total") == 1
    )
    assert (
        diag.get("tie_breaker_engaged_addressee_not_directed_total")
        == 1
    )
    assert (
        diag.get(
            "tie_breaker_rejected_confidence_low_addressee_directed_total"
        )
        == 1
    )
    # No false-positive on the "not addressed" reject counter.
    assert (
        diag.get(
            "tie_breaker_rejected_confidence_low_addressee_not_directed_total",
            0,
        )
        == 0
    )
