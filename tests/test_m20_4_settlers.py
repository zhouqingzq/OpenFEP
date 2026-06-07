"""Tests for M20.4 v1 §3 LLM-judge settlers.

Both `AddresseeTargetMatchLLMJudgeSettler` and
`ReactionAttributionMatchLLMJudgeSettler` follow the
`BoundaryHandledLLMJudgeSettler` pattern (M20.1):
- frozen system prompt + bounded user prompt
- injected LLM call (None in v1 → SettlerUnavailable)
- fail closed on invalid response
- magnitude: 1.0 (confirmed/violated) or 0.5 (ambiguous)
"""

from __future__ import annotations

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    SettledValue,
    SettlerUnavailable,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    AddresseeTargetMatchLLMJudgeSettler,
    M20_4_ENGINEERING_PROXY_LABEL,
    ReactionAttributionMatchLLMJudgeSettler,
)


def _commitment(*, observable: str) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id="cid_x",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_test",
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload={
            "hypothesis": {"confidence": 0.9},
            "hypothesis_commit_id": "abcd" * 10,
            "current_turn_id": "0",
            "inbound_bounded_excerpt": "hi",
            "ambiguity_band": "high",
        },
        target={},
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


# === AddresseeTargetMatchLLMJudgeSettler ================================


def test_addressee_settler_emits_settled_value_for_confirmed() -> None:
    def stub(system, user):
        return {
            "outcome": "confirmed",
            "rationale_span": "ok",
            "reason": "consistent with inbound",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {
            "now": "2026-06-06T00:00:00Z",
            "turn_index": 0,
        },
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    assert result.magnitude == 1.0
    assert result.settler_type == "llm_judge"
    assert result.engineering_proxy_label == M20_4_ENGINEERING_PROXY_LABEL


def test_addressee_settler_emits_settled_value_for_violated() -> None:
    def stub(system, user):
        return {
            "outcome": "violated",
            "rationale_span": "drift",
            "reason": "hypothesis says directed, inbound is to alice",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "violated"
    assert result.magnitude == 1.0


def test_addressee_settler_emits_settled_value_for_ambiguous() -> None:
    def stub(system, user):
        return {
            "outcome": "ambiguous",
            "rationale_span": "unsure",
            "reason": "cannot tell",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "ambiguous"
    assert result.magnitude == 0.5


def test_addressee_settler_fails_closed_on_llm_unavailable() -> None:
    """SettlerUnavailable when no LLM call is injected."""
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=None)
    try:
        settler.settle(
            _commitment(observable="addressee_target_match"),
            {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
        )
    except SettlerUnavailable as exc:
        assert "addressee_target_match" in str(exc)
    else:
        raise AssertionError("SettlerUnavailable not raised")


def test_addressee_settler_fails_closed_on_invalid_response() -> None:
    def stub(system, user):
        return {"outcome": "not_in_set"}  # invalid

    from segmentum.dialogue.runtime.active_commitment import NoSettlement

    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_llm_invalid_response"


def test_addressee_settler_fails_closed_on_non_mapping_response() -> None:
    from segmentum.dialogue.runtime.active_commitment import NoSettlement

    def stub(system, user):
        return "not a mapping"

    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_llm_invalid_response"


def test_addressee_settler_does_not_invent_evidence_refs() -> None:
    """An LLM response without `evidence_refs` → NoSettlement."""
    from segmentum.dialogue.runtime.active_commitment import NoSettlement

    def stub(system, user):
        return {"outcome": "confirmed", "evidence_refs": []}

    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "no_eligible_observation"


def test_addressee_settler_compares_inbound_turn_not_assistant_reply() -> None:
    """M20.4 §3 C2 fix: the bounded_excerpt is the INBOUND turn,
    not the assistant's reply. The user prompt must carry
    `inbound_bounded_excerpt` and NOT carry the assistant's
    reply.
    """
    captured = {}

    def stub(system, user):
        captured["system"] = system
        captured["user"] = user
        return {
            "outcome": "confirmed",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=stub)
    settler.settle(
        _commitment(observable="addressee_target_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert "inbound_bounded_excerpt" in captured["user"]
    # No assistant_reply field
    assert "assistant_reply" not in captured["user"]


# === ReactionAttributionMatchLLMJudgeSettler =========================


def test_reaction_settler_emits_settled_value_for_confirmed() -> None:
    def stub(system, user):
        return {
            "outcome": "confirmed",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="reaction_attribution_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"


def test_reaction_settler_emits_settled_value_for_violated() -> None:
    def stub(system, user):
        return {
            "outcome": "violated",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="reaction_attribution_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "violated"


def test_reaction_settler_emits_settled_value_for_ambiguous() -> None:
    def stub(system, user):
        return {
            "outcome": "ambiguous",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="reaction_attribution_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "ambiguous"
    assert result.magnitude == 0.5


def test_reaction_settler_fails_closed_on_llm_unavailable() -> None:
    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=None)
    try:
        settler.settle(
            _commitment(observable="reaction_attribution_match"),
            {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
        )
    except SettlerUnavailable as exc:
        assert "reaction_attribution_match" in str(exc)
    else:
        raise AssertionError("SettlerUnavailable not raised")


def test_reaction_settler_fails_closed_on_invalid_response() -> None:
    from segmentum.dialogue.runtime.active_commitment import NoSettlement

    def stub(system, user):
        return {"outcome": "garbage"}

    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        _commitment(observable="reaction_attribution_match"),
        {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_llm_invalid_response"


def test_reaction_settler_handles_missing_attributed_turn() -> None:
    """When `attributed_bounded_excerpt` is empty (older turn
    evicted from bus), the settler still works. The LLM
    judges based on the structural signals.
    """
    commitment = ActiveCommitment(
        commit_id="cid_r",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_r",
        layer="B_per_turn_commitment",
        observable="reaction_attribution_match",
        observable_payload={
            "hypothesis": {"confidence": 0.7},
            "hypothesis_commit_id": "abcd" * 10,
            "current_turn_id": "0",
            "attributed_turn_id": "turn_999",
            "attributed_bounded_excerpt": "",  # empty
            "ambiguity_band": "high",
        },
        target={},
        due_at={"kind": "next_turn"},
        priority=0.7,
        confidence=0.7,
        evidence_refs=("turn_0_user_utterance",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )

    def stub(system, user):
        return {
            "outcome": "ambiguous",
            "evidence_refs": ["turn_0_user_utterance"],
        }
    settler = ReactionAttributionMatchLLMJudgeSettler(llm_call=stub)
    result = settler.settle(
        commitment, {"now": "2026-06-06T00:00:00Z", "turn_index": 0}
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "ambiguous"


def test_settler_does_not_match_user_text() -> None:
    """The settler does not parse user text; it only calls the
    LLM with the bounded excerpts. Engineering layer
    validates shapes, not content.
    """
    import segmentum.dialogue.runtime.m20_4_attribution as m

    src = open(m.__file__, encoding="utf-8").read()
    for forbidden in ["re.search", "re.match", "re.compile", "re.findall"]:
        assert forbidden not in src


def test_settler_does_not_call_llm_when_not_injected() -> None:
    """SettlerUnavailable is raised (which the M20.1 scheduler
    catches and converts to NoSettlement) when no LLM is
    available.
    """
    settler = AddresseeTargetMatchLLMJudgeSettler(llm_call=None)
    try:
        settler.settle(
            _commitment(observable="addressee_target_match"),
            {"now": "2026-06-06T00:00:00Z", "turn_index": 0},
        )
    except SettlerUnavailable:
        pass
    else:
        raise AssertionError("expected SettlerUnavailable")
