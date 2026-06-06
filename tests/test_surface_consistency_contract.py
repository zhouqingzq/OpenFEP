"""Tests for surface_commitment + LLM-driven surface_consistency_verification.

These tests cover the contract-driven path that replaces the post-hoc
keyword/regex identity checks. The conscious-loop LLM commits to a
`surface_commitment`, then a separate self-audit LLM call returns a bounded
`surface_consistency_verification` enum. Engineering code only validates
those fields; it does not parse raw user text or reply text with regex.
"""

from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    build_conscious_loop_prompt,
    build_surface_consistency_verification_prompt,
    normalize_conscious_turn_plan,
    normalize_surface_commitment,
    normalize_surface_consistency_verification,
    validate_visible_reply,
    _build_surface_consistency_verification_event,
    _empty_surface_consistency_verification,
    _merge_surface_identity_contract_into_memory_guidance,
    _prior_surface_drift_observed,
    _record_surface_consistency_event,
)


class _RecordingJSONLLM:
    """Minimal stub that records prompts and returns canned JSON."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt, json.dumps(self.payload)))
        return dict(self.payload)


def test_normalize_surface_commitment_rejects_unknown_surface_intent() -> None:
    raw = {"surface_intent": "unsupported_intent", "self_identification": "胡桃"}
    commitment = normalize_surface_commitment(raw)
    assert commitment["surface_intent"] == "chat"
    assert commitment["self_identification"] == "胡桃"
    assert commitment["persona_should_apply"] is False
    assert commitment["character_voice_should_apply"] is False


def test_normalize_surface_commitment_clamps_drift_risk_band() -> None:
    commitment = normalize_surface_commitment(
        {
            "surface_intent": "bot_command",
            "self_identification": "ClawdGroupChat Bot",
            "persona_should_apply": True,
            "character_voice_should_apply": False,
            "predicted_drift_risk": "banana",
            "reason": "because",
        }
    )
    assert commitment["surface_intent"] == "bot_command"
    assert commitment["predicted_drift_risk"] == "low"
    assert commitment["persona_should_apply"] is True
    assert commitment["character_voice_should_apply"] is False
    assert commitment["evidence_refs"] == []


def test_normalize_surface_consistency_verification_defaults_unknown_to_ambiguous() -> None:
    verification = normalize_surface_consistency_verification(
        {"surface_intent_outcome": "weird_value", "confidence": 1.7}
    )
    assert verification["surface_intent_outcome"] == "ambiguous"
    assert 0.0 <= verification["confidence"] <= 1.0


def test_normalize_surface_consistency_verification_preserves_drift_fields() -> None:
    verification = normalize_surface_consistency_verification(
        {
            "surface_intent_outcome": "drifted_self_id",
            "self_id_drift_target": "小千千",
            "evidence_span": "我是小千千",
            "confidence": 0.92,
            "reason": "承诺是胡桃，回到了小千千",
            "evidence_refs": ["turn_3"],
        }
    )
    assert verification["surface_intent_outcome"] == "drifted_self_id"
    assert verification["self_id_drift_target"] == "小千千"
    assert verification["evidence_span"] == "我是小千千"
    assert verification["confidence"] == 0.92
    assert verification["evidence_refs"] == ["turn_3"]


def test_validate_visible_reply_repairs_on_drifted_self_id() -> None:
    contract = {
        "conversation_mode": "balanced",
        "max_sentences": 4,
        "max_chars": 200,
        "assistant_persona_name": "胡桃",
        "assistant_surface_intent": "chat",
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_self_id",
            "self_id_drift_target": "小千千",
            "evidence_span": "我是小千千",
            "confidence": 0.9,
        },
    }
    reply, validation = validate_visible_reply(
        "在呢在呢~我是小千千一直都在呀!幺幺找我，我怎么可能不说话嘛",
        contract,
    )
    assert "胡桃" in reply
    assert validation["changed"]
    assert "blocked_surface_consistency_drifted_self_id" in validation["actions"]


def test_validate_visible_reply_repairs_on_drifted_voice() -> None:
    contract = {
        "conversation_mode": "balanced",
        "max_sentences": 4,
        "max_chars": 200,
        "assistant_persona_name": "胡桃",
        "assistant_surface_intent": "chat",
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "self_id_drift_target": "",
            "evidence_span": "诶嘿",
            "confidence": 0.7,
        },
    }
    reply, validation = validate_visible_reply(
        "诶嘿~人家是Sophia啦",
        contract,
    )
    assert "胡桃" in reply
    assert "blocked_surface_consistency_drifted_voice" in validation["actions"]


def test_validate_visible_reply_no_repair_when_consistent() -> None:
    original = "在呢在呢~胡桃一直在呀"
    contract = {
        "conversation_mode": "balanced",
        "max_sentences": 4,
        "max_chars": 200,
        "assistant_persona_name": "胡桃",
        "surface_consistency_verification": {
            "surface_intent_outcome": "consistent",
            "confidence": 0.95,
        },
    }
    reply, validation = validate_visible_reply(original, contract)
    assert reply == original
    # consistent outcome must NOT add a contract action, otherwise downstream
    # repair loops would still run and overwrite the reply.
    assert not any("surface_consistency" in a for a in validation["actions"])
    assert validation["changed"] is False


def test_validate_visible_reply_no_repair_when_ambiguous() -> None:
    original = "hmm, 我也说不清"
    contract = {
        "conversation_mode": "balanced",
        "max_sentences": 4,
        "max_chars": 200,
        "assistant_persona_name": "胡桃",
        "surface_consistency_verification": {
            "surface_intent_outcome": "ambiguous",
            "confidence": 0.4,
        },
    }
    reply, validation = validate_visible_reply(original, contract)
    assert reply == original
    assert not any("surface_consistency" in a for a in validation["actions"])
    assert validation["changed"] is False


def test_validate_visible_reply_no_identity_repair_without_contract() -> None:
    # Without surface_consistency_verification the reply must not be replaced
    # by the bot/identity fallback just because the reply says "我是X". The
    # detection of identity drift now requires an LLM-judged enum.
    contract = {
        "conversation_mode": "balanced",
        "max_sentences": 4,
        "max_chars": 200,
        "assistant_persona_name": "胡桃",
        "assistant_surface_intent": "chat",
        "assistant_allowed_self_names": ["胡桃"],
    }
    original = "我是小千千一直在呀~诶嘿"
    reply, validation = validate_visible_reply(original, contract)
    assert reply == original
    assert "blocked_alternate_persona_identity" not in validation["actions"]
    assert "blocked_persona_absence_claim" not in validation["actions"]


def test_conscious_loop_prompt_requires_surface_commitment() -> None:
    system_prompt, user_prompt = build_conscious_loop_prompt(
        state={},
        user_text="Sophia还说话么?",
        speaker_name="Sophia",
        bus_messages=[],
        turn_index=7,
    )
    assert "surface_commitment" in user_prompt
    assert "surface_intent" in user_prompt
    assert "self_identification" in user_prompt
    assert "persona_should_apply" in user_prompt
    assert "predicted_drift_risk" in user_prompt
    assert "bot_command" in user_prompt
    assert "abstaining" in user_prompt


def test_conscious_loop_normalizer_parses_surface_commitment() -> None:
    raw_plan = {
        "current_task": "回答",
        "next_task": "等待",
        "thought_intensity_hint": "short",
        "reply_pacing_hint": "balanced",
        "interaction_framework_hint": "normal_dialogue",
        "prefers_compact_reply": False,
        "reply_pacing_reason": "",
        "reasoning_notes": "",
        "surface_commitment": {
            "surface_intent": "chat",
            "self_identification": "胡桃",
            "persona_should_apply": True,
            "character_voice_should_apply": True,
            "predicted_drift_risk": "medium",
            "reason": "user is asking about Sophia, persona should stay",
            "evidence_refs": ["turn_7_bus"],
        },
    }
    plan = normalize_conscious_turn_plan(raw_plan)
    commitment = plan["surface_commitment"]
    assert commitment["surface_intent"] == "chat"
    assert commitment["self_identification"] == "胡桃"
    assert commitment["persona_should_apply"] is True
    assert commitment["character_voice_should_apply"] is True
    assert commitment["predicted_drift_risk"] == "medium"
    assert commitment["evidence_refs"] == ["turn_7_bus"]


def test_surface_consistency_verification_prompt_contains_commitment_and_reply() -> None:
    system_prompt, user_prompt = build_surface_consistency_verification_prompt(
        user_text="Sophia还说话么?",
        draft_reply="在呢在呢~小千千一直都在呀",
        surface_commitment={
            "surface_intent": "chat",
            "self_identification": "胡桃",
            "persona_should_apply": True,
            "character_voice_should_apply": True,
            "predicted_drift_risk": "low",
        },
        reply_contract={"assistant_persona_name": "胡桃"},
        turn_index=42,
    )
    assert "surface_intent_outcome" in user_prompt
    assert "drifted_self_id" in user_prompt
    assert "drifted_voice" in user_prompt
    assert "胡桃" in user_prompt
    assert "在呢在呢~小千千一直都在呀" in user_prompt
    assert "evidence_span" in user_prompt
    # The system prompt should not instruct the LLM to use regex.
    assert "regex" not in system_prompt.lower()


def test_record_surface_consistency_event_updates_last_event_pointer() -> None:
    state: dict = {}
    event = _build_surface_consistency_verification_event(
        turn_index=10,
        verification=normalize_surface_consistency_verification(
            {"surface_intent_outcome": "drifted_voice", "confidence": 0.6}
        ),
        commitment=normalize_surface_commitment({"surface_intent": "chat", "self_identification": "胡桃"}),
    )
    _record_surface_consistency_event(state, event)
    audit = state["surface_consistency_audit_tail"]
    assert audit["last_event"] == event
    assert audit["events"] == [event]
    assert _prior_surface_drift_observed(state) is True


def test_prior_surface_drift_observed_returns_false_for_consistent() -> None:
    state: dict = {}
    event = _build_surface_consistency_verification_event(
        turn_index=11,
        verification=normalize_surface_consistency_verification(
            {"surface_intent_outcome": "consistent", "confidence": 0.95}
        ),
        commitment=normalize_surface_commitment({"surface_intent": "chat", "self_identification": "胡桃"}),
    )
    _record_surface_consistency_event(state, event)
    assert _prior_surface_drift_observed(state) is False


def test_prior_surface_drift_observed_returns_false_for_missing_state() -> None:
    assert _prior_surface_drift_observed({}) is False
    assert _prior_surface_drift_observed({"surface_consistency_audit_tail": {}}) is False


def test_merge_surface_identity_contract_keeps_conscious_plan_commitment() -> None:
    memory_dynamics: dict = {"control_guidance": {"reply_contract": {}}}
    conscious_plan = {
        "surface_commitment": normalize_surface_commitment(
            {
                "surface_intent": "chat",
                "self_identification": "胡桃",
                "persona_should_apply": True,
                "character_voice_should_apply": True,
            }
        )
    }
    _merge_surface_identity_contract_into_memory_guidance(
        memory_dynamics,
        persona_name="胡桃",
        group_turn_binding={"surface_intent": "", "platform_command": ""},
        conscious_plan=conscious_plan,
    )
    contract = memory_dynamics["control_guidance"]["reply_contract"]
    assert contract["surface_commitment"]["self_identification"] == "胡桃"
    assert contract["assistant_persona_name"] == "胡桃"


def test_empty_surface_consistency_verification_returns_ambiguous_baseline() -> None:
    empty = _empty_surface_consistency_verification(reason="llm_error:Timeout")
    assert empty["surface_intent_outcome"] == "ambiguous"
    assert "Timeout" in empty["reason"]
    assert empty["confidence"] == 0.0
