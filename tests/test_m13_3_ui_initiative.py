from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import (
    CONTEXT_ASSESSOR_MARKER,
    DELIVERY_ASSESSOR_MARKER,
    PROACTIVE_SURROGATE_USER_TEXT,
    assess_proactive_context_semantics,
    assess_proactive_delivery_semantics,
    build_proactive_thinking_user_text,
    default_initiative_state,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    normalize_proactive_context_assessment,
    normalize_proactive_delivery_assessment,
    proactive_delivered_text_is_safe,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from segmentum.dialogue.runtime.chat import ChatInterface


def _opted_in_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    state["m13_drive_state"] = m13
    state.update(overrides)
    return state


class _ContextUnsafeLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if CONTEXT_ASSESSOR_MARKER in system_prompt:
            if "lonely" in user_prompt.casefold() or "寂寞" in user_prompt:
                return {
                    "context_unsafe": True,
                    "unsafe_reason_codes": ["loneliness_attention_seeking"],
                    "trigger": "none",
                    "confidence": 0.9,
                    "proposed_topic": "",
                    "ordinary_language_intent": "",
                    "reason_codes": ["semantic_unsafe"],
                }
            return normalize_proactive_context_assessment({})
        return {}


class _DeliveryUnsafeLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if DELIVERY_ASSESSOR_MARKER in system_prompt:
            return {
                "allow_delivery": False,
                "violation_codes": ["subjective_loneliness_claim"],
                "confidence": 0.92,
                "reason_codes": ["semantic_unsafe_wording"],
            }
        return {"allow_delivery": True, "confidence": 0.9, "violation_codes": [], "reason_codes": []}


def test_proactive_initiative_disabled_by_default() -> None:
    initiative = default_initiative_state()
    assert initiative["enabled"] is False
    assert initiative["user_opt_in"] is False
    assert initiative["max_proactive_per_session"] == 1
    assert initiative["implicit_idle_delivery"] is False
    assert initiative["manual_continue_button"] is True


def test_no_proactive_message_without_user_opt_in() -> None:
    state = _opted_in_state()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    m13["initiative"] = default_initiative_state()
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_000,
        turn_index=2,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.proposal is None
    assert check.suppression_reason == "not_opted_in"


def test_idle_tick_creates_proposal_for_high_value_open_target() -> None:
    state = _opted_in_state(
        open_items=[
            {
                "id": "oi_1",
                "status": "open",
                "title": "M13 split",
                "next_check": "sketch test gates for path pull vs boredom",
            }
        ],
    )
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_100,
        turn_index=3,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.proposal is not None
    assert check.proposal.trigger == "open_item_next_check"
    assert any(event.get("type") == "M13ProactiveProposalEvent" for event in check.events)


def test_cooldown_and_session_cap_suppress_repeated_proactive_turns() -> None:
    state = _opted_in_state(
        open_items=[{"id": "oi_1", "status": "open", "title": "t", "next_check": "n"}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["proactive_count_this_session"] = 1
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_200,
        turn_index=4,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.suppression_reason == "session_limit_reached"


def test_suppression_reason_logged_when_no_high_value_target() -> None:
    state = _opted_in_state()
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_300,
        turn_index=2,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.suppression_reason == "no_high_value_target"
    assert any(event.get("type") == "M13ProactiveSuppressionEvent" for event in check.events)
    _, check_idle = evaluate_proactive_initiative(
        state,
        now=1_700_000_301,
        turn_index=2,
        manual_continue=False,
        implicit_idle_request=False,
        llm=_ContextUnsafeLLM(),
    )
    assert check_idle.suppression_reason == "implicit_idle_disabled"

    risky = _opted_in_state(
        open_items=[{"id": "oi", "status": "open", "title": "t", "next_check": "n"}],
        temporal_state={"last_user_text": "I feel lonely and needed to talk to you."},
    )
    _, check2 = evaluate_proactive_initiative(
        risky,
        now=1_700_000_302,
        turn_index=2,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check2.suppression_reason == "safety_risk"
    assert any(event.get("type") == "M13ProactiveContextAssessmentEvent" for event in check2.events)


def test_without_llm_recent_text_requires_insufficient_evidence_when_no_structural_target() -> None:
    state = _opted_in_state(temporal_state={"last_user_text": "我们明天再聊这个架构细节"})
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_303,
        turn_index=2,
        manual_continue=True,
        llm=None,
    )
    assert check.suppression_reason == "insufficient_evidence"


def test_context_assessor_can_propose_remind_without_keyword_cue() -> None:
    class _RemindLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if CONTEXT_ASSESSOR_MARKER in system_prompt:
                return {
                    "context_unsafe": False,
                    "unsafe_reason_codes": [],
                    "trigger": "explicit_remind_request",
                    "confidence": 0.8,
                    "proposed_topic": "architecture_followup",
                    "ordinary_language_intent": "Offer the concise reminder about the architecture thread.",
                    "reason_codes": ["user_authorized_followup"],
                }
            return {}

    state = _opted_in_state(
        temporal_state={"last_user_text": "这件事你过两小时帮我记一下，不用现在展开"},
    )
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_304,
        turn_index=2,
        manual_continue=True,
        llm=_RemindLLM(),
    )
    assert check.proposal is not None
    assert check.proposal.trigger == "explicit_remind_request"


def test_first_version_uses_manual_continue_not_implicit_idle_delivery() -> None:
    initiative = default_initiative_state()
    assert initiative["manual_continue_button"] is True
    assert initiative["implicit_idle_delivery"] is False
    state = _opted_in_state(
        open_items=[{"id": "x", "status": "open", "title": "t", "next_check": "step"}],
    )
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_400,
        turn_index=2,
        idle_seconds=999.0,
        implicit_idle_request=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.suppression_reason == "implicit_idle_disabled"


def test_implicit_idle_delivery_requires_explicit_opt_in_when_enabled() -> None:
    state = _opted_in_state(
        open_items=[{"id": "x", "status": "open", "title": "t", "next_check": "step"}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["implicit_idle_delivery"] = True
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_500,
        turn_index=2,
        idle_seconds=10.0,
        implicit_idle_request=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.suppression_reason == "idle_time_too_short"
    _, check2 = evaluate_proactive_initiative(
        state,
        now=1_700_000_501,
        turn_index=2,
        idle_seconds=200.0,
        implicit_idle_request=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check2.proposal is not None


def test_delivery_semantic_assessor_allows_casual_proactive_wording() -> None:
    llm = _ShortLLM()
    reply = "阳光这么好，心情都跟着亮堂了～"
    assessment = assess_proactive_delivery_semantics(
        llm,
        reply=reply,
        followup_replies=[],
        ordinary_language_intent="Follow up on the open item: later",
        trigger="open_item_next_check",
        turn_index=0,
    )
    normalized = normalize_proactive_delivery_assessment(assessment)
    assert normalized["allow_delivery"] is True
    assert proactive_delivered_text_is_safe(
        reply,
        llm=llm,
        ordinary_language_intent="Follow up on the open item: later",
        trigger="open_item_next_check",
        turn_index=0,
    )


def test_delivery_semantic_assessor_can_still_block_via_assessor() -> None:
    """Delivery gate remains; policy no longer targets loneliness/attention by default."""
    llm = _DeliveryUnsafeLLM()
    assessment = assess_proactive_delivery_semantics(
        llm,
        reply="example",
        followup_replies=[],
        ordinary_language_intent="test",
        trigger="open_item_next_check",
        turn_index=0,
    )
    assert normalize_proactive_delivery_assessment(assessment)["allow_delivery"] is False


def test_build_proactive_thinking_user_text_includes_intent() -> None:
    block = build_proactive_thinking_user_text(
        surrogate=PROACTIVE_SURROGATE_USER_TEXT,
        ordinary_language_intent="Follow up on the open item: draft gates",
        proposed_topic="plan",
        trigger="open_item_next_check",
    )
    assert "draft gates" in block
    assert "open_item_next_check" in block
    assert "非用户输入" in block


def test_turn_cooldown_suppresses_repeated_proactive_turns() -> None:
    state = _opted_in_state(
        open_items=[{"id": "oi_1", "status": "open", "title": "t", "next_check": "n"}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["last_proactive_turn_index"] = 1
    initiative["cooldown_turns"] = 2
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    _, check = evaluate_proactive_initiative(
        state,
        now=1_700_000_150,
        turn_index=2,
        manual_continue=True,
        llm=_ContextUnsafeLLM(),
    )
    assert check.suppression_reason == "cooldown_active"


class _ShortLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if CONTEXT_ASSESSOR_MARKER in system_prompt:
            return normalize_proactive_context_assessment({})
        if DELIVERY_ASSESSOR_MARKER in system_prompt:
            return normalize_proactive_delivery_assessment(
                {"allow_delivery": True, "confidence": 0.9, "violation_codes": []}
            )
        if "意识主循环" in system_prompt:
            return {
                "expectation_results": [],
                "memory_search_keywords": ["status"],
                "temporal_assessment": {},
            }
        if "上轮回复后果评估" in system_prompt:
            return {"reaction": "neutral", "confidence": 0.5, "reason_codes": []}
        return {
            "reply": "I thought of a cleaner next step for the open item. Want a short sketch?",
            "reply_action": "answer",
            "llm_thinking_result": {},
        }


class _UnsafeProactiveLLM(_ShortLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "思考与回复模块" in system_prompt:
            return {
                "reply": "我好寂寞，一直想和你聊聊。",
                "reply_action": "answer",
                "llm_thinking_result": {},
            }
        if DELIVERY_ASSESSOR_MARKER in system_prompt:
            return {
                "allow_delivery": False,
                "violation_codes": ["subjective_loneliness_claim"],
                "confidence": 0.95,
                "reason_codes": ["semantic_unsafe_wording"],
            }
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_proactive_turn_is_not_logged_as_user_message(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess"), llm=_ShortLLM())
    state = runtime.store.load()
    state["open_items"] = [
        {"id": "oi_1", "status": "open", "title": "plan", "next_check": "draft gates"}
    ]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    runtime.store.save(state)

    check = runtime.maybe_propose_proactive_turn(turn_index=0, manual_continue=True)
    assert check.get("proposal")
    proposal_id = str(check["proposal"]["proposal_id"])
    result = runtime.run_proactive_turn(proposal_id=proposal_id, turn_index=0, speaker_name="测试用户")
    assert result.reply
    assert result.diagnostics.get("not_user_requested_current_turn") is True

    log_path = runtime.store.root / "conversation_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    proactive_rows = [row for row in rows if row.get("event") == "proactive_turn"]
    assert proactive_rows
    assert proactive_rows[-1].get("not_user_requested_current_turn") is True
    assert proactive_rows[-1].get("user_text", "missing") == "missing"
    bus_types = [msg.get("type") for msg in result.diagnostics.get("bus_messages", [])]
    assert "UserUtteranceEvent" not in bus_types
    assert "M13ProactiveTurnRequestEvent" in bus_types


def test_proactive_generation_uses_safety_and_reply_validation(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess2"), llm=_ShortLLM())
    state = runtime.store.load()
    state["open_items"] = [{"id": "oi", "status": "open", "title": "t", "next_check": "n"}]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    runtime.store.save(state)
    check = runtime.maybe_propose_proactive_turn(turn_index=0, manual_continue=True)
    result = runtime.run_proactive_turn(
        proposal_id=str(check["proposal"]["proposal_id"]),
        turn_index=0,
    )
    assert result.diagnostics.get("reply_validation")
    bus_msg = next(
        msg
        for msg in result.diagnostics.get("bus_messages", [])
        if msg.get("type") == "M13ProactiveTurnRequestEvent"
    )
    assert PROACTIVE_SURROGATE_USER_TEXT in str(bus_msg.get("surrogate_context", ""))
    assert "Follow up on the open item" in str(bus_msg.get("ordinary_language_intent", ""))


def test_blocked_proactive_turn_rolls_back_state_and_skips_delivery_log(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess_block"), llm=_UnsafeProactiveLLM())
    state = runtime.store.load()
    state["temporal_state"] = {"last_user_text": "real user question about M13"}
    state["open_items"] = [{"id": "oi", "status": "open", "title": "t", "next_check": "n"}]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    runtime.store.save(state)

    check = runtime.maybe_propose_proactive_turn(turn_index=0, manual_continue=True)
    result = runtime.run_proactive_turn(
        proposal_id=str(check["proposal"]["proposal_id"]),
        turn_index=0,
    )
    assert not str(result.reply or "").strip()
    assert result.diagnostics.get("proactive_text_blocked") is True

    reloaded = runtime.store.load()
    assert reloaded["temporal_state"]["last_user_text"] == "real user question about M13"

    log_path = runtime.store.root / "conversation_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    proactive_rows = [row for row in rows if row.get("event") == "proactive_turn"]
    assert not proactive_rows
    assert any(
        row.get("event") == "m13_proactive_audit"
        and row.get("type") == "M13ProactiveSuppressionEvent"
        and row.get("reason") == "safety_risk"
        for row in rows
    )


def test_mvp_runtime_maybe_and_run_proactive_audit_events(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess3"), llm=_ShortLLM())
    state = runtime.store.load()
    state["open_items"] = [{"id": "oi", "status": "open", "title": "t", "next_check": "n"}]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    runtime.store.save(state)
    check = runtime.maybe_propose_proactive_turn(turn_index=1, manual_continue=True)
    types = [event.get("type") for event in check.get("events", [])]
    assert "M13ProactiveCheckEvent" in types
    assert "M13ProactiveProposalEvent" in types

    result = runtime.run_proactive_turn(
        proposal_id=str(check["proposal"]["proposal_id"]),
        turn_index=1,
    )
    assert result.reply
    log_path = runtime.store.root / "conversation_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("type") == "M13ProactiveGenerationEvent" for row in rows if row.get("event") == "m13_proactive_audit")
    assert any(row.get("event") == "proactive_turn" for row in rows)


def test_chat_interface_proactive_does_not_append_user_transcript(tmp_path: Path) -> None:
    class _Traits:
        def to_dict(self) -> dict[str, float]:
            return {"curiosity": 0.5}

    class _PP:
        openness = 0.5
        conscientiousness = 0.5
        extraversion = 0.5
        agreeableness = 0.5
        neuroticism = 0.5

    class _AgentStub:
        slow_variable_learner = type("Slow", (), {"state": type("S", (), {"traits": _Traits()})()})()
        self_model = type("M", (), {"personality_profile": _PP()})()

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess4"), llm=_ShortLLM())
    state = runtime.store.load()
    state["open_items"] = [{"id": "oi", "status": "open", "title": "t", "next_check": "n"}]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    runtime.store.save(state)

    iface = ChatInterface(use_llm=False, session_id="test_sess")
    iface.set_agent(_AgentStub(), persona_name="test")  # type: ignore[arg-type]
    iface._mvp_runtime = runtime  # type: ignore[attr-defined]
    iface._use_mvp_runtime = True  # type: ignore[attr-defined]
    iface._turn_index = 0

    check = iface.maybe_propose_proactive_turn(manual_continue=True)
    before = len(iface._transcript)
    resp = iface.run_proactive_turn(str(check["proposal"]["proposal_id"]), speaker_name="u1")
    assert resp.reply
    assert len(iface._transcript) == before + 1
    last = iface._transcript[-1]
    role = last.role if hasattr(last, "role") else last.get("role")
    assert role == "agent"
