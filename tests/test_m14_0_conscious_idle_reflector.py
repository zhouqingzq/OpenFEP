from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_boredom import normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import (
    gather_idle_structural_signals,
    normalize_idle_introspection_state,
    set_idle_introspection_user_opt_in,
)
from segmentum.dialogue.runtime.m13_initiative import (
    PROACTIVE_SURROGATE_USER_TEXT,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m14_idle_owners import (
    MAX_SELF_COGNITION_PATCHES_PER_SESSION,
    MemoryConsolidationOwner,
    OpenItemPatchOwner,
    SelfCognitionPatchOwner,
)
from segmentum.dialogue.runtime.m14_idle_reflector import (
    IDLE_INTROSPECTION_MARKER,
    apply_idle_drive_rules,
    build_idle_context,
    build_structural_idle_plan,
    empty_conscious_idle_plan,
    normalize_conscious_idle_plan,
    subjective_language_violations,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


def _opted_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [],
        "short_term_memory": [],
        "long_term_memory": [],
        "pending_expectations": [],
        "self_cognition": {"patch_history": []},
        "temporal_state": {
            "last_turn_at": 1_700_000_000,
            "last_user_turn_at": 1_700_000_000,
            "last_turn_index": 3,
            "last_reply": "prior reply",
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    state["m13_drive_state"] = m13
    state.update(overrides)
    return state


class _IdleLLM:
  def __init__(self, plan: dict[str, object] | None = None) -> None:
      self.plan = plan
      self.calls = 0

  def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
      self.calls += 1
      if IDLE_INTROSPECTION_MARKER not in system_prompt:
          if "M13" in system_prompt:
              return {"allow_delivery": True, "confidence": 0.9, "violation_codes": [], "reason_codes": []}
          return {}
      if self.plan is not None:
          return dict(self.plan)
      return {}


class _ProactiveRunLLM(_IdleLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if PROACTIVE_SURROGATE_USER_TEXT[:20] in user_prompt or "系统主动续写" in user_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "proactive",
                    "state_or_memory_used": [],
                    "response_choice": "answer",
                    "uncertainty": "",
                    "debug_summary": "ok",
                },
                "reply": "短跟进：open item 测试。",
                "reply_action": "answer",
                "disclosure_action": "none",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_idle_plan_is_empty_when_no_signal() -> None:
    ctx = build_idle_context(
        _opted_state(),
        m13_state=normalize_m13_drive_state(default_m13_drive_state()),
        structural_signals=gather_idle_structural_signals(
            _opted_state(), now=1_700_000_100, turn_index=2
        ),
        turn_index=2,
        now=1_700_000_100,
    )
    plan = build_structural_idle_plan(ctx, retrieved_ids=set())
    assert plan["reflection_focus"] is None
    assert plan["outreach_recommendation"]["should_outreach"] is False


def test_idle_plan_picks_open_item_focus_when_long_pending() -> None:
    state = _opted_state(
        open_items=[
            {
                "id": "oi_long",
                "status": "open",
                "title": "split",
                "next_check": "write acceptance tests for idle reflector",
            }
        ],
        short_term_memory=[
            {"id": "oi_long", "content": "context", "salience": 0.7, "kind": "open_item"},
        ],
    )
    ctx = build_idle_context(
        state,
        m13_state=normalize_m13_drive_state(state["m13_drive_state"]),  # type: ignore[arg-type]
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_200, turn_index=4),
        turn_index=4,
        now=1_700_000_200,
    )
    plan = build_structural_idle_plan(ctx, retrieved_ids={"oi_long"})
    focus = plan.get("reflection_focus")
    assert isinstance(focus, dict)
    assert focus.get("reflection_kind") == "open_item"


def test_idle_plan_picks_habit_calibration_when_boredom_high_and_stale() -> None:
    state = _opted_state()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    boredom = normalize_boredom_state(m13.get("boredom"))
    boredom["boredom_level"] = 0.82
    boredom["last_exploration_target"] = "fresh angle on travel thread"
    m13["boredom"] = boredom
    m13["affective_reward_proxy"]["path_feels_stale_proxy"] = True
    state["m13_drive_state"] = m13
    ctx = build_idle_context(
        state,
        m13_state=m13,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_300, turn_index=5),
        turn_index=5,
        now=1_700_000_300,
    )
    plan = build_structural_idle_plan(ctx, retrieved_ids={"mem_a"})
    plan = apply_idle_drive_rules(plan, idle_context=ctx, structural_signals=gather_idle_structural_signals(state, now=1_700_000_300, turn_index=5))
    focus = plan.get("reflection_focus")
    assert isinstance(focus, dict)
    assert focus.get("reflection_kind") == "habit_calibration"
    assert plan["outreach_recommendation"]["should_outreach"] is False


def test_self_cognition_patch_requires_evidence_refs() -> None:
    state = _opted_state()
    result = SelfCognitionPatchOwner.validate_and_commit(
        state,
        {
            "apply": True,
            "summary_delta": "operational note",
            "evidence_refs": [],
            "confidence": 0.9,
            "reason": "test",
        },
        retrieved_ids={"mem1"},
        turn_index=1,
        now=1,
        session_patches=0,
    )
    assert result.committed is False
    assert "missing_evidence_refs" in result.violation_codes


def test_self_cognition_patch_rejected_when_confidence_low() -> None:
    state = _opted_state()
    result = SelfCognitionPatchOwner.validate_and_commit(
        state,
        {
            "apply": True,
            "summary_delta": "operational note",
            "evidence_refs": ["mem1"],
            "confidence": 0.2,
            "reason": "test",
        },
        retrieved_ids={"mem1"},
        turn_index=1,
        now=1,
        session_patches=0,
    )
    assert result.committed is False
    assert "confidence_below_threshold" in result.violation_codes


def test_open_item_patch_updates_status_with_audit() -> None:
    state = _opted_state(
        open_items=[{"id": "oi_patch", "status": "open", "title": "t", "next_check": "old step"}],
    )
    result = OpenItemPatchOwner.validate_and_commit(
        state,
        [{"op": "update", "id": "oi_patch", "rationale": "refined next step after idle review"}],
        retrieved_ids={"oi_patch"},
        turn_index=3,
        now=3,
        session_patches=0,
    )
    assert result.committed is True
    item = next(row for row in state["open_items"] if row.get("id") == "oi_patch")  # type: ignore[index,union-attr]
    assert "refined" in str(item.get("next_check", ""))
    assert any(e.get("type") == "OpenItemPatchCommitEvent" for e in result.events)


def test_self_cognition_patch_committed_appends_history() -> None:
    state = _opted_state()
    result = SelfCognitionPatchOwner.validate_and_commit(
        state,
        {
            "apply": True,
            "summary_delta": "I stay cautious under uncertainty.",
            "evidence_refs": ["mem1"],
            "confidence": 0.85,
            "reason": "idle reflection",
        },
        retrieved_ids={"mem1"},
        turn_index=2,
        now=2,
        session_patches=0,
    )
    assert result.committed is True
    history = state["self_cognition"]["patch_history"]  # type: ignore[index]
    assert len(history) == 1


def test_memory_consolidation_proposal_becomes_intent(tmp_path: Path) -> None:
    plan = {
        "mode": "idle_introspection",
        "reflection_focus": None,
        "self_cognition_patch_proposal": {"apply": False, "evidence_refs": [], "confidence": 0.0},
        "memory_consolidation_proposals": [
            {
                "target": "short_term",
                "kind": "preference",
                "content": "consolidated preference",
                "confidence": 0.8,
                "evidence_refs": ["mem_c"],
            }
        ],
        "open_item_proposals": [],
        "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
        "thought_intensity_hint": "short",
    }
    state = _opted_state(
        short_term_memory=[{"id": "mem_c", "content": "user prefers concise replies", "salience": 0.6}],
    )
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "mem"), llm=_IdleLLM(plan))  # type: ignore[arg-type]
    runtime.store.save(state)  # type: ignore[arg-type]
    idle = runtime.run_idle_introspection_turn(
        now=1_700_000_400,
        turn_index=6,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_400, turn_index=6),
    )
    assert idle.ran_llm is True
    assert any(
        e.get("type") == "MemoryConsolidationIntentEvent" and e.get("committed")
        for e in idle.audit_events
    )


def test_outreach_recommendation_routes_through_m13_3(tmp_path: Path) -> None:
    plan = {
        "mode": "idle_introspection",
        "reflection_focus": {
            "topic": "follow up thread",
            "evidence_refs": ["oi_x"],
            "reflection_kind": "open_item",
        },
        "self_cognition_patch_proposal": {"apply": False, "evidence_refs": [], "confidence": 0.0},
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {
            "should_outreach": True,
            "reason": "open_item_followup",
            "suggested_intent": "Offer a short follow-up on the open item.",
            "trigger": "reflection_outreach",
        },
        "thought_intensity_hint": "short",
    }
    state = _opted_state(
        open_items=[{"id": "oi_x", "status": "open", "title": "t", "next_check": "continue"}],
    )
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "out"), llm=_ProactiveRunLLM(plan))  # type: ignore[arg-type]
    runtime.store.save(state)  # type: ignore[arg-type]
    idle = runtime.run_idle_introspection_turn(
        now=1_700_000_500,
        turn_index=7,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_500, turn_index=7),
    )
    assert idle.outreach_recommendation.get("should_outreach") is False
    assert idle.diagnostics.get("outreach_outcome") in {"low_value", "reflection_only"}
    logs = (runtime.store.root / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "IdleOutreachProposalEvent" not in logs
    assert "proactive_turn" not in logs


def test_outreach_failure_does_not_rollback_patches(tmp_path: Path) -> None:
    plan = {
        "mode": "idle_introspection",
        "reflection_focus": {
            "topic": "thread",
            "evidence_refs": ["mem_z"],
            "reflection_kind": "open_item",
        },
        "self_cognition_patch_proposal": {
            "apply": True,
            "summary_delta": "patch survives outreach failure",
            "evidence_refs": ["mem_z"],
            "confidence": 0.9,
            "reason": "test",
        },
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {
            "should_outreach": True,
            "reason": "open_item_followup",
            "suggested_intent": "Try outreach",
            "trigger": "reflection_outreach",
        },
        "thought_intensity_hint": "short",
    }
    state = _opted_state(
        short_term_memory=[{"id": "mem_z", "content": "z", "salience": 0.8}],
        open_items=[{"id": "mem_z", "status": "open", "next_check": "continue"}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    initiative = m13["initiative"]
    initiative["proactive_count_this_session"] = initiative["max_proactive_per_session"]
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "rb"), llm=_IdleLLM(plan))  # type: ignore[arg-type]
    runtime.store.save(state)  # type: ignore[arg-type]
    idle = runtime.run_idle_introspection_turn(
        now=1_700_000_600,
        turn_index=8,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_600, turn_index=8),
    )
    reloaded = runtime.store.load()
    history = reloaded["self_cognition"].get("patch_history", [])
    assert len(history) >= 1
    assert idle.diagnostics.get("outreach_outcome") in {"low_value", "reflection_only"}


def test_idle_plan_never_writes_to_m11_or_m12_ledgers(tmp_path: Path) -> None:
    state = _opted_state(
        m11_user_models={"u": {"claims": []}},
        m12_user_continuity={"profiles_by_user": {}},
    )
    plan = {
        "mode": "idle_introspection",
        "reflection_focus": None,
        "self_cognition_patch_proposal": {
            "apply": True,
            "summary_delta": "write m11_user_models hack",
            "evidence_refs": ["mem1"],
            "confidence": 0.9,
            "reason": "bad",
        },
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
        "thought_intensity_hint": "short",
    }
    state["short_term_memory"] = [{"id": "mem1", "content": "x", "salience": 0.9}]
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "m12"), llm=_IdleLLM(plan))  # type: ignore[arg-type]
    runtime.store.save(state)  # type: ignore[arg-type]
    before_m11 = dict(runtime.store.load().get("m11_user_models", {}))
    runtime.run_idle_introspection_turn(
        now=1_700_000_700,
        turn_index=9,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_700, turn_index=9),
    )
    after = runtime.store.load()
    assert after.get("m11_user_models") == before_m11
    assert not after["self_cognition"].get("patch_history")


def test_idle_plan_does_not_emit_subjective_language() -> None:
    raw = {
        "mode": "idle_introspection",
        "reflection_focus": {"topic": "我很无聊", "evidence_refs": ["a"], "reflection_kind": "none"},
        "self_cognition_patch_proposal": {"apply": False, "confidence": 0.0, "evidence_refs": []},
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {
            "should_outreach": True,
            "reason": "none",
            "suggested_intent": "I feel lonely and addicted",
        },
        "thought_intensity_hint": "short",
    }
    plan = normalize_conscious_idle_plan(raw)
    assert plan.get("reflection_focus") is None
    assert plan["outreach_recommendation"]["should_outreach"] is False
    assert subjective_language_violations("I feel lonely")


def test_session_caps_block_further_patches() -> None:
    state = _opted_state()
    for _ in range(MAX_SELF_COGNITION_PATCHES_PER_SESSION + 1):
        SelfCognitionPatchOwner.validate_and_commit(
            state,
            {
                "apply": True,
                "summary_delta": f"delta {_}",
                "evidence_refs": ["mem1"],
                "confidence": 0.9,
                "reason": "cap",
            },
            retrieved_ids={"mem1"},
            turn_index=1,
            now=1,
            session_patches=_,
        )
    history = state["self_cognition"]["patch_history"]  # type: ignore[index]
    assert len(history) == MAX_SELF_COGNITION_PATCHES_PER_SESSION


def test_path_feels_stale_biases_outreach_toward_false() -> None:
    state = _opted_state(
        open_items=[{"id": "oi", "status": "open", "next_check": "continue"}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    m13["affective_reward_proxy"]["path_feels_stale_proxy"] = True
    state["m13_drive_state"] = m13
    ctx = build_idle_context(
        state,
        m13_state=m13,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_800, turn_index=10),
        turn_index=10,
        now=1_700_000_800,
    )
    plan = build_structural_idle_plan(ctx, retrieved_ids={"oi"})
    plan = apply_idle_drive_rules(plan, idle_context=ctx, structural_signals=gather_idle_structural_signals(state, now=1_700_000_800, turn_index=10))
    assert plan["outreach_recommendation"]["should_outreach"] is False


def test_consecutive_idle_ticks_reflect_more_than_outreach(tmp_path: Path) -> None:
    state = _opted_state(
        open_items=[{"id": "oi", "status": "open", "next_check": "step"}],
        short_term_memory=[{"id": "oi", "content": "ctx", "salience": 0.7}],
    )
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    m13["affective_reward_proxy"]["path_feels_stale_proxy"] = True
    state["m13_drive_state"] = m13
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "seq"), llm=_IdleLLM({}))  # type: ignore[arg-type]
    runtime.store.save(state)  # type: ignore[arg-type]
    sig = gather_idle_structural_signals(state, now=1_700_001_000, turn_index=11)
    first = runtime.run_idle_introspection_turn(now=1_700_001_000, turn_index=11, structural_signals=sig)
    state2 = runtime.store.load()
    sig2 = gather_idle_structural_signals(state2, now=1_700_001_200, turn_index=12)
    second = runtime.run_idle_introspection_turn(now=1_700_001_200, turn_index=12, structural_signals=sig2)
    assert first.ran_llm is True
    assert second.ran_llm is True
    assert first.outreach_recommendation.get("should_outreach") is False
    assert second.outreach_recommendation.get("should_outreach") is False
