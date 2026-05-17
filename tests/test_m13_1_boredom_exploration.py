from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_boredom import (
    BOREDOM_USER_TEXT_ASSESSOR_MARKER,
    MAX_PROMPT_GUIDANCE_LINES,
    MAX_PROMPT_LINE_LENGTH,
    MAX_SEMANTIC_INFORMATION_GAIN_HINT,
    M13BoredomEvaluator,
    _closed_expectation_progress,
    _progress_signal,
    apply_post_turn_boredom_state,
    build_prompt_safe_guidance_lines,
    merge_exploration_guidance_into_control,
    normalize_boredom_state,
    normalize_boredom_user_text_assessment,
    prompt_safe_control_guidance_for_thinking,
    prompt_safe_m13_boredom_diagnostics,
    sanitize_drive_guidance_for_prompt,
)
from segmentum.dialogue.runtime.m13_drive import (
    M13DriveEvaluator,
    apply_post_turn_m13_state,
    default_m13_drive_state,
    merge_drive_guidance_into_control,
    normalize_m13_drive_state,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    _prompt_safe_state,
    build_thinking_prompt,
)

_BOREDOM_THINKING_PROMPT_MARKERS = (
    "boredom_level",
    "novelty_proxy",
    "information_gain_proxy",
    "repetition_pressure",
    "stale_turn_count",
    "engineering_proxy",
    "mvp_local_boredom_proxy",
    "exploration_bias",
    "exploration_target",
    "suppressed_repetition_actions",
    "exploration_cooldown",
    "exploration_suppressed",
    "preferred_exploration_mode",
    "I am bored",
    "我很无聊",
)


def _assert_drive_guidance_prompt_safe(guidance: dict[str, object]) -> None:
    blob = json.dumps(guidance, ensure_ascii=False).casefold()
    for marker in _BOREDOM_THINKING_PROMPT_MARKERS:
        assert marker.casefold() not in blob


def _stale_boredom_state() -> dict[str, object]:
    return {
        **default_m13_drive_state(),
        "recent_topic_fingerprints": [{"topic": "status|update", "turn_index": i} for i in range(6)],
        "recent_action_trace": [
            {
                "turn_id": f"t{i}",
                "turn_index": i,
                "user_id": "u1",
                "action": "answer",
                "topic_fingerprint": "status|update",
                "outcome_band": "uncertain",
            }
            for i in range(6)
        ],
        "boredom": {"stale_turn_count": 8, "exploration_cooldown": 0},
    }


class _BoredomSemanticLLM:
    """Stub LLM for boredom user-text semantics (not production keyword policy)."""

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if BOREDOM_USER_TEXT_ASSESSOR_MARKER not in system_prompt:
            return {}
        text = user_prompt
        if "LRU" in text or "怎么实现" in text:
            return normalize_boredom_user_text_assessment(
                {
                    "explicit_task_pressure": 0.45,
                    "information_gain_hint": 0.06,
                    "user_need_salience": 0.35,
                    "low_information_utterance": False,
                    "confidence": 0.88,
                    "reason_codes": ["direct_implementation_request"],
                }
            )
        if "为什么" in text:
            return normalize_boredom_user_text_assessment(
                {
                    "explicit_task_pressure": 0.0,
                    "information_gain_hint": 0.12,
                    "user_need_salience": 0.12,
                    "low_information_utterance": False,
                    "confidence": 0.8,
                    "reason_codes": ["analytic_question"],
                }
            )
        if "用户本轮发言:\n嗯" in text or text.rstrip().endswith("嗯"):
            return normalize_boredom_user_text_assessment(
                {
                    "explicit_task_pressure": 0.0,
                    "information_gain_hint": 0.0,
                    "user_need_salience": 0.0,
                    "low_information_utterance": True,
                    "confidence": 0.9,
                    "reason_codes": ["minimal_backchannel"],
                }
            )
        return normalize_boredom_user_text_assessment({})


def _base_control(**overrides: object) -> dict[str, object]:
    control: dict[str, object] = {
        "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
        "reply_contract": {},
        "conflict_level": 0.0,
        "repair_bias": 0.0,
        "clarification_bias": 0.0,
    }
    control.update(overrides)
    return control


def _status_turn_inputs(*, turn: int, memory_id: str) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    conscious = {"memory_search_keywords": ["status", "update"]}
    dynamics = {
        "recall_query": {"semantic_terms": ["status", "update"]},
        "control_guidance": _base_control(),
    }
    memories = [{"id": memory_id, "keywords": ["status"], "kind": "episode"}]
    return conscious, dynamics, memories


def test_boredom_suppressed_on_first_turn() -> None:
    evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    result = evaluator.evaluate(
        user_text="Python 里怎么实现一个 LRU cache？请给步骤。",
        user_id="u1",
        turn_id="t0",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["python", "lru", "cache"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["python"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=state,
    )
    assert result.boredom_band == "low"
    assert result.exploration_suppressed is True
    assert result.exploration_bias < 0.2


def test_boredom_suppressed_on_direct_task_after_first_turn() -> None:
    evaluator = M13BoredomEvaluator()
    stale = _stale_boredom_state()
    llm = _BoredomSemanticLLM()
    direct_task = evaluator.evaluate(
        user_text="Python 里怎么实现一个 LRU cache？请给步骤。",
        user_id="u1",
        turn_id="t_direct",
        turn_index=3,
        conscious_plan={"memory_search_keywords": ["python", "lru", "cache"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["python"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=stale,
        llm=llm,
    )
    casual_repeat = evaluator.evaluate(
        user_text="还是 status update",
        user_id="u1",
        turn_id="t_repeat",
        turn_index=3,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        m13_state=stale,
        llm=llm,
    )
    assert direct_task.exploration_suppressed is True
    assert direct_task.exploration_bias < 0.15
    assert casual_repeat.exploration_bias > direct_task.exploration_bias


def test_boredom_rises_for_repeated_action_and_same_topic() -> None:
    evaluator = M13BoredomEvaluator()
    drive_evaluator = M13DriveEvaluator()
    state = default_m13_drive_state()
    user_id = "repeat_user"
    levels: list[float] = []
    for turn in range(8):
        conscious, dynamics, memories = _status_turn_inputs(turn=turn, memory_id=f"m{turn}")
        drive_eval = drive_evaluator.evaluate(
            user_text="项目 status update 呢",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            conscious_plan=conscious,
            memory_dynamics=dynamics,
            retrieved_memories=memories,
            response_style_prior={},
            habit_traits={},
            relationship_value_context={},
            m13_state=state,
        )
        boredom = evaluator.evaluate(
            user_text="项目 status update 呢",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            conscious_plan=conscious,
            memory_dynamics=dynamics,
            retrieved_memories=memories,
            m13_state=state,
            m13_drive_evaluation=drive_eval,
        )
        levels.append(boredom.boredom_level)
        state, _ = apply_post_turn_m13_state(
            state,
            evaluation=drive_eval,
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            selected_action=drive_eval.top_behavioral_pull_action,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan={"expectation_results": []},
            memory_candidates_applied=[],
        )
        state, _ = apply_post_turn_boredom_state(
            state,
            boredom=boredom,
            conscious_plan=conscious,
            retrieved_memories=memories,
            turn_index=turn + 1,
        )
    assert max(levels) > levels[0] + 0.05
    assert boredom.repetition_pressure >= 0.2


def test_new_information_gain_reduces_boredom() -> None:
    evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    stale_state = normalize_m13_drive_state(state)
    for turn in range(5):
        conscious, dynamics, memories = _status_turn_inputs(turn=turn, memory_id="m0")
        stale_state.setdefault("recent_topic_fingerprints", []).append(
            {"topic": "status|update", "turn_index": turn}
        )
        stale_state.setdefault("recent_action_trace", []).append(
            {
                "turn_id": f"t{turn}",
                "turn_index": turn,
                "user_id": "u1",
                "action": "answer",
                "topic_fingerprint": "status|update",
                "outcome_band": "uncertain",
            }
        )
    stale = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t_stale",
        turn_index=6,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        m13_state=stale_state,
    )
    fresh = evaluator.evaluate(
        user_text="请对比三种架构方案并给出设计权衡，我需要做决定。",
        user_id="u1",
        turn_id="t_fresh",
        turn_index=6,
        conscious_plan={"memory_search_keywords": ["architecture", "tradeoff", "decision", "design"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["architecture"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m_new", "keywords": ["architecture"]}],
        m13_state=stale_state,
    )
    assert fresh.information_gain_proxy > stale.information_gain_proxy
    assert fresh.boredom_level < stale.boredom_level


def test_high_boredom_adds_exploration_guidance_not_visible_diagnostic() -> None:
    evaluator = M13BoredomEvaluator()
    boredom = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=4,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        m13_state={
            **default_m13_drive_state(),
            "recent_topic_fingerprints": [{"topic": "status|update", "turn_index": i} for i in range(6)],
            "recent_action_trace": [
                {
                    "turn_id": f"t{i}",
                    "turn_index": i,
                    "user_id": "u1",
                    "action": "answer",
                    "topic_fingerprint": "status|update",
                    "outcome_band": "uncertain",
                }
                for i in range(6)
            ],
            "boredom": {"stale_turn_count": 6, "exploration_cooldown": 0},
        },
        m13_drive_evaluation=None,
    )
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_exploration_guidance_into_control(memory_dynamics, boredom)
    guidance = memory_dynamics["control_guidance"]["drive_guidance"]  # type: ignore[index]
    lines = guidance.get("prompt_safe_lines") or []
    assert lines
    _assert_drive_guidance_prompt_safe(guidance)


def test_boredom_suppressed_under_repair_or_conflict_pressure() -> None:
    evaluator = M13BoredomEvaluator()
    result = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["status"]},
            "control_guidance": _base_control(conflict_level=0.8, repair_bias=0.7),
        },
        retrieved_memories=[],
        m13_state=default_m13_drive_state(),
    )
    assert result.exploration_suppressed is True
    assert result.exploration_bias < 0.15


def test_exploration_cooldown_prevents_every_turn_topic_shift() -> None:
    evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    state["boredom"] = normalize_boredom_state({"exploration_cooldown": 3, "stale_turn_count": 8})
    result = evaluator.evaluate(
        user_text="还是老样子",
        user_id="u1",
        turn_id="t1",
        turn_index=5,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=state,
    )
    assert result.exploration_suppressed is True
    updated, _ = apply_post_turn_boredom_state(
        state,
        boredom=result,
        conscious_plan={"memory_search_keywords": ["status"]},
        retrieved_memories=[],
        turn_index=5,
    )
    assert int(updated["boredom"]["exploration_cooldown"]) == 2


def test_boredom_event_and_patch_are_auditable() -> None:
    evaluator = M13BoredomEvaluator()
    boredom = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=4,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        m13_state=_stale_boredom_state(),
    )
    assert boredom.events
    event = boredom.events[0]
    assert event["type"] == "M13BoredomEvaluationEvent"
    assert event.get("event_id")
    assert event.get("engineering_proxy_label") == "mvp_local_boredom_proxy"
    assert boredom.boredom_level >= 0.35

    state, patch_events = apply_post_turn_boredom_state(
        _stale_boredom_state(),
        boredom=boredom,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        turn_index=4,
    )
    assert "boredom" in state
    commits = [e for e in patch_events if e.get("type") == "M13BoredomPatchCommit"]
    proposals = [e for e in patch_events if e.get("type") == "M13BoredomPatchProposal"]
    assert commits
    assert proposals
    diag = prompt_safe_m13_boredom_diagnostics(boredom)
    assert "boredom_level" not in diag
    assert diag.get("engineering_proxy_label") == "mvp_local_boredom_proxy"


def test_retrieve_specific_memory_mode_when_stale_with_evidence() -> None:
    evaluator = M13BoredomEvaluator()
    drive_evaluator = M13DriveEvaluator()
    state = default_m13_drive_state()
    conscious = {"memory_search_keywords": ["status", "update"]}
    dynamics = {
        "recall_query": {"semantic_terms": ["status", "update"]},
        "control_guidance": _base_control(),
    }
    memories = [{"id": "m0", "keywords": ["status"], "kind": "episode"}]
    state["boredom"] = normalize_boredom_state({"stale_turn_count": 6, "recent_retrieval_ids": ["m0"]})
    drive_eval = drive_evaluator.evaluate(
        user_text="还是老样子",
        user_id="u1",
        turn_id="t0",
        turn_index=4,
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        retrieved_memories=memories,
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
        evidence_judgment={"epistemic_stance": "known"},
    )
    topic = drive_eval.topic_fingerprint
    action = drive_eval.top_behavioral_pull_action
    state["recent_topic_fingerprints"] = [{"topic": topic, "turn_index": i} for i in range(4)]
    state["recent_action_trace"] = [
        {
            "turn_id": f"t{i}",
            "turn_index": i,
            "user_id": "u1",
            "action": action,
            "topic_fingerprint": topic,
            "outcome_band": "uncertain",
        }
        for i in range(2)
    ]
    result = evaluator.evaluate(
        user_text="还是老样子",
        user_id="u1",
        turn_id="t1",
        turn_index=4,
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        retrieved_memories=memories,
        m13_state=state,
        m13_drive_evaluation=drive_eval,
        evidence_judgment={"epistemic_stance": "known"},
    )
    assert result.preferred_exploration_mode in {
        "retrieve_specific_memory",
        "summarize_and_choose_next_target",
        "shift_from_repetition_to_progress",
    }
    assert result.repetition_pressure >= 0.15
    assert result.boredom_level >= 0.35


def test_boredom_multiturn_same_topic_raises_exploration_bias() -> None:
    evaluator = M13BoredomEvaluator()
    drive_evaluator = M13DriveEvaluator()
    state = default_m13_drive_state()
    biases: list[float] = []
    levels: list[float] = []
    user_id = "mt_user"
    for turn in range(9):
        conscious, dynamics, memories = _status_turn_inputs(turn=turn, memory_id=f"m{turn}")
        drive_eval = drive_evaluator.evaluate(
            user_text="还是 status update",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            conscious_plan=conscious,
            memory_dynamics=dynamics,
            retrieved_memories=memories,
            response_style_prior={},
            habit_traits={},
            relationship_value_context={},
            m13_state=state,
        )
        boredom = evaluator.evaluate(
            user_text="还是 status update",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            conscious_plan=conscious,
            memory_dynamics=dynamics,
            retrieved_memories=memories,
            m13_state=state,
            m13_drive_evaluation=drive_eval,
        )
        biases.append(boredom.exploration_bias)
        levels.append(boredom.boredom_level)
        state, _ = apply_post_turn_m13_state(
            state,
            evaluation=drive_eval,
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            selected_action="answer",
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan={},
            memory_candidates_applied=[],
        )
        state, _ = apply_post_turn_boredom_state(
            state,
            boredom=boredom,
            conscious_plan=conscious,
            retrieved_memories=memories,
            turn_index=turn + 1,
        )

    assert max(biases[2:6]) > biases[0] + 0.05
    peak_level = max(levels)
    fresh = evaluator.evaluate(
        user_text="我们换到 travel plan：请分析下周行程取舍并做决定。",
        user_id=user_id,
        turn_id="turn_fresh",
        turn_index=10,
        conscious_plan={"memory_search_keywords": ["travel", "plan", "decision"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["travel", "plan"]},
            "control_guidance": _base_control(),
        },
        retrieved_memories=[{"id": "travel_1", "keywords": ["travel"]}],
        m13_state=state,
    )
    assert fresh.information_gain_proxy > 0.2
    assert fresh.boredom_level < peak_level - 0.03
    assert int(state["boredom"]["exploration_cooldown"]) > 0
    peak_bias = max(biases[2:6])
    assert fresh.exploration_bias < peak_bias or fresh.exploration_suppressed


def test_drive_guidance_is_capped_to_six_prompt_safe_lines() -> None:
    drive_evaluator = M13DriveEvaluator()
    boredom_evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    evaluation = drive_evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=2,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    boredom = boredom_evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=2,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=state,
        m13_drive_evaluation=evaluation,
    )
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_drive_guidance_into_control(
        memory_dynamics,
        evaluation,
        boredom_evaluation=boredom,
    )
    guidance = memory_dynamics["control_guidance"]["drive_guidance"]  # type: ignore[index]
    lines = guidance.get("prompt_safe_lines") or []
    assert len(lines) <= MAX_PROMPT_GUIDANCE_LINES
    assert all(len(line) <= MAX_PROMPT_LINE_LENGTH for line in lines)
    _assert_drive_guidance_prompt_safe(guidance)

    extra = build_prompt_safe_guidance_lines(
        drive_summary="a" * 200,
        drive_caution="b" * 200,
        exploration_hint="c" * 200,
        exploration_mode="offer_new_angle",
    )
    assert len(extra) <= MAX_PROMPT_GUIDANCE_LINES
    assert all(len(line) <= MAX_PROMPT_LINE_LENGTH for line in extra)


def test_drive_guidance_sanitizer_strips_engineering_keys() -> None:
    dirty = {
        "preferred_reply_actions": ["answer"],
        "exploration_bias": 0.88,
        "boredom_level": 0.9,
        "prompt_safe_lines": ["Prefer a small next step when helpful."],
    }
    clean = sanitize_drive_guidance_for_prompt(dirty)
    _assert_drive_guidance_prompt_safe(clean)
    assert clean.get("preferred_reply_actions") == ["answer"]
    assert "prompt_safe_lines" in clean


def test_thinking_prompt_excludes_boredom_internals(tmp_path: Path) -> None:
    state = default_m13_drive_state()
    state["boredom"] = normalize_boredom_state({"boredom_level": 0.9, "stale_turn_count": 10})
    safe = _prompt_safe_state({"m13_drive_state": state}, user_id="u1")
    boredom_evaluator = M13BoredomEvaluator()
    boredom = boredom_evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=2,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=state,
    )
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_exploration_guidance_into_control(memory_dynamics, boredom)
    safe_control = prompt_safe_control_guidance_for_thinking(memory_dynamics["control_guidance"])
    _, user_prompt = build_thinking_prompt(
        state=safe,
        user_text="嗯",
        conscious_plan={},
        retrieved_memories=[],
        turn_index=2,
        memory_guidance={"control_guidance": safe_control},
    )
    lowered = user_prompt.casefold()
    for marker in _BOREDOM_THINKING_PROMPT_MARKERS:
        assert marker.casefold() not in lowered


def test_progress_signal_ignores_empty_closed_expectation_ids() -> None:
    empty_list = _progress_signal(
        conscious_plan={},
        evidence_judgment=None,
        memory_dynamics={"expectation_impact": {"closed_expectation_ids": []}},
    )
    missing = _progress_signal(
        conscious_plan={},
        evidence_judgment=None,
        memory_dynamics={},
    )
    assert empty_list == missing
    assert _closed_expectation_progress({"closed_expectation_ids": []}) == 0.0
    assert _closed_expectation_progress({"closed_expectation_ids": ["exp_1"]}) == 0.2


def test_information_gain_semantic_hint_is_capped_and_structured_dominates() -> None:
    evaluator = M13BoredomEvaluator()
    llm = _BoredomSemanticLLM()
    baseline_state = default_m13_drive_state()
    baseline_state["boredom"] = normalize_boredom_state({"recent_plan_terms": ["status"]})
    hint_only = evaluator.evaluate(
        user_text="为什么？",
        user_id="u1",
        turn_id="t_hint",
        turn_index=2,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=baseline_state,
        llm=llm,
    )
    structured = evaluator.evaluate(
        user_text="为什么？",
        user_id="u1",
        turn_id="t_struct",
        turn_index=2,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        m13_state=baseline_state,
        llm=llm,
        m12_payload={
            "state_after": {
                "conflict_records": [
                    {"resolution_status": "open"},
                    {"resolution_status": "open"},
                    {"resolution_status": "open"},
                ]
            }
        },
    )
    assert hint_only.information_gain_proxy <= MAX_SEMANTIC_INFORMATION_GAIN_HINT + 0.01
    assert structured.information_gain_proxy > hint_only.information_gain_proxy
    assert any(event.get("type") == "M13BoredomUserTextAssessmentEvent" for event in hint_only.events)


def test_merge_exploration_includes_mode_line_when_hint_present() -> None:
    evaluator = M13BoredomEvaluator()
    boredom = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=4,
        conscious_plan={"memory_search_keywords": ["status", "update"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[{"id": "m0", "keywords": ["status"]}],
        m13_state=_stale_boredom_state(),
    )
    assert boredom.ordinary_language_hint
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_exploration_guidance_into_control(memory_dynamics, boredom)
    lines = memory_dynamics["control_guidance"]["drive_guidance"]["prompt_safe_lines"]  # type: ignore[index]
    assert any("exploration tendency" in line.casefold() for line in lines)


def test_mvp_runtime_wires_boredom_without_visible_diagnostic(tmp_path: Path) -> None:
    class ShortLLM(_BoredomSemanticLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt:
                return {
                    "expectation_results": [],
                    "memory_search_keywords": ["status"],
                    "temporal_assessment": {},
                }
            if BOREDOM_USER_TEXT_ASSESSOR_MARKER in system_prompt:
                return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            return {"reply": "好的。", "reply_action": "answer", "llm_thinking_result": {}}

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "p_boredom"), llm=ShortLLM())
    result = runtime.run_turn("嗯", turn_index=0)
    diag = result.diagnostics.get("m13_boredom_evaluation", {})
    assert isinstance(diag, dict)
    assert "boredom_level" not in diag
    forbidden = ("boredom_level", "我很无聊", "i am bored")
    for term in forbidden:
        assert term not in result.reply.casefold()
