from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import (
    CANDIDATE_REPLY_ACTIONS,
    MAX_PATH_PATTERNS,
    PATTERN_DECAY_PER_TURN,
    M13DriveEvaluator,
    _bounded_float,
    apply_post_turn_m13_state,
    build_topic_fingerprint,
    collect_allowed_reply_actions,
    default_m13_drive_state,
    evict_path_patterns,
    merge_drive_guidance_into_control,
    normalize_m13_drive_state,
    prompt_safe_m13_state_summary,
    prompt_safe_m13_turn_diagnostics,
    resolve_m13_safety_repair,
    should_trigger_m13_rollback,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    SYSTEM_FILE_DEFAULTS,
    _prompt_safe_state,
    build_thinking_prompt,
)

_M13_INTERNAL_MARKERS = (
    "habit_precision",
    "success_proxy_count",
    "failure_proxy_count",
    "mean_control_cost_discount",
    "traction_by_action",
    "cue_precision_by_topic",
    "relation_path_precision",
    "path_patterns_by_action",
    "behavioral_pull",
)


def _seed_pattern(
    state: dict[str, object],
    *,
    action: str,
    user_id: str,
    habit_precision: float,
    topic_fingerprint: str = "python|prototype",
    success: int = 3,
    support: int = 4,
) -> None:
    patterns = state.setdefault("path_patterns_by_action", [])
    assert isinstance(patterns, list)
    patterns.append(
        {
            "action": action,
            "user_id": user_id,
            "topic_fingerprint": topic_fingerprint,
            "support_count": support,
            "success_proxy_count": success,
            "failure_proxy_count": 0,
            "habit_precision": habit_precision,
            "mean_control_cost_discount": 0.1,
            "last_seen_turn": 1,
            "source_evidence_ids": ["ltm_1"],
            "status": "active",
        }
    )


def test_m13_drive_state_defaults_load_for_existing_persona(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "persona")
    state = store.load()
    assert "m13_drive_state" in SYSTEM_FILE_DEFAULTS
    assert "m13_drive_state" in state
    normalized = normalize_m13_drive_state(state.get("m13_drive_state"))
    assert normalized["version"] == 1
    assert normalized["recent_action_trace"] == []
    assert normalize_m13_drive_state(None)["version"] == 1


def test_behavioral_pull_increases_for_repeated_success_proxy_action() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    conscious = {"memory_search_keywords": ["python", "prototype"]}
    memory_dynamics = {
        "recall_query": {"semantic_terms": ["python"]},
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        },
    }
    topic = build_topic_fingerprint(
        conscious_plan=conscious,
        memory_dynamics=memory_dynamics,
        user_text="继续聊 Python 原型",
    )
    _seed_pattern(state, action="answer", user_id="alice", habit_precision=0.55, topic_fingerprint=topic)
    result = evaluator.evaluate(
        user_text="继续聊 Python 原型",
        user_id="alice",
        turn_id="turn_0002",
        turn_index=1,
        conscious_plan=conscious,
        memory_dynamics=memory_dynamics,
        retrieved_memories=[
            {"id": "ltm_1", "keywords": ["python", "prototype"], "kind": "preference", "content": "偏好 Python"},
        ],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
        entity_binding={},
        evidence_judgment={"allowed_reply_actions": ["direct_share"]},
    )
    answer_pull = result.scores_by_action["answer"]["behavioral_pull"]
    clarify_pull = result.scores_by_action.get("clarify", {}).get("behavioral_pull", 0.0)
    assert answer_pull > clarify_pull
    assert result.top_behavioral_pull_action == "answer"


def test_behavioral_pull_reduces_control_cost_without_forcing_action() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    memory_dynamics = {
        "recall_query": {"semantic_terms": ["what", "meaning"]},
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        },
    }
    conscious = {"memory_search_keywords": ["what", "meaning"]}
    topic = build_topic_fingerprint(
        conscious_plan=conscious,
        memory_dynamics=memory_dynamics,
        user_text="这是什么？",
    )
    _seed_pattern(state, action="clarify", user_id="bob", habit_precision=0.2, topic_fingerprint=topic)
    row = state["path_patterns_by_action"][0]
    row["mean_control_cost_discount"] = 0.45
    result = evaluator.evaluate(
        user_text="这是什么？",
        user_id="bob",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan=conscious,
        memory_dynamics=memory_dynamics,
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    clarify_pull = result.scores_by_action.get("clarify", {}).get("behavioral_pull", 0.0)
    answer_pull = result.scores_by_action.get("answer", {}).get("behavioral_pull", 0.0)
    assert clarify_pull > answer_pull
    assert result.top_behavioral_pull_action == "clarify"
    assert result.selected_action == "clarify"


def test_m13_evaluation_event_is_added_to_mvp_bus_diagnostics(tmp_path: Path) -> None:
    class TurnLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt:
                return {
                    "pending_expectations_to_verify": [],
                    "expectation_results": [],
                    "current_task": "回应",
                    "next_task": "",
                    "memory_search_keywords": ["python"],
                    "needs_self_cognition_update": False,
                    "temporal_assessment": {"time_gap_label": "first_turn"},
                }
            if "思考与回复模块" in system_prompt:
                return {
                    "thought_type": "short",
                    "llm_thinking_result": {"debug_summary": "ok"},
                    "reply": "好的。",
                    "reply_action": "answer",
                }
            return {}

    store = MVPStateStore(tmp_path / "p")
    store.ensure_files()
    runtime = MVPDialogueRuntime(store=store, llm=TurnLLM())
    result = runtime.run_turn("说说 Python", turn_index=0, speaker_name="测试")
    bus_types = [msg.get("type") for msg in result.diagnostics.get("bus_messages", [])]
    assert "M13DriveEvaluationEvent" in bus_types
    assert result.diagnostics.get("m13_drive_evaluation")


def test_m13_patch_commit_records_source_reason_confidence() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="项目进展如何",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["项目", "进展"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["项目", "进展"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[{"id": "m1", "keywords": ["项目"], "content": "上周讨论过项目"}],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, events = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action=evaluation.top_behavioral_pull_action,
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=[{"id": "m1"}],
    )
    proposals = [e for e in events if e.get("type") == "M13DrivePatchProposal"]
    commits = [e for e in events if e.get("type") == "M13DrivePatchCommit"]
    assert proposals
    assert commits
    assert proposals[0].get("reason")
    assert proposals[0].get("confidence", 0) >= 0.6
    assert commits[0].get("owner") == "MVPDialogueRuntime"
    assert updated.get("last_patch_id")


def test_uncertain_outcome_does_not_increment_failure_proxy_count() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="maybe",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={},
        memory_dynamics={
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            }
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, _ = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action="answer",
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={},
        memory_candidates_applied=[],
    )
    pattern = updated["path_patterns_by_action"][0]
    assert int(pattern.get("failure_proxy_count", 0) or 0) == 0
    assert pattern.get("success_proxy_count", 0) == 0


def test_cold_start_silent_turn_does_not_strengthen_habit() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="hi",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={},
        memory_dynamics={
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            }
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, events = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action="answer",
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={},
        memory_candidates_applied=[],
    )
    commits = [e for e in events if e.get("type") == "M13DrivePatchCommit"]
    assert not commits
    patterns = updated.get("path_patterns_by_action", [])
    assert patterns
    assert float(patterns[0].get("habit_precision", 0.0) or 0.0) == 0.0
    assert int(patterns[0].get("success_proxy_count", 0) or 0) == 0


def test_prompt_receives_drive_guidance_not_raw_scores() -> None:
    seeded = default_m13_drive_state()
    _seed_pattern(seeded, action="answer", user_id="u1", habit_precision=0.62)
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="项目进展如何",
        user_id="u1",
        turn_id="t1",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["项目"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["项目"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=seeded,
    )
    memory_dynamics: dict[str, object] = {
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        }
    }
    merge_drive_guidance_into_control(memory_dynamics, evaluation)
    guidance = memory_dynamics["control_guidance"]["drive_guidance"]  # type: ignore[index]
    assert "preferred_reply_actions" in guidance
    assert "behavioral_pull" not in json.dumps(guidance, ensure_ascii=False)
    assert "habit_precision" not in json.dumps(guidance, ensure_ascii=False)

    safe_state = _prompt_safe_state({"m13_drive_state": seeded}, user_id="u1")
    _, user_prompt = build_thinking_prompt(
        state=safe_state,
        user_text="项目进展如何",
        conscious_plan={},
        retrieved_memories=[],
        turn_index=0,
        memory_guidance={
            "control_guidance": memory_dynamics["control_guidance"],
        },
    )
    assert "drive_guidance" in user_prompt or "preferred_reply_actions" in user_prompt
    assert "behavioral_pull=" not in user_prompt
    for marker in _M13_INTERNAL_MARKERS:
        assert marker not in user_prompt


def test_thinking_prompt_does_not_contain_m13_internal_keys() -> None:
    state = default_m13_drive_state()
    _seed_pattern(state, action="answer", user_id="alice", habit_precision=0.62)
    safe = _prompt_safe_state({"m13_drive_state": state}, user_id="alice")
    summary = safe["m13_drive_state"]
    assert "path_patterns_by_action" not in summary
    assert "habit_precision" not in json.dumps(summary, ensure_ascii=False)
    _, user_prompt = build_thinking_prompt(
        state=safe,
        user_text="继续聊 Python",
        conscious_plan={},
        retrieved_memories=[],
        turn_index=0,
    )
    for marker in _M13_INTERNAL_MARKERS:
        assert marker not in user_prompt


def test_prompt_safe_m13_state_summary_omits_pattern_tables() -> None:
    state = default_m13_drive_state()
    _seed_pattern(state, action="answer", user_id="bob", habit_precision=0.4)
    state["traction_by_action"]["answer|bob"] = 0.4
    summary = prompt_safe_m13_state_summary(state, user_id="bob")
    assert "traction_summary_for_user" in summary
    assert "path_patterns_by_action" not in summary
    assert "habit_precision" not in json.dumps(summary, ensure_ascii=False)


def test_visible_reply_does_not_expose_behavioral_pull_terms(tmp_path: Path) -> None:
    class ShortLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt:
                return {
                    "expectation_results": [],
                    "memory_search_keywords": [],
                    "temporal_assessment": {},
                }
            return {"reply": "我明白了。", "reply_action": "answer", "llm_thinking_result": {}}

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "p2"), llm=ShortLLM())
    result = runtime.run_turn("嗯", turn_index=0)
    forbidden = ("behavioral_pull", "habit_precision", "policy precision", "free energy", "candidate cost")
    lowered = result.reply.casefold()
    for term in forbidden:
        assert term not in lowered


def test_drive_guidance_cannot_enable_blocked_reply_action() -> None:
    state = default_m13_drive_state()
    _seed_pattern(state, action="self_disclose", user_id="u1", habit_precision=0.95)
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="秘密是什么",
        user_id="u1",
        turn_id="t1",
        turn_index=0,
        conscious_plan={},
        memory_dynamics={
            "control_guidance": {
                "sharing_policy": {
                    "allow_direct_disclosure": False,
                    "allow_abstract_sharing": True,
                },
                "reply_contract": {},
            }
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
        evidence_judgment={"allowed_reply_actions": ["deflect", "deny_knowledge"]},
    )
    memory_dynamics = {
        "control_guidance": {
            "sharing_policy": {
                "allow_direct_disclosure": False,
                "allow_abstract_sharing": True,
            },
            "reply_contract": {},
        }
    }
    merge_drive_guidance_into_control(memory_dynamics, evaluation, evidence_judgment={"allowed_reply_actions": ["deflect", "deny_knowledge"]})
    preferred = memory_dynamics["control_guidance"]["drive_guidance"]["preferred_reply_actions"]  # type: ignore[index]
    allowed = collect_allowed_reply_actions(
        evidence_judgment={"allowed_reply_actions": ["deflect", "deny_knowledge"]},
        memory_dynamics=memory_dynamics,
    )
    assert "self_disclose" not in preferred
    assert set(preferred).issubset(allowed)


def test_topic_fingerprint_is_deterministic_without_llm() -> None:
    conscious = {"memory_search_keywords": ["Python", "原型"]}
    dynamics = {"recall_query": {"semantic_terms": ["python"]}}
    binding = {"target_person": "小明", "pronoun_bindings": {"他": "小明"}}
    memories = [{"keywords": ["偏好"]}]
    first = build_topic_fingerprint(
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        entity_binding=binding,
        retrieved_memories=memories,
        user_text="Python 原型怎么做",
    )
    second = build_topic_fingerprint(
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        entity_binding=binding,
        retrieved_memories=memories,
        user_text="Python 原型怎么做",
    )
    assert first == second
    assert first.count("|") <= 7
    empty = build_topic_fingerprint()
    assert empty == "topic:unknown"


def test_topic_fingerprint_source_priority_preserves_order() -> None:
    fp = build_topic_fingerprint(
        conscious_plan={"memory_search_keywords": ["alpha", "beta"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["gamma"]}},
        user_text="delta epsilon",
    )
    terms = fp.split("|")
    assert terms.index("alpha") < terms.index("gamma")
    assert terms.index("gamma") < terms.index("delta")


def test_behavioral_pull_multiturn_repetition_scenario() -> None:
    """Repeated same-topic turns strengthen traction; a new topic does not inherit it."""
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    user_id = "repeat_user"
    base_control = {
        "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
        "reply_contract": {},
    }
    status_dynamics = {
        "recall_query": {"semantic_terms": ["status", "update"]},
        "control_guidance": dict(base_control),
    }
    status_conscious = {"memory_search_keywords": ["status", "update"]}
    pulls: list[float] = []
    for turn in range(8):
        evaluation = evaluator.evaluate(
            user_text="项目 status update 呢",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn,
            conscious_plan=status_conscious,
            memory_dynamics=status_dynamics,
            retrieved_memories=[{"id": f"m{turn}", "keywords": ["status"]}],
            response_style_prior={},
            habit_traits={},
            relationship_value_context={},
            m13_state=state,
        )
        pulls.append(evaluation.scores_by_action["answer"]["behavioral_pull"])
        state, _ = apply_post_turn_m13_state(
            state,
            evaluation=evaluation,
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn,
            selected_action=evaluation.top_behavioral_pull_action,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan={"expectation_results": [{"status": "confirmed"}]},
            memory_candidates_applied=[{"id": f"m{turn}"}],
        )
        guidance: dict[str, object] = {"control_guidance": dict(status_dynamics["control_guidance"])}  # type: ignore[arg-type]
        merge_drive_guidance_into_control(guidance, evaluation)  # type: ignore[arg-type]
        drive = guidance["control_guidance"]["drive_guidance"]  # type: ignore[index]
        assert drive.get("advisory_only") is True
    assert pulls[-1] > pulls[0]
    status_traction = state["traction_by_action"].get("answer|repeat_user", 0.0)
    assert status_traction > 0.0

    travel_dynamics = {
        "recall_query": {"semantic_terms": ["travel", "plan"]},
        "control_guidance": dict(base_control),
    }
    travel_eval = evaluator.evaluate(
        user_text="travel plan for next week",
        user_id=user_id,
        turn_id="turn_0005",
        turn_index=4,
        conscious_plan={"memory_search_keywords": ["travel", "plan"]},
        memory_dynamics=travel_dynamics,
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    travel_pull = travel_eval.scores_by_action["answer"]["behavioral_pull"]
    assert travel_pull < pulls[-1]
    assert len(pulls) == 8


def test_resolve_m13_safety_repair_matches_rollback_repair_signals() -> None:
    assert resolve_m13_safety_repair(reply_validation={"changed": True}) is True
    assert (
        resolve_m13_safety_repair(
            post_reply_observer={"needs_followup": True, "followup_type": "clarify"}
        )
        is True
    )
    rollback, reason = should_trigger_m13_rollback(
        reply_validation={"changed": True},
        post_reply_observer={"needs_followup": True, "followup_type": "clarify"},
        safety_repair=resolve_m13_safety_repair(
            reply_validation={"changed": True},
            post_reply_observer={"needs_followup": True, "followup_type": "clarify"},
        ),
    )
    assert rollback is True
    assert reason == "rollback_safety_repair"


def test_needs_followup_without_type_does_not_trigger_safety_repair() -> None:
    assert (
        resolve_m13_safety_repair(
            post_reply_observer={"needs_followup": True, "followup_type": "none"}
        )
        is False
    )
    rollback, _ = should_trigger_m13_rollback(
        post_reply_observer={"needs_followup": True, "followup_type": "none"},
        safety_repair=False,
    )
    assert rollback is False


def test_should_trigger_m13_rollback_on_expectation_violated() -> None:
    rollback, reason = should_trigger_m13_rollback(
        conscious_plan={"expectation_results": [{"status": "violated"}]},
    )
    assert rollback is True
    assert reason == "rollback_expectation_violated"


def test_rollback_reverses_recent_positive_habit_patch() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="项目进展如何",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["项目", "进展"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["项目", "进展"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[{"id": "m1", "keywords": ["项目"], "content": "上周讨论过项目"}],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    state, _ = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action=evaluation.top_behavioral_pull_action,
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=[{"id": "m1"}],
    )
    pattern = state["path_patterns_by_action"][0]
    habit_after_positive = float(pattern["habit_precision"])
    assert habit_after_positive > 0.0

    state, rollback_events = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0002",
        turn_index=1,
        selected_action=evaluation.top_behavioral_pull_action,
        reply_validation={"changed": True},
        post_reply_observer={"needs_followup": False},
        conscious_plan={},
        memory_candidates_applied=[],
        safety_repair=True,
    )
    rolled_back = state["path_patterns_by_action"][0]
    assert float(rolled_back["habit_precision"]) < habit_after_positive
    rollback_commits = [
        e for e in rollback_events if e.get("type") == "M13DrivePatchCommit"
    ]
    assert rollback_commits
    assert str(rollback_commits[0].get("reason", "")).startswith("rollback_")


def test_path_patterns_evict_oldest_when_over_max() -> None:
    patterns: list[dict[str, object]] = []
    for index in range(MAX_PATH_PATTERNS + 5):
        patterns.append(
            {
                "action": "answer",
                "user_id": f"user_{index}",
                "topic_fingerprint": f"topic_{index}",
                "support_count": 1,
                "success_proxy_count": 0,
                "failure_proxy_count": 0,
                "habit_precision": 0.1,
                "mean_control_cost_discount": 0.0,
                "last_seen_turn": index,
                "source_evidence_ids": [],
                "status": "active",
            }
        )
    evict_path_patterns(patterns)
    assert len(patterns) == MAX_PATH_PATTERNS
    assert int(patterns[0]["last_seen_turn"]) == 5
    assert int(patterns[-1]["last_seen_turn"]) == MAX_PATH_PATTERNS + 4


def test_relationship_value_context_boosts_empathize() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    kwargs = dict(
        user_text="我最近压力很大",
        user_id="bob",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["压力"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["压力"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        m13_state=state,
    )
    plain = evaluator.evaluate(**kwargs, relationship_value_context={})
    with_relationship = evaluator.evaluate(
        **kwargs,
        relationship_value_context={
            "active_relationship_value_memories": [
                {
                    "summary": "用户希望被理解",
                    "prediction_constraint": "优先共情而不是说教",
                }
            ],
        },
    )
    assert (
        with_relationship.scores_by_action["empathize"]["behavioral_pull"]
        > plain.scores_by_action["empathize"]["behavioral_pull"]
    )


def test_memory_support_prefers_kind_aligned_memories() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    memories = [
        {"id": "f1", "kind": "fact", "keywords": ["压力"], "content": "事实片段"},
        {"id": "r1", "kind": "relationship", "keywords": ["压力"], "content": "关系片段"},
    ]
    empathize = evaluator.evaluate(
        user_text="压力很大",
        user_id="u1",
        turn_id="t1",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["压力"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["压力"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=memories,
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    answer = evaluator.evaluate(
        user_text="压力很大",
        user_id="u1",
        turn_id="t1",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["压力"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["压力"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=memories,
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    assert (
        empathize.scores_by_action["empathize"]["memory_support"]
        >= answer.scores_by_action["answer"]["memory_support"]
    )


def test_action_match_without_anchor_does_not_strengthen_habit() -> None:
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="随便聊聊",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={},
        memory_dynamics={
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            }
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, events = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action=evaluation.top_behavioral_pull_action,
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={},
        memory_candidates_applied=[],
    )
    commits = [e for e in events if e.get("type") == "M13DrivePatchCommit"]
    assert not commits
    pattern = updated["path_patterns_by_action"][0]
    assert float(pattern.get("habit_precision", 0.0) or 0.0) == 0.0


def test_prompt_safe_m13_turn_diagnostics_omits_internal_terms() -> None:
    state = default_m13_drive_state()
    _seed_pattern(state, action="answer", user_id="u1", habit_precision=0.5)
    evaluation = M13DriveEvaluator().evaluate(
        user_text="项目",
        user_id="u1",
        turn_id="t1",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["项目"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["项目"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    payload = json.dumps(prompt_safe_m13_turn_diagnostics(evaluation), ensure_ascii=False)
    for marker in _M13_INTERNAL_MARKERS:
        assert marker not in payload


def test_rollback_window_eviction_at_limit() -> None:
    state = default_m13_drive_state()
    for i in range(12):
        state.setdefault("rollback_window", []).append(
            {
                "patch_id": f"patch_{i}",
                "action": "answer",
                "user_id": "u1",
                "topic_fingerprint": f"topic_{i}",
                "previous_habit_precision": 0.3,
                "previous_control_discount": 0.1,
                "confidence": 0.7,
            }
        )
    # Simulate a positive update which appends and caps at 8.
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="test",
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={"memory_search_keywords": ["test"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["test"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[{"id": "m1", "keywords": ["test"]}],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, _ = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        turn_id="turn_0001",
        turn_index=0,
        selected_action="answer",
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=[{"id": "m1"}],
    )
    window = updated.get("rollback_window", [])
    assert len(window) <= 8


def test_pattern_decay_applied_to_existing_patterns() -> None:
    state = default_m13_drive_state()
    state["path_patterns_by_action"] = [
        {
            "action": "answer",
            "user_id": "u1",
            "topic_fingerprint": "old|topic",
            "support_count": 1,
            "success_proxy_count": 0,
            "failure_proxy_count": 0,
            "habit_precision": 0.5,
            "mean_control_cost_discount": 0.3,
            "last_seen_turn": 0,
            "source_evidence_ids": [],
            "status": "active",
        }
    ]
    evaluator = M13DriveEvaluator()
    evaluation = evaluator.evaluate(
        user_text="new topic",
        user_id="u2",
        turn_id="turn_0002",
        turn_index=1,
        conscious_plan={"memory_search_keywords": ["new"]},
        memory_dynamics={
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
            },
        },
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    updated, _ = apply_post_turn_m13_state(
        state,
        evaluation=evaluation,
        user_id="u2",
        turn_id="turn_0002",
        turn_index=1,
        selected_action="deflect",
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={},
        memory_candidates_applied=[],
    )
    old_pattern = updated["path_patterns_by_action"][0]
    assert float(old_pattern["habit_precision"]) < 0.5
    assert float(old_pattern["habit_precision"]) >= 0.5 - PATTERN_DECAY_PER_TURN - 1e-6
    assert float(old_pattern["mean_control_cost_discount"]) < 0.3


def test_boredom_suppressed_under_high_conflict_pressure() -> None:
    from segmentum.dialogue.runtime.m13_boredom import M13BoredomEvaluator

    state = default_m13_drive_state()
    evaluator = M13BoredomEvaluator()
    result = evaluator.evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={
            "recall_query": {"semantic_terms": ["status"]},
            "control_guidance": {
                "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
                "reply_contract": {},
                "conflict_level": 0.85,
                "repair_bias": 0.8,
                "clarification_bias": 0.75,
            },
        },
        retrieved_memories=[],
        m13_state=state,
    )
    assert result.exploration_suppressed is True
    assert result.boredom_band == "low"
