from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import (
    CANDIDATE_REPLY_ACTIONS,
    M13DriveEvaluator,
    apply_post_turn_m13_state,
    build_topic_fingerprint,
    collect_allowed_reply_actions,
    default_m13_drive_state,
    merge_drive_guidance_into_control,
    normalize_m13_drive_state,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    SYSTEM_FILE_DEFAULTS,
    build_thinking_prompt,
)


def _seed_pattern(
    state: dict[str, object],
    *,
    action: str,
    user_id: str,
    habit_precision: float,
    success: int = 3,
    support: int = 4,
) -> None:
    patterns = state.setdefault("path_patterns_by_action", [])
    assert isinstance(patterns, list)
    patterns.append(
        {
            "action": action,
            "user_id": user_id,
            "topic_fingerprint": "python|prototype",
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
    _seed_pattern(state, action="answer", user_id="alice", habit_precision=0.55)
    evaluator = M13DriveEvaluator()
    conscious = {"memory_search_keywords": ["python", "prototype"]}
    memory_dynamics = {
        "recall_query": {"semantic_terms": ["python"]},
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        },
    }
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
    _seed_pattern(state, action="clarify", user_id="bob", habit_precision=0.2)
    row = state["path_patterns_by_action"][0]
    row["mean_control_cost_discount"] = 0.45
    evaluator = M13DriveEvaluator()
    memory_dynamics = {
        "recall_query": {},
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        },
    }
    result = evaluator.evaluate(
        user_text="这是什么？",
        user_id="bob",
        turn_id="turn_0001",
        turn_index=0,
        conscious_plan={},
        memory_dynamics=memory_dynamics,
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    assert result.top_behavioral_pull_action in CANDIDATE_REPLY_ACTIONS
    assert "drive_guidance" not in str(result.events[0])


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
        user_text="你好",
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


def test_prompt_receives_drive_guidance_not_raw_scores() -> None:
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
        m13_state=default_m13_drive_state(),
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

    _, user_prompt = build_thinking_prompt(
        state={},
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
    """Eight deterministic turns: traction rises; guidance stays advisory-only."""
    state = default_m13_drive_state()
    evaluator = M13DriveEvaluator()
    user_id = "repeat_user"
    memory_dynamics = {
        "recall_query": {"semantic_terms": ["status"]},
        "control_guidance": {
            "sharing_policy": {"allow_direct_disclosure": True, "allow_abstract_sharing": True},
            "reply_contract": {},
        },
    }
    conscious = {"memory_search_keywords": ["status", "update"]}
    pulls: list[float] = []
    for turn in range(8):
        evaluation = evaluator.evaluate(
            user_text="项目 status update 呢",
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn,
            conscious_plan=conscious,
            memory_dynamics=memory_dynamics,
            retrieved_memories=[],
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
            selected_action="answer",
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan={"expectation_results": [{"status": "confirmed"}]},
            memory_candidates_applied=[{"id": f"m{turn}"}],
        )
        guidance: dict[str, object] = {"control_guidance": dict(memory_dynamics["control_guidance"])}  # type: ignore[arg-type]
        merge_drive_guidance_into_control(guidance, evaluation)  # type: ignore[arg-type]
        drive = guidance["control_guidance"]["drive_guidance"]  # type: ignore[index]
        assert drive.get("advisory_only") is True
    assert pulls[-1] >= pulls[0]
    traction = state["traction_by_action"].get("answer|repeat_user", 0.0)
    assert traction > 0.0
