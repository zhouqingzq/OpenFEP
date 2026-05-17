from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import (
    M13DriveEvaluator,
    _bounded_float,
    apply_post_turn_m13_state,
    default_m13_drive_state,
    normalize_m13_drive_state,
)
from segmentum.dialogue.runtime.m13_reward import (
    MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA,
    MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA,
    MAX_SINGLE_TURN_TOLERANCE_DELTA,
    M13RewardEvaluator,
    apply_post_turn_m13_reward_state,
    apply_reward_pull_connection,
    compute_net_affective_reward_proxy,
    create_pending_settlement,
    default_affective_reward_proxy_state,
    evaluate_pre_turn_reward_proxy,
    list_assessable_pending_rows,
    merge_affective_guidance_into_control,
    normalize_affective_reward_proxy_state,
    normalize_user_reaction_assessment,
    observation_channels_from_bus,
    path_id_for,
    prompt_safe_m13_reward_diagnostics,
    prompt_safe_m13_reward_ui_labels,
    settle_pending_m13_actions,
)
from segmentum.dialogue.runtime.m13_boredom import M13BoredomEvaluator
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore, build_thinking_prompt

_FORBIDDEN_UI_TERMS = (
    "reward",
    "tolerance",
    "opponent process",
    "addiction",
    "craving",
    "behavioral_pull",
)
_FORBIDDEN_PROMPT_TERMS = (
    *_FORBIDDEN_UI_TERMS,
    "net_affective_reward_proxy",
    "pleasure decreased",
    "I am addicted",
)


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


def _llm_uptake_assessment() -> dict[str, object]:
    return {"reaction": "uptake", "confidence": 0.72, "reason_codes": ["semantic_uptake"]}


def _llm_correction_assessment() -> dict[str, object]:
    return {"reaction": "correction", "confidence": 0.8, "reason_codes": ["semantic_correction"]}


def _uptake_assessments_for_state(state: dict[str, object], *, turn_index: int) -> dict[str, object]:
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    assessments: dict[str, object] = {}
    for row in list_assessable_pending_rows(reward, turn_index=turn_index):
        pending_id = str(row.get("pending_id", ""))
        if pending_id:
            assessments[pending_id] = _llm_uptake_assessment()
    return assessments


def _status_inputs(turn: int) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    conscious = {"memory_search_keywords": ["status", "update"]}
    dynamics = {
        "recall_query": {"semantic_terms": ["status", "update"]},
        "control_guidance": _base_control(),
    }
    memories = [{"id": f"m{turn}", "keywords": ["status"], "kind": "episode"}]
    return conscious, dynamics, memories


def test_reward_proxy_state_defaults_are_backward_compatible() -> None:
    state = default_m13_drive_state()
    assert "affective_reward_proxy" in state
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    assert reward["engineering_proxy_label"] == "mvp_local_affective_reward_proxy"
    legacy = {
        **default_m13_drive_state(),
        "pending_settlements": [{"pending_id": "legacy_1", "prior_turn_index": 0}],
        "tolerance_by_path": [{"path_id": "answer|u1|status", "tolerance": 0.2}],
    }
    normalized = normalize_m13_drive_state(legacy)
    nested = normalize_affective_reward_proxy_state(normalized["affective_reward_proxy"])
    assert nested["pending_settlements"]
    assert nested["tolerance_by_path"]
    assert normalized["pending_settlements"] == []


def test_next_turn_settlement_records_uncertain_when_evidence_insufficient() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["pending_settlements"] = [
        create_pending_settlement(
            turn_index=1,
            action="answer",
            topic_fingerprint="status|update",
            reply_summary="ok",
            predicted_reward=0.3,
            predicted_relief=0.1,
            information_gain_proxy=0.1,
            evidence_refs=[],
            reply_validation={},
            post_reply_observer={},
            conscious_plan={},
            memory_candidates_applied=[],
            evidence_judgment=None,
            safety_repair=False,
            repetition_pressure=0.0,
            conflict_level=0.0,
        )
    ]
    state["affective_reward_proxy"] = reward
    updated, settlements, events = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
    )
    assert settlements
    assert settlements[0].outcome_band == "uncertain"
    assert "insufficient_settlement_evidence" in settlements[0].reason_codes
    assert any(event.get("type") == "M13RewardSettlementEvent" for event in events)
    assert updated["affective_reward_proxy"]["pending_settlements"] == []


def test_stale_pending_settlement_expires_without_update() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["pending_settlements"] = [
        {
            **create_pending_settlement(
                turn_index=0,
                action="answer",
                topic_fingerprint="old|topic",
                reply_summary="old",
                predicted_reward=0.4,
                predicted_relief=0.1,
                information_gain_proxy=0.0,
                evidence_refs=[],
                reply_validation={},
                post_reply_observer={},
                conscious_plan={},
                memory_candidates_applied=[],
                evidence_judgment=None,
                safety_repair=False,
                repetition_pressure=0.0,
                conflict_level=0.0,
            ),
            "expires_after_turns": 2,
        }
    ]
    state["affective_reward_proxy"] = reward
    updated, settlements, events = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=5,
        turn_id="turn_0006",
    )
    assert not settlements
    assert any(event.get("type") == "M13RewardSettlementExpired" for event in events)
    assert updated["affective_reward_proxy"]["pending_settlements"] == []


def test_normalize_user_reaction_assessment_rejects_invalid_reaction() -> None:
    normalized = normalize_user_reaction_assessment(
        {"reaction": "addicted", "confidence": 1.5, "reason_codes": ["x"] * 10}
    )
    assert normalized["reaction"] == "unclear"
    assert normalized["confidence"] <= 1.0
    assert len(normalized["reason_codes"]) <= 4


def test_polite_user_text_without_llm_assessment_does_not_imply_uptake() -> None:
    """Regression: regex keyword cues removed; short polite text alone is not uptake."""
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["pending_settlements"] = [
        create_pending_settlement(
            turn_index=1,
            action="answer",
            topic_fingerprint="status|update",
            reply_summary="summary",
            predicted_reward=0.3,
            predicted_relief=0.1,
            information_gain_proxy=0.0,
            evidence_refs=[],
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False, "followup_type": "none"},
            conscious_plan={},
            memory_candidates_applied=[],
            evidence_judgment=None,
            safety_repair=False,
            repetition_pressure=0.0,
            conflict_level=0.0,
        )
    ]
    state["affective_reward_proxy"] = reward
    _, settlements, _ = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
        user_reaction_assessments={},
    )
    assert settlements[0].outcome_band == "uncertain"
    assert "llm_user_uptake" not in settlements[0].reason_codes


def test_llm_uptake_assessment_can_support_positive_settlement() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    pending = create_pending_settlement(
        turn_index=1,
        action="answer",
        topic_fingerprint="status|update",
        reply_summary="summary",
        predicted_reward=0.3,
        predicted_relief=0.1,
        information_gain_proxy=0.0,
        evidence_refs=[],
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False, "followup_type": "none"},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        repetition_pressure=0.0,
        conflict_level=0.0,
    )
    reward["pending_settlements"] = [pending]
    state["affective_reward_proxy"] = reward
    _, settlements, _ = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
        user_reaction_assessments={str(pending["pending_id"]): _llm_uptake_assessment()},
    )
    assert settlements[0].outcome_band == "positive"
    assert "llm_user_uptake" in settlements[0].reason_codes


def test_repeated_success_raises_predicted_reward_and_tolerance() -> None:
    state = default_m13_drive_state()
    user_id = "u1"
    action = "answer"
    topic = "status|update"
    for turn in range(1, 5):
        reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
        reward["pending_settlements"] = [
            create_pending_settlement(
                turn_index=turn - 1,
                action=action,
                topic_fingerprint=topic,
                reply_summary="好的",
                predicted_reward=0.2,
                predicted_relief=0.1,
                information_gain_proxy=0.1,
                evidence_refs=[],
                reply_validation={"changed": False},
                post_reply_observer={"needs_followup": False, "followup_type": "none"},
                conscious_plan={"expectation_results": [{"status": "confirmed"}]},
                memory_candidates_applied=["m1"],
                evidence_judgment={"epistemic_stance": "known"},
                safety_repair=False,
                repetition_pressure=0.25,
                conflict_level=0.0,
            )
        ]
        state["affective_reward_proxy"] = reward
        state, settlements, _ = settle_pending_m13_actions(
            state,
            user_id=user_id,
            turn_index=turn,
            turn_id=f"turn_{turn:04d}",
            user_reaction_assessments=_uptake_assessments_for_state(state, turn_index=turn),
        )
        assert settlements[-1].outcome_band == "positive"
    path_rows = state["affective_reward_proxy"]["tolerance_by_path"]
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic)
    row = next(item for item in path_rows if item.get("path_id") == pid)
    assert float(row["predicted_reward"]) >= 0.2
    assert float(row["tolerance"]) > 0.0
    assert int(row.get("support_count", 0) or 0) >= 4


def test_net_reward_proxy_drops_when_reward_becomes_predicted() -> None:
    predicted = 0.5
    baseline = 0.35
    tolerance = 0.2
    high_gain_net = compute_net_affective_reward_proxy(
        observed_reward_proxy=0.55,
        relief_proxy=0.1,
        information_gain_proxy=0.2,
        predicted_reward=0.1,
        reward_baseline=baseline,
        tolerance=0.05,
        opponent_strength=0.0,
    )
    low_gain_net = compute_net_affective_reward_proxy(
        observed_reward_proxy=0.55,
        relief_proxy=0.1,
        information_gain_proxy=0.2,
        predicted_reward=predicted,
        reward_baseline=baseline,
        tolerance=tolerance,
        opponent_strength=0.0,
    )
    assert low_gain_net < high_gain_net


def test_net_reward_proxy_can_be_negative_when_penalties_dominate() -> None:
    net = compute_net_affective_reward_proxy(
        observed_reward_proxy=0.1,
        relief_proxy=0.0,
        information_gain_proxy=0.0,
        predicted_reward=0.8,
        reward_baseline=0.35,
        tolerance=0.3,
        opponent_strength=0.2,
    )
    assert net < 0.0


def test_bare_ok_without_prior_signals_stays_uncertain() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["pending_settlements"] = [
        create_pending_settlement(
            turn_index=1,
            action="answer",
            topic_fingerprint="status|update",
            reply_summary="ok",
            predicted_reward=0.3,
            predicted_relief=0.1,
            information_gain_proxy=0.0,
            evidence_refs=[],
            reply_validation={},
            post_reply_observer={},
            conscious_plan={},
            memory_candidates_applied=[],
            evidence_judgment=None,
            safety_repair=False,
            repetition_pressure=0.0,
            conflict_level=0.0,
        )
    ]
    state["affective_reward_proxy"] = reward
    _, settlements, _ = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
    )
    assert settlements[0].outcome_band == "uncertain"


def test_negative_settlement_rolls_back_habit_and_lowers_predicted() -> None:
    state = default_m13_drive_state()
    user_id = "u1"
    action = "answer"
    topic = "status|update"
    patterns = [
        {
            "action": action,
            "user_id": user_id,
            "topic_fingerprint": topic,
            "support_count": 3,
            "habit_precision": 0.42,
            "mean_control_cost_discount": 0.2,
            "last_seen_turn": 1,
        }
    ]
    state["path_patterns_by_action"] = patterns
    state["rollback_window"] = [
        {
            "patch_id": "m13_patch_test",
            "action": action,
            "user_id": user_id,
            "topic_fingerprint": topic,
            "previous_habit_precision": 0.32,
            "previous_control_discount": 0.16,
            "confidence": 0.8,
        }
    ]
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    pending = create_pending_settlement(
        turn_index=1,
        action=action,
        topic_fingerprint=topic,
        reply_summary="wrong",
        predicted_reward=0.55,
        predicted_relief=0.2,
        information_gain_proxy=0.0,
        evidence_refs=[],
        reply_validation={"changed": True},
        post_reply_observer={"needs_followup": True, "followup_type": "clarify"},
        conscious_plan={"expectation_results": [{"status": "violated"}]},
        memory_candidates_applied=[],
        evidence_judgment=None,
        safety_repair=True,
        repetition_pressure=0.0,
        conflict_level=0.0,
    )
    reward["pending_settlements"] = [pending]
    state["affective_reward_proxy"] = reward
    updated, settlements, events = settle_pending_m13_actions(
        state,
        user_id=user_id,
        turn_index=2,
        turn_id="turn_0003",
        user_reaction_assessments={str(pending["pending_id"]): _llm_correction_assessment()},
    )
    assert settlements[0].outcome_band == "negative"
    assert "llm_user_correction" in settlements[0].reason_codes
    row = updated["path_patterns_by_action"][0]
    assert float(row["habit_precision"]) < 0.42
    assert float(row["habit_precision"]) >= 0.32
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic)
    path_reward = next(
        item for item in updated["affective_reward_proxy"]["tolerance_by_path"] if item["path_id"] == pid
    )
    assert float(path_reward["predicted_reward"]) < 0.55
    assert any(event.get("reason") == "rollback_negative_settlement" for event in events)


def test_observation_channels_from_bus_and_settlement_adjustment() -> None:
    bus = [
        {
            "type": "ObservationEvent",
            "channels": {
                "conflict_tension": 0.7,
                "emotional_tone": 0.4,
                "semantic_content": 0.5,
                "topic_novelty": 0.4,
            },
        }
    ]
    channels = observation_channels_from_bus(bus)
    assert channels["conflict_tension"] == 0.7
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["pending_settlements"] = [
        create_pending_settlement(
            turn_index=1,
            action="answer",
            topic_fingerprint="status|update",
            reply_summary="ok",
            predicted_reward=0.4,
            predicted_relief=0.1,
            information_gain_proxy=0.0,
            evidence_refs=[],
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False, "followup_type": "none"},
            conscious_plan={},
            memory_candidates_applied=[],
            evidence_judgment=None,
            safety_repair=False,
            repetition_pressure=0.0,
            conflict_level=0.0,
        )
    ]
    state["affective_reward_proxy"] = reward
    _, settlements, _ = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
        observation_channels=channels,
    )
    assert "observation_conflict_pressure" in settlements[0].reason_codes


def test_behavioral_pull_can_increase_while_reward_proxy_drops() -> None:
    drive_evaluator = M13DriveEvaluator()
    reward_evaluator = M13RewardEvaluator()
    boredom_evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    user_id = "pull_user"
    nets: list[float] = []
    pulls: list[float] = []
    for turn in range(9):
        conscious, dynamics, memories = _status_inputs(turn)
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
        boredom = boredom_evaluator.evaluate(
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
        if turn > 0:
            state, _, _ = settle_pending_m13_actions(
                state,
                user_id=user_id,
                turn_index=turn + 1,
                turn_id=f"turn_{turn:04d}",
                boredom_information_gain=boredom.information_gain_proxy,
                user_reaction_assessments=_uptake_assessments_for_state(state, turn_index=turn + 1),
            )
        action = drive_eval.top_behavioral_pull_action
        pull = float(drive_eval.scores_by_action[action]["behavioral_pull"])
        reward_eval = reward_evaluator.evaluate(
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            user_id=user_id,
            action=action,
            topic_fingerprint=drive_eval.topic_fingerprint,
            m13_state=state,
            conscious_plan=conscious,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False, "followup_type": "none"},
            memory_candidates_applied=["m1"],
            evidence_judgment={"epistemic_stance": "known"},
            safety_repair=False,
            information_gain_proxy=boredom.information_gain_proxy,
            repetition_pressure=boredom.repetition_pressure,
            conflict_level=0.0,
            behavioral_pull=pull,
        )
        nets.append(reward_eval.net_affective_reward_proxy)
        pulls.append(pull)
        state, _ = apply_post_turn_m13_state(
            state,
            evaluation=drive_eval,
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            selected_action=action,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan=conscious,
            memory_candidates_applied=["m1"],
        )
        state, _ = apply_post_turn_m13_reward_state(
            state,
            evaluation=reward_eval,
            user_id=user_id,
            action=action,
            topic_fingerprint=drive_eval.topic_fingerprint,
            turn_index=turn + 1,
            reply_summary="好的",
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan=conscious,
            memory_candidates_applied=["m1"],
            evidence_judgment={"epistemic_stance": "known"},
            safety_repair=False,
            repetition_pressure=boredom.repetition_pressure,
            conflict_level=0.0,
            behavioral_pull=pull,
        )
    assert min(nets[4:]) <= nets[0] - 0.08
    assert pulls[-1] >= pulls[0] - 0.01
    reward_state = state["affective_reward_proxy"]
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=drive_eval.topic_fingerprint)
    path_row = next(item for item in reward_state["tolerance_by_path"] if item.get("path_id") == pid)
    assert float(path_row.get("predicted_reward", 0.0)) >= 0.15
    assert float(path_row.get("tolerance", 0.0)) > 0.0


def test_repeated_path_reward_drops_while_pull_does_not_drop() -> None:
    """Core acceptance: repeated same-path turns lower net proxy without lowering pull."""
    drive_evaluator = M13DriveEvaluator()
    reward_evaluator = M13RewardEvaluator()
    boredom_evaluator = M13BoredomEvaluator()
    state = default_m13_drive_state()
    user_id = "repeat_user"
    nets: list[float] = []
    pulls: list[float] = []
    for turn in range(10):
        conscious, dynamics, memories = _status_inputs(turn)
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
        boredom = boredom_evaluator.evaluate(
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
        if turn > 0:
            state, _, _ = settle_pending_m13_actions(
                state,
                user_id=user_id,
                turn_index=turn + 1,
                turn_id=f"turn_{turn:04d}",
                boredom_information_gain=boredom.information_gain_proxy,
                user_reaction_assessments=_uptake_assessments_for_state(state, turn_index=turn + 1),
            )
        action = drive_eval.top_behavioral_pull_action
        pull = float(drive_eval.scores_by_action[action]["behavioral_pull"])
        reward_eval = reward_evaluator.evaluate(
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            user_id=user_id,
            action=action,
            topic_fingerprint=drive_eval.topic_fingerprint,
            m13_state=state,
            conscious_plan=conscious,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False, "followup_type": "none"},
            memory_candidates_applied=["m1"],
            evidence_judgment={"epistemic_stance": "known"},
            safety_repair=False,
            information_gain_proxy=boredom.information_gain_proxy,
            repetition_pressure=boredom.repetition_pressure,
            conflict_level=0.0,
            behavioral_pull=pull,
        )
        nets.append(reward_eval.net_affective_reward_proxy)
        pulls.append(pull)
        state, _ = apply_post_turn_m13_state(
            state,
            evaluation=drive_eval,
            user_id=user_id,
            turn_id=f"turn_{turn:04d}",
            turn_index=turn + 1,
            selected_action=action,
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan=conscious,
            memory_candidates_applied=["m1"],
        )
        state, _ = apply_post_turn_m13_reward_state(
            state,
            evaluation=reward_eval,
            user_id=user_id,
            action=action,
            topic_fingerprint=drive_eval.topic_fingerprint,
            turn_index=turn + 1,
            reply_summary="好的",
            reply_validation={"changed": False},
            post_reply_observer={"needs_followup": False},
            conscious_plan=conscious,
            memory_candidates_applied=["m1"],
            evidence_judgment={"epistemic_stance": "known"},
            safety_repair=False,
            repetition_pressure=boredom.repetition_pressure,
            conflict_level=0.0,
            behavioral_pull=pull,
        )
    assert nets[-1] < nets[1]
    assert pulls[-1] >= pulls[1] - 0.01
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=drive_eval.topic_fingerprint)
    path_row = next(
        item for item in state["affective_reward_proxy"]["tolerance_by_path"] if item.get("path_id") == pid
    )
    assert float(path_row.get("predicted_reward", 0.0)) >= 0.15
    assert float(path_row.get("tolerance", 0.0)) > 0.0


def test_opponent_strength_increases_caution_not_visible_self_punishment() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["opponent_strength"] = 0.5
    state["affective_reward_proxy"] = reward
    drive_evaluator = M13DriveEvaluator()
    drive_eval = drive_evaluator.evaluate(
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
    pre = evaluate_pre_turn_reward_proxy(
        turn_id="t1",
        turn_index=2,
        user_id="u1",
        m13_state=state,
        m13_evaluation=drive_eval,
        information_gain_proxy=0.1,
        repetition_pressure=0.2,
        conflict_level=0.0,
    )
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_affective_guidance_into_control(memory_dynamics, pre)
    guidance = memory_dynamics["control_guidance"]["affective_drive_guidance"]  # type: ignore[index]
    hint = str(guidance.get("ordinary_language_hint", ""))
    assert hint
    assert "punish" not in hint.casefold()
    assert "addict" not in hint.casefold()


def test_prompt_guidance_uses_plain_language_not_reward_jargon(tmp_path: Path) -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["opponent_strength"] = 0.4
    pid = path_id_for(action="answer", user_id="u1", topic_fingerprint="status|update")
    reward["tolerance_by_path"] = [
        {
            "path_id": pid,
            "action": "answer",
            "topic_fingerprint": "status|update",
            "predicted_reward": 0.5,
            "tolerance": 0.35,
            "support_count": 4,
            "last_updated_turn": 3,
        }
    ]
    state["affective_reward_proxy"] = reward
    drive_eval = M13DriveEvaluator().evaluate(
        user_text="嗯",
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        conscious_plan={"memory_search_keywords": ["status"]},
        memory_dynamics={"recall_query": {"semantic_terms": ["status"]}, "control_guidance": _base_control()},
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
        m13_state=state,
    )
    pre = evaluate_pre_turn_reward_proxy(
        turn_id="t1",
        turn_index=3,
        user_id="u1",
        m13_state=state,
        m13_evaluation=drive_eval,
        information_gain_proxy=0.05,
        repetition_pressure=0.35,
        conflict_level=0.0,
    )
    memory_dynamics: dict[str, object] = {"control_guidance": _base_control()}
    merge_affective_guidance_into_control(memory_dynamics, pre)
    blob = json.dumps(memory_dynamics["control_guidance"], ensure_ascii=False).casefold()
    for term in _FORBIDDEN_PROMPT_TERMS:
        assert term.casefold() not in blob


def test_m13_reward_patch_commit_is_auditable() -> None:
    evaluator = M13RewardEvaluator()
    state = default_m13_drive_state()
    evaluation = evaluator.evaluate(
        turn_id="t1",
        turn_index=2,
        user_id="u1",
        action="answer",
        topic_fingerprint="status|update",
        m13_state=state,
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False, "followup_type": "none"},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        information_gain_proxy=0.2,
        repetition_pressure=0.1,
        conflict_level=0.0,
        behavioral_pull=0.4,
    )
    _, events = apply_post_turn_m13_reward_state(
        state,
        evaluation=evaluation,
        user_id="u1",
        action="answer",
        topic_fingerprint="status|update",
        turn_index=2,
        reply_summary="好的",
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        repetition_pressure=0.1,
        conflict_level=0.0,
        behavioral_pull=0.4,
    )
    commits = [event for event in events if event.get("type") == "M13RewardPatchCommit"]
    assert commits


def test_ui_visible_labels_do_not_expose_reward_or_addiction_terms() -> None:
    labels = prompt_safe_m13_reward_ui_labels()
    blob = json.dumps(labels, ensure_ascii=False).casefold()
    for term in _FORBIDDEN_UI_TERMS:
        assert term.casefold() not in blob


def test_single_turn_update_deltas_are_bounded() -> None:
    old = 0.2
    new = _bounded_float(0.2 + MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA + 0.01)
    from segmentum.dialogue.runtime.m13_reward import _bounded_delta

    capped = _bounded_delta(old, new, max_delta=MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA)
    assert capped <= old + MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA + 1e-6
    tol_cap = _bounded_delta(0.1, 0.9, max_delta=MAX_SINGLE_TURN_TOLERANCE_DELTA)
    assert tol_cap <= 0.1 + MAX_SINGLE_TURN_TOLERANCE_DELTA + 1e-6
    opp_cap = _bounded_delta(0.1, 0.9, max_delta=MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA)
    assert opp_cap <= 0.1 + MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA + 1e-6


def test_turn_index_regression_clears_stale_pending_without_settlement() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    reward["last_seen_turn_index"] = 8
    reward["pending_settlements"] = [
        create_pending_settlement(
            turn_index=7,
            action="answer",
            topic_fingerprint="status|update",
            reply_summary="old",
            predicted_reward=0.4,
            predicted_relief=0.1,
            information_gain_proxy=0.0,
            evidence_refs=[],
            reply_validation={},
            post_reply_observer={},
            conscious_plan={},
            memory_candidates_applied=[],
            evidence_judgment=None,
            safety_repair=False,
            repetition_pressure=0.0,
            conflict_level=0.0,
        )
    ]
    state["affective_reward_proxy"] = reward
    updated, settlements, events = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=0,
        turn_id="turn_0001",
    )
    assert not settlements
    assert updated["affective_reward_proxy"]["pending_settlements"] == []
    assert any(event.get("reason") == "session_turn_index_regressed" for event in events)
    assert int(updated["affective_reward_proxy"]["settlement_generation"]) >= 1


def test_multiple_pending_use_distinct_assessments() -> None:
    state = default_m13_drive_state()
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    pending_a = create_pending_settlement(
        turn_index=1,
        action="answer",
        topic_fingerprint="topic|a",
        reply_summary="a",
        predicted_reward=0.3,
        predicted_relief=0.1,
        information_gain_proxy=0.0,
        evidence_refs=[],
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False, "followup_type": "none"},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        repetition_pressure=0.0,
        conflict_level=0.0,
    )
    pending_b = create_pending_settlement(
        turn_index=1,
        action="answer",
        topic_fingerprint="topic|b",
        reply_summary="b",
        predicted_reward=0.3,
        predicted_relief=0.1,
        information_gain_proxy=0.0,
        evidence_refs=[],
        reply_validation={"changed": True},
        post_reply_observer={"needs_followup": True, "followup_type": "clarify"},
        conscious_plan={"expectation_results": [{"status": "violated"}]},
        memory_candidates_applied=[],
        evidence_judgment=None,
        safety_repair=True,
        repetition_pressure=0.0,
        conflict_level=0.0,
    )
    reward["pending_settlements"] = [pending_a, pending_b]
    state["affective_reward_proxy"] = reward
    _, settlements, _ = settle_pending_m13_actions(
        state,
        user_id="u1",
        turn_index=2,
        turn_id="turn_0003",
        user_reaction_assessments={
            str(pending_a["pending_id"]): _llm_uptake_assessment(),
            str(pending_b["pending_id"]): _llm_correction_assessment(),
        },
    )
    assert len(settlements) == 2
    by_id = {item.pending_id: item for item in settlements}
    assert "llm_user_uptake" in by_id[str(pending_a["pending_id"])].reason_codes
    assert "llm_user_correction" in by_id[str(pending_b["pending_id"])].reason_codes


def test_apply_reward_pull_connection_boosts_habit_when_net_high() -> None:
    state = default_m13_drive_state()
    state["path_patterns_by_action"] = [
        {
            "action": "answer",
            "user_id": "u1",
            "topic_fingerprint": "status|update",
            "support_count": 2,
            "habit_precision": 0.4,
            "mean_control_cost_discount": 0.2,
            "last_seen_turn": 1,
        }
    ]
    evaluator = M13RewardEvaluator()
    evaluation = evaluator.evaluate(
        turn_id="t1",
        turn_index=2,
        user_id="u1",
        action="answer",
        topic_fingerprint="status|update",
        m13_state=state,
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False, "followup_type": "none"},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        information_gain_proxy=0.2,
        repetition_pressure=0.1,
        conflict_level=0.0,
        behavioral_pull=0.5,
    )
    assert evaluation.net_affective_reward_proxy >= 0.45
    updated = apply_reward_pull_connection(
        state,
        evaluation=evaluation,
        behavioral_pull=0.5,
    )
    row = updated["path_patterns_by_action"][0]
    assert float(row["habit_precision"]) > 0.4


def test_positive_settlement_caps_predicted_reward_delta_per_turn() -> None:
    state = default_m13_drive_state()
    user_id = "u1"
    action = "answer"
    topic = "status|update"
    reward = normalize_affective_reward_proxy_state(state["affective_reward_proxy"])
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic)
    reward["tolerance_by_path"] = [
        {
            "path_id": pid,
            "action": action,
            "topic_fingerprint": topic,
            "predicted_reward": 0.1,
            "tolerance": 0.0,
            "support_count": 1,
            "last_updated_turn": 1,
        }
    ]
    pending = create_pending_settlement(
        turn_index=1,
        action=action,
        topic_fingerprint=topic,
        reply_summary="ok",
        predicted_reward=0.1,
        predicted_relief=0.0,
        information_gain_proxy=0.0,
        evidence_refs=[],
        reply_validation={"changed": False},
        post_reply_observer={"needs_followup": False, "followup_type": "none"},
        conscious_plan={"expectation_results": [{"status": "confirmed"}]},
        memory_candidates_applied=["m1"],
        evidence_judgment={"epistemic_stance": "known"},
        safety_repair=False,
        repetition_pressure=0.0,
        conflict_level=0.0,
    )
    reward["pending_settlements"] = [pending]
    state["affective_reward_proxy"] = reward
    updated, settlements, _ = settle_pending_m13_actions(
        state,
        user_id=user_id,
        turn_index=2,
        turn_id="turn_0003",
        user_reaction_assessments={str(pending["pending_id"]): _llm_uptake_assessment()},
    )
    assert settlements[0].outcome_band == "positive"
    row = next(
        item
        for item in updated["affective_reward_proxy"]["tolerance_by_path"]
        if item.get("path_id") == pid
    )
    assert float(row["predicted_reward"]) <= 0.1 + MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA + 1e-6


def test_opponent_strength_raises_boredom_exploration_bias() -> None:
    low_state = default_m13_drive_state()
    high_state = default_m13_drive_state()
    high_reward = normalize_affective_reward_proxy_state(high_state["affective_reward_proxy"])
    high_reward["opponent_strength"] = 0.5
    high_state["affective_reward_proxy"] = high_reward
    drive_evaluator = M13DriveEvaluator()
    conscious = {"memory_search_keywords": ["status"]}
    dynamics = {
        "recall_query": {"semantic_terms": ["status"]},
        "control_guidance": _base_control(),
    }
    drive_kwargs = dict(
        user_text="还是 status",
        user_id="u1",
        turn_id="t1",
        turn_index=2,
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        retrieved_memories=[],
        response_style_prior={},
        habit_traits={},
        relationship_value_context={},
    )
    drive_low = drive_evaluator.evaluate(m13_state=low_state, **drive_kwargs)
    drive_high = drive_evaluator.evaluate(m13_state=high_state, **drive_kwargs)
    boredom_kwargs = dict(
        user_text="还是 status",
        user_id="u1",
        turn_id="t1",
        turn_index=2,
        conscious_plan=conscious,
        memory_dynamics=dynamics,
        retrieved_memories=[],
    )
    boredom_evaluator = M13BoredomEvaluator()
    boredom_low = boredom_evaluator.evaluate(
        **boredom_kwargs,
        m13_state=low_state,
        m13_drive_evaluation=drive_low,
    )
    boredom_high = boredom_evaluator.evaluate(
        **boredom_kwargs,
        m13_state=high_state,
        m13_drive_evaluation=drive_high,
    )
    assert boredom_high.exploration_bias > boredom_low.exploration_bias


def test_mvp_runtime_wires_settlement_and_reward(tmp_path: Path) -> None:
    class ShortLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt:
                return {
                    "expectation_results": [{"status": "confirmed"}],
                    "memory_search_keywords": ["status"],
                    "temporal_assessment": {},
                }
            if "上轮回复后果评估" in system_prompt:
                return {
                    "reaction": "uptake",
                    "confidence": 0.7,
                    "reason_codes": ["continues_thread"],
                }
            return {"reply": "好的。", "reply_action": "answer", "llm_thinking_result": {}}

    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "p_reward"), llm=ShortLLM())
    first = runtime.run_turn("项目 status", turn_index=0)
    second = runtime.run_turn("好的，继续", turn_index=1)
    bus_types = [msg.get("type") for msg in second.diagnostics.get("bus_messages", [])]
    assert "M13RewardSettlementAssessorEvent" in bus_types
    assert "M13RewardSettlementEvent" in bus_types
    diag = second.diagnostics.get("m13_reward_evaluation", {})
    assert isinstance(diag, dict)
    diag_blob = json.dumps(diag, ensure_ascii=False).casefold()
    assert "net_affective_reward_proxy" not in diag
    assert "tolerance" not in diag_blob
    assert "addiction" not in diag_blob
    assert "behavioral_pull" not in diag_blob
