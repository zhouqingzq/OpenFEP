from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.m14_idle_reflector import normalize_conscious_idle_plan
from segmentum.dialogue.runtime.m19_self_expectation import (
    apply_m19_traction_proposals_to_m13,
    apply_self_expectation_post_turn,
    build_self_repair_guidance,
    build_shadow_validation,
    default_self_expectation_state,
    infer_matching_target_contexts,
)
from segmentum.dialogue.runtime.m13_drive import normalize_m13_drive_state
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    build_conscious_loop_prompt,
    normalize_conscious_turn_plan,
)


class _M19LoopLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "pending_expectations_to_verify" in user_prompt:
            turn_marker = "turn_index: "
            turn_index = int(user_prompt.split(turn_marker, 1)[1].splitlines()[0].strip())
            proposal = {
                "proposal_id": f"self_exp_{turn_index}",
                "target_context": "short_casual_reply",
                "expected_outcome": "casual_turn_stays_light_and_short",
                "expected_reply_quality": "compact",
                "confidence": 0.82,
                "evidence_refs": [f"turn:{turn_index}:proposal"],
                "reason_codes": ["compact_context"],
                "engineering_proxy_label": "test_m19",
            }
            outcome_results: list[dict[str, object]] = []
            if turn_index == 1:
                outcome_results.append(
                    {
                        "source_expectation_id": "self_exp_0",
                        "target_context": "short_casual_reply",
                        "status": "violated",
                        "evidence_refs": ["turn:1:outcome"],
                        "reason_codes": ["too_heavy"],
                    }
                )
            elif turn_index == 2:
                outcome_results.append(
                    {
                        "source_expectation_id": "self_exp_1",
                        "target_context": "short_casual_reply",
                        "status": "violated",
                        "evidence_refs": ["turn:2:outcome"],
                        "reason_codes": ["too_heavy_again"],
                    }
                )
            elif turn_index == 3:
                outcome_results.append(
                    {
                        "source_expectation_id": "self_exp_2",
                        "target_context": "short_casual_reply",
                        "status": "confirmed",
                        "evidence_refs": ["turn:3:outcome"],
                        "reason_codes": ["lighter_fit"],
                    }
                )
            elif turn_index == 4:
                outcome_results.append(
                    {
                        "source_expectation_id": "self_exp_3",
                        "target_context": "short_casual_reply",
                        "status": "confirmed",
                        "evidence_refs": ["turn:4:outcome"],
                        "reason_codes": ["lighter_fit_repeat"],
                    }
                )
            return {
                "current_task": "reply",
                "memory_search_keywords": ["casual", "light"],
                "self_response_expectation_proposals": [proposal],
                "self_expectation_outcome_results": outcome_results,
                "reply_pacing_hint": "casual_fast",
                "prefers_compact_reply": True,
            }
        if "llm_thinking_result" in user_prompt and '"reply_action"' in user_prompt:
            return {
                "thought_type": "short",
                "llm_thinking_result": {
                    "user_intent_read": "light chat",
                    "state_or_memory_used": ["self repair guidance when active"],
                    "response_choice": "answer",
                    "uncertainty": "",
                    "debug_summary": "ok",
                },
                "reply": "好呀，来个轻一点的答复。",
                "reply_action": "answer",
                "disclosure_action": "none",
                "new_expectations": [],
                "memory_writes": [],
                "self_cognition_patch": {"apply": False},
                "open_item_writes": [],
                "scheduled_outreach_requests": [],
                "habit_updates": [],
                "memory_dynamics_note": "",
            }
        if '"reply"' in user_prompt and "draft_reply" in user_prompt:
            return {"reply": "好呀，来个轻一点的答复。"}
        if '"needs_followup"' in user_prompt:
            return {"needs_followup": False, "followup_type": "none"}
        return {}


def test_conscious_prompt_mentions_m19_fields() -> None:
    _, user_prompt = build_conscious_loop_prompt(
        state={"self_expectation_state": {"active_mismatch_focus_topk": []}},
        user_text="随便聊聊",
        bus_messages=[],
        turn_index=1,
    )
    assert "self_response_expectation_proposals" in user_prompt
    assert "self_expectation_outcome_results" in user_prompt
    assert "self_expectation_state" in user_prompt


def test_normalize_conscious_turn_plan_rejects_unknown_self_expectation_target_context() -> None:
    normalized = normalize_conscious_turn_plan(
        {
            "self_response_expectation_proposals": [
                {
                    "proposal_id": "self_exp_ok",
                    "target_context": "short_casual_reply",
                    "expected_outcome": "ok",
                    "expected_reply_quality": "compact",
                    "confidence": 0.8,
                    "evidence_refs": ["a"],
                },
                {
                    "proposal_id": "self_exp_bad",
                    "target_context": "made_up_context",
                    "expected_outcome": "bad",
                    "expected_reply_quality": "compact",
                    "confidence": 0.8,
                    "evidence_refs": ["b"],
                },
            ]
        }
    )
    proposals = normalized["self_response_expectation_proposals"]
    assert len(proposals) == 1
    assert proposals[0]["proposal_id"] == "self_exp_ok"


def test_idle_plan_accepts_self_expectation_review_rows() -> None:
    plan = normalize_conscious_idle_plan(
        {
            "reflection_focus": {
                "topic": "review",
                "evidence_refs": ["self_exp_1"],
                "reflection_kind": "self_expectation_calibration",
            },
            "self_expectation_review_proposals": [
                {
                    "source_expectation_id": "self_exp_1",
                    "target_context": "short_casual_reply",
                    "review_status": "stale",
                    "evidence_refs": ["self_exp_1"],
                    "reason_codes": ["no_repeat"],
                }
            ],
            "self_cognition_patch_proposal": {"apply": False},
            "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
        }
    )
    assert plan["reflection_focus"]["reflection_kind"] == "self_expectation_calibration"
    assert plan["self_expectation_review_proposals"][0]["review_status"] == "stale"


def test_infer_matching_target_contexts_uses_pacing_without_proposal() -> None:
    contexts = infer_matching_target_contexts(
        {
            "reply_pacing_hint": "casual_fast",
            "prefers_compact_reply": True,
            "self_response_expectation_proposals": [],
        },
        self_state=default_self_expectation_state(),
    )
    assert "short_casual_reply" in contexts


def test_build_self_repair_guidance_reads_slow_repair_priors() -> None:
    guidance = build_self_repair_guidance(
        {
            "self_expectation_state": default_self_expectation_state(),
            "self_cognition": {
                "repair_priors": [
                    {
                        "id": "repair_prior_1",
                        "target_context": "short_casual_reply",
                        "preferred_intervention": "prefer_short_casual_surface_form",
                        "confidence": 0.8,
                        "status": "active",
                    }
                ]
            },
        },
        conscious_plan={
            "reply_pacing_hint": "casual_fast",
            "prefers_compact_reply": True,
            "self_response_expectation_proposals": [],
        },
    )
    assert guidance["repair_bias_delta"] > 0
    assert guidance["reply_action_biases"].get("answer", 0.0) > 0


def test_settlement_without_outcome_stays_uncertain() -> None:
    state = {
        "self_expectation_state": {
            **default_self_expectation_state(),
            "repair_expectations": [
                {
                    "expectation_id": "self_repair_1",
                    "source_mismatch_key": "short_casual_reply:outcome_too_heavy_for_context",
                    "target_context": "short_casual_reply",
                    "intervention": "prefer_short_casual_surface_form",
                    "prediction_error_reduction_target": 0.2,
                    "status": "active",
                    "created_turn_index": 0,
                    "opportunity_window": 4,
                    "expires_after_opportunities": 2,
                    "evidence_refs": ["self_exp_0"],
                    "opportunities_seen": 0,
                }
            ],
        }
    }
    result = apply_self_expectation_post_turn(
        state,
        conscious_plan={
            "reply_pacing_hint": "casual_fast",
            "prefers_compact_reply": True,
            "self_response_expectation_proposals": [],
            "self_expectation_outcome_results": [],
        },
        control_guidance={"repair_bias": 0.1, "conflict_level": 0.1},
        reward_prediction_error_proxy=0.05,
        reward_event_id="m13_reward_test",
        now=100,
        turn_index=2,
    )
    settlements = state["self_expectation_state"]["settlements_tail"]
    assert settlements
    assert settlements[-1]["status"] == "uncertain"


def test_shadow_validation_is_advisory_only() -> None:
    shadow = build_shadow_validation(
        {
            "expectation_id": "self_repair_1",
            "intervention": "prefer_short_casual_surface_form",
        },
        prediction_error_before=0.35,
        prediction_error_after=0.12,
        control_guidance={"repair_bias": 0.2},
    )
    assert shadow["advisory_only"] is True
    assert shadow["estimated_prediction_error_delta"] > 0


def test_shadow_validation_does_not_force_confirmed_settlement() -> None:
    state = {
        "self_expectation_state": {
            **default_self_expectation_state(),
            "repair_expectations": [
                {
                    "expectation_id": "self_repair_1",
                    "source_mismatch_key": "short_casual_reply:outcome_too_heavy_for_context",
                    "target_context": "short_casual_reply",
                    "intervention": "prefer_short_casual_surface_form",
                    "prediction_error_reduction_target": 0.2,
                    "status": "active",
                    "created_turn_index": 0,
                    "opportunity_window": 4,
                    "expires_after_opportunities": 2,
                    "evidence_refs": ["self_exp_0"],
                    "opportunities_seen": 0,
                }
            ],
        }
    }
    result = apply_self_expectation_post_turn(
        state,
        conscious_plan={
            "reply_pacing_hint": "casual_fast",
            "prefers_compact_reply": True,
            "self_response_expectation_proposals": [],
            "self_expectation_outcome_results": [],
        },
        control_guidance={"repair_bias": 0.4, "conflict_level": 0.1},
        reward_prediction_error_proxy=0.03,
        reward_event_id="m13_reward_test",
        now=100,
        turn_index=2,
    )
    settlement = state["self_expectation_state"]["settlements_tail"][-1]
    assert settlement["status"] == "uncertain"
    assert settlement.get("shadow_validation")
    assert any(event["type"] == "SelfRepairShadowValidationEvent" for event in result.events)


def test_apply_m19_traction_proposals_updates_m13_state() -> None:
    m13_state = normalize_m13_drive_state({})
    updated, events = apply_m19_traction_proposals_to_m13(
        m13_state,
        [
            {
                "proposal_id": "m19_traction_1",
                "intervention": "prefer_short_casual_surface_form",
                "status": "confirmed",
                "traction_delta": 0.05,
            }
        ],
        user_id="user_a",
        topic_fingerprint="casual|light",
        turn_index=3,
    )
    assert events
    assert updated["path_patterns_by_action"]
    assert updated["traction_by_action"]


def test_m19_end_to_end_runtime_promotes_slow_self_cognition(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=_M19LoopLLM(),
        persona_name="test",
    )
    last_result = None
    for turn_index in range(5):
        last_result = runtime.run_turn("随便聊聊", turn_index=turn_index, now=1_700_000_000 + turn_index * 60)
    assert last_result is not None
    control = last_result.diagnostics["memory_dynamics"]["control_guidance"]
    assert "self_repair_guidance" in control
    state = runtime.store.load()
    self_state = state["self_expectation_state"]
    assert self_state["mismatch_memory_fast"]
    assert self_state["repair_expectations"]
    assert any(row["status"] == "confirmed" for row in self_state["settlements_tail"])
    assert any(abs(float(row["prediction_error_delta"])) > 0 for row in self_state["settlements_tail"])
    cognition = state["self_cognition"]
    assert cognition["calibrated_tendencies"]
    assert cognition["repair_priors"]
