from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import (
    _pick_structural_target,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    record_target_assessor_reject_backoff,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m13_memory_efe import (
    ACTIVE_GRACE_SECONDS,
    ENGINEERING_PROXY_LABEL,
    F_MEMORY_CAP,
    _PENDING_SETTLEMENT_SUPPRESSION,
    apply_memory_efe_state,
    apply_memory_efe_state_with_store_lock,
    build_memory_efe_outreach_proposal,
    build_memory_efe_outreach_proposal_input,
    evaluate_memory_efe,
    merge_memory_efe_guidance_into_control,
    normalize_expectations_for_efe,
    register_memory_efe_outreach_settlement,
    settle_memory_efe_outreach,
)
from segmentum.dialogue.runtime.m14_idle_reflector import (
    IDLE_INTROSPECTION_MARKER,
    apply_idle_drive_rules,
    empty_conscious_idle_plan,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_700_000_000


def _opted_in_m13() -> dict[str, object]:
    m13 = default_m13_drive_state()
    m13 = set_initiative_user_opt_in(m13, enabled=True)  # type: ignore[arg-type]
    initiative = normalize_initiative_state(m13["initiative"])  # type: ignore[index]
    initiative["implicit_idle_delivery"] = True
    initiative["enabled"] = True
    m13["initiative"] = initiative
    return m13


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "pending_expectations": [],
        "open_items": [],
        "temporal_state": {
            "last_user_turn_at": NOW - 7200,
            "last_turn_at": NOW - 7200,
            "last_turn_index": 4,
        },
        "m13_drive_state": _opted_in_m13(),
    }
    state.update(overrides)
    return state


def _overdue_expectation(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "exp_a",
        "verify_on": "later",
        "status": "pending",
        "content": "user said they would send the benchmark result",
        "confidence": 0.95,
        "due_at_epoch": NOW - 7200,
        "expected_window_seconds": 900,
        "evidence_refs": ["mem_a"],
        "bound_memory_ids": ["ltm_a"],
    }
    row.update(overrides)
    return row


def test_no_open_expectation_idle_suppresses_outreach() -> None:
    result = evaluate_memory_efe(_state(), phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.eligible_for_efe == []
    assert result.should_outreach is False
    assert "no_efe_eligible_expectation" in result.suppression_reasons
    assert set(result.efe_by_policy) == {"wait", "reflect", "outreach"}


def test_vague_verify_later_is_diagnostic_only() -> None:
    state = _state(pending_expectations=[{"id": "exp_vague", "verify_on": "later", "content": "later maybe"}])
    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert normalized.eligible_for_efe == []
    assert len(normalized.diagnostic_only) == 1
    assert normalized.diagnostic_only[0].ineligibility_reason == "verify_later_without_concrete_due"
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.should_outreach is False


def test_generic_memory_dynamics_expectation_is_diagnostic_only() -> None:
    state = _state(
        pending_expectations=[
            {
                "id": "exp_generic",
                "source": "memory_dynamics_adapter",
                "verify_on": "memory_dynamics_idle",
                "status": "pending",
                "content": "用户会继续抱怨天气或询问消暑方法，或者转其他闲聊话题",
                "confidence": 0.8,
                "created_at": NOW - 120,
                "evidence_refs": ["stm_turn_weather"],
                "bound_memory_ids": ["stm_turn_weather"],
            }
        ]
    )

    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert normalized.eligible_for_efe == []
    assert normalized.diagnostic_only[0].ineligibility_reason == "generic_low_resolution_expectation"
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.should_outreach is False
    assert "generic_low_resolution_expectation" in result.reason_codes


def test_generic_social_continuation_with_strict_trace_is_diagnostic_only() -> None:
    state = _state(
        pending_expectations=[
            {
                "id": "exp_generic_social",
                "source": "memory_dynamics_adapter",
                "verify_on": "memory_dynamics_idle",
                "status": "pending",
                "content": "用户会继续闲聊或表示惊讶/认可，或者道晚安",
                "confidence": 0.86,
                "created_at": NOW - 120,
                "evidence_refs": ["stm_turn_chat"],
                "bound_memory_ids": ["stm_turn_chat"],
            }
        ]
    )

    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")

    assert normalized.eligible_for_efe == []
    diagnostic = normalized.diagnostic_only[0]
    assert diagnostic.ineligibility_reason == "generic_low_resolution_expectation"
    assert diagnostic.resolution_class == "generic_social_continuation"
    assert diagnostic.specificity_score == 0.0
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.should_outreach is False
    assert "generic_low_resolution_expectation" in result.reason_codes


def test_mixed_specific_and_generic_memory_expectation_remains_eligible_with_discounted_resolution() -> None:
    state = _state(
        pending_expectations=[
            {
                "id": "exp_mixed_moon",
                "source": "memory_dynamics_adapter",
                "verify_on": "memory_dynamics_idle",
                "status": "pending",
                "content": "用户会追问满月夜具体忙什么（往生堂业务/仪式/守夜），或继续开月亮玩笑",
                "confidence": 0.86,
                "created_at": NOW - 120,
                "evidence_refs": ["stm_turn_moon"],
                "bound_memory_ids": ["stm_turn_moon"],
            }
        ]
    )

    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")

    assert len(normalized.eligible_for_efe) == 1
    eligible = normalized.eligible_for_efe[0]
    assert eligible.resolution_class == "mixed_specific_and_generic"
    assert eligible.testable_branch_count == 1
    assert eligible.generic_branch_count == 1
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.policy_costs["resolution_class"] == "mixed_specific_and_generic"
    assert result.policy_costs["specificity_score"] < 1.0


def test_assessor_reject_backoff_suppresses_same_memory_efe_target() -> None:
    m13 = record_target_assessor_reject_backoff(
        _opted_in_m13(),  # type: ignore[arg-type]
        expectation_id="exp_specific",
        now=NOW,
        reason_code="delivery_assessor_reject",
    )
    state = _state(
        m13_drive_state=m13,
        pending_expectations=[
            {
                "id": "exp_specific",
                "source": "memory_dynamics_adapter",
                "verify_on": "memory_dynamics_idle",
                "status": "pending",
                "content": "用户会回复是否已经按刚才约定提交 benchmark 结果",
                "confidence": 0.95,
                "created_at": NOW - 120,
                "evidence_refs": ["stm_turn_benchmark"],
                "bound_memory_ids": ["stm_turn_benchmark"],
            }
        ],
    )

    result = evaluate_memory_efe(state, phase="idle", now=NOW + 30, turn_index=5, user_active=False)

    assert result.traceable_expectation_id == "exp_specific"
    assert result.selected_policy == "outreach"
    assert result.should_outreach is False
    assert "target_assessor_reject_backoff" in result.suppression_reasons


def test_overdue_scheduled_open_item_can_make_outreach_lowest_policy() -> None:
    state = _state(
        open_items=[
            {
                "id": "oi_due",
                "status": "open",
                "title": "benchmark follow-up",
                "scheduled_intent_id": "intent_benchmark",
                "due_at_epoch": NOW - 10_000,
                "expected_window_seconds": 900,
                "evidence_refs": ["mem_oi"],
                "confidence": 0.95,
            }
        ]
    )
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.social_prediction_error > 0
    assert result.traceable_expectation_id == "oi_due"
    assert set(result.efe_by_policy) == {"wait", "reflect", "outreach"}
    assert result.selected_policy == "outreach"
    assert result.outreach_margin >= 0.08
    assert result.should_outreach is True


def test_open_item_next_user_turn_emerges_without_calendar_due() -> None:
    created = NOW - 5000
    state = _state(
        open_items=[
            {
                "id": "item_001",
                "status": "open",
                "content": "Who is 老爹 in the prior thread?",
                "next_check": "next_user_turn",
                "created_at": created,
                "evidence_refs": ["stm_turn_old"],
            }
        ],
        temporal_state={"last_user_turn_at": NOW - 120, "last_turn_at": NOW - 120},
    )
    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert [e.expectation_id for e in normalized.eligible_for_efe] == ["item_001"]
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.social_prediction_error > 0
    assert result.traceable_expectation_id == "item_001"


def test_open_item_bare_due_at_without_scheduled_intent_is_diagnostic_only() -> None:
    state = _state(
        open_items=[
            {
                "id": "oi_alarm",
                "status": "open",
                "due_at_epoch": NOW - 10_000,
                "next_check": "later",
            }
        ]
    )
    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert normalized.eligible_for_efe == []
    assert normalized.diagnostic_only[0].ineligibility_reason == "vague_or_missing_traceable_next_check"


def test_idle_next_user_turn_over_response_window_is_eligible() -> None:
    state = _state(
        pending_expectations=[
            _overdue_expectation(
                id="exp_next",
                verify_on="next_user_turn",
                created_at=NOW - 2000,
                due_at_epoch=0,
            )
        ],
        temporal_state={"last_user_turn_at": NOW - 3000, "last_turn_at": NOW - 3000},
    )
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert [e.expectation_id for e in result.eligible_for_efe] == ["exp_next"]
    assert result.social_prediction_error > 0


def test_memory_dynamics_idle_expectation_becomes_eligible_without_scheduled_due() -> None:
    state = _state(
        pending_expectations=[
            {
                "id": "exp_mem_dyn",
                "source": "memory_dynamics_adapter",
                "verify_on": "memory_dynamics_idle",
                "status": "pending",
                "content": "continue the unresolved science versus supernatural tension",
                "confidence": 0.95,
                "created_at": NOW - 120,
                "evidence_refs": ["stm_turn_mem_dyn"],
                "bound_memory_ids": ["stm_turn_mem_dyn"],
            }
        ],
        short_term_memory=[
            {
                "id": "stm_turn_mem_dyn",
                "content": "The user is stuck on the science versus supernatural tension.",
                "salience": 0.8,
            }
        ],
        temporal_state={"last_user_turn_at": NOW - 120, "last_turn_at": NOW - 120},
    )
    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert [e.expectation_id for e in normalized.eligible_for_efe] == ["exp_mem_dyn"]
    assert normalized.eligible_for_efe[0].source_kind == "memory_dynamics_expectation"


def test_in_turn_violation_repairs_and_never_outreaches() -> None:
    state = _state(pending_expectations=[_overdue_expectation(status="violated")])
    result = evaluate_memory_efe(
        state,
        phase="in_turn",
        now=NOW,
        turn_index=5,
        user_active=True,
        memory_dynamics={"control_guidance": {"repair_bias": 0.5}},
        conscious_plan={"expectation_results": [{"id": "exp_a", "status": "violated"}]},
    )
    assert result.selected_policy == "continue_reply"
    assert result.efe_by_policy.keys() == {"continue_reply"}
    assert result.reply_angle_bias == "repair_expectation"
    assert result.should_outreach is False


def test_idle_phase_never_includes_continue_reply() -> None:
    result = evaluate_memory_efe(_state(pending_expectations=[_overdue_expectation()]), phase="idle", now=NOW, turn_index=5, user_active=False)
    assert "continue_reply" not in result.efe_by_policy


def test_repetition_tension_active_user_sets_new_angle_without_outreach() -> None:
    class _Boredom:
        repetition_pressure = 0.42
        information_gain_proxy = 0.2
        progress_signal = 0.0

    result = evaluate_memory_efe(
        _state(),
        phase="in_turn",
        now=NOW,
        turn_index=5,
        user_active=True,
        m13_boredom_evaluation=_Boredom(),
    )
    assert result.reply_angle_bias == "new_angle"
    assert result.should_outreach is False


def test_scheduled_and_queued_outreach_suppress_duplicates() -> None:
    state = _state(pending_expectations=[_overdue_expectation(id="exp_dup")])
    scheduled = {"intent_id": "exp_dup", "kind": "scheduled_outreach", "status": "prepared", "due_at_epoch": NOW - 10}
    queued = {"source_intent_id": "exp_dup", "status": "pending"}
    result = evaluate_memory_efe(
        state,
        phase="idle",
        now=NOW,
        turn_index=5,
        user_active=False,
        structural_signals={"scheduled_intents": [scheduled], "queued_outreach": [queued]},
    )
    assert result.should_outreach is False
    assert "scheduled_outreach_already_active" in result.suppression_reasons
    assert "queued_outreach_already_pending" in result.suppression_reasons


def test_boundary_cost_high_suppresses_outreach_with_cost_diagnostics() -> None:
    state = _state(pending_expectations=[_overdue_expectation(boundary_strength="hard")])
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.policy_costs["boundary_cost"] == 0.45
    assert result.policy_costs["boundary_cost_high"] is True
    assert result.should_outreach is False
    assert "boundary_cost_high" in result.suppression_reasons


def test_recall_failure_only_when_linked_memory_not_retrieved() -> None:
    state = _state(pending_expectations=[_overdue_expectation(id="exp_recall")])
    retrieved = [{"id": "ltm_a", "content": "benchmark"}]
    with_recall = evaluate_memory_efe(
        state,
        phase="in_turn",
        now=NOW,
        turn_index=5,
        user_active=True,
        retrieved_memories=retrieved,
    )
    without_recall = evaluate_memory_efe(
        state,
        phase="in_turn",
        now=NOW,
        turn_index=5,
        user_active=True,
        retrieved_memories=[],
    )
    assert without_recall.epistemic_prediction_error > 0
    assert with_recall.epistemic_prediction_error == 0.0
    assert without_recall.eligible_for_efe[0].precision_approx.get("recall_salience_approx", 0) > 0


def test_reflect_preferred_when_recall_failure_without_outreach_resolution() -> None:
    state = _state(pending_expectations=[_overdue_expectation(id="exp_recall")])
    result = evaluate_memory_efe(
        state,
        phase="idle",
        now=NOW,
        turn_index=5,
        user_active=False,
        retrieved_memories=[],
    )
    assert result.epistemic_prediction_error > 0
    assert result.efe_by_policy["reflect"] <= result.efe_by_policy["outreach"]
    assert result.selected_policy == "reflect"


def test_not_opted_in_suppresses_should_outreach() -> None:
    m13 = default_m13_drive_state()
    m13 = set_initiative_user_opt_in(m13, enabled=False)  # type: ignore[arg-type]
    state = _state(pending_expectations=[_overdue_expectation()], m13_drive_state=m13)
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert "not_opted_in" in result.suppression_reasons
    assert result.should_outreach is False


def test_vague_open_item_next_check_is_not_efe_eligible() -> None:
    state = _state(open_items=[{"id": "oi", "status": "open", "next_check": "regular"}])
    normalized = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    assert normalized.eligible_for_efe == []
    assert normalized.diagnostic_only[0].ineligibility_reason == "vague_or_missing_traceable_next_check"


def test_outreach_proposal_not_delivered_does_not_clear_prediction_error() -> None:
    state = _state(pending_expectations=[_overdue_expectation()])
    evaluation = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    m13, _ = apply_memory_efe_state(state["m13_drive_state"], evaluation)  # type: ignore[arg-type]
    m13, events = register_memory_efe_outreach_settlement(
        m13,
        evaluation=evaluation,
        proposal_id="prop_x",
        delivery_status="failed",
        now=NOW,
    )
    assert events
    m13, settle_events = settle_memory_efe_outreach(m13, turn_index=6, now=NOW + 10, delivery_failures={"prop_x": "failed"})
    assert settle_events[-1]["observed_resolution"] == 0.0
    assert settle_events[-1]["outcome_band"] == "unresolved"
    assert normalize_m13_drive_state(m13)["memory_efe"]["f_memory"] == evaluation.f_memory


def test_delivered_but_no_later_observation_expires_uncertain() -> None:
    state = _state(pending_expectations=[_overdue_expectation()])
    evaluation = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    m13, _ = register_memory_efe_outreach_settlement(
        state["m13_drive_state"],  # type: ignore[arg-type]
        evaluation=evaluation,
        proposal_id="prop_y",
        delivery_status="delivered",
        now=NOW,
    )
    m13, events = settle_memory_efe_outreach(m13, turn_index=99, now=NOW + 90_000)
    assert events[-1]["outcome_band"] == "uncertain"
    assert "ttl_expired_without_observation" in events[-1]["reason_codes"]


def test_m13_3_single_entry_and_locked_memory_efe_priority() -> None:
    state = _state(open_items=[{"id": "oi", "status": "open", "next_check": "regular"}])
    state["m13_drive_state"] = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    evaluation = evaluate_memory_efe(
        _state(
            open_items=[
                {
                    "id": "oi_due",
                    "status": "open",
                    "scheduled_intent_id": "intent_benchmark",
                    "due_at_epoch": NOW - 10_000,
                    "expected_window_seconds": 900,
                    "evidence_refs": ["mem_oi"],
                    "confidence": 0.95,
                }
            ]
        ),
        phase="idle",
        now=NOW,
        turn_index=5,
        user_active=False,
    )
    assert evaluation.should_outreach is True
    initiative = normalize_initiative_state(normalize_m13_drive_state(state["m13_drive_state"])["initiative"])  # type: ignore[arg-type]
    proposal = build_memory_efe_outreach_proposal(evaluation, now=NOW, initiative=initiative)
    assert proposal is not None
    checked_state, check = evaluate_proactive_initiative(state, now=NOW, turn_index=5, locked_proposal=proposal)
    assert check.proposal is not None
    assert check.proposal.trigger == "memory_efe_outreach"
    assert checked_state["m13_drive_state"]["initiative"]["pending_proactive_proposal"]["trigger"] == "memory_efe_outreach"  # type: ignore[index]


def test_prompt_safe_guidance_excludes_raw_equations_and_subjective_words() -> None:
    state = _state(pending_expectations=[_overdue_expectation()])
    result = evaluate_memory_efe(state, phase="in_turn", now=NOW, turn_index=5, user_active=True)
    memory_dynamics: dict[str, object] = {"control_guidance": {"conflict_level": 0.4, "repair_bias": 0.2}}
    merge_memory_efe_guidance_into_control(memory_dynamics, result)
    guidance = memory_dynamics["control_guidance"]["memory_efe_guidance"]  # type: ignore[index]
    text = json.dumps(guidance, ensure_ascii=False).casefold()
    assert "efe(" not in text
    assert "lonely" not in text
    assert "raw" not in text
    assert guidance["advisory_only"] is True  # type: ignore[index]


def test_determinism_same_state_timestamp_same_values() -> None:
    state = _state(pending_expectations=[_overdue_expectation()])
    a = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    b = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert a.efe_by_policy == b.efe_by_policy
    assert a.reason_codes == b.reason_codes
    assert a.traceable_expectation_id == b.traceable_expectation_id


def test_user_active_race_guard_inside_active_grace_suppresses_outreach() -> None:
    state = _state(
        pending_expectations=[_overdue_expectation()],
        temporal_state={"last_user_turn_at": NOW - ACTIVE_GRACE_SECONDS + 1, "last_turn_at": NOW - ACTIVE_GRACE_SECONDS + 1},
    )
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.should_outreach is False
    assert "user_active" in result.suppression_reasons


def test_f_memory_cap_and_top_k_ignore_low_extra_expectations() -> None:
    rows = [_overdue_expectation(id=f"exp_{i}", due_at_epoch=NOW - 10_000 - i) for i in range(8)]
    result = evaluate_memory_efe(_state(pending_expectations=rows), phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.f_memory <= F_MEMORY_CAP
    baseline = result.selected_policy
    rows.append(_overdue_expectation(id="exp_low", confidence=0.01, due_at_epoch=NOW - 1))
    with_extra = evaluate_memory_efe(_state(pending_expectations=rows), phase="idle", now=NOW, turn_index=5, user_active=False)
    assert with_extra.selected_policy == baseline


def test_relationship_precision_applied_once_outside_precision() -> None:
    base = _state(pending_expectations=[_overdue_expectation()])
    high = _state(pending_expectations=[_overdue_expectation()])
    m13 = normalize_m13_drive_state(high["m13_drive_state"])  # type: ignore[arg-type]
    m13["relation_path_precision"] = {"u": 1.0}
    high["m13_drive_state"] = m13
    base_e = evaluate_memory_efe(base, phase="idle", now=NOW, turn_index=5, user_active=False).eligible_for_efe[0]
    high_e = evaluate_memory_efe(high, phase="idle", now=NOW, turn_index=5, user_active=False).eligible_for_efe[0]
    assert high_e.relationship_weight > base_e.relationship_weight
    assert high_e.precision == base_e.precision


def test_traceable_selection_tie_breaks_by_due_then_id() -> None:
    state = _state(
        pending_expectations=[
            _overdue_expectation(id="b", due_at_epoch=NOW - 3600),
            _overdue_expectation(id="a", due_at_epoch=NOW - 3600),
            _overdue_expectation(id="z", due_at_epoch=NOW - 7200),
        ]
    )
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.traceable_expectation_id == "z"

    tied = _state(
        pending_expectations=[
            _overdue_expectation(id="b", due_at_epoch=NOW - 3600),
            _overdue_expectation(id="a", due_at_epoch=NOW - 3600),
        ]
    )
    assert evaluate_memory_efe(tied, phase="idle", now=NOW, turn_index=5, user_active=False).traceable_expectation_id == "a"


def test_m14_plan_outreach_fails_closed_without_memory_efe_allowance() -> None:
    plan = empty_conscious_idle_plan()
    plan["outreach_recommendation"] = {
        "should_outreach": True,
        "reason": "open_item_followup",
        "suggested_intent": "try normal reflection outreach",
    }
    merged = apply_idle_drive_rules(
        plan,
        idle_context={"boredom": {"band": "high"}, "affective_reward_proxy": {}, "memory_efe": {"should_outreach": False}},
        structural_signals={},
    )
    assert merged["outreach_recommendation"]["should_outreach"] is False


def test_apply_memory_efe_state_with_store_lock_persists_snapshot(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "sess")
    state = store.load()
    state["pending_expectations"] = [_overdue_expectation()]
    store.save(state)
    evaluation = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    apply_memory_efe_state_with_store_lock(store, evaluation)
    reloaded = store.load()
    assert reloaded["m13_drive_state"]["memory_efe"]["engineering_proxy_label"] == ENGINEERING_PROXY_LABEL
    assert reloaded["m13_drive_state"]["memory_efe"]["traceable_expectation_id"] == "exp_a"


def test_expired_traceable_expectation_is_diagnostic_only_not_eligible() -> None:
    state = _state(pending_expectations=[_overdue_expectation(status="expired")])
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.eligible_for_efe == []
    assert result.diagnostic_only
    assert result.diagnostic_only[0].ineligibility_reason == "expectation_expired"


def test_build_proposal_input_is_structural_only() -> None:
    result = evaluate_memory_efe(_state(pending_expectations=[_overdue_expectation()]), phase="idle", now=NOW, turn_index=5, user_active=False)
    payload = build_memory_efe_outreach_proposal_input(result)
    if result.should_outreach:
        assert payload is not None
        assert payload["trigger"] == "memory_efe_outreach"
        assert "ordinary_language_intent" in payload
        assert "reply" not in payload


def test_pending_memory_efe_settlement_suppresses_duplicate_outreach() -> None:
    state = _state(pending_expectations=[_overdue_expectation(id="exp_dup")])
    evaluation = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    m13, _ = register_memory_efe_outreach_settlement(
        state["m13_drive_state"],  # type: ignore[arg-type]
        evaluation=evaluation,
        proposal_id="prop_z",
        delivery_status="queued",
        now=NOW,
    )
    state["m13_drive_state"] = m13
    result = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    assert result.should_outreach is False
    assert _PENDING_SETTLEMENT_SUPPRESSION in result.suppression_reasons


def test_boredom_structural_target_blocked_when_memory_efe_active() -> None:
    state = _state(pending_expectations=[_overdue_expectation()])
    evaluation = evaluate_memory_efe(state, phase="idle", now=NOW, turn_index=5, user_active=False)
    m13, _ = apply_memory_efe_state(state["m13_drive_state"], evaluation)  # type: ignore[arg-type]
    m13["boredom"] = {
        "boredom_level": 0.82,
        "last_exploration_target": "stale topic",
        "recent_plan_terms": ["stale"],
    }
    state["m13_drive_state"] = m13
    assert _pick_structural_target(state, m13) is None


def test_run_turn_surfaces_memory_efe_repair_guidance(tmp_path: Path) -> None:
    class _MemEfeTurnLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "意识主循环" in system_prompt:
                return {
                    "expectation_results": [{"id": "exp_a", "status": "violated"}],
                    "memory_search_keywords": ["benchmark"],
                    "temporal_assessment": {"elapsed": "recent"},
                    "task_focus": "reply",
                }
            return {"reply": "收到。", "reply_action": "answer", "llm_thinking_result": {}}

    store = MVPStateStore(tmp_path / "memefe_turn")
    state = store.load()
    state["pending_expectations"] = [_overdue_expectation(status="violated")]
    state["long_term_memory"] = [{"id": "ltm_a", "content": "benchmark", "keywords": ["benchmark"]}]
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_MemEfeTurnLLM())  # type: ignore[arg-type]
    result = runtime.run_turn("刚才那个预期不对。", turn_index=1, now=NOW)
    control = result.diagnostics["memory_dynamics"]["control_guidance"]  # type: ignore[index]
    memefe = control.get("memory_efe_guidance", {})
    assert memefe.get("reply_angle_bias") == "repair_expectation"
    assert result.diagnostics["m13_memory_efe_evaluation"]["reply_angle_bias"] == "repair_expectation"  # type: ignore[index]


def test_idle_llm_outreach_cannot_bypass_memory_efe_without_eligible_expectation(tmp_path: Path) -> None:
    class _IdleOutreachLLM:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if IDLE_INTROSPECTION_MARKER in system_prompt:
                return {
                    "reflection_focus": {
                        "topic": "open follow-up",
                        "evidence_refs": ["oi"],
                        "reflection_kind": "open_item",
                    },
                    "outreach_recommendation": {
                        "should_outreach": True,
                        "reason": "open_item_followup",
                        "suggested_intent": "Check in on the open item.",
                    },
                    "self_cognition_patch_proposal": empty_conscious_idle_plan()["self_cognition_patch_proposal"],
                    "memory_consolidation_proposals": [],
                    "open_item_proposals": [],
                }
            return {}

    store = MVPStateStore(tmp_path / "memefe_idle")
    state = store.load()
    state.update(
        {
            "open_items": [{"id": "oi", "status": "open", "title": "draft", "next_check": "later"}],
            "temporal_state": {
                "last_user_turn_at": NOW - 7200,
                "last_turn_at": NOW - 7200,
                "last_turn_index": 2,
            },
            "m13_drive_state": _opted_in_m13(),
        }
    )
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_IdleOutreachLLM())  # type: ignore[arg-type]
    runtime.set_initiative_user_opt_in(True)
    runtime.set_idle_introspection_opt_in(True)
    from segmentum.dialogue.runtime.m13_idle import gather_idle_structural_signals

    signals = gather_idle_structural_signals(state, now=NOW, turn_index=3)
    idle_result = runtime.run_idle_introspection_turn(
        now=NOW,
        turn_index=3,
        structural_signals=signals,
    )
    assert idle_result.outreach_recommendation.get("should_outreach") is False
    reloaded = store.load()
    assert reloaded["m13_drive_state"]["memory_efe"]["should_outreach"] is False  # type: ignore[index]


def test_memory_dynamics_expectation_stale_after_user_turn_not_eligible() -> None:
    state = _state(
        pending_expectations=[
            {
                "id": "exp_turn0_weather_response",
                "content": "user will respond to weather joke",
                "verify_on": "memory_dynamics_idle",
                "source": "memory_dynamics_adapter",
                "created_at": NOW - 7200,
                "created_turn_index": 0,
                "evidence_refs": ["stm_turn_0"],
                "bound_memory_ids": ["stm_turn_0"],
                "confidence": 0.7,
            }
        ],
        temporal_state={
            "last_user_turn_at": NOW - 60,
            "last_turn_at": NOW - 60,
            "last_turn_index": 1,
        },
    )
    bundle = normalize_expectations_for_efe(state, now=NOW, phase="idle")
    eligible_ids = [row.expectation_id for row in bundle.eligible_for_efe]
    assert "exp_turn0_weather_response" not in eligible_ids


def test_diagnostic_only_pool_emits_noisy_reason_not_active_cleanup() -> None:
    state = _state(
        pending_expectations=[
            _overdue_expectation(id="exp_live"),
            {"id": "exp_old", "status": "expired", "content": "stale", "verify_on": "later"},
        ]
    )
    result = evaluate_memory_efe(state, now=NOW, phase="idle", turn_index=5, user_active=False)
    assert result.eligible_for_efe
    assert "diagnostic_expectation_pool_noisy" in result.reason_codes
    assert "cleanup_filtered_low_traceability_candidates" not in result.reason_codes
