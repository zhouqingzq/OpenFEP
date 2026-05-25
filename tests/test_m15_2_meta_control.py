from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import (
    ProactiveTarget,
    build_proposal_from_target,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, aggregate_fe_components, build_episode
from segmentum.dialogue.runtime.m15_meta_control import (
    apply_reflection_focus_intent,
    consume_recall_breadth_intent,
    detect_and_emit_intents,
)


NOW = 1_700_000_000


def _state(*, profile: str = "streamlit_open_chat") -> dict[str, object]:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["enabled"] = True
    initiative["implicit_idle_delivery"] = True
    initiative["proactive_policy_profile"] = profile
    m13["initiative"] = initiative
    return {
        "pending_expectations": [],
        "open_items": [],
        "short_term_memory": [],
        "long_term_memory": [],
        "temporal_state": {"last_turn_index": 4, "last_user_turn_at": NOW - 7200},
        "m13_drive_state": m13,
    }


def _append_episode(
    ledger: EpisodeLedger,
    state: dict[str, object],
    *,
    at: int,
    turn_index: int,
    trigger: str,
    delta_failure: bool,
) -> None:
    base = aggregate_fe_components(state)
    before = {**base, "expectation_prediction_error_proxy": 0.5}
    after = {**base, "expectation_prediction_error_proxy": 0.5 if delta_failure else 0.0}
    ledger.append(
        build_episode(
            at=at,
            turn_index=turn_index,
            phase="proactive_turn",
            state=state,
            action="proactive_outreach",
            action_trigger=trigger,
            evidence_refs=[f"ev_{turn_index}"],
            components_before=before,
            components_after=after,
            outcome_summary="violated" if delta_failure else "confirmed",
        )
    )


def test_repeated_failure_detector_fires_after_k_consecutive_zero_delta_fe(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    for idx in range(3):
        _append_episode(ledger, state, at=NOW + idx, turn_index=idx, trigger="memory_efe_outreach", delta_failure=True)
    result = detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    assert any(event["type"] == "RepeatedFailurePathDetectedEvent" for event in result.events)
    active = state["m13_drive_state"]["meta_control_intents"]["active"]  # type: ignore[index]
    assert active[-1]["intent_kind"] == "suppress_action_trigger_for_n_turns"


def test_repeated_failure_does_not_fire_on_mixed_outcomes(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="memory_efe_outreach", delta_failure=True)
    _append_episode(ledger, state, at=NOW + 1, turn_index=2, trigger="memory_efe_outreach", delta_failure=False)
    _append_episode(ledger, state, at=NOW + 2, turn_index=3, trigger="memory_efe_outreach", delta_failure=True)
    result = detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    assert not any(event["type"] == "RepeatedFailurePathDetectedEvent" for event in result.events)


def test_detectors_emit_audits_on_bounded_default_but_no_intents(tmp_path: Path) -> None:
    state = _state(profile="bounded_default")
    ledger = EpisodeLedger(tmp_path)
    for idx in range(3):
        _append_episode(ledger, state, at=NOW + idx, turn_index=idx, trigger="memory_efe_outreach", delta_failure=True)
    result = detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    assert any(event["type"] == "RepeatedFailurePathDetectedEvent" for event in result.events)
    meta = state["m13_drive_state"].get("meta_control_intents", {})  # type: ignore[union-attr]
    assert meta.get("active", []) == []


def test_suppress_intent_blocks_matching_trigger_in_m13_3(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    for idx in range(3):
        _append_episode(ledger, state, at=NOW + idx, turn_index=idx, trigger="scheduled_outreach", delta_failure=True)
    detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    initiative = normalize_initiative_state(state["m13_drive_state"]["initiative"])  # type: ignore[index]
    proposal = build_proposal_from_target(
        ProactiveTarget(
            trigger="scheduled_outreach",
            evidence_refs=["ev_sched"],
            proposed_topic="scheduled follow-up",
            ordinary_language_intent="Follow up on the scheduled item.",
            source_kind="scheduled_intent",
        ),
        now=NOW + 20,
        initiative=initiative,
    )
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW + 20,
        turn_index=5,
        locked_proposal=proposal,
    )
    assert check.proposal is None
    assert check.suppression_reason_code == "meta_control_trigger_suppressed"
    assert any(event.get("type") == "MetaControlInterventionAppliedEvent" for event in check.events)
    assert state["m13_drive_state"]["meta_control_intents"]["active"]  # type: ignore[index]
    assert state["m13_drive_state"]["initiative"]["last_suppression_reason_code"] == "meta_control_trigger_suppressed"  # type: ignore[index]


def test_expired_suppress_intent_does_not_get_restored_by_initiative_gate(tmp_path: Path) -> None:
    state = _state()
    state["m13_drive_state"]["meta_control_intents"] = {  # type: ignore[index]
        "active": [
            {
                "intent_id": "intent_old",
                "at": NOW,
                "turn_index": 1,
                "detector": "RepeatedFailurePathDetector",
                "intent_kind": "suppress_action_trigger_for_n_turns",
                "payload": {"action_trigger": "scheduled_outreach", "ttl_turns": 3},
                "evidence_refs": ["ev_old"],
                "detector_evidence_event_ids": ["ep_old"],
                "expires_at": NOW + 1000,
            }
        ],
        "consumed": [],
    }
    initiative = normalize_initiative_state(state["m13_drive_state"]["initiative"])  # type: ignore[index]
    proposal = build_proposal_from_target(
        ProactiveTarget(
            trigger="scheduled_outreach",
            evidence_refs=["ev_sched"],
            proposed_topic="scheduled follow-up",
            ordinary_language_intent="Follow up on the scheduled item.",
            source_kind="scheduled_intent",
        ),
        now=NOW + 20,
        initiative=initiative,
    )
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW + 20,
        turn_index=99,
        locked_proposal=proposal,
    )
    assert check.proposal is not None
    assert state["m13_drive_state"]["meta_control_intents"]["active"] == []  # type: ignore[index]
    assert state["m13_drive_state"]["meta_control_intents"]["consumed"]  # type: ignore[index]


def test_suppress_intent_expires_after_ttl_turns(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    for idx in range(3):
        _append_episode(ledger, state, at=NOW + idx, turn_index=idx, trigger="scheduled_outreach", delta_failure=True)
    detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    from segmentum.dialogue.runtime.m15_meta_control import expire_meta_control_intents

    events = expire_meta_control_intents(state, now=NOW + 1000, turn_index=8)
    assert any(event["type"] == "MetaControlInterventionExpiredEvent" for event in events)
    assert state["m13_drive_state"]["meta_control_intents"]["active"] == []  # type: ignore[index]


def test_self_consistency_tension_requires_stable_ticks_and_evidence_intersection(tmp_path: Path) -> None:
    state = _state()
    state["m12_user_continuity"] = {
        "identity_tensions": [
            {"tension_id": "tension_42", "confidence": 0.71, "evidence_refs": ["m12_ref"]}
        ]
    }
    state["m12_1_user_personality"] = {
        "plain_language_report": {"contradictions": [{"evidence_refs": ["m12_ref"]}]}
    }
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="x", delta_failure=False)
    first = detect_and_emit_intents(state, ledger, now=NOW + 1, turn_index=5, source="idle_cognitive_tick")
    second = detect_and_emit_intents(state, ledger, now=NOW + 2, turn_index=6, source="idle_cognitive_tick")
    assert not any(event["type"] == "SelfConsistencyTensionDetectedEvent" for event in first.events)
    assert any(event["type"] == "SelfConsistencyTensionDetectedEvent" for event in second.events)


def test_self_consistency_tension_uses_evidence_ref_intersection_not_content(tmp_path: Path) -> None:
    state = _state()
    state["m12_user_continuity"] = {
        "identity_tensions": [
            {"tension_id": "same_words", "confidence": 0.9, "evidence_refs": ["a"]}
        ]
    }
    state["m12_1_user_personality"] = {
        "plain_language_report": {"contradictions": [{"evidence_refs": ["b"], "text": "same_words"}]}
    }
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="x", delta_failure=False)
    for idx in range(2):
        result = detect_and_emit_intents(state, ledger, now=NOW + idx, turn_index=idx, source="idle_cognitive_tick")
    assert not any(event["type"] == "SelfConsistencyTensionDetectedEvent" for event in result.events)


def test_self_consistency_stability_counter_does_not_increment_on_background_tick(tmp_path: Path) -> None:
    state = _state()
    state["m12_user_continuity"] = {
        "identity_tensions": [
            {"tension_id": "tension_bg", "confidence": 0.9, "evidence_refs": ["m12_ref"]}
        ]
    }
    state["m12_1_user_personality"] = {
        "plain_language_report": {"contradictions": [{"evidence_refs": ["m12_ref"]}]}
    }
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="x", delta_failure=False)
    first = detect_and_emit_intents(state, ledger, now=NOW + 1, turn_index=5, source="background_tick")
    second = detect_and_emit_intents(state, ledger, now=NOW + 2, turn_index=6, source="background_tick")
    assert not any(event["type"] == "SelfConsistencyTensionDetectedEvent" for event in [*first.events, *second.events])


def test_request_reflection_focus_only_applies_when_plan_focus_is_null(tmp_path: Path) -> None:
    state = _state()
    state["m13_drive_state"]["meta_control_intents"] = {  # type: ignore[index]
        "active": [
            {
                "intent_id": "intent_focus",
                "at": NOW,
                "turn_index": 1,
                "detector": "SelfConsistencyTensionDetector",
                "intent_kind": "request_reflection_focus",
                "payload": {"focus_topic": "self_consistency:tension_42", "suggested_reflection_kind": "self_consistency"},
                "evidence_refs": ["m12_ref"],
                "detector_evidence_event_ids": ["tension_42"],
                "expires_at": NOW + 1000,
            }
        ],
        "consumed": [],
    }
    plan, events = apply_reflection_focus_intent(
        state,
        {"reflection_focus": None},
        now=NOW + 1,
        turn_index=2,
    )
    assert plan["reflection_focus"]["topic"] == "self_consistency:tension_42"
    assert plan["reflection_focus"]["evidence_refs"] == ["m12_ref"]
    assert "m12_user_continuity" not in json.dumps(plan)
    assert any(event.get("applied_to") == "m14_reflector" for event in events)
    assert any(event["type"] == "MetaControlInterventionExpiredEvent" and event["reason"] == "consumed" for event in events)


def test_request_reflection_focus_preserves_existing_focus(tmp_path: Path) -> None:
    state = _state()
    state["m13_drive_state"]["meta_control_intents"] = {  # type: ignore[index]
        "active": [
            {
                "intent_id": "intent_focus",
                "at": NOW,
                "turn_index": 1,
                "detector": "SelfConsistencyTensionDetector",
                "intent_kind": "request_reflection_focus",
                "payload": {"focus_topic": "self_consistency:tension_42", "suggested_reflection_kind": "self_consistency"},
                "evidence_refs": ["m12_ref"],
                "detector_evidence_event_ids": ["tension_42"],
                "expires_at": NOW + 1000,
            }
        ],
        "consumed": [],
    }
    plan, events = apply_reflection_focus_intent(
        state,
        {"reflection_focus": {"topic": "existing", "evidence_refs": ["x"], "reflection_kind": "open_item"}},
        now=NOW + 1,
        turn_index=2,
    )
    assert plan["reflection_focus"]["topic"] == "existing"
    assert any(event.get("applied_effect_summary") == "existing_focus_preserved" for event in events)


def _tick(reason: str = "no_high_value_target") -> dict[str, object]:
    return {
        "type": "IdleCognitiveTickEvent",
        "at": NOW,
        "turn_index": 1,
        "reject_reason": reason,
        "bands": {
            "boredom_band": "low",
            "reward_band": "low",
            "behavior_band": "low",
            "relation_band": "low",
        },
    }


def test_stall_detector_requires_same_reject_and_flat_bands(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="x", delta_failure=False)
    result = None
    for idx in range(6):
        tick = _tick("no_high_value_target")
        tick["at"] = NOW + idx
        tick["turn_index"] = idx
        result = detect_and_emit_intents(
            state,
            ledger,
            now=NOW + idx,
            turn_index=idx,
            source="idle_cognitive_tick",
            current_idle_tick_event=tick,
        )
    assert result is not None
    assert any(event["type"] == "MetaControlStallDetectedEvent" for event in result.events)


def test_stall_bias_recall_breadth_capped_and_consumed_after_one_tick(tmp_path: Path) -> None:
    state = _state()
    state["m13_drive_state"]["meta_control_intents"] = {  # type: ignore[index]
        "active": [
            {
                "intent_id": "intent_recall",
                "at": NOW,
                "turn_index": 1,
                "detector": "MetaControlStallDetector",
                "intent_kind": "bias_idle_recall_breadth",
                "payload": {"new_top_k": 99, "ttl_ticks": 1, "ticks_seen": 0},
                "evidence_refs": [],
                "detector_evidence_event_ids": [],
                "expires_at": NOW + 1000,
            }
        ],
        "consumed": [],
    }
    top_k, events = consume_recall_breadth_intent(state, now=NOW + 1, turn_index=2, default_top_k=8)
    assert top_k == 12
    assert any(event["type"] == "MetaControlInterventionAppliedEvent" for event in events)
    assert state["m13_drive_state"]["meta_control_intents"]["active"] == []  # type: ignore[index]


def test_detectors_do_not_run_during_user_turn(tmp_path: Path) -> None:
    state = _state()
    state["m13_ui_turn_in_progress"] = True
    ledger = EpisodeLedger(tmp_path)
    _append_episode(ledger, state, at=NOW, turn_index=1, trigger="x", delta_failure=True)
    result = detect_and_emit_intents(state, ledger, now=NOW, turn_index=1, source="idle_cognitive_tick")
    assert result.events == []
    assert result.intents == []


def test_detector_errors_are_caught_and_audited(tmp_path: Path) -> None:
    class BadLedger:
        def recent(self, _limit: int) -> list[object]:
            raise RuntimeError("bad ledger")

    state = _state()
    result = detect_and_emit_intents(state, BadLedger(), now=NOW, turn_index=1, source="idle_cognitive_tick")  # type: ignore[arg-type]
    assert any(event["type"] == "MetaControlDetectorErrorEvent" for event in result.events)


def test_no_module_can_send_reply_directly_from_meta_control(tmp_path: Path) -> None:
    state = _state()
    ledger = EpisodeLedger(tmp_path)
    for idx in range(3):
        _append_episode(ledger, state, at=NOW + idx, turn_index=idx, trigger="memory_efe_outreach", delta_failure=True)
    result = detect_and_emit_intents(state, ledger, now=NOW + 10, turn_index=4, source="idle_cognitive_tick")
    blob = json.dumps([event for event in result.events], ensure_ascii=False).casefold()
    assert "reply" not in blob
    assert "visible_text" not in blob
