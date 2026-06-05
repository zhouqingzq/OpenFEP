from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.mind_debug_bundle import build_mind_debug_bundle_text
from segmentum.dialogue.runtime.mvp_loop import MVPStateStore


def test_build_mind_debug_bundle_includes_traceability_and_verdicts(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "session")
    state = store.load()
    state.update(
        {
            "temporal_state": {
                "last_turn_index": 2,
                "last_turn_at": 1_700_000_100,
                "last_user_turn_at": 1_700_000_000,
                "last_time_gap_label": "long_gap",
            },
            "open_items": [
                {
                    "id": "item_001",
                    "status": "open",
                    "title": "follow up benchmark",
                    "next_check": "later",
                    "evidence_refs": ["mem_1"],
                    "bound_memory_ids": ["mem_1"],
                    "created_at": 1_700_000_000,
                }
            ],
            "pending_expectations": [
                {
                    "id": "exp_001",
                    "status": "pending",
                    "content": "user will ask about boundary",
                    "evidence_refs": ["mem_2"],
                    "bound_memory_ids": ["mem_2"],
                },
                {
                    "id": "exp_old",
                    "status": "expired",
                    "content": "stale expectation",
                }
            ],
            "m13_drive_state": {
                "initiative": {
                    "user_opt_in": True,
                    "enabled": True,
                    "proactive_policy_profile": "bounded_default",
                    "idle_introspection": {
                        "enabled": True,
                        "user_opt_in": True,
                        "last_skip_reason": "idle_time_too_short",
                    },
                }
            },
        }
    )
    store.save(state)
    store.append_log(
        {
            "event": "m13_proactive_audit",
            "type": "IdleCognitiveTickEvent",
            "at": 1_700_000_200,
            "idle_seconds": 120.0,
            "reject_reason": "generic_self_only_open_item",
            "retrieved_ids": ["mem_1"],
            "memory_efe_should_outreach": False,
        }
    )
    store.append_log(
        {
            "event": "m14_idle_audit",
            "type": "IdleIntrospectionPlanEvent",
            "at": 1_700_000_150,
            "plan": {
                "outreach_recommendation": {
                    "should_outreach": True,
                    "reason": "traceable_focus",
                    "suggested_intent": "Follow up item_001",
                }
            },
        }
    )

    text = build_mind_debug_bundle_text(
        session_root=store.root,
        persona_name="test_persona",
        session_id="sess_1",
        state=state,
        observability={
            "m13_5_last_idle_cognitive_tick": {
                "at": 1_700_000_200,
                "idle_seconds": 120.0,
                "reject_reason": "generic_self_only_open_item",
                "retrieved_ids": ["mem_1"],
                "memory_efe_should_outreach": False,
            },
            "m14_3_open_item_traceability_suggestions": 1,
            "health_ticks_today": 0,
            "environment_event_status_counts": {"acked_count": 1, "pending_count": 0},
            "environment_events_terminal_ratio": 1.0,
            "environment_event_backlog_count": 0,
            "stale_environment_event_backlog_count": 0,
            "latest_turn_latency": {
                "latency_mode": "normal",
                "latency_mode_reasons": ["task_or_technical_marker"],
                "blocking_llm_calls": 5,
                "turn_total_duration_ms": 1200.0,
                "slowest_stage": {"stage": "conscious_loop"},
                "skipped_llm_stage_count": 3,
                "turn_latency_trace": [
                    {"stage": "conscious_loop", "duration_ms": 900.0},
                    {"stage": "thinking_reply", "duration_ms": 300.0},
                ],
            },
            "scheduler_skip_reason": "idle_time_too_short",
            "cognitive_selector_skip_reason": "generic_self_only_open_item",
            "delivery_skip_reason": "",
            "m15_meta_control": {
                "cleanup_consumed": [
                    {
                        "intent_kind": "cleanup_pending_expectation_backlog",
                        "consumed_at": 1_700_000_210,
                        "ops_delta": {"expired_pending_expectations": 1},
                    }
                ]
            },
        },
        ui_hints={"pending_user_message": "hello"},
        turn_index=2,
    )

    assert "Path B Mind Debug Bundle" in text
    assert "item_001" in text
    assert "exp_001" in text
    assert "pending_expectations_raw_total=2 active_total=1 strict_trace_active=1" in text
    assert "folded_non_active=1" in text
    assert "recently_applied_cleanup cleanup_pending_expectation_backlog" in text
    assert "environment_events_terminal_ratio: 1.0" in text
    assert "stale_environment_event_backlog_count: 0" in text
    assert "latest_turn_latency_reasons: task_or_technical_marker" in text
    assert "latest_turn_latency_trace: conscious_loop:900.0ms; thinking_reply:300.0ms" in text
    assert "scheduler_skip_reason: idle_time_too_short" in text
    assert "cognitive_selector_skip_reason: generic_self_only_open_item" in text
    assert "generic_self_only_open_item" in text
    assert "intro_should_outreach: True" in text
    assert "pending_user_message: hello" in text
    assert "## Diagnose verdicts" in text
    assert "## Recent audit tail" in text
    assert "## Proactive target timeline" in text


def test_debug_bundle_shows_separate_proactive_timeline_fields(tmp_path: Path) -> None:
    text = build_mind_debug_bundle_text(
        session_root=tmp_path / "session2",
        persona_name="test_persona",
        session_id="sess_2",
        state={},
        observability={
            "latest_selector_target": {"at": 100, "traceable_expectation_id": "exp_sel"},
            "latest_attempted_target": {"at": 200, "traceable_expectation_id": "exp_try"},
            "latest_delivered_target": {"at": 300, "proposal_id": "prop_ok"},
            "latest_suppressed_target": {"at": 250, "reason_code": "delivery_assessor_reject"},
            "latest_pipeline_suppression": {"at": 250, "reason_code": "delivery_assessor_reject"},
        },
        ui_hints={},
        turn_index=1,
    )
    assert "exp_sel" in text
    assert "exp_try" in text
    assert "prop_ok" in text
    assert "delivery_assessor_reject" in text


def test_debug_bundle_includes_m19_chain_summary(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "session3")
    state = store.load()
    state.update(
        {
            "self_expectation_state": {
                "expectations_tail": [
                    {
                        "expectation_id": "self_exp_1",
                        "target_context": "short_casual_reply",
                        "expected_outcome": "keep it light",
                        "expected_reply_quality": "light",
                    }
                ],
                "mismatches_tail": [
                    {
                        "mismatch_id": "mismatch_1",
                        "mismatch_type": "outcome_too_heavy_for_context",
                        "target_context": "short_casual_reply",
                        "severity": 0.72,
                    }
                ],
                "mismatch_memory_fast": [
                    {
                        "mismatch_key": "short_casual_reply|outcome_too_heavy_for_context",
                        "mismatch_type": "outcome_too_heavy_for_context",
                        "target_context": "short_casual_reply",
                        "weighted_support": 1.35,
                        "last_prediction_error_proxy": 0.28,
                        "status": "active",
                    }
                ],
                "active_mismatch_focus_topk": [
                    {
                        "mismatch_key": "short_casual_reply|outcome_too_heavy_for_context",
                        "mismatch_type": "outcome_too_heavy_for_context",
                        "target_context": "short_casual_reply",
                        "weighted_support": 1.35,
                        "last_prediction_error_proxy": 0.28,
                        "status": "active",
                    }
                ],
                "last_prediction_error_proxy": 0.28,
                "repair_expectations": [
                    {
                        "expectation_id": "repair_exp_1",
                        "target_context": "short_casual_reply",
                        "intervention": "prefer_short_casual_surface_form",
                        "status": "pending",
                        "verify_on": "next_similar_turn",
                        "source_mismatch_key": "short_casual_reply|outcome_too_heavy_for_context",
                    }
                ],
                "settlements_tail": [
                    {
                        "settlement_id": "self_settlement_1",
                        "expectation_id": "repair_exp_1",
                        "source_mismatch_key": "short_casual_reply|outcome_too_heavy_for_context",
                        "matched_context": "short_casual_reply",
                        "status": "confirmed",
                        "prediction_error_delta": 0.12,
                    }
                ],
                "traction_proposals_tail": [],
                "observations_tail": [
                    {
                        "observation_id": "self_obs_1",
                        "target_context": "short_casual_reply",
                        "review_status": "reinforced",
                    }
                ],
            },
            "self_cognition": {
                "calibrated_tendencies": [
                    {
                        "id": "cal_tend_1",
                        "target_context": "short_casual_reply",
                        "confidence": 0.81,
                        "source_mismatch_key": "short_casual_reply|outcome_too_heavy_for_context",
                        "status": "active",
                    }
                ],
                "repair_priors": [
                    {
                        "id": "repair_prior_1",
                        "target_context": "short_casual_reply",
                        "preferred_intervention": "prefer_short_casual_surface_form",
                        "confidence": 0.84,
                        "status": "active",
                    }
                ],
            },
        }
    )
    store.save(state)
    store.append_log(
        {
            "event": "m13_proactive_audit",
            "type": "SelfExpectationMismatchObservedEvent",
            "at": 1_700_000_300,
            "status": "violated",
            "target_context": "short_casual_reply",
        }
    )
    store.append_log(
        {
            "event": "m13_proactive_audit",
            "type": "SelfRepairExpectationCreatedEvent",
            "at": 1_700_000_301,
            "expectation_id": "repair_exp_1",
            "target_context": "short_casual_reply",
        }
    )
    store.append_log(
        {
            "event": "m13_proactive_audit",
            "type": "SelfRepairSettlementEvent",
            "at": 1_700_000_302,
            "settlement_id": "self_settlement_1",
            "status": "confirmed",
            "target_context": "short_casual_reply",
        }
    )
    store.append_log(
        {
            "event": "m14_idle_audit",
            "type": "SelfExpectationSlowPromotionProposalEvent",
            "at": 1_700_000_303,
            "reason": "m19_self_expectation_promotion",
        }
    )

    text = build_mind_debug_bundle_text(
        session_root=store.root,
        persona_name="test_persona",
        session_id="sess_3",
        state=state,
        observability={},
        ui_hints={},
        turn_index=3,
    )

    assert "## M19 self-expectation" in text
    assert "active_focus=1 active_repairs=1 active_repair_priors=1 active_calibrated_tendencies=1" in text
    assert "log_counts: mismatch=1 confirmed_outcome=0 repair_created=1" in text
    assert "settlement_confirmed=1" in text
    assert "slow_promotion=1" in text
    assert "focus `short_casual_reply|outcome_too_heavy_for_context`" in text
    assert "repair `repair_exp_1`" in text
    assert "settlement `self_settlement_1`" in text
    assert "repair_prior `repair_prior_1`" in text
    assert "calibrated_tendency `cal_tend_1`" in text
    assert "SelfRepairSettlementEvent" in text
