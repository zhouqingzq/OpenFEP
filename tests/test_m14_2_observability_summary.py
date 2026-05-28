from __future__ import annotations

import time
from types import MethodType

from segmentum.dialogue.runtime.chat import ChatInterface


def test_observability_summary_counts_active_intents_and_pending_outreach_only() -> None:
    chat = ChatInterface.__new__(ChatInterface)
    now = int(time.time())

    chat.read_self_loop_daemon_status = MethodType(lambda self: {"status": "running"}, chat)
    chat.read_idle_introspection_status = MethodType(
        lambda self: {
            "reflection_count_this_session": 2,
            "max_per_session": 4,
            "last_introspection_at": now - 10,
            "last_skip_reason": "budget_ok",
            "last_outreach_outcome": "queued",
        },
        chat,
    )
    chat.read_background_continuity_status = MethodType(
        lambda self: {
            "llm_calls_today": 1,
            "llm_calls_budget_per_day": 80,
            "tokens_used_today": 256,
            "tokens_budget_per_day": 30000,
            "last_budget_block_reason": "",
        },
        chat,
    )
    chat.read_m14_2_environment_events = MethodType(
        lambda self, limit=20: [
            {"event_type": "ClockWakeEvent", "status": "acked", "at": now},
            {"event_type": "UserMessageCommittedEvent", "status": "pending", "at": now - 600},
            {"event_type": "UIPingEvent", "status": "pending", "at": now},
        ],
        chat,
    )
    chat.read_m14_2_scheduled_intents = MethodType(
        lambda self: [
            {"status": "pending"},
            {"status": "prepared"},
            {"status": "awaiting_delivery"},
            {"status": "delivered"},
            {"status": "expired"},
        ],
        chat,
    )
    chat.read_queued_outreach = MethodType(
        lambda self: [
            {"status": "pending"},
            {"status": "delivered"},
            {"status": "expired"},
            {"status": "pending"},
        ],
        chat,
    )
    chat.read_conversation_log = MethodType(
        lambda self, limit=300: [
            {
                "event": "m14_2_audit",
                "type": "SelfLoopDaemonHealthEvent",
                "at": now,
                "llm_available": False,
                "llm_unavailable_reason": "llm_unavailable",
                "background_ran_llm": False,
            },
            {
                "event": "m13_proactive_audit",
                "type": "M13ProactiveProposalEvent",
                "trigger": "memory_efe_outreach",
                "traceable_expectation_id": "exp_a",
                "ordinary_language_intent": "follow up",
                "source_kind": "open_item",
            },
            {
                "event": "m13_proactive_audit",
                "type": "M13ProactiveSuppressionEvent",
                "reason": "safety_risk",
                "reason_code": "delivery_assessor_reject",
                "reason_stage": "post_generation",
            },
            {
                "event": "turn_latency",
                "type": "MVPDialogTurnLatencyEvent",
                "at": now,
                "latency_mode": "fast_chat",
                "latency_mode_reasons": ["low_risk_short_chat"],
                "blocking_llm_calls": 2,
                "total_llm_duration_ms": 123.5,
                "turn_total_duration_ms": 180.0,
                "slowest_stage": {"stage": "thinking_reply", "duration_ms": 90.0},
                "turn_latency_trace": [
                    {"stage": "conscious_loop", "duration_ms": 33.5, "success": True, "error_type": ""},
                    {"stage": "thinking_reply", "duration_ms": 90.0, "success": True, "error_type": ""},
                ],
                "skipped_llm_stages": [{"stage": "m11_user_model"}],
            },
            {
                "event": "m13_proactive_audit",
                "type": "IdleCognitiveTickEvent",
                "at": now + 1,
                "reject_reason": "diagnostic_expectation_pool_noisy",
            },
        ],
        chat,
    )

    summary = chat.read_m14_2_observability_summary()

    assert summary["scheduled_intents_active"] == 3
    assert summary["queued_outreach"] == 2
    assert summary["clock_wake_acked_today"] == 1
    assert summary["user_message_pending"] == 1
    assert summary["ui_audit_pending"] == 1
    assert summary["environment_event_status_counts"]["acked_count"] == 1
    assert summary["environment_event_status_counts"]["pending_count"] == 2
    assert summary["environment_events_terminal_ratio"] == round(1 / 3, 4)
    assert summary["environment_event_backlog_count"] == 2
    assert summary["stale_environment_event_backlog_count"] == 1
    assert summary["daemon_llm_available"] is False
    assert summary["daemon_llm_unavailable_reason"] == "llm_unavailable"
    assert summary["m14_3_last_proactive_target"]["trigger"] == "memory_efe_outreach"
    assert summary["m14_3_last_proactive_suppression"]["reason_code"] == "delivery_assessor_reject"
    assert summary["latest_turn_latency"]["latency_mode"] == "fast_chat"
    assert summary["latest_turn_latency"]["blocking_llm_calls"] == 2
    assert summary["latest_turn_latency"]["slowest_stage"]["stage"] == "thinking_reply"
    assert summary["latest_turn_latency"]["latency_mode_reasons"] == ["low_risk_short_chat"]
    assert summary["latest_turn_latency"]["turn_latency_trace"][0]["stage"] == "conscious_loop"
    assert summary["scheduler_skip_reason"] == "budget_ok"
    assert summary["cognitive_selector_skip_reason"] == "diagnostic_expectation_pool_noisy"
    assert summary["delivery_skip_reason"] == "delivery_assessor_reject"


def test_observability_summary_clamps_idle_reflection_count() -> None:
    chat = ChatInterface.__new__(ChatInterface)

    chat.read_self_loop_daemon_status = MethodType(lambda self: {"status": "running"}, chat)
    chat.read_idle_introspection_status = MethodType(
        lambda self: {"reflection_count_this_session": 10, "max_per_session": 4},
        chat,
    )
    chat.read_background_continuity_status = MethodType(lambda self: {}, chat)
    chat.read_m14_2_environment_events = MethodType(lambda self, limit=20: [], chat)
    chat.read_m14_2_scheduled_intents = MethodType(lambda self: [], chat)
    chat.read_queued_outreach = MethodType(lambda self: [], chat)
    chat.read_conversation_log = MethodType(lambda self, limit=300: [], chat)

    summary = chat.read_m14_2_observability_summary()

    assert summary["reflection_count"] == 4
    assert summary["reflection_max"] == 4
