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
    assert "generic_self_only_open_item" in text
    assert "intro_should_outreach: True" in text
    assert "pending_user_message: hello" in text
    assert "## Diagnose verdicts" in text
    assert "## Recent audit tail" in text
