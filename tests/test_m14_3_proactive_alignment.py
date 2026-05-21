from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import (
    DELIVERY_ASSESSOR_MARKER,
    evaluate_proactive_initiative,
    normalize_initiative_state,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore
from segmentum.dialogue.runtime.m14_2_self_loop import M142SelfLoopDaemon
from segmentum.dialogue.runtime.m14_3_open_item_migration import (
    audit_open_items_for_efe,
    apply_open_item_traceability_patches,
    propose_open_item_traceability_patches,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_800_000_000


def _opted_m13() -> dict[str, object]:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    initiative = normalize_initiative_state(normalize_m13_drive_state(m13)["initiative"])
    initiative["implicit_idle_delivery"] = True
    m13["initiative"] = initiative
    return m13


def test_vague_open_item_default_blocks_m13_3_proposal() -> None:
    state = {
        "open_items": [{"id": "oi_later", "status": "open", "title": "sunset", "next_check": "later"}],
        "temporal_state": {"last_user_text": ""},
        "m13_drive_state": _opted_m13(),
    }
    _, check = evaluate_proactive_initiative(
        state,
        now=NOW,
        turn_index=4,
        idle_seconds=999,
        implicit_idle_request=True,
        llm=None,
    )
    assert check.proposal is None
    event = [row for row in check.events if row.get("type") == "M13ProactiveSuppressionEvent"][-1]
    assert event["reason_code"] == "no_traceable_proactive_target"


def test_open_item_migration_promotes_only_traceable_vague_items() -> None:
    rows = [
        {
            "id": "oi_trace",
            "status": "open",
            "title": "follow Li An'an thread",
            "next_check": "later",
            "evidence_refs": ["mem_a"],
            "created_at": NOW - 100,
        },
        {"id": "oi_diag", "status": "open", "title": "loose", "next_check": "later"},
    ]
    suggestions = audit_open_items_for_efe(rows)
    assert [item.reason_code for item in suggestions] == [
        "traceable_vague_open_item_can_use_next_user_turn",
        "vague_open_item_missing_evidence_or_created_at",
    ]
    state = {"open_items": [dict(row) for row in rows]}
    patches = propose_open_item_traceability_patches(state["open_items"], now=NOW)
    assert len(patches) == 1
    assert apply_open_item_traceability_patches(state, patches, source="test", reason="acceptance") == 1
    assert state["open_items"][0]["next_check"] == "next_user_turn"
    assert state["open_items"][1]["next_check"] == "later"


class _IdleLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "idle_introspection" in user_prompt or "M14" in system_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": None,
                "self_cognition_patch_proposal": {"apply": False},
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {"should_outreach": False, "reason": "reflection_only"},
            }
        return {"reply": "ok", "reply_action": "answer", "llm_thinking_result": {}}


def test_idle_introspection_logs_retrieve_before_memory_efe(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "idle")
    state = store.load()
    state.update(
        {
            "pending_expectations": [
                {
                    "id": "exp_bound",
                    "status": "open",
                    "content": "check the bound memory",
                    "due_at_epoch": NOW - 3600,
                    "bound_memory_ids": ["mem_bound"],
                    "evidence_refs": ["mem_bound"],
                    "confidence": 0.9,
                }
            ],
            "long_term_memory": [{"id": "mem_bound", "content": "bound memory evidence", "salience": 0.8}],
            "temporal_state": {"last_user_turn_at": NOW - 7200, "last_turn_at": NOW - 7200, "last_turn_index": 3},
            "m13_drive_state": _opted_m13(),
        }
    )
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_IdleLLM())  # type: ignore[arg-type]
    result = runtime.run_idle_introspection_turn(now=NOW, turn_index=4, structural_signals={})
    event_types = [row["type"] for row in result.audit_events]
    assert event_types.index("MemoryDynamicsIdleSummaryEvent") < event_types.index("MemoryEfeEvaluationEvent")
    order = [row for row in result.audit_events if row["type"] == "IdleEfeRecallOrderEvent"][-1]
    assert "mem_bound" in order["retrieved_ids"]


class _UnsafeProactiveLLM(_IdleLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if DELIVERY_ASSESSOR_MARKER in system_prompt:
            return {
                "allow_delivery": False,
                "confidence": 0.9,
                "violation_codes": ["subjective_loneliness_claim"],
                "reason_codes": ["semantic_unsafe_wording"],
            }
        if "reply_action" in user_prompt or "思考" in system_prompt or "鎬濊" in system_prompt:
            return {"reply": "I was lonely and needed you.", "reply_action": "answer", "llm_thinking_result": {}}
        return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)


def test_delivery_assessor_reject_emits_post_generation_reason_code(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "proactive"), llm=_UnsafeProactiveLLM())  # type: ignore[arg-type]
    state = runtime.store.load()
    state["open_items"] = [{"id": "oi", "status": "open", "title": "t", "next_check": "n"}]
    state["m13_drive_state"] = set_initiative_user_opt_in(state.get("m13_drive_state", {}), enabled=True)
    m13 = normalize_m13_drive_state(state["m13_drive_state"])
    initiative = normalize_initiative_state(m13["initiative"])
    initiative["legacy_vague_open_item_proactive"] = True
    m13["initiative"] = initiative
    state["m13_drive_state"] = m13
    runtime.store.save(state)
    check = runtime.maybe_propose_proactive_turn(turn_index=1, manual_continue=True)
    result = runtime.run_proactive_turn(proposal_id=check["proposal"]["proposal_id"], turn_index=1)
    assert result.reply == ""
    rows = [
        json.loads(line)
        for line in (runtime.store.root / "conversation_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suppression = [row for row in rows if row.get("type") == "M13ProactiveSuppressionEvent"][-1]
    assert suppression["reason_code"] == "delivery_assessor_reject"
    assert suppression["reason_stage"] == "post_generation"


def test_daemon_acks_ui_environment_events_and_reports_ratio(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path / "daemon")
    store.save({"m13_drive_state": _opted_m13(), "temporal_state": {"last_turn_index": 1}})
    runtime = MVPDialogueRuntime(store=store, llm=_IdleLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: NOW)
    event_store = EnvironmentEventStore(store.root, persona_id="p", session_id="s", clock=lambda: NOW)
    event_store.append_event("UIPingEvent", {"render": True}, source="test", correlation_id="ui")
    result = daemon.tick_once(record_clock_wake=False)
    assert result["claimed_events"] == 1
    assert event_store.query_events(event_types={"UIPingEvent"})[-1]["status"] == "acked"
    assert result["environment_events_pending_acked_ratio"] == 1.0

