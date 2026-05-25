from __future__ import annotations

import time
from pathlib import Path

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state, set_initiative_user_opt_in
from segmentum.dialogue.runtime.m13_memory_efe import normalize_expectations_for_efe
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    enqueue_outreach_proposal,
    load_queued_outreach,
    set_background_continuity_opt_in,
)
from segmentum.dialogue.runtime.m14_7_recall_scoring import score_recall_candidate
from segmentum.dialogue.runtime.m15_3_cleanup_control import (
    CleanupOwner,
    cleanup_ineligibility_reason,
    detect_cleanup_intents,
    is_strictly_traceable,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


NOW = 1_900_000_000


def _m13(*, profile: str = "bounded_default") -> dict[str, object]:
    m13 = set_initiative_user_opt_in(default_m13_drive_state(), enabled=True)
    initiative = normalize_initiative_state(normalize_m13_drive_state(m13)["initiative"])
    initiative["proactive_policy_profile"] = profile
    initiative["implicit_idle_delivery"] = True
    m13["initiative"] = initiative
    return m13


def _low_open(index: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": f"oi_{index}",
        "status": "open",
        "title": f"unanchored open item {index}",
        "next_check": "next_user_turn",
        "created_at": NOW - 10 * 86400,
    }
    row.update(overrides)
    return row


def _low_pending(index: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": f"exp_{index}",
        "status": "pending",
        "content": f"unanchored pending expectation {index}",
        "verify_on": "next_user_turn",
        "created_at": NOW - 5 * 86400,
    }
    row.update(overrides)
    return row


def test_strict_traceability_rejects_self_referential_evidence() -> None:
    row = {"id": "item_001", "status": "open", "evidence_refs": ["item_001"], "next_check": "next_user_turn"}

    assert is_strictly_traceable(row) is False
    assert cleanup_ineligibility_reason(row, now=NOW, phase="idle") == "self_referential_evidence_only"

    anchored = {**row, "evidence_refs": ["turn_7"]}
    assert is_strictly_traceable(anchored) is True


def test_cleanup_detection_is_audit_only_on_bounded_default() -> None:
    state = {
        "open_items": [_low_open(i) for i in range(6)],
        "pending_expectations": [],
        "m13_drive_state": _m13(profile="bounded_default"),
    }

    result = detect_cleanup_intents(state, now=NOW, turn_index=4, source="idle_cognitive_tick")

    assert any(event["type"] == "OpenItemBacklogDetectedEvent" for event in result.events)
    assert result.intents == []
    meta = state["m13_drive_state"]["meta_control_intents"]  # type: ignore[index]
    assert meta["cleanup_active"] == []
    assert all(row["status"] == "open" for row in state["open_items"])  # type: ignore[index]


def test_cleanup_owner_marks_rows_without_deleting(monkeypatch) -> None:
    monkeypatch.setenv("SEGMENTUM_CLEANUP_CONTROL_APPLY", "1")
    state = {
        "open_items": [_low_open(i) for i in range(6)],
        "pending_expectations": [_low_pending(i) for i in range(6)],
        "m13_drive_state": _m13(profile="bounded_default"),
    }

    detection = detect_cleanup_intents(
        state,
        now=NOW,
        turn_index=4,
        source="idle_cognitive_tick",
        current_idle_tick_event={"reject_reason": "no_high_value_target"},
    )
    result = CleanupOwner.apply_intents(state, now=NOW, turn_index=4, source="idle_cognitive_tick")

    assert len(detection.intents) == 3
    assert len(state["open_items"]) == 6
    assert len(state["pending_expectations"]) == 6
    assert all(row["status"] == "diagnostic_only" for row in state["open_items"])  # type: ignore[index]
    assert all(row["status"] == "expired" for row in state["pending_expectations"])  # type: ignore[index]
    assert result.ops["diagnostic_open_items"] == 6
    assert result.ops["expired_pending_expectations"] == 6
    assert any(event["type"] == "CleanupRunEvent" and event["row_deletion_count"] == 0 for event in result.events)


def test_structural_merge_does_not_use_content_similarity() -> None:
    state = {
        "open_items": [
            _low_open(1, content="same text", created_at=NOW - 100, evidence_refs=["turn_a"], bound_memory_ids=["mem_a"]),
            _low_open(2, content="same text", created_at=NOW - 50, evidence_refs=["turn_a"], bound_memory_ids=["mem_a"]),
            _low_open(3, content="same text"),
            _low_open(4, content="same text"),
        ],
        "pending_expectations": [],
        "m13_drive_state": _m13(profile="streamlit_open_chat"),
    }
    intent = {
        "intent_id": "cleanup_structural",
        "at": NOW,
        "turn_index": 4,
        "detector": "test",
        "intent_kind": "cleanup_open_item_backlog",
        "payload": {"candidate_ids": ["oi_1", "oi_2", "oi_3", "oi_4"]},
        "expires_at": NOW + 100,
    }
    state["m13_drive_state"]["meta_control_intents"] = {"cleanup_active": [intent]}  # type: ignore[index]

    result = CleanupOwner.apply_intents(state, now=NOW, turn_index=4, source="idle_cognitive_tick")

    statuses = {row["id"]: row["status"] for row in state["open_items"]}  # type: ignore[index]
    assert statuses["oi_2"] == "merged_into:oi_1"
    assert statuses["oi_3"] == "diagnostic_only"
    assert statuses["oi_4"] == "diagnostic_only"
    assert result.ops["merged_duplicates"] == 1


def test_recall_scoring_filters_cleanup_statuses() -> None:
    active = {"id": "mem_a", "content": "benchmark result", "salience": 0.9, "precision": 0.9}
    diagnostic = {**active, "status": "diagnostic_only"}
    merged = {**active, "status": "merged_into:mem_root"}
    deprioritized = {**active, "recall_deprioritized_until": NOW + 3600}

    assert score_recall_candidate(active, query=["benchmark"], now=NOW, retrieved_context={"phase": "idle"}) > 0
    assert score_recall_candidate(diagnostic, query=["benchmark"], now=NOW, retrieved_context={"phase": "idle"}) == 0
    assert score_recall_candidate(merged, query=["benchmark"], now=NOW, retrieved_context={"phase": "idle"}) == 0
    assert score_recall_candidate(deprioritized, query=["benchmark"], now=NOW, retrieved_context={"phase": "idle"}) == 0


def test_memory_efe_rejects_self_reference_and_seeds_bound_memory_floor() -> None:
    self_ref_state = {
        "open_items": [
            {
                "id": "item_self",
                "status": "open",
                "content": "self referenced only",
                "next_check": "next_user_turn",
                "created_at": NOW - 2000,
                "evidence_refs": ["item_self"],
            }
        ],
        "pending_expectations": [],
        "temporal_state": {"last_user_turn_at": NOW - 5000},
        "m13_drive_state": _m13(),
    }
    self_ref = normalize_expectations_for_efe(self_ref_state, now=NOW, phase="idle")
    assert self_ref.eligible_for_efe == []
    assert self_ref.diagnostic_only[0].ineligibility_reason == "self_referential_evidence_only"

    bound_state = {
        "pending_expectations": [
            {
                "id": "exp_low_bound",
                "status": "pending",
                "content": "check the low scoring anchor",
                "due_at_epoch": NOW - 5000,
                "expected_window_seconds": 900,
                "evidence_refs": ["turn_anchor"],
                "bound_memory_ids": ["ltm_low"],
                "confidence": 0.9,
            }
        ],
        "long_term_memory": [
            {"id": "ltm_low", "content": "orthogonal", "salience": 0.1, "precision": 0.1, "value_proxy": 0.1}
        ],
        "temporal_state": {"last_user_turn_at": NOW - 5000},
        "m13_drive_state": _m13(),
    }
    seeded = normalize_expectations_for_efe(bound_state, now=NOW, phase="idle")
    assert [row.expectation_id for row in seeded.eligible_for_efe] == ["exp_low_bound"]
    assert seeded.bound_recall_seed_ids == ["ltm_low"]
    assert seeded.bound_recall_floor_bypassed_ids == ["ltm_low"]


def test_queued_outreach_delivery_defaults_to_active_session_only(tmp_path: Path) -> None:
    persona_root = tmp_path / "persona"
    source_root = persona_root / "sessions" / "old_tab"
    current_root = persona_root / "sessions" / "current_tab"
    source_root.mkdir(parents=True)
    current_root.mkdir(parents=True)
    now = int(time.time())
    enqueue_outreach_proposal(
        source_root,
        proposal={
            "proposal_id": "prop_old_tab",
            "trigger": "scheduled_outreach",
            "ordinary_language_intent": "follow up from old tab",
            "proposed_topic": "scheduled outreach",
            "session_id": "old_tab",
            "evidence_refs": ["turn_old"],
        },
        now=now - 120,
        ttl_seconds=3600,
        due_at=now - 60,
        source_intent_id="sint_old_tab",
    )
    store = MVPStateStore(current_root, shared_root=persona_root)
    m13 = set_background_continuity_opt_in(_m13(), enabled=True, runner_kind="inline")
    store.save({"m13_drive_state": m13, "temporal_state": {"last_user_turn_at": now - 7200}})
    runtime = MVPDialogueRuntime(store=store, llm=None)

    result = runtime.maybe_drain_queued_outreach(turn_index=4, now=now)

    assert result["drained"] is False
    assert result["reason"] == "empty_queue"
    assert load_queued_outreach(source_root)[0]["status"] == "pending"
    assert load_queued_outreach(current_root) == []
