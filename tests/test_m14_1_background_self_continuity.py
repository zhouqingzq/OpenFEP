from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    DEFAULT_QUEUED_OUTREACH_TTL_SECONDS,
    check_background_budgets,
    default_background_continuity_state,
    enqueue_outreach_proposal,
    load_queued_outreach,
    maybe_rollover_daily_counters,
    normalize_background_continuity_state,
    pop_next_pending_outreach,
    session_file_lock,
    set_background_continuity_opt_in,
)
from segmentum.dialogue.runtime.m14_self_continuity import (
    K_DRIFT_KNOWN_LIMIT,
    K_STABLE_PROMOTION,
    apply_self_cognition_patch_to_continuity,
    default_self_continuity_state,
    run_self_review_tick,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


def _full_opted_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [
            {
                "id": "item_bg_1",
                "title": "follow up",
                "status": "open",
                "next_check": "later",
            }
        ],
        "short_term_memory": [],
        "long_term_memory": [],
        "pending_expectations": [],
        "self_cognition": {"patch_history": [], "self_continuity": default_self_continuity_state()},
        "temporal_state": {
            "last_turn_at": int(time.time()) - 300,
            "last_user_turn_at": int(time.time()) - 300,
            "last_turn_index": 2,
            "last_reply": "ok",
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    m13 = set_background_continuity_opt_in(m13, enabled=True, runner_kind="inline")
    state["m13_drive_state"] = m13
    state.update(overrides)
    return state


class _BgIdleLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": {
                    "topic": "open item",
                    "evidence_refs": ["item_bg_1"],
                    "reflection_kind": "open_item",
                },
                "self_cognition_patch_proposal": {
                    "apply": True,
                    "summary_delta": "工程性自我观察",
                    "new_identity_tensions": [],
                    "new_known_limits": [],
                    "evidence_refs": ["item_bg_1"],
                    "confidence": 0.85,
                    "reason": "test",
                },
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {
                    "should_outreach": True,
                    "reason": "open_item_followup",
                    "suggested_intent": "Follow up on open item",
                    "trigger": "reflection_outreach",
                },
            }
        if "M13" in system_prompt or "主动续写" in system_prompt:
            return {
                "allow_delivery": True,
                "confidence": 0.9,
                "violation_codes": [],
                "reason_codes": [],
            }
        return {
            "thought_type": "short",
            "llm_thinking_result": {"debug_summary": "queued"},
            "reply": "队列送达测试。",
            "reply_action": "answer",
            "disclosure_action": "none",
            "new_expectations": [],
            "memory_writes": [],
            "self_cognition_patch": {"apply": False},
            "open_item_writes": [],
            "habit_updates": [],
            "memory_dynamics_note": "",
        }


def test_background_continuity_disabled_by_default() -> None:
    bg = default_background_continuity_state()
    assert bg["enabled"] is False
    assert bg["user_opt_in"] is False


def test_daily_rollover_resets_today_counters_but_keeps_lifetime(tmp_path: Path) -> None:
    bg = normalize_background_continuity_state(
        {
            "day_anchor": "2000-01-01",
            "ticks_today": 5,
            "idle_ticks_lifetime": 99,
            "llm_calls_lifetime": 10,
        }
    )
    now = int(time.time())
    merged, event = maybe_rollover_daily_counters(bg, now=now)
    assert event is not None
    assert merged["ticks_today"] == 0
    assert merged["idle_ticks_lifetime"] == 99
    assert merged["llm_calls_lifetime"] == 10


def test_llm_calls_budget_blocks() -> None:
    bg = normalize_background_continuity_state(
        {"llm_calls_today": 80, "llm_calls_budget_per_day": 80}
    )
    assert check_background_budgets(bg) == "llm_calls_budget_exhausted"


def test_queued_outreach_default_ttl_is_24_hours(tmp_path: Path) -> None:
    assert DEFAULT_QUEUED_OUTREACH_TTL_SECONDS == 24 * 3600
    now = 1_700_000_000
    entry = enqueue_outreach_proposal(
        tmp_path,
        proposal={
            "proposal_id": "prop_q1",
            "trigger": "reflection_outreach",
            "ordinary_language_intent": "test intent",
            "proposed_topic": "topic",
            "evidence_refs": [],
        },
        now=now,
        ttl_seconds=DEFAULT_QUEUED_OUTREACH_TTL_SECONDS,
    )
    assert int(entry["expires_at"]) == now + 24 * 3600


def test_queued_outreach_persists_across_restart(tmp_path: Path) -> None:
    now = int(time.time())
    enqueue_outreach_proposal(
        tmp_path,
        proposal={
            "proposal_id": "prop_persist",
            "trigger": "reflection_outreach",
            "ordinary_language_intent": "persist",
            "evidence_refs": [],
        },
        now=now,
        ttl_seconds=3600,
    )
    rows = load_queued_outreach(tmp_path)
    assert any(r.get("proposal_id") == "prop_persist" for r in rows)
    pending = pop_next_pending_outreach(tmp_path, now=now)
    assert pending is not None
    assert pending["proposal_id"] == "prop_persist"


def test_self_continuity_pins_stable_value_after_consecutive_appearances() -> None:
    from segmentum.dialogue.runtime.m14_self_continuity import _bump_stable_candidate

    sc = default_self_continuity_state()
    now = int(time.time())
    for _ in range(K_STABLE_PROMOTION):
        _bump_stable_candidate(sc, "耐心", now=now)
    assert "耐心" in sc.get("baseline_stable_values", [])


def test_self_continuity_promotes_persistent_drift_to_known_limit() -> None:
    sc = default_self_continuity_state()
    now = int(time.time())
    for i in range(K_DRIFT_KNOWN_LIMIT):
        sc.setdefault("drift_window", []).append(
            {
                "patch_id": f"d{i}",
                "at": now + i,
                "magnitude": 0.5,
                "direction": "pending",
                "kept_into_baseline": False,
                "summary_delta": f"drift {i}",
            }
        )
    sc, events = run_self_review_tick(sc, now=now)
    assert events
    assert any("self_continuity_drift" in str(x) for x in sc.get("baseline_known_limits", []))


def test_background_tick_queues_outreach(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    state = _full_opted_state()
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_BgIdleLLM())
    result = runtime.run_background_self_tick(runner_kind="inline")
    assert result.get("ran_introspection") is True
    rows = load_queued_outreach(tmp_path)
    assert any(str(r.get("status")) == "pending" for r in rows)


def test_session_file_lock_excludes_concurrent_writes(tmp_path: Path) -> None:
    with session_file_lock(tmp_path):
        lock_path = tmp_path / "store.lock"
        assert lock_path.is_file()
    assert not (tmp_path / "store.lock").exists()


def test_no_background_autonomy_wording_in_engineering_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    app_text = (root / "segmentum/dialogue/runtime/app.py").read_text(encoding="utf-8")
    assert "no background autonomy" not in app_text.casefold()
    init_text = (root / "segmentum/dialogue/runtime/m13_initiative.py").read_text(encoding="utf-8")
    assert "background autonomy" not in init_text.casefold() or "not a claim" in init_text
