from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    DEFAULT_QUEUED_OUTREACH_TTL_SECONDS,
    MAX_QUEUED_OUTREACH_TTL_SECONDS,
    MIN_QUEUED_OUTREACH_TTL_SECONDS,
    check_background_budgets,
    default_background_continuity_state,
    enqueue_outreach_proposal,
    expire_queued_outreach,
    load_queued_outreach,
    maybe_rollover_daily_counters,
    normalize_background_continuity_state,
    pop_next_pending_outreach,
    read_runner_lock,
    release_runner_lock,
    session_file_lock,
    set_background_continuity_opt_in,
    try_acquire_runner_lock,
)
from segmentum.dialogue.runtime.m14_self_continuity import (
    K_DRIFT_KNOWN_LIMIT,
    K_STABLE_PROMOTION,
    MIN_BASELINE_UPDATE_CONFIDENCE,
    apply_self_cognition_patch_to_continuity,
    default_self_continuity_state,
    run_self_review_tick,
)
from segmentum.dialogue.runtime.m14_1_self_runner import BackgroundSelfRunner
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


def _full_opted_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [
            {
                "id": "item_bg_1",
                "title": "follow up",
                "status": "open",
                "next_check": "next_user_turn",
                "evidence_refs": ["mem_bg_1"],
                "created_at": int(time.time()) - 1000,
            }
        ],
        "short_term_memory": [{"id": "mem_bg_1", "content": "traceable background follow-up memory"}],
        "long_term_memory": [],
        "pending_expectations": [],
        "self_cognition": {"patch_history": [], "self_continuity": default_self_continuity_state()},
        "temporal_state": {
            "last_turn_at": int(time.time()) - 7200,
            "last_user_turn_at": int(time.time()) - 7200,
            "last_turn_index": 2,
            "last_reply": "ok",
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    m13 = set_background_continuity_opt_in(m13, enabled=True, runner_kind="inline")
    m13["initiative"]["implicit_idle_delivery"] = True
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
                    "evidence_refs": ["mem_bg_1"],
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


class _LargeIdleLLM(_BgIdleLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        payload = super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            payload = dict(payload)
            payload["padding"] = "x" * 20_000
        return payload


class _LowConfidenceIdleLLM(_BgIdleLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        payload = super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            patch = dict(payload["self_cognition_patch_proposal"])  # type: ignore[index]
            patch["confidence"] = MIN_BASELINE_UPDATE_CONFIDENCE - 0.05
            patch["summary_delta"] = "low confidence background delta"
            payload = dict(payload)
            payload["self_cognition_patch_proposal"] = patch
            payload["outreach_recommendation"] = {
                "should_outreach": False,
                "reason": "reflection_only",
                "suggested_intent": "",
                "trigger": "reflection_outreach",
            }
        return payload


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


def test_background_runner_requires_parent_opt_ins() -> None:
    state = {"m13_drive_state": default_m13_drive_state()}
    m13 = set_background_continuity_opt_in(state["m13_drive_state"], enabled=True, runner_kind="inline")  # type: ignore[arg-type]
    bg = m13["initiative"]["background_continuity"]  # type: ignore[index]
    assert bg["enabled"] is True
    assert m13["initiative"]["user_opt_in"] is False  # type: ignore[index]


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


def test_queued_outreach_preserves_traceability_fields(tmp_path: Path) -> None:
    now = 1_700_000_000
    entry = enqueue_outreach_proposal(
        tmp_path,
        proposal={
            "proposal_id": "prop_trace",
            "trigger": "memory_efe_outreach",
            "ordinary_language_intent": "Follow up on the unresolved expectation: traceable memory",
            "proposed_topic": "traceable memory",
            "trigger_evidence_refs": ["mem_1", "exp_1"],
            "traceable_expectation_id": "exp_1",
            "source_kind": "pending_expectation",
            "selection_reason_codes": ["memory_efe_should_outreach"],
        },
        now=now,
        ttl_seconds=3600,
    )
    assert entry["evidence_refs"] == ["mem_1", "exp_1"]
    assert entry["traceable_expectation_id"] == "exp_1"
    assert entry["source_kind"] == "pending_expectation"
    assert entry["selection_reason_codes"] == ["memory_efe_should_outreach"]


def test_queued_outreach_ttl_clamped() -> None:
    low = normalize_background_continuity_state({"queued_outreach_ttl_seconds": 1})
    high = normalize_background_continuity_state({"queued_outreach_ttl_seconds": 999999})
    assert low["queued_outreach_ttl_seconds"] == MIN_QUEUED_OUTREACH_TTL_SECONDS
    assert high["queued_outreach_ttl_seconds"] == MAX_QUEUED_OUTREACH_TTL_SECONDS


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


def test_queued_outreach_expires_after_ttl(tmp_path: Path) -> None:
    now = 1_700_000_000
    enqueue_outreach_proposal(
        tmp_path,
        proposal={"proposal_id": "prop_expire", "ordinary_language_intent": "x"},
        now=now,
        ttl_seconds=3600,
    )
    events = expire_queued_outreach(tmp_path, now=now + 3601)
    rows = load_queued_outreach(tmp_path)
    assert events and events[0]["type"] == "QueuedOutreachExpiredEvent"
    assert rows[0]["status"] == "expired"


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


def test_queued_outreach_drain_uses_delivery_path_and_counts_llm(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_BgIdleLLM())
    runtime.run_background_self_tick(runner_kind="inline")
    result = runtime.maybe_drain_queued_outreach(turn_index=3)
    rows = load_queued_outreach(tmp_path)
    bg = store.load()["m13_drive_state"]["initiative"]["background_continuity"]
    assert result["drained"] is True
    assert rows[0]["status"] == "delivered"
    assert result["llm_calls_delta"] >= 1
    assert bg["llm_calls_today"] >= 2
    assert bg["tokens_used_today"] > 1


def test_current_session_can_relay_due_outbox_from_sibling_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEGMENTUM_QUEUE_INCLUDE_OTHER_SESSIONS", "1")
    persona_root = tmp_path / "persona"
    source_root = persona_root / "sessions" / "old_tab"
    current_root = persona_root / "sessions" / "current_tab"
    source_root.mkdir(parents=True)
    store = MVPStateStore(current_root, shared_root=persona_root)
    store.save(_full_opted_state())
    now = int(time.time())
    enqueue_outreach_proposal(
        source_root,
        proposal={
            "proposal_id": "prop_old_tab",
            "trigger": "scheduled_outreach",
            "ordinary_language_intent": "follow up from old tab",
            "proposed_topic": "scheduled outreach",
            "persona_id": "胡桃",
            "session_id": "old_tab",
        },
        now=now - 120,
        ttl_seconds=3600,
        due_at=now - 60,
        source_intent_id="sint_old_tab",
    )
    runtime = MVPDialogueRuntime(store=store, llm=_BgIdleLLM())

    result = runtime.maybe_drain_queued_outreach(turn_index=5, now=now)

    assert result["drained"] is True
    source_rows = load_queued_outreach(source_root)
    current_rows = load_queued_outreach(current_root)
    assert source_rows[0]["status"] == "relayed"
    assert source_rows[0]["relayed_to_session_id"] == "current_tab"
    assert current_rows[0]["status"] == "delivered"
    log = (current_root / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "OutboxEntryRelayedEvent" in log


def test_background_tick_records_estimated_tokens_not_constant_one(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_LargeIdleLLM())
    runtime.run_background_self_tick(runner_kind="inline")
    bg = store.load()["m13_drive_state"]["initiative"]["background_continuity"]
    assert bg["llm_calls_today"] == 1
    assert bg["tokens_used_today"] > 1000


def test_low_confidence_background_patch_does_not_pollute_current_self_view(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_LowConfidenceIdleLLM())
    runtime.run_background_self_tick(runner_kind="inline")
    cognition = store.load()["self_cognition"]
    continuity = cognition["self_continuity"]
    assert "low confidence background delta" not in str(cognition.get("current_self_view", ""))
    assert continuity["drift_window"]


def test_session_file_lock_excludes_concurrent_writes(tmp_path: Path) -> None:
    with session_file_lock(tmp_path):
        lock_path = tmp_path / "store.lock"
        assert lock_path.is_file()
    assert not (tmp_path / "store.lock").exists()


def test_runner_lock_is_atomic_and_reports_collision(tmp_path: Path) -> None:
    ok, info = try_acquire_runner_lock(tmp_path, runner_kind="inline", now=1)
    assert ok is True
    assert info is not None
    ok2, existing = try_acquire_runner_lock(tmp_path, runner_kind="cli", now=2)
    assert ok2 is False
    assert existing is not None and existing.pid == info.pid
    release_runner_lock(tmp_path)


def test_background_runner_collision_event_without_starting_second_thread(tmp_path: Path) -> None:
    class RuntimeStub:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def append_background_audit(self, event: dict[str, object]) -> None:
            self.events.append(event)

        def record_streamlit_ping(self) -> None:
            pass

        def inline_runner_should_stop(self, *, idle_death_seconds: int) -> bool:
            return False

    runtime = RuntimeStub()
    ok, _info = try_acquire_runner_lock(tmp_path, runner_kind="inline", now=1)
    assert ok is True
    try:
        colliding = BackgroundSelfRunner(runtime, session_root=tmp_path, tick_interval_seconds=30)
        colliding.start()
        assert any(e.get("type") == "BackgroundRunnerCollisionEvent" for e in runtime.events)
    finally:
        release_runner_lock(tmp_path)
    assert read_runner_lock(tmp_path) is None


def test_inline_dev_fallback_runner_bumps_background_ticks_each_iteration(tmp_path: Path) -> None:
    """Field gap reproduction: the inline_dev_fallback runner must call
    ``run_background_self_tick`` each loop iteration in addition to the M14.2
    event-driven ``tick_once``. Without this wiring,
    ``background_ticks_today`` stays at 0 unless a ``ScheduledIntent`` is due —
    the failure mode seen on Hu Tao live sessions (M15.3 prompt §6 out-of-scope
    list).
    """
    state = _full_opted_state()
    state["open_items"] = []
    store = MVPStateStore(tmp_path)
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_BgIdleLLM())
    runner = BackgroundSelfRunner(
        runtime,
        session_root=tmp_path,
        persona_id="p",
        session_id="s",
        runner_kind="inline_dev_fallback",
        tick_interval_seconds=30,
    )
    runner.start()
    try:
        deadline = time.monotonic() + 8.0
        ticks_today = 0
        while time.monotonic() < deadline:
            bg = (
                store.load()
                .get("m13_drive_state", {})
                .get("initiative", {})
                .get("background_continuity", {})
            )
            ticks_today = int(bg.get("ticks_today", 0) or 0)
            if ticks_today >= 1:
                break
            time.sleep(0.05)
        assert ticks_today >= 1, "inline_dev_fallback runner did not bump background_ticks_today"
    finally:
        runner.stop(drain_wait_seconds=2.0)


def test_cli_accepts_persona_session_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "segmentum.dialogue.runtime.m14_1_self_runner",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "--persona" in result.stdout
    assert "--session" in result.stdout


def test_no_background_autonomy_wording_in_engineering_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    app_text = (root / "segmentum/dialogue/runtime/app.py").read_text(encoding="utf-8")
    assert "no background autonomy" not in app_text.casefold()
    init_text = (root / "segmentum/dialogue/runtime/m13_initiative.py").read_text(encoding="utf-8")
    assert "background autonomy" not in init_text.casefold() or "not a claim" in init_text


def test_background_tick_persists_skip_reason(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "bg_skip"), llm=None)
    runtime.set_initiative_user_opt_in(True)
    runtime.set_background_continuity_opt_in(True)
    runtime.set_idle_introspection_opt_in(True)
    result = runtime.run_background_self_tick(runner_kind="cli")
    assert result.get("skip_reason") in {"llm_unavailable", "no_structural_signal"}
    state = runtime.store.load()
    bg = state["m13_drive_state"]["initiative"]["background_continuity"]
    assert bg.get("last_background_skip_reason") in {"llm_unavailable", "no_structural_signal"}


def test_seed_mvp_session_resets_per_session_counters(tmp_path: Path) -> None:
    from segmentum.dialogue.runtime.chat import _reset_per_session_counters
    from segmentum.dialogue.runtime.mvp_loop import SYSTEM_FILE_NAMES

    persona_root = tmp_path / "persona"
    session_root = persona_root / "sessions" / "sess_new"
    persona_root.mkdir(parents=True)
    payload = {
        "initiative": {
            "proactive_count_this_session": 9,
            "idle_introspection": {"reflection_count_this_session": 99, "max_per_session": 4},
            "background_continuity": {"ticks_today": 50, "llm_calls_today": 40},
        }
    }
    session_root.mkdir(parents=True)
    (session_root / SYSTEM_FILE_NAMES["m13_drive_state"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _reset_per_session_counters(session_root)
    reloaded = json.loads((session_root / SYSTEM_FILE_NAMES["m13_drive_state"]).read_text(encoding="utf-8"))
    assert reloaded["initiative"]["proactive_count_this_session"] == 0
    assert reloaded["initiative"]["idle_introspection"]["reflection_count_this_session"] == 0
    assert reloaded["initiative"]["background_continuity"]["ticks_today"] == 0
