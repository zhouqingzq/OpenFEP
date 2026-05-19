"""Diagnose why Path B idle reflection / background ticks show zero (CLI)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from segmentum.dialogue.runtime.m13_idle import gather_idle_structural_signals
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    read_runner_lock,
    runner_lock_is_alive,
)
from segmentum.dialogue.runtime.m14_2_self_loop import default_session_root
from segmentum.dialogue.runtime.mvp_loop import MVPStateStore, SYSTEM_FILE_NAMES


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _tail_jsonl(path: Path, *, limit: int = 40) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fmt_ts(ts: int) -> str:
    if ts <= 0:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def diagnose_session(session_root: Path) -> int:
    session_root = session_root.resolve()
    print(f"session_root: {session_root}")
    if not session_root.is_dir():
        print("VERDICT: session directory missing — check --persona / --session or --session-root")
        return 2

    store = MVPStateStore(session_root)
    state = store.load()
    m13 = state.get("m13_drive_state", {})
    initiative = m13.get("initiative", {}) if isinstance(m13, dict) else {}
    if not isinstance(initiative, dict):
        initiative = {}
    idle = initiative.get("idle_introspection", {})
    if not isinstance(idle, dict):
        idle = {}
    bg = initiative.get("background_continuity", {})
    if not isinstance(bg, dict):
        bg = {}

    temporal = state.get("temporal_state", {})
    if not isinstance(temporal, dict):
        temporal = {}
    now = int(time.time())
    turn_index = int(temporal.get("last_turn_index", 0) or 0)
    last_user = int(temporal.get("last_user_turn_at", 0) or 0)
    idle_elapsed = max(0, now - last_user) if last_user > 0 else -1

    print("\n## Opt-in / counters")
    print(f"  initiative.user_opt_in: {initiative.get('user_opt_in')}")
    print(f"  idle_introspection.enabled: {idle.get('enabled')}")
    print(f"  idle_introspection.user_opt_in: {idle.get('user_opt_in')}")
    print(f"  idle last_skip_reason: {idle.get('last_skip_reason') or '—'}")
    print(f"  reflections this session: {idle.get('reflection_count_this_session', 0)}/{idle.get('max_per_session', '?')}")
    print(f"  last_introspection_at: {_fmt_ts(int(idle.get('last_introspection_at', 0) or 0))}")
    print(f"  background_continuity.user_opt_in: {bg.get('user_opt_in')}")
    print(f"  background runner_kind: {bg.get('runner_kind', 'none')}")
    print(f"  ticks_today / lifetime: {bg.get('ticks_today', 0)} / {bg.get('idle_ticks_lifetime', 0)}")
    print(f"  llm_calls_today: {bg.get('llm_calls_today', 0)}/{bg.get('llm_calls_budget_per_day', '?')}")
    print(f"  tokens_used_today: {bg.get('tokens_used_today', 0)}/{bg.get('tokens_budget_per_day', '?')}")
    print(f"  last_budget_block_reason: {bg.get('last_budget_block_reason') or '—'}")
    print(f"  last_user_turn idle_elapsed: {idle_elapsed}s (need >= {idle.get('idle_threshold_seconds', 90)} for UI idle path)")

    lock = read_runner_lock(session_root)
    alive = runner_lock_is_alive(lock)
    print("\n## Daemon lock (runner.lock)")
    if lock is None:
        print("  no runner.lock — daemon not running (or never acquired lock)")
    else:
        print(f"  pid={lock.pid} host={lock.host} kind={lock.runner_kind} started={_fmt_ts(lock.started_at)}")
        print(f"  process_alive: {alive}")

    signals = gather_idle_structural_signals(state, now=now, turn_index=turn_index)
    sig = signals.to_dict()
    print("\n## Structural pre-filter (would background/UI call LLM?)")
    print(f"  should_run_llm: {sig.get('should_run_llm')}")
    print(f"  boredom_band: {sig.get('boredom_band')} (level={sig.get('boredom_level')})")
    print(f"  open_items_concrete_count: {sig.get('open_items_concrete_count')}")
    print(f"  unsettled_pending_settlement_count: {sig.get('unsettled_pending_settlement_count')}")
    print(f"  just_outreached_recently: {sig.get('just_outreached_recently')}")

    log_path = session_root / "conversation_log.jsonl"
    rows = _tail_jsonl(log_path, limit=60)
    bg_audit = [r for r in rows if r.get("event") == "m14_1_background_audit"]
    idle_audit = [r for r in rows if r.get("event") == "m13_idle_audit"]

    print("\n## Recent audit (conversation_log.jsonl tail)")
    print(f"  m14_1_background_audit rows (tail): {len(bg_audit)}")
    for row in bg_audit[-5:]:
        print(f"    - type={row.get('type')} skip={row.get('skip_reason', '')} at={_fmt_ts(int(row.get('at', 0) or 0))}")
    print(f"  m13_idle_audit rows (tail): {len(idle_audit)}")
    for row in idle_audit[-5:]:
        print(f"    - type={row.get('type')} skip={row.get('skip_reason', row.get('reason', ''))}")

    print("\n## Verdict")
    verdicts: list[str] = []

    if not bool(initiative.get("user_opt_in")) or not bool(idle.get("user_opt_in")):
        verdicts.append("M13 opt-in off — enable proactive + idle introspection in sidebar.")
    if bool(bg.get("user_opt_in")) and lock is None:
        verdicts.append("M14.1 opted in but no daemon lock — start: python -m segmentum.dialogue.runtime.m14_2_self_loop --persona … --session …")
    if bool(bg.get("user_opt_in")) and lock is not None and not alive:
        verdicts.append("Stale runner.lock (dead PID) — remove runner.lock or restart daemon.")
    if int(bg.get("ticks_today", 0) or 0) == 0 and not bg_audit:
        verdicts.append("No background ticks recorded — daemon likely never ran on THIS session_root.")
    if not idle_audit:
        verdicts.append("No m13_idle_audit — Streamlit idle hook was not firing (fixed in app if you pulled latest).")
    if idle_elapsed >= 0 and idle_elapsed < int(idle.get("idle_threshold_seconds", 90) or 90):
        verdicts.append(f"User not idle long enough ({idle_elapsed}s) — wait without sending messages.")
    if not sig.get("should_run_llm"):
        verdicts.append("Structural gate closed — need open_items w/ concrete next_check, boredom medium/high, or unsettled settlements.")
    if bg_audit and all(r.get("skip_reason") == "no_structural_signal" for r in bg_audit[-3:] if r.get("type") == "BackgroundIdleTickEvent"):
        verdicts.append("Daemon ticks run but skip LLM: no_structural_signal (boredom still low, etc.).")

    if not verdicts:
        if int(idle.get("reflection_count_this_session", 0) or 0) > 0 or bg_audit:
            verdicts.append("Pipeline has run — if panel still 0, UI may be reading a different session_root than this path.")
        else:
            verdicts.append("Config looks eligible; wait for idle + daemon tick or rerun Streamlit after idle hook.")

    for i, line in enumerate(verdicts, 1):
        print(f"  {i}. {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose MVP idle reflection / M14.1 background ticks.")
    parser.add_argument("--session-root", type=Path, default=None, help="Full path to sessions/<id> folder")
    parser.add_argument("--persona", default="", help="Persona id (mvp_personas/<persona>/sessions/…)")
    parser.add_argument("--session", default="m56_live", help="Session id (default m56_live)")
    args = parser.parse_args(argv)
    if args.session_root:
        root = args.session_root
    elif args.persona:
        root = default_session_root(args.persona, args.session or "m56_live")
    else:
        parser.error("provide --session-root or --persona")
        return 2
    return diagnose_session(root)


if __name__ == "__main__":
    raise SystemExit(main())
