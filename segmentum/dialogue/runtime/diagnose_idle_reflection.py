"""Diagnose Path B idle reflection / proactive background activity (CLI)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from segmentum.dialogue.runtime.m13_idle import gather_idle_structural_signals
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    read_runner_lock,
    runner_lock_is_alive,
)
from segmentum.dialogue.runtime.m14_2_self_loop import default_session_root
from segmentum.dialogue.runtime.mvp_loop import MVPStateStore


VERDICT_CODES = frozenset(
    {
        "PROACTIVE_PIPELINE_NEVER_TICKED",
        "PROACTIVE_PIPELINE_TICKING_BUT_NO_TARGET",
        "PROACTIVE_PIPELINE_TICKING_NEEDS_TRACEABILITY",
        "PROACTIVE_PIPELINE_DELIVERED_RECENTLY",
        "IDLE_INTRO_NEVER_RAN",
        "IDLE_INTRO_PLAN_SELECTOR_MISMATCH",
        "DAEMON_PROCESS_DEAD",
        "DAEMON_PROCESS_ALIVE_NO_BACKGROUND_TICKS",
        "DAEMON_PROCESS_ALIVE_HEALTHY",
        "SESSION_ROOT_MISMATCH",
        "CONFIG_OPT_IN_OFF",
        "USER_NOT_IDLE_LONG_ENOUGH",
    }
)

IDLE_BACKGROUND_CHANNELS = frozenset(
    {
        "m13_idle_audit",
        "m14_idle_audit",
        "m14_4_implicit_idle_audit",
        "m14_1_background_audit",
        "m14_2_audit",
    }
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _fmt_ts(ts: int) -> str:
    if ts <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def summarize_log(path: Path) -> dict[str, Any]:
    """Stream the complete conversation log and retain compact diagnostics."""
    counts: dict[str, int] = {channel: 0 for channel in IDLE_BACKGROUND_CHANNELS}
    counts["IdleCognitiveTickEvent"] = 0
    latest_skip: dict[str, Any] = {}
    latest_stateful_skip: dict[str, Any] = {}
    latest_tick: dict[str, Any] = {}
    latest_mismatch: dict[str, Any] = {}
    latest_delivery: dict[str, Any] = {}
    latest_health: dict[str, Any] = {}
    latest_intro_plan: dict[str, Any] = {}

    for row in _iter_jsonl(path):
        event = str(row.get("event", "") or "")
        typ = str(row.get("type", "") or "")
        if event in counts:
            counts[event] += 1
        if typ == "IdleCognitiveTickEvent":
            counts["IdleCognitiveTickEvent"] += 1
            latest_tick = dict(row)
        if typ == "IdlePlanStructuralMismatchEvent":
            latest_mismatch = dict(row)
        if typ == "IdleIntrospectionPlanEvent":
            latest_intro_plan = dict(row)
        if typ == "SelfLoopDaemonHealthEvent":
            latest_health = dict(row)
        if event == "proactive_turn" or typ == "M13ProactiveGenerationEvent":
            latest_delivery = dict(row)
        if "skip_reason" in row or "reason_code" in row or "suppression_reason_code" in row:
            latest_skip = dict(row)
        if event in {"m13_idle_audit", "m14_idle_audit", "m14_4_implicit_idle_audit"} and (
            "skip_reason" in row or "reason_code" in row or "suppression_reason_code" in row
        ):
            latest_stateful_skip = dict(row)

    return {
        "counts": counts,
        "latest_skip": latest_skip,
        "latest_stateful_skip": latest_stateful_skip,
        "latest_tick": latest_tick,
        "latest_mismatch": latest_mismatch,
        "latest_delivery": latest_delivery,
        "latest_health": latest_health,
        "latest_intro_plan": latest_intro_plan,
    }


def verdicts_for_session(
    *,
    state: dict[str, Any],
    log_summary: dict[str, Any],
    lock_alive: bool,
    has_lock: bool,
    idle_elapsed: int,
    structural_should_run: bool,
) -> list[str]:
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

    counts = dict(log_summary.get("counts", {}))
    verdicts: list[str] = []
    if not bool(initiative.get("user_opt_in")) or not bool(idle.get("user_opt_in")):
        verdicts.append("CONFIG_OPT_IN_OFF")
    if bool(bg.get("user_opt_in")):
        if not has_lock or not lock_alive:
            verdicts.append("DAEMON_PROCESS_DEAD")
        elif int(bg.get("ticks_today", 0) or 0) == 0 and counts.get("m14_2_audit", 0) > 0:
            verdicts.append("DAEMON_PROCESS_ALIVE_NO_BACKGROUND_TICKS")
        else:
            verdicts.append("DAEMON_PROCESS_ALIVE_HEALTHY")
    if idle_elapsed >= 0 and idle_elapsed < int(idle.get("idle_threshold_seconds", 90) or 90):
        verdicts.append("USER_NOT_IDLE_LONG_ENOUGH")
    if counts.get("IdleCognitiveTickEvent", 0) == 0 and counts.get("m14_4_implicit_idle_audit", 0) == 0:
        verdicts.append("PROACTIVE_PIPELINE_NEVER_TICKED")

    latest_tick = log_summary.get("latest_tick", {})
    if isinstance(latest_tick, dict) and latest_tick:
        reject = str(latest_tick.get("reject_reason", "") or "")
        if reject in {"generic_self_only_open_item", "all_targets_traceability_failed", "no_recall_hit_for_eligible"}:
            verdicts.append("PROACTIVE_PIPELINE_TICKING_NEEDS_TRACEABILITY")
        elif reject:
            verdicts.append("PROACTIVE_PIPELINE_TICKING_BUT_NO_TARGET")

    if log_summary.get("latest_delivery"):
        verdicts.append("PROACTIVE_PIPELINE_DELIVERED_RECENTLY")
    if counts.get("m14_idle_audit", 0) == 0 and not log_summary.get("latest_intro_plan"):
        verdicts.append("IDLE_INTRO_NEVER_RAN")
    if log_summary.get("latest_mismatch"):
        verdicts.append("IDLE_INTRO_PLAN_SELECTOR_MISMATCH")
    if not structural_should_run and "PROACTIVE_PIPELINE_TICKING_BUT_NO_TARGET" not in verdicts:
        verdicts.append("PROACTIVE_PIPELINE_TICKING_BUT_NO_TARGET")

    return [code for code in dict.fromkeys(verdicts) if code in VERDICT_CODES]


def diagnose_session(session_root: Path) -> int:
    session_root = session_root.resolve()
    print(f"session_root: {session_root}")
    if not session_root.is_dir():
        print("VERDICT: SESSION_ROOT_MISMATCH")
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
    print(f"  idle last_skip_reason (state, may be stale): {idle.get('last_skip_reason') or '-'}")
    print(f"  reflections this session: {idle.get('reflection_count_this_session', 0)}/{idle.get('max_per_session', '?')}")
    print(f"  last_introspection_at: {_fmt_ts(int(idle.get('last_introspection_at', 0) or 0))}")
    print(f"  background_continuity.user_opt_in: {bg.get('user_opt_in')}")
    print(f"  background runner_kind: {bg.get('runner_kind', 'none')}")
    print(f"  ticks_today / lifetime: {bg.get('ticks_today', 0)} / {bg.get('idle_ticks_lifetime', 0)}")
    print(f"  last_user_turn idle_elapsed: {idle_elapsed}s")

    lock = read_runner_lock(session_root)
    alive = runner_lock_is_alive(lock)
    print("\n## Daemon lock")
    if lock is None:
        print("  no runner.lock")
    else:
        print(f"  pid={lock.pid} host={lock.host} kind={lock.runner_kind} started={_fmt_ts(lock.started_at)}")
        print(f"  process_alive: {alive}")

    signals = gather_idle_structural_signals(state, now=now, turn_index=turn_index)
    sig = signals.to_dict()
    print("\n## Structural pre-filter")
    print(f"  should_run_llm: {sig.get('should_run_llm')}")
    print(f"  boredom_band: {sig.get('boredom_band')} (level={sig.get('boredom_level')})")
    print(f"  open_items_concrete_count: {sig.get('open_items_concrete_count')}")
    print(f"  unsettled_pending_settlement_count: {sig.get('unsettled_pending_settlement_count')}")

    log_summary = summarize_log(session_root / "conversation_log.jsonl")
    counts = dict(log_summary.get("counts", {}))
    print("\n## Audit counts (full conversation_log.jsonl stream)")
    for key in sorted(counts):
        print(f"  {key}: {counts.get(key, 0)}")
    latest_skip = log_summary.get("latest_stateful_skip") or log_summary.get("latest_skip") or {}
    if isinstance(latest_skip, dict) and latest_skip:
        code = latest_skip.get("reason_code") or latest_skip.get("suppression_reason_code") or latest_skip.get("skip_reason")
        print(f"  latest emitted skip: type={latest_skip.get('type')} code={code}")

    verdicts = verdicts_for_session(
        state=state,
        log_summary=log_summary,
        lock_alive=alive,
        has_lock=lock is not None,
        idle_elapsed=idle_elapsed,
        structural_should_run=bool(sig.get("should_run_llm")),
    )
    if not verdicts:
        verdicts = ["PROACTIVE_PIPELINE_TICKING_BUT_NO_TARGET"]

    print("\n## Verdict")
    for index, code in enumerate(verdicts, 1):
        print(f"  {index}. {code}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose MVP idle reflection / M14 background ticks.")
    parser.add_argument("--session-root", type=Path, default=None, help="Full path to sessions/<id> folder")
    parser.add_argument("--persona", default="", help="Persona id (mvp_personas/<persona>/sessions/...)")
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
