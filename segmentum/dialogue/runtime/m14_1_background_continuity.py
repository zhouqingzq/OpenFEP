"""M14.1 background self-continuity: budgets, queue, file locks (Path B)."""

from __future__ import annotations

import copy
import ctypes
import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

M14_1_ENGINEERING_PROXY_LABEL = "mvp_local_background_self_continuity"

MIN_TICK_SECONDS = 30
DEFAULT_TICK_INTERVAL_SECONDS = 90
DEFAULT_TOKENS_BUDGET_PER_DAY = 30_000
DEFAULT_WALLCLOCK_BUDGET_PER_DAY_SECONDS = 600
DEFAULT_MAX_TICKS_PER_DAY = 400
DEFAULT_LLM_CALLS_BUDGET_PER_DAY = 80
DEFAULT_QUEUED_OUTREACH_TTL_SECONDS = 24 * 3600
MIN_QUEUED_OUTREACH_TTL_SECONDS = 3600
MAX_QUEUED_OUTREACH_TTL_SECONDS = 72 * 3600
MAX_QUEUED_OUTREACH = 8
INLINE_RUNNER_IDLE_DEATH_SECONDS = 1800
LOCK_TIMEOUT_SECONDS = 5.0

_TRANSIENT_OUTREACH_SUPPRESSION = frozenset({"cooldown_active", "user_active", "user_typing"})


def default_background_continuity_state() -> dict[str, Any]:
    return {
        "enabled": False,
        "user_opt_in": False,
        "runner_kind": "none",
        "tick_interval_seconds": DEFAULT_TICK_INTERVAL_SECONDS,
        "queued_outreach_ttl_seconds": DEFAULT_QUEUED_OUTREACH_TTL_SECONDS,
        "tokens_used_today": 0,
        "tokens_budget_per_day": DEFAULT_TOKENS_BUDGET_PER_DAY,
        "wallclock_used_today_seconds": 0.0,
        "wallclock_budget_per_day_seconds": DEFAULT_WALLCLOCK_BUDGET_PER_DAY_SECONDS,
        "ticks_today": 0,
        "max_ticks_per_day": DEFAULT_MAX_TICKS_PER_DAY,
        "llm_calls_today": 0,
        "llm_calls_budget_per_day": DEFAULT_LLM_CALLS_BUDGET_PER_DAY,
        "self_reviews_today": 0,
        "day_anchor": "",
        "last_tick_at": 0,
        "last_budget_block_reason": "",
        "last_streamlit_ping_at": 0,
        "idle_ticks_lifetime": 0,
        "self_reviews_lifetime": 0,
        "llm_calls_lifetime": 0,
        "tokens_used_lifetime": 0,
        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
    }


def normalize_background_continuity_state(raw: Any) -> dict[str, Any]:
    base = default_background_continuity_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["enabled"] = bool(merged.get("enabled"))
    merged["user_opt_in"] = bool(merged.get("user_opt_in"))
    merged["runner_kind"] = str(merged.get("runner_kind", "none") or "none")[:16]
    merged["tick_interval_seconds"] = max(
        MIN_TICK_SECONDS,
        int(merged.get("tick_interval_seconds", DEFAULT_TICK_INTERVAL_SECONDS) or DEFAULT_TICK_INTERVAL_SECONDS),
    )
    ttl = int(merged.get("queued_outreach_ttl_seconds", DEFAULT_QUEUED_OUTREACH_TTL_SECONDS) or DEFAULT_QUEUED_OUTREACH_TTL_SECONDS)
    merged["queued_outreach_ttl_seconds"] = max(
        MIN_QUEUED_OUTREACH_TTL_SECONDS,
        min(MAX_QUEUED_OUTREACH_TTL_SECONDS, ttl),
    )
    merged["tokens_budget_per_day"] = max(1000, int(merged.get("tokens_budget_per_day", DEFAULT_TOKENS_BUDGET_PER_DAY) or 1))
    merged["wallclock_budget_per_day_seconds"] = max(
        60.0, float(merged.get("wallclock_budget_per_day_seconds", DEFAULT_WALLCLOCK_BUDGET_PER_DAY_SECONDS) or 60)
    )
    merged["max_ticks_per_day"] = max(1, int(merged.get("max_ticks_per_day", DEFAULT_MAX_TICKS_PER_DAY) or 1))
    merged["llm_calls_budget_per_day"] = max(1, int(merged.get("llm_calls_budget_per_day", DEFAULT_LLM_CALLS_BUDGET_PER_DAY) or 1))
    for key in (
        "tokens_used_today",
        "ticks_today",
        "llm_calls_today",
        "self_reviews_today",
        "last_tick_at",
        "idle_ticks_lifetime",
        "self_reviews_lifetime",
        "llm_calls_lifetime",
        "tokens_used_lifetime",
        "last_streamlit_ping_at",
    ):
        merged[key] = max(0, int(merged.get(key, 0) or 0))
    merged["wallclock_used_today_seconds"] = max(0.0, float(merged.get("wallclock_used_today_seconds", 0) or 0))
    merged["day_anchor"] = str(merged.get("day_anchor", "") or "")[:16]
    merged["last_budget_block_reason"] = str(merged.get("last_budget_block_reason", "") or "")[:64]
    return merged


def merge_background_continuity_into_initiative(initiative: dict[str, Any]) -> dict[str, Any]:
    merged = dict(initiative)
    merged["background_continuity"] = normalize_background_continuity_state(merged.get("background_continuity"))
    return merged


def _local_day_anchor(now: int) -> str:
    return date.fromtimestamp(now).isoformat()


def maybe_rollover_daily_counters(bg: dict[str, Any], *, now: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    today = _local_day_anchor(now)
    if bg.get("day_anchor") == today:
        return bg, None
    event = {
        "type": "BackgroundDailyRolloverEvent",
        "at": now,
        "previous_day_anchor": bg.get("day_anchor", ""),
        "new_day_anchor": today,
        "previous_ticks_today": bg.get("ticks_today", 0),
        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
    }
    bg = dict(bg)
    bg["day_anchor"] = today
    bg["tokens_used_today"] = 0
    bg["wallclock_used_today_seconds"] = 0.0
    bg["ticks_today"] = 0
    bg["llm_calls_today"] = 0
    bg["self_reviews_today"] = 0
    bg["last_budget_block_reason"] = ""
    return bg, event


def estimate_token_usage(*, system_prompt: str = "", user_prompt: str = "", response: Any = None) -> int:
    parts = [system_prompt, user_prompt]
    if response is not None:
        try:
            parts.append(json.dumps(response, ensure_ascii=False))
        except TypeError:
            parts.append(str(response))
    text = "".join(parts)
    return max(1, len(text) // 4)


def check_background_budgets(bg: Mapping[str, Any]) -> str:
    if int(bg.get("ticks_today", 0) or 0) >= int(bg.get("max_ticks_per_day", 1) or 1):
        return "max_ticks_per_day"
    if int(bg.get("llm_calls_today", 0) or 0) >= int(bg.get("llm_calls_budget_per_day", 1) or 1):
        return "llm_calls_budget_exhausted"
    if int(bg.get("tokens_used_today", 0) or 0) >= int(bg.get("tokens_budget_per_day", 1) or 1):
        return "tokens_budget_exhausted"
    if float(bg.get("wallclock_used_today_seconds", 0) or 0) >= float(
        bg.get("wallclock_budget_per_day_seconds", 1) or 1
    ):
        return "wallclock_budget_exhausted"
    return ""


def record_background_llm_usage(
    bg: dict[str, Any],
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    response: Any = None,
) -> dict[str, Any]:
    merged = dict(bg)
    tokens = estimate_token_usage(system_prompt=system_prompt, user_prompt=user_prompt, response=response)
    merged["llm_calls_today"] = int(merged.get("llm_calls_today", 0) or 0) + 1
    merged["llm_calls_lifetime"] = int(merged.get("llm_calls_lifetime", 0) or 0) + 1
    merged["tokens_used_today"] = int(merged.get("tokens_used_today", 0) or 0) + tokens
    merged["tokens_used_lifetime"] = int(merged.get("tokens_used_lifetime", 0) or 0) + tokens
    return merged


class BackgroundBudgetExhausted(RuntimeError):
    """Raised before a background-owned LLM call would exceed a daily budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _JSONLLM(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class BackgroundLLMMeter:
    """Budget-checking LLM wrapper for background-owned call chains."""

    def __init__(self, llm: _JSONLLM, bg: Mapping[str, Any]) -> None:
        self._llm = llm
        self.bg = normalize_background_continuity_state(bg)
        self.llm_calls_delta = 0
        self.tokens_delta = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        block = check_background_budgets(self.bg)
        if block:
            raise BackgroundBudgetExhausted(block)
        response = self._llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        before_tokens = int(self.bg.get("tokens_used_today", 0) or 0)
        self.bg = record_background_llm_usage(
            self.bg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
        )
        self.llm_calls_delta += 1
        self.tokens_delta += int(self.bg.get("tokens_used_today", 0) or 0) - before_tokens
        return response


def record_background_tick(bg: dict[str, Any], *, wallclock_seconds: float, ran_introspection: bool) -> dict[str, Any]:
    merged = dict(bg)
    merged["ticks_today"] = int(merged.get("ticks_today", 0) or 0) + 1
    merged["wallclock_used_today_seconds"] = float(merged.get("wallclock_used_today_seconds", 0) or 0) + max(
        0.0, wallclock_seconds
    )
    if ran_introspection:
        merged["idle_ticks_lifetime"] = int(merged.get("idle_ticks_lifetime", 0) or 0) + 1
    return merged


@contextmanager
def session_file_lock(session_root: Path, *, timeout: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    session_root.mkdir(parents=True, exist_ok=True)
    lock_path = session_root / "store.lock"
    deadline = time.monotonic() + timeout
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}".encode())
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
    if not acquired:
        raise TimeoutError(f"could not acquire store lock: {lock_path}")
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass(frozen=True)
class RunnerLockInfo:
    pid: int
    host: str
    runner_kind: str
    started_at: int


def read_runner_lock(session_root: Path) -> RunnerLockInfo | None:
    path = session_root / "runner.lock"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return RunnerLockInfo(
        pid=int(payload.get("pid", 0) or 0),
        host=str(payload.get("host", "") or ""),
        runner_kind=str(payload.get("runner_kind", "") or ""),
        started_at=int(payload.get("started_at", 0) or 0),
    )


def try_acquire_runner_lock(session_root: Path, *, runner_kind: str, now: int) -> tuple[bool, RunnerLockInfo | None]:
    session_root.mkdir(parents=True, exist_ok=True)
    lock_path = session_root / "runner.lock"
    existing = read_runner_lock(session_root)
    if existing is not None and _pid_alive(existing.pid):
        return False, existing
    if existing is not None:
        lock_path.unlink(missing_ok=True)
    info = RunnerLockInfo(pid=os.getpid(), host=socket.gethostname()[:64], runner_kind=runner_kind, started_at=now)
    payload = json.dumps(
        {
            "pid": info.pid,
            "host": info.host,
            "runner_kind": info.runner_kind,
            "started_at": info.started_at,
        },
        ensure_ascii=False,
    )
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, read_runner_lock(session_root)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return True, info


def release_runner_lock(session_root: Path) -> None:
    path = session_root / "runner.lock"
    info = read_runner_lock(session_root)
    if info is not None and info.pid == os.getpid():
        path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def queued_outreach_path(session_root: Path) -> Path:
    return session_root / "queued_outreach.jsonl"


def load_queued_outreach(session_root: Path) -> list[dict[str, Any]]:
    path = queued_outreach_path(session_root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def save_queued_outreach(session_root: Path, rows: list[dict[str, Any]]) -> None:
    path = queued_outreach_path(session_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows[-MAX_QUEUED_OUTREACH * 2 :]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enqueue_outreach_proposal(
    session_root: Path,
    *,
    proposal: Mapping[str, Any],
    now: int,
    ttl_seconds: int,
    drive_snapshot: Mapping[str, Any] | None = None,
    due_at: int | None = None,
    source_intent_id: str = "",
) -> dict[str, Any]:
    rows = load_queued_outreach(session_root)
    source_intent_id = str(source_intent_id or proposal.get("source_intent_id", "") or "")
    if source_intent_id:
        for row in rows:
            if str(row.get("source_intent_id", "")) == source_intent_id:
                return dict(row)
    entry = {
        "proposal_id": str(proposal.get("proposal_id", "") or ""),
        "created_at": now,
        "due_at": int(due_at if due_at is not None else proposal.get("due_at", now) or now),
        "expires_at": now + max(MIN_QUEUED_OUTREACH_TTL_SECONDS, int(ttl_seconds)),
        "trigger": str(proposal.get("trigger", "reflection_outreach") or "reflection_outreach"),
        "source_intent_id": source_intent_id,
        "persona_id": str(proposal.get("persona_id", "") or ""),
        "session_id": str(proposal.get("session_id", "") or ""),
        "ordinary_language_intent": str(proposal.get("ordinary_language_intent", "") or "")[:240],
        "proposed_topic": str(proposal.get("proposed_topic", "") or "")[:120],
        "evidence_refs": list(proposal.get("evidence_refs", []) or [])[:8],
        "drive_snapshot_compact": dict(drive_snapshot or {}),
        "status": "pending",
        "delivery_policy": {
            "require_m13_3_assessor": True,
            "max_visible_messages": 1,
            "no_direct_generation": True,
        },
        "delivery_attempts": 0,
        "last_delivery_attempt_at": 0,
        "last_suppression_reason": "",
        "source": "queued_outreach",
    }
    rows.append(entry)
    pending = [r for r in rows if str(r.get("status", "")) == "pending"]
    while len(pending) > MAX_QUEUED_OUTREACH:
        oldest = min(pending, key=lambda r: int(r.get("created_at", 0) or 0))
        oldest["status"] = "expired"
        pending = [r for r in rows if str(r.get("status", "")) == "pending"]
    save_queued_outreach(session_root, rows)
    return entry


def expire_queued_outreach(session_root: Path, *, now: int) -> list[dict[str, Any]]:
    rows = load_queued_outreach(session_root)
    events: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        if str(row.get("status", "")) != "pending":
            continue
        if int(row.get("expires_at", 0) or 0) <= now:
            row["status"] = "expired"
            changed = True
            events.append(
                {
                    "type": "QueuedOutreachExpiredEvent",
                    "at": now,
                    "proposal_id": row.get("proposal_id", ""),
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
    if changed:
        save_queued_outreach(session_root, rows)
    return events


def pop_next_pending_outreach(session_root: Path, *, now: int) -> dict[str, Any] | None:
    expire_queued_outreach(session_root, now=now)
    rows = load_queued_outreach(session_root)
    pending = [r for r in rows if str(r.get("status", "")) == "pending"]
    if not pending:
        return None
    pending.sort(key=lambda r: int(r.get("created_at", 0) or 0))
    return dict(pending[0])


def update_queued_outreach_status(
    session_root: Path,
    proposal_id: str,
    status: str,
    *,
    now: int | None = None,
    suppression_reason: str = "",
) -> None:
    rows = load_queued_outreach(session_root)
    for row in rows:
        if str(row.get("proposal_id", "")) == proposal_id:
            row["status"] = status
            if now is not None and status in {"delivered", "suppressed", "expired"}:
                row["last_delivery_attempt_at"] = int(now)
            if suppression_reason:
                row["last_suppression_reason"] = str(suppression_reason)[:160]
    save_queued_outreach(session_root, rows)


def record_queued_outreach_delivery_attempt(session_root: Path, proposal_id: str, *, now: int) -> None:
    rows = load_queued_outreach(session_root)
    for row in rows:
        if str(row.get("proposal_id", "")) == proposal_id:
            row["delivery_attempts"] = int(row.get("delivery_attempts", 0) or 0) + 1
            row["last_delivery_attempt_at"] = int(now)
    save_queued_outreach(session_root, rows)


def outreach_suppression_is_transient(reason: str) -> bool:
    return reason in _TRANSIENT_OUTREACH_SUPPRESSION


def set_background_continuity_opt_in(
    m13_state: dict[str, Any],
    *,
    enabled: bool,
    runner_kind: str = "none",
) -> dict[str, Any]:
    from segmentum.dialogue.runtime.m13_initiative import merge_initiative_into_m13_state, normalize_initiative_state

    state = merge_initiative_into_m13_state(m13_state)
    initiative = merge_background_continuity_into_initiative(normalize_initiative_state(state.get("initiative")))
    bg = normalize_background_continuity_state(initiative.get("background_continuity"))
    bg["user_opt_in"] = bool(enabled)
    bg["enabled"] = bool(enabled)
    bg["runner_kind"] = str(runner_kind if enabled else "none")
    if not enabled:
        bg["last_budget_block_reason"] = "disabled"
    initiative["background_continuity"] = bg
    state["initiative"] = initiative
    return state
