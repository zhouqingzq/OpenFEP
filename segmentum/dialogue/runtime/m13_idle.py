"""MVP-local M13.4 idle tick and introspection entry (plumbing only).

Structural pre-filters and audit events. Conscious idle reflection content is M14.0.
No LLM calls in M13.4 stub path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_boredom import boredom_band, normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import _mapping, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import merge_initiative_into_m13_state, normalize_initiative_state
from segmentum.dialogue.runtime.m13_reward import (
    list_assessable_pending_rows,
    normalize_affective_reward_proxy_state,
)

ENGINEERING_PROXY_LABEL = "mvp_local_idle_introspection"

DEFAULT_IDLE_INTROSPECTION_IDLE_THRESHOLD_SECONDS = 90
DEFAULT_IDLE_INTROSPECTION_MIN_INTERVAL_SECONDS = 180
DEFAULT_IDLE_INTROSPECTION_MAX_PER_SESSION = 4
JUST_OUTREACHED_WINDOW_SECONDS = 120
IDLE_AUDIT_THROTTLE_SECONDS = 60

IDLE_SKIP_REASONS: frozenset[str] = frozenset(
    {
        "not_opted_in",
        "disabled",
        "user_active",
        "idle_time_too_short",
        "min_interval_too_short",
        "session_limit_reached",
        "no_structural_signal",
        "just_outreached_recently",
    }
)

_NEXT_USER_TURN_MARKERS: frozenset[str] = frozenset({"next_user_turn", "next_turn"})


def _idle_user_activity_timestamp(temporal: Mapping[str, Any]) -> int:
    """Elapsed idle time is measured from the last user-initiated turn when known."""
    user_at = int(temporal.get("last_user_turn_at", 0) or 0)
    if user_at > 0:
        return user_at
    return int(temporal.get("last_turn_at", 0) or 0)


def should_persist_idle_audit_events(
    idle: Mapping[str, Any],
    *,
    skip_reason: str,
    now: int,
    force: bool = False,
) -> bool:
    """Throttle repetitive skip audits on Streamlit reruns (state still updates)."""
    if force or not skip_reason:
        return True
    noisy = frozenset({"idle_time_too_short", "min_interval_too_short", "user_active"})
    if skip_reason not in noisy:
        return True
    last_reason = str(idle.get("last_audit_logged_skip_reason", "") or "")
    last_at = int(idle.get("last_audit_logged_at", 0) or 0)
    if skip_reason != last_reason:
        return True
    return (now - last_at) >= IDLE_AUDIT_THROTTLE_SECONDS


def mark_idle_audit_logged(idle: dict[str, Any], *, skip_reason: str, now: int) -> None:
    idle["last_audit_logged_at"] = now
    idle["last_audit_logged_skip_reason"] = skip_reason


def default_idle_introspection_state() -> dict[str, Any]:
    return {
        "enabled": False,
        "user_opt_in": False,
        "idle_threshold_seconds": DEFAULT_IDLE_INTROSPECTION_IDLE_THRESHOLD_SECONDS,
        "min_interval_seconds": DEFAULT_IDLE_INTROSPECTION_MIN_INTERVAL_SECONDS,
        "max_per_session": DEFAULT_IDLE_INTROSPECTION_MAX_PER_SESSION,
        "reflection_count_this_session": 0,
        "last_introspection_at": 0,
        "last_skip_reason": "",
        "last_audit_logged_at": 0,
        "last_audit_logged_skip_reason": "",
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def normalize_idle_introspection_state(raw: Any) -> dict[str, Any]:
    base = default_idle_introspection_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["enabled"] = bool(merged.get("enabled"))
    merged["user_opt_in"] = bool(merged.get("user_opt_in"))
    merged["idle_threshold_seconds"] = max(
        30,
        int(
            merged.get("idle_threshold_seconds", DEFAULT_IDLE_INTROSPECTION_IDLE_THRESHOLD_SECONDS)
            or DEFAULT_IDLE_INTROSPECTION_IDLE_THRESHOLD_SECONDS
        ),
    )
    merged["min_interval_seconds"] = max(
        30,
        int(
            merged.get("min_interval_seconds", DEFAULT_IDLE_INTROSPECTION_MIN_INTERVAL_SECONDS)
            or DEFAULT_IDLE_INTROSPECTION_MIN_INTERVAL_SECONDS
        ),
    )
    merged["max_per_session"] = max(
        1,
        int(merged.get("max_per_session", DEFAULT_IDLE_INTROSPECTION_MAX_PER_SESSION) or DEFAULT_IDLE_INTROSPECTION_MAX_PER_SESSION),
    )
    merged["reflection_count_this_session"] = max(0, int(merged.get("reflection_count_this_session", 0) or 0))
    merged["last_introspection_at"] = int(merged.get("last_introspection_at", 0) or 0)
    merged["last_audit_logged_at"] = int(merged.get("last_audit_logged_at", 0) or 0)
    merged["last_audit_logged_skip_reason"] = str(merged.get("last_audit_logged_skip_reason", "") or "")[:64]
    reason = str(merged.get("last_skip_reason", "") or "")[:64]
    merged["last_skip_reason"] = reason if reason in IDLE_SKIP_REASONS or reason == "" else reason[:64]
    return merged


def merge_idle_introspection_into_initiative(initiative: dict[str, Any]) -> dict[str, Any]:
    merged = normalize_initiative_state(initiative)
    idle_raw = merged.get("idle_introspection")
    merged["idle_introspection"] = normalize_idle_introspection_state(idle_raw)
    return merged


def set_idle_introspection_user_opt_in(m13_state: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    state = merge_initiative_into_m13_state(m13_state)
    initiative = merge_idle_introspection_into_initiative(state.get("initiative", {}))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    parent_opted_in = bool(initiative.get("user_opt_in"))
    idle["user_opt_in"] = bool(enabled) and parent_opted_in
    idle["enabled"] = bool(enabled) and parent_opted_in
    if not idle["enabled"]:
        idle["last_skip_reason"] = "disabled" if not enabled else "not_opted_in"
    initiative["idle_introspection"] = idle
    state["initiative"] = initiative
    return state


def disable_idle_introspection_on_proactive_off(m13_state: dict[str, Any]) -> dict[str, Any]:
    state = merge_initiative_into_m13_state(m13_state)
    initiative = merge_idle_introspection_into_initiative(state.get("initiative", {}))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    idle["user_opt_in"] = False
    idle["enabled"] = False
    idle["last_skip_reason"] = "disabled"
    initiative["idle_introspection"] = idle
    state["initiative"] = initiative
    return state


@dataclass(frozen=True)
class IdleStructuralSignals:
    open_items_concrete_count: int
    boredom_band: str
    boredom_level: float
    unsettled_pending_settlement_count: int
    path_feels_stale_proxy: bool
    elapsed_since_last_turn_seconds: float
    just_outreached_recently: bool
    last_turn_was_proactive: bool

    def should_run_llm(self) -> bool:
        has_open = self.open_items_concrete_count > 0
        bored = self.boredom_band in {"medium", "high"}
        unsettled = self.unsettled_pending_settlement_count > 0
        return (has_open or bored or unsettled) and not self.just_outreached_recently

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_items_concrete_count": self.open_items_concrete_count,
            "boredom_band": self.boredom_band,
            "boredom_level": round(self.boredom_level, 4),
            "unsettled_pending_settlement_count": self.unsettled_pending_settlement_count,
            "path_feels_stale_proxy": self.path_feels_stale_proxy,
            "elapsed_since_last_turn_seconds": round(self.elapsed_since_last_turn_seconds, 3),
            "just_outreached_recently": self.just_outreached_recently,
            "last_turn_was_proactive": self.last_turn_was_proactive,
            "should_run_llm": self.should_run_llm(),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }


def _open_item_has_concrete_next_check(item: Mapping[str, Any]) -> bool:
    status = str(item.get("status", "open")).strip().lower()
    if status not in {"open", "pending", "active", ""}:
        return False
    next_check = str(item.get("next_check", item.get("next_step", "")) or "").strip().lower()
    if not next_check:
        return False
    if next_check in _NEXT_USER_TURN_MARKERS:
        return False
    return True


def gather_idle_structural_signals(
    state: Mapping[str, Any],
    *,
    now: int,
    turn_index: int,
) -> IdleStructuralSignals:
    open_items = state.get("open_items", []) or []
    concrete = 0
    if isinstance(open_items, list):
        for item in open_items:
            if isinstance(item, Mapping) and _open_item_has_concrete_next_check(item):
                concrete += 1

    m13_state = normalize_m13_drive_state(state.get("m13_drive_state"))
    boredom_state = normalize_boredom_state(m13_state.get("boredom"))
    level = float(boredom_state.get("boredom_level", 0.0) or 0.0)
    band = boredom_band(level)
    reward_state = normalize_affective_reward_proxy_state(m13_state.get("affective_reward_proxy"))
    path_feels_stale = bool(reward_state.get("path_feels_stale_proxy"))
    assessable_pending = list_assessable_pending_rows(reward_state, turn_index=turn_index)
    unsettled = len(assessable_pending)

    temporal = _mapping(state.get("temporal_state"))
    last_activity_at = _idle_user_activity_timestamp(temporal)
    elapsed = float(max(0, now - last_activity_at)) if last_activity_at > 0 else 0.0

    initiative = normalize_initiative_state(m13_state.get("initiative"))
    last_proactive_at = int(initiative.get("last_proactive_turn_at", 0) or 0)
    just_outreached = (
        last_proactive_at > 0 and (now - last_proactive_at) < JUST_OUTREACHED_WINDOW_SECONDS
    )
    last_proactive_index = int(initiative.get("last_proactive_turn_index", -1) or -1)
    last_turn_index = int(temporal.get("last_turn_index", -1) or -1)
    last_turn_was_proactive = last_proactive_index >= 0 and last_proactive_index == last_turn_index

    return IdleStructuralSignals(
        open_items_concrete_count=concrete,
        boredom_band=band,
        boredom_level=level,
        unsettled_pending_settlement_count=unsettled,
        path_feels_stale_proxy=path_feels_stale,
        elapsed_since_last_turn_seconds=elapsed,
        just_outreached_recently=just_outreached,
        last_turn_was_proactive=last_turn_was_proactive,
    )


def _idle_audit_base(
    *,
    turn_index: int,
    now: int,
    idle: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "at": now,
        "idle_introspection.enabled": bool(idle.get("enabled")),
        "idle_introspection.user_opt_in": bool(idle.get("user_opt_in")),
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


@dataclass
class IdleTickCheckResult:
    skip_reason: str
    events: list[dict[str, Any]] = field(default_factory=list)
    structural_signals: IdleStructuralSignals | None = None
    state_fields_read: list[str] = field(default_factory=list)


def evaluate_idle_tick(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    user_active: bool = False,
) -> tuple[dict[str, Any], IdleTickCheckResult]:
    """Cheap gate before structural pre-filter or LLM. Always emits IdleTickEvent."""
    m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
    initiative = merge_idle_introspection_into_initiative(m13_state.get("initiative"))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    fields_read = [
        "initiative.idle_introspection.enabled",
        "initiative.idle_introspection.user_opt_in",
        "temporal_state.last_user_turn_at",
        "temporal_state.last_turn_at",
        "open_items",
        "m13_drive_state.boredom",
        "m13_drive_state.affective_reward_proxy",
    ]
    events: list[dict[str, Any]] = [
        {
            "type": "IdleTickEvent",
            **_idle_audit_base(turn_index=turn_index, now=now, idle=idle),
            "skip_reason": "",
        }
    ]

    def finish(reason: str) -> tuple[dict[str, Any], IdleTickCheckResult]:
        idle["last_skip_reason"] = reason
        initiative["idle_introspection"] = idle
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        events[0]["skip_reason"] = reason
        if reason:
            events.append(
                {
                    "type": "IdleIntrospectionSuppressionEvent",
                    **_idle_audit_base(turn_index=turn_index, now=now, idle=idle),
                    "reason": reason,
                }
            )
        return state, IdleTickCheckResult(
            skip_reason=reason,
            events=events,
            state_fields_read=fields_read,
        )

    if not bool(initiative.get("user_opt_in")):
        return finish("not_opted_in")
    if not idle.get("enabled") or not idle.get("user_opt_in"):
        return finish("not_opted_in" if not idle.get("user_opt_in") else "disabled")
    if user_active:
        return finish("user_active")

    temporal = _mapping(state.get("temporal_state"))
    last_user_activity_at = _idle_user_activity_timestamp(temporal)
    if last_user_activity_at <= 0:
        return finish("idle_time_too_short")
    elapsed = now - last_user_activity_at
    if elapsed < int(idle.get("idle_threshold_seconds", DEFAULT_IDLE_INTROSPECTION_IDLE_THRESHOLD_SECONDS)):
        return finish("idle_time_too_short")

    last_intro = int(idle.get("last_introspection_at", 0) or 0)
    if last_intro > 0 and (now - last_intro) < int(
        idle.get("min_interval_seconds", DEFAULT_IDLE_INTROSPECTION_MIN_INTERVAL_SECONDS)
    ):
        return finish("min_interval_too_short")

    if int(idle.get("reflection_count_this_session", 0) or 0) >= int(
        idle.get("max_per_session", DEFAULT_IDLE_INTROSPECTION_MAX_PER_SESSION)
    ):
        return finish("session_limit_reached")

    signals = gather_idle_structural_signals(state, now=now, turn_index=turn_index)
    events[0]["structural_signals"] = signals.to_dict()
    return state, IdleTickCheckResult(
        skip_reason="",
        events=events,
        structural_signals=signals,
        state_fields_read=fields_read,
    )


def evaluate_idle_structural_pre_filter(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    signals: IdleStructuralSignals,
) -> tuple[dict[str, Any], IdleTickCheckResult]:
    """After tick gates pass; may emit IdleIntrospectionSkipEvent without LLM."""
    m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
    initiative = merge_idle_introspection_into_initiative(m13_state.get("initiative"))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    events: list[dict[str, Any]] = []

    if not signals.should_run_llm():
        reason = "just_outreached_recently" if signals.just_outreached_recently else "no_structural_signal"
        idle["last_skip_reason"] = reason
        initiative["idle_introspection"] = idle
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        events.append(
            {
                "type": "IdleIntrospectionSkipEvent",
                **_idle_audit_base(turn_index=turn_index, now=now, idle=idle),
                "skip_reason": reason,
                "structural_signals": signals.to_dict(),
            }
        )
        return state, IdleTickCheckResult(skip_reason=reason, events=events, structural_signals=signals)

    return state, IdleTickCheckResult(skip_reason="", events=events, structural_signals=signals)


def mark_idle_introspection_consumed(m13_state: dict[str, Any], *, now: int) -> dict[str, Any]:
    state = merge_initiative_into_m13_state(m13_state)
    initiative = merge_idle_introspection_into_initiative(state.get("initiative"))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    idle["reflection_count_this_session"] = int(idle.get("reflection_count_this_session", 0) or 0) + 1
    idle["last_introspection_at"] = now
    idle["last_skip_reason"] = ""
    initiative["idle_introspection"] = idle
    state["initiative"] = initiative
    return state
