"""M14.4 Streamlit implicit idle proactive delivery helpers.

This module only wires the existing M13.3 proposal/delivery path for an open
Streamlit chat page. It does not choose a new target and does not generate text
outside ``run_proactive_turn``.

After M16.4, callers must respect ``m16_streamlit_legacy.streamlit_scheduling_allowed()``.
The entrypoint below returns a suppressed result when the legacy scheduler is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_initiative import DEFAULT_IDLE_THRESHOLD_SECONDS
from segmentum.dialogue.runtime.m16_streamlit_legacy import streamlit_scheduling_allowed

ENGINEERING_PROXY_LABEL = "mvp_local_streamlit_implicit_idle"
DEFAULT_IMPLICIT_IDLE_ATTEMPT_INTERVAL_SECONDS = 45.0


@dataclass
class ImplicitIdleProactiveResult:
    attempted: bool = False
    delivered: bool = False
    rerun_requested: bool = False
    idle_seconds: float = 0.0
    idle_threshold_seconds: float = float(DEFAULT_IDLE_THRESHOLD_SECONDS)
    proactive_policy_profile: str = "bounded_default"
    suppression_reason_code: str = ""
    proposal_id: str = ""
    response: Any | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "delivered": self.delivered,
            "rerun_requested": self.rerun_requested,
            "idle_seconds": round(float(self.idle_seconds), 3),
            "idle_threshold_seconds": round(float(self.idle_threshold_seconds), 3),
            "proactive_policy_profile": self.proactive_policy_profile,
            "suppression_reason_code": self.suppression_reason_code,
            "proposal_id": self.proposal_id,
            "events": list(self.events),
        }


def _mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _session_get(session_state: Any, key: str, default: Any = None) -> Any:
    if session_state is None:
        return default
    getter = getattr(session_state, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(session_state, Mapping):
        return session_state.get(key, default)
    return getattr(session_state, key, default)


def _session_set(session_state: Any, key: str, value: Any) -> None:
    if session_state is None:
        return
    try:
        session_state[key] = value
        return
    except Exception:
        pass
    try:
        setattr(session_state, key, value)
    except Exception:
        return


def _epoch(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def compute_idle_seconds(state: Mapping[str, Any] | None, *, now: float) -> float:
    temporal = _mapping(_mapping(state).get("temporal_state"))
    last_user = _epoch(temporal.get("last_user_turn_at"))
    if last_user <= 0:
        last_user = _epoch(temporal.get("last_turn_at"))
    if last_user <= 0:
        return 0.0
    return max(0.0, float(now) - last_user)


def should_attempt_implicit_idle_proactive(
    session_state: Any,
    *,
    now_mono: float,
    idle_seconds: float,
    initiative_status: Mapping[str, Any],
    mvp_runtime_active: bool,
    has_agent: bool,
    pending_user_message: Any = None,
    pending_proactive: bool = False,
    user_typing: bool = False,
    min_attempt_interval_seconds: float = DEFAULT_IMPLICIT_IDLE_ATTEMPT_INTERVAL_SECONDS,
) -> tuple[bool, str]:
    if not has_agent:
        return False, "agent_not_loaded"
    if not mvp_runtime_active:
        return False, "mvp_runtime_inactive"
    if not bool(initiative_status.get("user_opt_in")):
        return False, "not_opted_in"
    if not bool(initiative_status.get("enabled")):
        return False, "initiative_disabled"
    if not bool(initiative_status.get("implicit_idle_delivery")):
        return False, "delivery_channel_unavailable"
    if pending_user_message is not None or _session_get(session_state, "pending_user_message") is not None:
        return False, "pending_user_message"
    if pending_proactive or bool(_session_get(session_state, "pending_proactive_continue", False)):
        return False, "pending_proactive_continue"
    if user_typing:
        return False, "user_active"
    if bool(_session_get(session_state, "m13_ui_turn_in_progress", False)):
        return False, "turn_in_progress"
    threshold = float(initiative_status.get("idle_threshold_seconds", DEFAULT_IDLE_THRESHOLD_SECONDS) or DEFAULT_IDLE_THRESHOLD_SECONDS)
    if float(idle_seconds) < threshold:
        return False, "idle_time_too_short"
    last = float(_session_get(session_state, "_implicit_idle_last_attempt_mono", 0.0) or 0.0)
    if last > 0 and float(now_mono) - last < float(min_attempt_interval_seconds):
        return False, "implicit_idle_attempt_throttled"
    return True, ""


def _event(
    *,
    idle_seconds: float,
    idle_threshold_seconds: float,
    proactive_policy_profile: str,
    suppression_reason_code: str = "",
    proposal_id: str = "",
) -> dict[str, Any]:
    return {
        "type": "M14ImplicitIdleProactiveCheckEvent",
        "idle_seconds": round(float(idle_seconds), 3),
        "idle_threshold_seconds": round(float(idle_threshold_seconds), 3),
        "proactive_policy_profile": proactive_policy_profile,
        "suppression_reason_code": str(suppression_reason_code or "")[:96],
        "proposal_id": str(proposal_id or "")[:120],
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def _record_event(chat_iface: Any, event: Mapping[str, Any]) -> None:
    recorder = getattr(chat_iface, "append_m14_4_implicit_idle_audit", None)
    if callable(recorder):
        recorder(dict(event))


def run_streamlit_implicit_idle_proactive(
    chat_iface: Any,
    *,
    session_state: Any,
    now: float | None = None,
    now_mono: float | None = None,
    pending_user_message: Any = None,
    pending_proactive: bool = False,
    user_typing: bool = False,
    speaker_name: str = "",
    min_attempt_interval_seconds: float = DEFAULT_IMPLICIT_IDLE_ATTEMPT_INTERVAL_SECONDS,
) -> ImplicitIdleProactiveResult:
    if not streamlit_scheduling_allowed():
        result = ImplicitIdleProactiveResult(
            suppression_reason_code="m16_legacy_scheduler_off",
            proactive_policy_profile="bounded_default",
        )
        _record_event(
            chat_iface,
            _event(
                idle_seconds=0.0,
                idle_threshold_seconds=float(DEFAULT_IDLE_THRESHOLD_SECONDS),
                proactive_policy_profile=result.proactive_policy_profile,
                suppression_reason_code=result.suppression_reason_code,
            ),
        )
        return result
    now = float(time.time() if now is None else now)
    now_mono = float(time.monotonic() if now_mono is None else now_mono)
    state = chat_iface.read_mvp_state_dict() if hasattr(chat_iface, "read_mvp_state_dict") else {}
    initiative = chat_iface.read_initiative_status() if hasattr(chat_iface, "read_initiative_status") else {}
    initiative = dict(initiative or {})
    idle_seconds = compute_idle_seconds(_mapping(state), now=now)
    threshold = float(initiative.get("idle_threshold_seconds", DEFAULT_IDLE_THRESHOLD_SECONDS) or DEFAULT_IDLE_THRESHOLD_SECONDS)
    profile = str(initiative.get("proactive_policy_profile", "bounded_default") or "bounded_default")

    _session_set(session_state, "last_implicit_idle_attempt_at", int(now))
    _session_set(session_state, "idle_seconds_at_last_check", round(float(idle_seconds), 3))

    allowed, reason = should_attempt_implicit_idle_proactive(
        session_state,
        now_mono=now_mono,
        idle_seconds=idle_seconds,
        initiative_status=initiative,
        mvp_runtime_active=bool(getattr(chat_iface, "mvp_runtime_active", False)),
        has_agent=bool(chat_iface.has_agent()) if hasattr(chat_iface, "has_agent") else False,
        pending_user_message=pending_user_message,
        pending_proactive=pending_proactive,
        user_typing=user_typing,
        min_attempt_interval_seconds=min_attempt_interval_seconds,
    )
    if not allowed:
        event = _event(
            idle_seconds=idle_seconds,
            idle_threshold_seconds=threshold,
            proactive_policy_profile=profile,
            suppression_reason_code=reason,
        )
        _record_event(chat_iface, event)
        _session_set(session_state, "last_implicit_idle_suppression_reason_code", reason)
        return ImplicitIdleProactiveResult(
            attempted=False,
            idle_seconds=idle_seconds,
            idle_threshold_seconds=threshold,
            proactive_policy_profile=profile,
            suppression_reason_code=reason,
            events=[event],
        )

    _session_set(session_state, "_implicit_idle_last_attempt_mono", now_mono)
    _session_set(session_state, "m13_ui_turn_in_progress", True)
    try:
        tick: Mapping[str, Any] | None = None
        run_tick = getattr(chat_iface, "run_idle_cognitive_tick", None)
        if callable(run_tick):
            tick = run_tick(idle_seconds=idle_seconds)
            tick_target = _mapping(_mapping(tick).get("selected_target"))
            if not tick_target:
                reason = str(_mapping(tick).get("reject_reason") or "") or "no_high_value_target"
                event = _event(
                    idle_seconds=idle_seconds,
                    idle_threshold_seconds=threshold,
                    proactive_policy_profile=profile,
                    suppression_reason_code=reason,
                )
                _record_event(chat_iface, event)
                _session_set(session_state, "last_implicit_idle_suppression_reason_code", reason)
                return ImplicitIdleProactiveResult(
                    attempted=True,
                    idle_seconds=idle_seconds,
                    idle_threshold_seconds=threshold,
                    proactive_policy_profile=profile,
                    suppression_reason_code=reason,
                    events=[*_mapping(tick).get("events", []), event] if isinstance(_mapping(tick).get("events"), list) else [event],
                )
        check = chat_iface.maybe_propose_proactive_turn(
            implicit_idle_request=True,
            idle_seconds=idle_seconds,
            user_typing=user_typing,
            preselected_target=_mapping(_mapping(tick).get("selected_target")) if tick is not None else None,
        )
        proposal = check.get("proposal") if isinstance(check, Mapping) else None
        if not isinstance(proposal, Mapping) or not proposal.get("proposal_id"):
            reason = ""
            if isinstance(check, Mapping):
                reason = str(check.get("suppression_reason_code") or check.get("suppression_reason") or "")
            reason = reason or "no_traceable_proactive_target"
            event = _event(
                idle_seconds=idle_seconds,
                idle_threshold_seconds=threshold,
                proactive_policy_profile=profile,
                suppression_reason_code=reason,
            )
            _record_event(chat_iface, event)
            _session_set(session_state, "last_implicit_idle_suppression_reason_code", reason)
            return ImplicitIdleProactiveResult(
                attempted=True,
                idle_seconds=idle_seconds,
                idle_threshold_seconds=threshold,
                proactive_policy_profile=profile,
                suppression_reason_code=reason,
                events=[event],
            )
        proposal_id = str(proposal["proposal_id"])
        response = chat_iface.run_proactive_turn(proposal_id, speaker_name=str(speaker_name or ""))
        delivered = bool(str(getattr(response, "reply", "") or "").strip())
        reason = ""
        diagnostics = getattr(response, "diagnostics", {})
        if isinstance(diagnostics, Mapping):
            reason = str(diagnostics.get("reason_code") or diagnostics.get("suppression_reason") or "")
        event = _event(
            idle_seconds=idle_seconds,
            idle_threshold_seconds=threshold,
            proactive_policy_profile=profile,
            suppression_reason_code="" if delivered else reason or "empty_generation",
            proposal_id=proposal_id,
        )
        _record_event(chat_iface, event)
        _session_set(session_state, "last_implicit_idle_suppression_reason_code", "" if delivered else event["suppression_reason_code"])
        _session_set(session_state, "last_implicit_idle_proposal_id", proposal_id)
        return ImplicitIdleProactiveResult(
            attempted=True,
            delivered=delivered,
            rerun_requested=delivered,
            idle_seconds=idle_seconds,
            idle_threshold_seconds=threshold,
            proactive_policy_profile=profile,
            suppression_reason_code="" if delivered else str(event["suppression_reason_code"]),
            proposal_id=proposal_id,
            response=response,
            events=[event],
        )
    finally:
        _session_set(session_state, "m13_ui_turn_in_progress", False)
