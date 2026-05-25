"""M15.2 bounded meta-control detectors and intervention intents.

This module is an engineering scheduler layer only.  It reads ledger/state,
emits bounded bias intents, and never sends replies or writes memory/M12 state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import uuid
from typing import Any, Mapping

from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, MemoryDynamicsEpisode


ENGINEERING_PROXY_LABEL = "mvp_local_meta_control"
K_CONSECUTIVE_FAILURES = 3
LOOKBACK_EPISODES = 16
DELTA_FE_FAILURE_FLOOR = 0.0
MIN_TENSION_CONFIDENCE = 0.65
MIN_IDLE_TICKS_STABLE_TENSION = 2
STALL_TICK_WINDOW = 6
SAME_REJECT_REASON_RATIO = 0.8
MAX_ACTIVE_INTENTS = 8
MAX_CONSUMED_INTENTS = 16
MAX_RECALL_TOP_K = 12
DEFAULT_IDLE_TOP_K = 8

INTENT_KINDS = frozenset(
    {
        "suppress_action_trigger_for_n_turns",
        "request_reflection_focus",
        "bias_idle_recall_breadth",
    }
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _epoch(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _walk_evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        refs.extend(_string_list(value.get("evidence_refs"), limit=16))
        for child in value.values():
            refs.extend(_walk_evidence_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_walk_evidence_refs(child))
    return _string_list(refs, limit=32)


@dataclass(frozen=True)
class MetaControlInterventionIntent:
    intent_id: str
    at: int
    turn_index: int
    detector: str
    intent_kind: str
    payload: dict[str, Any]
    evidence_refs: list[str] = field(default_factory=list)
    detector_evidence_event_ids: list[str] = field(default_factory=list)
    expires_at: int = 0
    engineering_proxy_label: str = ENGINEERING_PROXY_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "at": self.at,
            "turn_index": self.turn_index,
            "detector": self.detector,
            "intent_kind": self.intent_kind,
            "payload": dict(self.payload),
            "evidence_refs": list(self.evidence_refs[:8]),
            "detector_evidence_event_ids": list(self.detector_evidence_event_ids[:8]),
            "expires_at": self.expires_at,
            "engineering_proxy_label": self.engineering_proxy_label,
        }

    @staticmethod
    def from_mapping(row: Mapping[str, Any]) -> "MetaControlInterventionIntent":
        return MetaControlInterventionIntent(
            intent_id=str(row.get("intent_id", "")),
            at=_epoch(row.get("at")),
            turn_index=_epoch(row.get("turn_index")),
            detector=str(row.get("detector", "")),
            intent_kind=str(row.get("intent_kind", "")),
            payload=dict(_mapping(row.get("payload"))),
            evidence_refs=_string_list(row.get("evidence_refs"), limit=8),
            detector_evidence_event_ids=_string_list(row.get("detector_evidence_event_ids"), limit=8),
            expires_at=_epoch(row.get("expires_at")),
        )


@dataclass(frozen=True)
class MetaControlDetectionResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    intents: list[MetaControlInterventionIntent] = field(default_factory=list)


def _meta_state(m13_state: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(m13_state.get("meta_control_intents"))
    active = [dict(row) for row in raw.get("active", []) or [] if isinstance(row, Mapping)]
    consumed = [dict(row) for row in raw.get("consumed", []) or [] if isinstance(row, Mapping)]
    return {
        "active": active[-MAX_ACTIVE_INTENTS:],
        "consumed": consumed[-MAX_CONSUMED_INTENTS:],
        "idle_tick_history": [dict(row) for row in raw.get("idle_tick_history", []) or [] if isinstance(row, Mapping)][-STALL_TICK_WINDOW:],
        "tension_observation_counts": dict(_mapping(raw.get("tension_observation_counts"))),
        "recent_detections": [dict(row) for row in raw.get("recent_detections", []) or [] if isinstance(row, Mapping)][-16:],
    }


def _apply_enabled(state: Mapping[str, Any]) -> bool:
    if str(os.environ.get("SEGMENTUM_META_CONTROL_APPLY", "") or "").strip() == "1":
        return True
    m13 = _mapping(state.get("m13_drive_state"))
    initiative = _mapping(m13.get("initiative"))
    return str(initiative.get("proactive_policy_profile", "") or "") == "streamlit_open_chat"


def _has_active_kind(meta: Mapping[str, Any], intent_kind: str) -> bool:
    return any(str(row.get("intent_kind", "")) == intent_kind for row in meta.get("active", []) or [])


def _store_intent(state: dict[str, Any], intent: MetaControlInterventionIntent) -> None:
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    meta["active"] = [*meta["active"], intent.to_dict()][-MAX_ACTIVE_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13


def expire_meta_control_intents(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    tick: bool = False,
) -> list[dict[str, Any]]:
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    kept: list[dict[str, Any]] = []
    consumed = list(meta.get("consumed", []))
    events: list[dict[str, Any]] = []
    for row in meta.get("active", []) or []:
        intent = MetaControlInterventionIntent.from_mapping(row)
        payload = dict(intent.payload)
        expired = False
        if intent.intent_kind == "suppress_action_trigger_for_n_turns":
            ttl = min(5, max(1, int(payload.get("ttl_turns", 3) or 3)))
            expired = int(turn_index) > int(intent.turn_index) + ttl
        elif intent.intent_kind == "bias_idle_recall_breadth":
            ttl_ticks = min(3, max(1, int(payload.get("ttl_ticks", 1) or 1)))
            used = int(payload.get("ticks_seen", 0) or 0)
            expired = tick and used >= ttl_ticks
        elif intent.expires_at:
            expired = int(now) >= int(intent.expires_at)
        if expired:
            expired_row = intent.to_dict()
            expired_row["expired_at"] = now
            expired_row["expiration_reason"] = "ttl"
            consumed.append(expired_row)
            events.append(
                {
                    "type": "MetaControlInterventionExpiredEvent",
                    "intent_id": intent.intent_id,
                    "expired_at": now,
                    "reason": "ttl",
                    "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
                }
            )
        else:
            kept.append(intent.to_dict())
    meta["active"] = kept[-MAX_ACTIVE_INTENTS:]
    meta["consumed"] = consumed[-MAX_CONSUMED_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13
    return events


def _detection_event_base(now: int, turn_index: int, source: str) -> dict[str, Any]:
    return {
        "at": now,
        "turn_index": turn_index,
        "source": source,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def _detect_repeated_failure(
    state: dict[str, Any],
    ledger: EpisodeLedger,
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    apply_intents: bool,
) -> tuple[list[dict[str, Any]], list[MetaControlInterventionIntent]]:
    episodes = ledger.recent(LOOKBACK_EPISODES)
    by_trigger: dict[str, list[MemoryDynamicsEpisode]] = {}
    for episode in episodes:
        trigger = str(episode.action_trigger or "")
        if trigger:
            by_trigger.setdefault(trigger, []).append(episode)
    events: list[dict[str, Any]] = []
    intents: list[MetaControlInterventionIntent] = []
    if _has_active_kind(meta, "suppress_action_trigger_for_n_turns"):
        return events, intents
    for trigger, rows in sorted(by_trigger.items()):
        streak: list[MemoryDynamicsEpisode] = []
        for episode in reversed(rows):
            if episode.delta_fe_proxy >= DELTA_FE_FAILURE_FLOOR:
                streak.append(episode)
            else:
                break
        if len(streak) < K_CONSECUTIVE_FAILURES:
            continue
        intent = MetaControlInterventionIntent(
            intent_id=_new_id("m15_meta_intent"),
            at=now,
            turn_index=turn_index,
            detector="RepeatedFailurePathDetector",
            intent_kind="suppress_action_trigger_for_n_turns",
            payload={"action_trigger": trigger, "ttl_turns": 3},
            evidence_refs=_string_list([ref for episode in streak for ref in episode.evidence_refs], limit=8),
            detector_evidence_event_ids=[episode.episode_id for episode in streak[:8]],
            expires_at=now + 86400,
        )
        event = {
            **_detection_event_base(now, turn_index, source),
            "type": "RepeatedFailurePathDetectedEvent",
            "action_trigger": trigger,
            "lookback_window": LOOKBACK_EPISODES,
            "failure_count": len(streak),
            "mean_delta_fe": round(sum(row.delta_fe_proxy for row in streak) / len(streak), 6),
            "emitted_intent_id": intent.intent_id if apply_intents else "",
        }
        events.append(event)
        if apply_intents:
            _store_intent(state, intent)
            intents.append(intent)
        break
    return events, intents


def _m12_tensions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    m12 = _mapping(state.get("m12_user_continuity"))
    rows = m12.get("identity_tensions", [])
    if not rows:
        rows = m12.get("conflict_records", [])
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        refs = _string_list(row.get("evidence_refs") or row.get("evidence_quote_ids"), limit=8)
        tid = str(row.get("tension_id") or row.get("conflict_id") or row.get("id") or (refs[0] if refs else ""))
        confidence = _bounded_float(row.get("confidence"), default=_bounded_float(row.get("severity"), default=0.0))
        if tid and confidence >= MIN_TENSION_CONFIDENCE:
            out.append({"id": tid, "confidence": confidence, "evidence_refs": refs})
    return out


def _m12_1_report_refs(state: Mapping[str, Any]) -> list[str]:
    m12_1 = _mapping(state.get("m12_1_user_personality"))
    refs = _walk_evidence_refs(m12_1.get("plain_language_report"))
    refs.extend(_walk_evidence_refs(m12_1.get("latest_reports_by_user")))
    return _string_list(refs, limit=32)


def _detect_self_consistency(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    apply_intents: bool,
) -> tuple[list[dict[str, Any]], list[MetaControlInterventionIntent], dict[str, int]]:
    events: list[dict[str, Any]] = []
    intents: list[MetaControlInterventionIntent] = []
    counts = {str(k): int(v or 0) for k, v in _mapping(meta.get("tension_observation_counts")).items()}
    if _has_active_kind(meta, "request_reflection_focus"):
        return events, intents, counts
    if source != "idle_cognitive_tick":
        return events, intents, counts
    report_refs = set(_m12_1_report_refs(state))
    for tension in _m12_tensions(state):
        refs = set(_string_list(tension.get("evidence_refs"), limit=8))
        if not refs or not (refs & report_refs):
            continue
        tid = str(tension.get("id", ""))
        counts[tid] = int(counts.get(tid, 0)) + 1
        if counts[tid] < MIN_IDLE_TICKS_STABLE_TENSION:
            continue
        intent = MetaControlInterventionIntent(
            intent_id=_new_id("m15_meta_intent"),
            at=now,
            turn_index=turn_index,
            detector="SelfConsistencyTensionDetector",
            intent_kind="request_reflection_focus",
            payload={
                "focus_topic": f"self_consistency:{tid}"[:120],
                "suggested_reflection_kind": "self_consistency",
            },
            evidence_refs=sorted(refs)[:8],
            detector_evidence_event_ids=[tid],
            expires_at=now + 86400,
        )
        events.append(
            {
                **_detection_event_base(now, turn_index, source),
                "type": "SelfConsistencyTensionDetectedEvent",
                "tension_id": tid,
                "confidence": round(_bounded_float(tension.get("confidence")), 6),
                "stable_tick_count": counts[tid],
                "evidence_refs": sorted(refs)[:8],
                "emitted_intent_id": intent.intent_id if apply_intents else "",
            }
        )
        if apply_intents:
            _store_intent(state, intent)
            intents.append(intent)
        break
    return events, intents, counts


def _detect_stall(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    current_idle_tick_event: Mapping[str, Any] | None,
    apply_intents: bool,
) -> tuple[list[dict[str, Any]], list[MetaControlInterventionIntent], list[dict[str, Any]]]:
    history = [dict(row) for row in meta.get("idle_tick_history", []) or [] if isinstance(row, Mapping)]
    if current_idle_tick_event is not None:
        history.append(
            {
                "at": int(current_idle_tick_event.get("at", now) or now),
                "turn_index": int(current_idle_tick_event.get("turn_index", turn_index) or turn_index),
                "reject_reason": str(current_idle_tick_event.get("reject_reason", "") or ""),
                "bands": dict(_mapping(current_idle_tick_event.get("bands"))),
            }
        )
    history = history[-STALL_TICK_WINDOW:]
    events: list[dict[str, Any]] = []
    intents: list[MetaControlInterventionIntent] = []
    if len(history) < STALL_TICK_WINDOW or _has_active_kind(meta, "bias_idle_recall_breadth"):
        return events, intents, history
    band_keys = ("boredom_band", "reward_band", "behavior_band", "relation_band")
    first_bands = _mapping(history[0].get("bands"))
    flat = all(_mapping(row.get("bands")).get(key, "") == first_bands.get(key, "") for row in history for key in band_keys)
    reasons = [str(row.get("reject_reason", "") or "") for row in history if str(row.get("reject_reason", "") or "")]
    if not flat or not reasons:
        return events, intents, history
    top_reason = max(set(reasons), key=reasons.count)
    ratio = reasons.count(top_reason) / STALL_TICK_WINDOW
    if ratio < SAME_REJECT_REASON_RATIO:
        return events, intents, history
    intent = MetaControlInterventionIntent(
        intent_id=_new_id("m15_meta_intent"),
        at=now,
        turn_index=turn_index,
        detector="MetaControlStallDetector",
        intent_kind="bias_idle_recall_breadth",
        payload={"new_top_k": min(MAX_RECALL_TOP_K, DEFAULT_IDLE_TOP_K + 3), "ttl_ticks": 1, "ticks_seen": 0},
        evidence_refs=[],
        detector_evidence_event_ids=[str(row.get("at", "")) for row in history],
        expires_at=now + 3600,
    )
    events.append(
        {
            **_detection_event_base(now, turn_index, source),
            "type": "MetaControlStallDetectedEvent",
            "reject_reason": top_reason,
            "window_size": STALL_TICK_WINDOW,
            "ratio": round(ratio, 6),
            "emitted_intent_id": intent.intent_id if apply_intents else "",
        }
    )
    if apply_intents:
        _store_intent(state, intent)
        intents.append(intent)
    return events, intents, history


def detect_and_emit_intents(
    state: dict[str, Any],
    ledger: EpisodeLedger,
    *,
    now: int,
    turn_index: int,
    source: str,
    current_idle_tick_event: Mapping[str, Any] | None = None,
) -> MetaControlDetectionResult:
    if bool(state.get("m13_ui_turn_in_progress")):
        return MetaControlDetectionResult(events=[], intents=[])
    events: list[dict[str, Any]] = []
    intents: list[MetaControlInterventionIntent] = []
    expiration_events = expire_meta_control_intents(
        state,
        now=now,
        turn_index=turn_index,
        tick=current_idle_tick_event is not None,
    )
    events.extend(expiration_events)
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    apply_intents = _apply_enabled(state)
    try:
        det_events, det_intents = _detect_repeated_failure(
            state,
            ledger,
            now=now,
            turn_index=turn_index,
            source=source,
            meta=meta,
            apply_intents=apply_intents,
        )
        events.extend(det_events)
        intents.extend(det_intents)
    except Exception as exc:
        events.append({**_detection_event_base(now, turn_index, source), "type": "MetaControlDetectorErrorEvent", "detector": "RepeatedFailurePathDetector", "error_type": type(exc).__name__})
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    try:
        det_events, det_intents, counts = _detect_self_consistency(
            state,
            now=now,
            turn_index=turn_index,
            source=source,
            meta=meta,
            apply_intents=apply_intents,
        )
        events.extend(det_events)
        intents.extend(det_intents)
        m13 = _mapping(state.get("m13_drive_state"))
        meta = _meta_state(m13)
        meta["tension_observation_counts"] = counts
        m13["meta_control_intents"] = meta
        state["m13_drive_state"] = m13
    except Exception as exc:
        events.append({**_detection_event_base(now, turn_index, source), "type": "MetaControlDetectorErrorEvent", "detector": "SelfConsistencyTensionDetector", "error_type": type(exc).__name__})
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    try:
        det_events, det_intents, history = _detect_stall(
            state,
            now=now,
            turn_index=turn_index,
            source=source,
            meta=meta,
            current_idle_tick_event=current_idle_tick_event,
            apply_intents=apply_intents,
        )
        events.extend(det_events)
        intents.extend(det_intents)
        m13 = _mapping(state.get("m13_drive_state"))
        meta = _meta_state(m13)
        meta["idle_tick_history"] = history[-STALL_TICK_WINDOW:]
        recent = list(meta.get("recent_detections", []))
        recent.extend([event for event in events if str(event.get("type", "")).endswith("DetectedEvent")])
        meta["recent_detections"] = recent[-16:]
        m13["meta_control_intents"] = meta
        state["m13_drive_state"] = m13
    except Exception as exc:
        events.append({**_detection_event_base(now, turn_index, source), "type": "MetaControlDetectorErrorEvent", "detector": "MetaControlStallDetector", "error_type": type(exc).__name__})
    return MetaControlDetectionResult(events=events, intents=intents)


def consume_recall_breadth_intent(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    default_top_k: int = DEFAULT_IDLE_TOP_K,
) -> tuple[int, list[dict[str, Any]]]:
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    events: list[dict[str, Any]] = []
    top_k = int(default_top_k)
    active: list[dict[str, Any]] = []
    for row in meta.get("active", []) or []:
        intent = MetaControlInterventionIntent.from_mapping(row)
        if intent.intent_kind != "bias_idle_recall_breadth":
            active.append(intent.to_dict())
            continue
        payload = dict(intent.payload)
        top_k = max(5, min(MAX_RECALL_TOP_K, int(payload.get("new_top_k", default_top_k) or default_top_k)))
        payload["ticks_seen"] = int(payload.get("ticks_seen", 0) or 0) + 1
        updated = MetaControlInterventionIntent(
            **{**intent.to_dict(), "payload": payload}
        ).to_dict()
        active.append(updated)
        events.append(
            {
                "type": "MetaControlInterventionAppliedEvent",
                "intent_id": intent.intent_id,
                "applied_to": "m13_5_tick",
                "applied_at": now,
                "applied_effect_summary": f"recall_top_k={top_k}",
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        break
    meta["active"] = active[-MAX_ACTIVE_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13
    events.extend(expire_meta_control_intents(state, now=now, turn_index=turn_index, tick=True))
    return top_k, events


def apply_reflection_focus_intent(
    state: dict[str, Any],
    plan: Mapping[str, Any],
    *,
    now: int,
    turn_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mutable = dict(plan)
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    active: list[dict[str, Any]] = []
    consumed = list(meta.get("consumed", []))
    events: list[dict[str, Any]] = []
    applied = False
    for row in meta.get("active", []) or []:
        intent = MetaControlInterventionIntent.from_mapping(row)
        if applied or intent.intent_kind != "request_reflection_focus":
            active.append(intent.to_dict())
            continue
        payload = dict(intent.payload)
        if not mutable.get("reflection_focus"):
            mutable["reflection_focus"] = {
                "topic": str(payload.get("focus_topic", "") or "")[:120],
                "evidence_refs": list(intent.evidence_refs[:8]),
                "reflection_kind": str(payload.get("suggested_reflection_kind", "self_consistency") or "self_consistency"),
            }
            summary = "reflection_focus_set"
        else:
            summary = "existing_focus_preserved"
        consumed_row = intent.to_dict()
        consumed_row["consumed_at"] = now
        consumed_row["consumed_by"] = "m14_reflector"
        consumed.append(consumed_row)
        events.append(
            {
                "type": "MetaControlInterventionAppliedEvent",
                "intent_id": intent.intent_id,
                "applied_to": "m14_reflector",
                "applied_at": now,
                "applied_effect_summary": summary,
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        events.append(
            {
                "type": "MetaControlInterventionExpiredEvent",
                "intent_id": intent.intent_id,
                "expired_at": now,
                "reason": "consumed",
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        applied = True
    meta["active"] = active[-MAX_ACTIVE_INTENTS:]
    meta["consumed"] = consumed[-MAX_CONSUMED_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13
    return mutable, events


def apply_trigger_suppression_intent(
    state: dict[str, Any],
    *,
    action_trigger: str,
    now: int,
    turn_index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    events = expire_meta_control_intents(state, now=now, turn_index=turn_index, tick=False)
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    for row in meta.get("active", []) or []:
        intent = MetaControlInterventionIntent.from_mapping(row)
        if intent.intent_kind != "suppress_action_trigger_for_n_turns":
            continue
        if str(intent.payload.get("action_trigger", "") or "") != str(action_trigger or ""):
            continue
        events.append(
            {
                "type": "MetaControlInterventionAppliedEvent",
                "intent_id": intent.intent_id,
                "applied_to": "m13_initiative",
                "applied_at": now,
                "applied_effect_summary": f"suppress_action_trigger={action_trigger}",
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        return state, intent.to_dict(), events
    return state, None, events
