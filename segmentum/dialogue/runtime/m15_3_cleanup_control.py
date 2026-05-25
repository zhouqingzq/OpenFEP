"""M15.3 cleanup meta-control for open items and pending expectations.

The module is structural only: it detects backlog pressure, emits bounded
cleanup intents, and routes all mutations through CleanupOwner.  It never
deletes rows and never compares natural-language content for sameness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import uuid
from typing import Any, Mapping


ENGINEERING_PROXY_LABEL = "mvp_local_cleanup_meta_control"

MAX_ACTIVE_CLEANUP_INTENTS = 8
MAX_CONSUMED_CLEANUP_INTENTS = 24
MAX_RECENT_CLEANUP_DETECTIONS = 24
MAX_CLEANUP_ROWS_PER_RUN = 8
MAX_MERGES_PER_RUN = 4
MAX_EXPIRES_PER_RUN = 6
MAX_DIAGNOSTIC_MARKS_PER_RUN = 6
MAX_RECALL_DEPRIORITIZE_PER_RUN = 6
DEFAULT_DEPRIORITIZE_TTL_SECONDS = 24 * 3600
RECENT_ROW_GRACE_SECONDS = 3600

OPEN_BACKLOG_MIN_ROWS = 8
PENDING_BACKLOG_MIN_ROWS = 8
LOW_TRACEABILITY_RATIO_FLOOR = 0.50
DUPLICATE_LOCAL_ID_MIN = 2
STALE_OPEN_ITEM_SECONDS = 14 * 86400
STALE_PENDING_SECONDS = 3 * 86400
EXPIRED_NEXT_USER_TURN_GRACE_TURNS = 2
EXPIRED_IDLE_EXPECTATION_SECONDS = 24 * 3600

RECALL_BURDEN_WINDOW = 5
NO_TARGET_RATIO = 0.80
LOW_TRACE_RETRIEVAL_RATIO = 0.50
RECALL_BURDEN_MIN_LOW_TRACE_IDS = 4

NO_TARGET_REJECT_REASONS = frozenset(
    {
        "no_high_value_target",
        "no_eligible_expectation",
        "cleanup_filtered_low_traceability_candidates",
    }
)

CLEANUP_INTENT_KINDS = frozenset(
    {
        "cleanup_open_item_backlog",
        "cleanup_pending_expectation_backlog",
        "deprioritize_low_traceability_recall_burden",
    }
)

CLEANUP_FILTER_PHASES = frozenset({"idle", "proactive", "memory_efe"})


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _epoch(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    raw = value if isinstance(value, (list, tuple, set)) else [value] if value is not None else []
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("open_item_id") or row.get("expectation_id") or "").strip()[:120]


def _status(row: Mapping[str, Any], *, default: str = "") -> str:
    return str(row.get("status", default) or default).strip().lower()


def _created_at(row: Mapping[str, Any]) -> int:
    return _epoch(row.get("created_at_epoch") or row.get("created_at") or row.get("at"))


def _created_turn_index(row: Mapping[str, Any]) -> int | None:
    raw = row.get("created_turn_index", row.get("turn_index"))
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def strict_evidence_refs(row: Mapping[str, Any], *, limit: int = 8) -> list[str]:
    """Evidence refs that are external anchors, never row-id fallbacks."""
    refs = _string_list(row.get("evidence_refs"), limit=limit)
    refs.extend(_string_list(row.get("evidence_ref"), limit=limit))
    refs.extend(_string_list(row.get("source_event_id"), limit=limit))
    return list(dict.fromkeys(refs))[:limit]


def strict_bound_memory_ids(row: Mapping[str, Any], *, limit: int = 8) -> list[str]:
    ids = _string_list(row.get("bound_memory_ids"), limit=limit)
    ids.extend(_string_list(row.get("memory_id"), limit=limit))
    ids.extend(_string_list(row.get("source_memory_id"), limit=limit))
    return list(dict.fromkeys(ids))[:limit]


def explicit_scheduled_anchor_refs(row: Mapping[str, Any], *, limit: int = 8) -> list[str]:
    refs = strict_evidence_refs(row, limit=limit)
    refs.extend(
        _string_list(row.get("source_intent_id") or row.get("scheduled_intent_id") or row.get("intent_id"), limit=limit)
    )
    return list(dict.fromkeys(refs))[:limit]


def is_self_referential_evidence_only(row: Mapping[str, Any]) -> bool:
    row_ids = {
        str(row.get("id") or "").strip(),
        str(row.get("open_item_id") or "").strip(),
        str(row.get("expectation_id") or "").strip(),
    }
    row_ids = {item for item in row_ids if item}
    refs = strict_evidence_refs(row)
    bounds = strict_bound_memory_ids(row)
    if not row_ids or not (refs or bounds):
        return False
    return all(ref in row_ids for ref in [*refs, *bounds])


def is_strictly_traceable(row: Mapping[str, Any]) -> bool:
    if is_self_referential_evidence_only(row):
        return False
    refs = strict_evidence_refs(row)
    bounds = strict_bound_memory_ids(row)
    if refs or bounds:
        return True
    if str(row.get("source_kind", "") or "") == "scheduled_outreach":
        return bool(explicit_scheduled_anchor_refs(row))
    return False


def is_cleanup_protected(row: Mapping[str, Any], *, now: int) -> bool:
    if bool(row.get("pin")):
        return True
    if str(row.get("scheduled_intent_id") or row.get("intent_id") or "").strip():
        return True
    created = _created_at(row)
    if created and now - created < RECENT_ROW_GRACE_SECONDS:
        return True
    if is_strictly_traceable(row):
        return True
    return False


def summarize_strict_traceability(state: Mapping[str, Any]) -> dict[str, int]:
    open_rows = _open_item_rows(state)
    pending_rows = _pending_expectation_rows(state)
    open_strict = sum(1 for row in open_rows if is_strictly_traceable(row))
    pending_strict = sum(1 for row in pending_rows if is_strictly_traceable(row))
    return {
        "open_items_total": len(open_rows),
        "open_items_strict_trace": open_strict,
        "open_items_duplicate_local_ids": len(_duplicate_local_ids(open_rows)),
        "pending_expectations_total": len(pending_rows),
        "pending_expectations_strict_trace": pending_strict,
        "pending_expectations_duplicate_local_ids": len(_duplicate_local_ids(pending_rows)),
    }


def cleanup_ineligibility_reason(
    row: Mapping[str, Any],
    *,
    now: int,
    phase: str,
    expectation: bool = True,
) -> str:
    status = _status(row)
    if status == "expired" and expectation:
        return "expectation_expired"
    if status.startswith("merged_into:"):
        return "expectation_merged" if expectation else "cleanup_filtered_low_traceability_candidates"
    if status == "diagnostic_only":
        return "low_traceability_cleanup_deprioritized"
    until = _epoch(row.get("recall_deprioritized_until"))
    if until and int(now) < until and str(phase or "") in CLEANUP_FILTER_PHASES:
        return "low_traceability_cleanup_deprioritized"
    if is_self_referential_evidence_only(row):
        return "self_referential_evidence_only"
    return ""


def cleanup_recall_suppression_reason(
    row: Mapping[str, Any],
    *,
    now: int,
    phase: str,
) -> str:
    status = _status(row)
    if status in {"archived", "expired", "diagnostic_only"}:
        return status
    if status.startswith("merged_into:"):
        return "merged"
    until = _epoch(row.get("recall_deprioritized_until"))
    if until and int(now) < until and str(phase or "") in CLEANUP_FILTER_PHASES:
        return "recall_deprioritized_until"
    return ""


@dataclass(frozen=True)
class CleanupInterventionIntent:
    intent_id: str
    at: int
    turn_index: int
    detector: str
    intent_kind: str
    payload: dict[str, Any]
    evidence_refs: list[str] = field(default_factory=list)
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
            "expires_at": self.expires_at,
            "engineering_proxy_label": self.engineering_proxy_label,
        }

    @staticmethod
    def from_mapping(row: Mapping[str, Any]) -> "CleanupInterventionIntent":
        return CleanupInterventionIntent(
            intent_id=str(row.get("intent_id", "")),
            at=_epoch(row.get("at")),
            turn_index=_epoch(row.get("turn_index")),
            detector=str(row.get("detector", "")),
            intent_kind=str(row.get("intent_kind", "")),
            payload=_mapping(row.get("payload")),
            evidence_refs=_string_list(row.get("evidence_refs"), limit=8),
            expires_at=_epoch(row.get("expires_at")),
        )


@dataclass(frozen=True)
class CleanupDetectionResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    intents: list[CleanupInterventionIntent] = field(default_factory=list)


@dataclass(frozen=True)
class CleanupRunResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    ops: dict[str, int] = field(default_factory=dict)


def _meta_state(m13_state: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(m13_state.get("meta_control_intents"))
    return {
        **raw,
        "cleanup_active": [
            dict(row) for row in raw.get("cleanup_active", []) or [] if isinstance(row, Mapping)
        ][-MAX_ACTIVE_CLEANUP_INTENTS:],
        "cleanup_consumed": [
            dict(row) for row in raw.get("cleanup_consumed", []) or [] if isinstance(row, Mapping)
        ][-MAX_CONSUMED_CLEANUP_INTENTS:],
        "cleanup_recent_detections": [
            dict(row) for row in raw.get("cleanup_recent_detections", []) or [] if isinstance(row, Mapping)
        ][-MAX_RECENT_CLEANUP_DETECTIONS:],
        "recent_idle_cognitive_ticks": [
            dict(row) for row in raw.get("recent_idle_cognitive_ticks", []) or [] if isinstance(row, Mapping)
        ][-RECALL_BURDEN_WINDOW:],
    }


def _cleanup_apply_enabled(state: Mapping[str, Any]) -> bool:
    cleanup_env = str(os.environ.get("SEGMENTUM_CLEANUP_CONTROL_APPLY", "") or "").strip()
    if cleanup_env == "0":
        return False
    if cleanup_env == "1":
        return True
    if str(os.environ.get("SEGMENTUM_META_CONTROL_APPLY", "") or "").strip() == "1":
        return True
    return True


def _has_active_cleanup_kind(meta: Mapping[str, Any], intent_kind: str) -> bool:
    return any(str(row.get("intent_kind", "")) == intent_kind for row in meta.get("cleanup_active", []) or [])


def _store_cleanup_intent(state: dict[str, Any], intent: CleanupInterventionIntent) -> None:
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    meta["cleanup_active"] = [*meta["cleanup_active"], intent.to_dict()][-MAX_ACTIVE_CLEANUP_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13


def expire_cleanup_intents(state: dict[str, Any], *, now: int) -> list[dict[str, Any]]:
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    kept: list[dict[str, Any]] = []
    consumed = list(meta.get("cleanup_consumed", []))
    events: list[dict[str, Any]] = []
    for row in meta.get("cleanup_active", []) or []:
        intent = CleanupInterventionIntent.from_mapping(row)
        expired = bool(intent.expires_at and int(now) >= int(intent.expires_at))
        if expired:
            consumed_row = intent.to_dict()
            consumed_row["expired_at"] = now
            consumed_row["expiration_reason"] = "ttl"
            consumed.append(consumed_row)
            events.append(
                {
                    "type": "CleanupIntentExpiredEvent",
                    "intent_id": intent.intent_id,
                    "at": now,
                    "reason": "ttl",
                    "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
                }
            )
        else:
            kept.append(intent.to_dict())
    meta["cleanup_active"] = kept[-MAX_ACTIVE_CLEANUP_INTENTS:]
    meta["cleanup_consumed"] = consumed[-MAX_CONSUMED_CLEANUP_INTENTS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13
    return events


def _record_idle_tick(meta: dict[str, Any], tick_event: Mapping[str, Any] | None) -> None:
    if not isinstance(tick_event, Mapping):
        return
    recent = list(meta.get("recent_idle_cognitive_ticks", []))
    recent.append(
        {
            "at": _epoch(tick_event.get("at")),
            "reject_reason": str(tick_event.get("reject_reason", "") or ""),
            "retrieved_ids": _string_list(tick_event.get("retrieved_ids"), limit=12),
        }
    )
    meta["recent_idle_cognitive_ticks"] = recent[-RECALL_BURDEN_WINDOW:]


def _low_traceable_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if not is_strictly_traceable(row)]


def _duplicate_local_ids(rows: list[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        row_id = _row_id(row)
        if row_id:
            counts[row_id] = counts.get(row_id, 0) + 1
    return sorted(row_id for row_id, count in counts.items() if count >= DUPLICATE_LOCAL_ID_MIN)


def _row_lookup(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for row in [*(_open_item_rows(state)), *(_pending_expectation_rows(state))]:
        row_id = _row_id(row)
        if row_id:
            lookup[row_id] = row
    return lookup


def _pending_turn_grace_expired(row: Mapping[str, Any], *, turn_index: int) -> bool:
    verify_on = str(row.get("verify_on", row.get("verify", "")) or "").strip().casefold()
    if verify_on not in {"next_user_turn", "next turn", "next-turn"}:
        return False
    created_turn = _created_turn_index(row)
    if created_turn is None:
        return False
    return turn_index - created_turn >= EXPIRED_NEXT_USER_TURN_GRACE_TURNS


def _pending_idle_expectation_expired(row: Mapping[str, Any], *, now: int) -> bool:
    verify_on = str(row.get("verify_on", row.get("verify", "")) or "").strip().casefold()
    if verify_on != "memory_dynamics_idle":
        return False
    created = _created_at(row)
    return bool(created and now - created >= EXPIRED_IDLE_EXPECTATION_SECONDS)


def _detection_event_base(now: int, turn_index: int, source: str) -> dict[str, Any]:
    return {
        "at": now,
        "turn_index": turn_index,
        "source": source,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def _append_cleanup_op_event(
    events: list[dict[str, Any]],
    *,
    run_id: str,
    intent_id: str,
    at: int,
    turn_index: int,
    op: str,
    source_ids: list[str],
    retained_id: str,
    new_status: str,
    reason_code: str,
) -> None:
    events.append(
        {
            "type": "CleanupOpEvent",
            "run_id": run_id,
            "intent_id": intent_id,
            "at": at,
            "turn_index": turn_index,
            "op": op,
            "source_ids": source_ids[:8],
            "retained_id": retained_id,
            "new_status": new_status,
            "reason_code": reason_code,
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }
    )


def _open_item_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in state.get("open_items", []) or []
        if isinstance(row, Mapping) and _status(row, default="open") in {"", "open", "active", "pending"}
    ]


def _pending_expectation_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in state.get("pending_expectations", []) or []
        if isinstance(row, Mapping) and _status(row, default="pending") in {"", "pending", "active", "uncertain", "due"}
    ]


def _backlog_should_fire(*, row_count: int, low_count: int, dup_ids: list[str], min_rows: int) -> bool:
    ratio = low_count / max(1, row_count)
    if row_count >= min_rows and ratio >= LOW_TRACEABILITY_RATIO_FLOOR:
        return True
    if dup_ids:
        return True
    return False


def _detect_open_item_backlog(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    apply_intents: bool,
) -> tuple[list[dict[str, Any]], list[CleanupInterventionIntent]]:
    rows = _open_item_rows(state)
    if not rows or _has_active_cleanup_kind(meta, "cleanup_open_item_backlog"):
        return [], []
    low = [row for row in _low_traceable_rows(rows) if not is_cleanup_protected(row, now=now)]
    dup_ids = _duplicate_local_ids(rows)
    stale = [
        _row_id(row)
        for row in rows
        if not is_cleanup_protected(row, now=now)
        and _created_at(row)
        and now - _created_at(row) >= STALE_OPEN_ITEM_SECONDS
    ]
    if not _backlog_should_fire(row_count=len(rows), low_count=len(low), dup_ids=dup_ids, min_rows=OPEN_BACKLOG_MIN_ROWS) and not stale:
        return [], []
    candidate_ids = list(
        dict.fromkeys([_row_id(row) for row in low if _row_id(row)] + stale + dup_ids)
    )[:MAX_CLEANUP_ROWS_PER_RUN]
    if not candidate_ids:
        return [], []
    intent = CleanupInterventionIntent(
        intent_id=_new_id("m15_cleanup_intent"),
        at=now,
        turn_index=turn_index,
        detector="OpenItemBacklogDetector",
        intent_kind="cleanup_open_item_backlog",
        payload={
            "candidate_ids": candidate_ids,
            "low_traceability_count": len(low),
            "duplicate_local_ids": dup_ids[:8],
            "stale_count": len(stale),
        },
        evidence_refs=[],
        expires_at=now + 86400,
    )
    event = {
        **_detection_event_base(now, turn_index, source),
        "type": "OpenItemBacklogDetectedEvent",
        "open_item_count": len(rows),
        "low_traceability_count": len(low),
        "duplicate_local_id_count": len(dup_ids),
        "stale_count": len(stale),
        "candidate_ids": candidate_ids[:8],
        "emitted_intent_id": intent.intent_id if apply_intents else "",
    }
    if apply_intents:
        _store_cleanup_intent(state, intent)
        return [event], [intent]
    return [event], []


def _detect_pending_expectation_backlog(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    apply_intents: bool,
) -> tuple[list[dict[str, Any]], list[CleanupInterventionIntent]]:
    rows = _pending_expectation_rows(state)
    if not rows or _has_active_cleanup_kind(meta, "cleanup_pending_expectation_backlog"):
        return [], []
    low = [row for row in _low_traceable_rows(rows) if not is_cleanup_protected(row, now=now)]
    dup_ids = _duplicate_local_ids(rows)
    stale = [
        _row_id(row)
        for row in rows
        if not is_cleanup_protected(row, now=now)
        and (
            (_created_at(row) and now - _created_at(row) >= STALE_PENDING_SECONDS)
            or _pending_turn_grace_expired(row, turn_index=turn_index)
            or _pending_idle_expectation_expired(row, now=now)
        )
    ]
    turn_grace = [_row_id(row) for row in rows if _pending_turn_grace_expired(row, turn_index=turn_index)]
    if not _backlog_should_fire(row_count=len(rows), low_count=len(low), dup_ids=dup_ids, min_rows=PENDING_BACKLOG_MIN_ROWS) and not stale:
        return [], []
    candidate_ids = list(
        dict.fromkeys([_row_id(row) for row in low if _row_id(row)] + stale + dup_ids + turn_grace)
    )[:MAX_CLEANUP_ROWS_PER_RUN]
    if not candidate_ids:
        return [], []
    intent = CleanupInterventionIntent(
        intent_id=_new_id("m15_cleanup_intent"),
        at=now,
        turn_index=turn_index,
        detector="PendingExpectationBacklogDetector",
        intent_kind="cleanup_pending_expectation_backlog",
        payload={
            "candidate_ids": candidate_ids,
            "low_traceability_count": len(low),
            "duplicate_local_ids": dup_ids[:8],
            "stale_count": len(stale),
        },
        evidence_refs=[],
        expires_at=now + 86400,
    )
    event = {
        **_detection_event_base(now, turn_index, source),
        "type": "PendingExpectationBacklogDetectedEvent",
        "pending_expectation_count": len(rows),
        "low_traceability_count": len(low),
        "duplicate_local_id_count": len(dup_ids),
        "stale_count": len(stale),
        "candidate_ids": candidate_ids[:8],
        "emitted_intent_id": intent.intent_id if apply_intents else "",
    }
    if apply_intents:
        _store_cleanup_intent(state, intent)
        return [event], [intent]
    return [event], []


def _recall_burden_metrics(
    state: Mapping[str, Any],
    recent_ticks: list[Mapping[str, Any]],
) -> tuple[float, float, list[str]]:
    if len(recent_ticks) < RECALL_BURDEN_WINDOW:
        return 0.0, 0.0, []
    lookup = _row_lookup(state)
    no_target = sum(
        1 for tick in recent_ticks[-RECALL_BURDEN_WINDOW:] if str(tick.get("reject_reason", "") or "") in NO_TARGET_REJECT_REASONS
    )
    no_target_ratio = no_target / float(RECALL_BURDEN_WINDOW)
    low_trace_hits = 0
    total_retrieved = 0
    low_trace_ids: list[str] = []
    for tick in recent_ticks[-RECALL_BURDEN_WINDOW:]:
        for retrieved_id in _string_list(tick.get("retrieved_ids"), limit=12):
            total_retrieved += 1
            row = lookup.get(retrieved_id)
            if row is None or not is_strictly_traceable(row):
                low_trace_hits += 1
                if retrieved_id not in low_trace_ids:
                    low_trace_ids.append(retrieved_id)
    retrieval_ratio = low_trace_hits / float(max(1, total_retrieved))
    return no_target_ratio, retrieval_ratio, low_trace_ids


def _detect_low_traceability_recall_burden(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    meta: Mapping[str, Any],
    apply_intents: bool,
    current_idle_tick_event: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[CleanupInterventionIntent]]:
    if _has_active_cleanup_kind(meta, "deprioritize_low_traceability_recall_burden"):
        return [], []
    recent_ticks = list(meta.get("recent_idle_cognitive_ticks", []))
    no_target_ratio, retrieval_ratio, low_trace_ids = _recall_burden_metrics(state, recent_ticks)
    if no_target_ratio < NO_TARGET_RATIO:
        return [], []
    if retrieval_ratio < LOW_TRACE_RETRIEVAL_RATIO:
        return [], []
    open_rows = _open_item_rows(state)
    pending_rows = _pending_expectation_rows(state)
    low_rows = [
        row
        for row in [*open_rows, *pending_rows]
        if not is_strictly_traceable(row) and not is_cleanup_protected(row, now=now)
    ]
    low_ids = list(dict.fromkeys(_row_id(row) for row in low_rows if _row_id(row)))
    if len(low_rows) < RECALL_BURDEN_MIN_LOW_TRACE_IDS:
        return [], []
    reject_reason = str(_mapping(current_idle_tick_event).get("reject_reason", "") or "")
    if source == "idle_cognitive_tick" and reject_reason not in NO_TARGET_REJECT_REASONS:
        return [], []
    intent = CleanupInterventionIntent(
        intent_id=_new_id("m15_cleanup_intent"),
        at=now,
        turn_index=turn_index,
        detector="LowTraceabilityRecallBurdenDetector",
        intent_kind="deprioritize_low_traceability_recall_burden",
        payload={
            "candidate_ids": low_ids[:MAX_CLEANUP_ROWS_PER_RUN],
            "ttl_seconds": DEFAULT_DEPRIORITIZE_TTL_SECONDS,
            "no_target_ratio": round(no_target_ratio, 3),
            "low_traceability_retrieval_ratio": round(retrieval_ratio, 3),
        },
        evidence_refs=[],
        expires_at=now + 3600,
    )
    event = {
        **_detection_event_base(now, turn_index, source),
        "type": "LowTraceabilityRecallBurdenDetectedEvent",
        "window": RECALL_BURDEN_WINDOW,
        "no_target_ratio": round(no_target_ratio, 3),
        "low_traceability_retrieval_ratio": round(retrieval_ratio, 3),
        "retrieved_candidate_ids": low_trace_ids[:8],
        "candidate_count": len(low_ids),
        "idle_reject_reason": reject_reason,
        "emitted_intent_id": intent.intent_id if apply_intents else "",
    }
    if apply_intents:
        _store_cleanup_intent(state, intent)
        return [event], [intent]
    return [event], []


def detect_cleanup_intents(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    source: str,
    current_idle_tick_event: Mapping[str, Any] | None = None,
) -> CleanupDetectionResult:
    if bool(state.get("m13_ui_turn_in_progress")):
        return CleanupDetectionResult(events=[], intents=[])
    events = expire_cleanup_intents(state, now=now)
    intents: list[CleanupInterventionIntent] = []
    apply_intents = _cleanup_apply_enabled(state)
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    if source == "idle_cognitive_tick":
        _record_idle_tick(meta, current_idle_tick_event)
        m13["meta_control_intents"] = meta
        state["m13_drive_state"] = m13
    for detector, name in (
        (_detect_open_item_backlog, "OpenItemBacklogDetector"),
        (_detect_pending_expectation_backlog, "PendingExpectationBacklogDetector"),
    ):
        try:
            det_events, det_intents = detector(
                state,
                now=now,
                turn_index=turn_index,
                source=source,
                meta=meta,
                apply_intents=apply_intents,
            )
            events.extend(det_events)
            intents.extend(det_intents)
        except Exception as exc:
            events.append(
                {
                    **_detection_event_base(now, turn_index, source),
                    "type": "CleanupDetectorErrorEvent",
                    "detector": name,
                    "error_type": type(exc).__name__,
                }
            )
        m13 = _mapping(state.get("m13_drive_state"))
        meta = _meta_state(m13)
    try:
        det_events, det_intents = _detect_low_traceability_recall_burden(
            state,
            now=now,
            turn_index=turn_index,
            source=source,
            meta=meta,
            apply_intents=apply_intents,
            current_idle_tick_event=current_idle_tick_event,
        )
        events.extend(det_events)
        intents.extend(det_intents)
    except Exception as exc:
        events.append(
            {
                **_detection_event_base(now, turn_index, source),
                "type": "CleanupDetectorErrorEvent",
                "detector": "LowTraceabilityRecallBurdenDetector",
                "error_type": type(exc).__name__,
            }
        )
    m13 = _mapping(state.get("m13_drive_state"))
    meta = _meta_state(m13)
    recent = list(meta.get("cleanup_recent_detections", []))
    recent.extend([event for event in events if str(event.get("type", "")).endswith("DetectedEvent")])
    meta["cleanup_recent_detections"] = recent[-MAX_RECENT_CLEANUP_DETECTIONS:]
    m13["meta_control_intents"] = meta
    state["m13_drive_state"] = m13
    return CleanupDetectionResult(events=events, intents=intents)


def _row_kind_lists(state: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    open_items = state.setdefault("open_items", [])
    pending = state.setdefault("pending_expectations", [])
    if not isinstance(open_items, list):
        open_items = []
        state["open_items"] = open_items
    if not isinstance(pending, list):
        pending = []
        state["pending_expectations"] = pending
    return open_items, pending


def _structural_signature(row: Mapping[str, Any], *, kind: str) -> tuple[Any, ...] | None:
    refs = tuple(strict_evidence_refs(row, limit=12))
    bounds = tuple(strict_bound_memory_ids(row, limit=12))
    if not refs and not bounds:
        return None
    if kind == "open_item":
        return (kind, _status(row, default="open"), str(row.get("next_check", "") or ""), refs, bounds)
    return (kind, _status(row, default="pending"), str(row.get("verify_on", row.get("verify", "")) or ""), refs, bounds)


def _matches_candidate(row: Mapping[str, Any], candidate_ids: set[str]) -> bool:
    return not candidate_ids or _row_id(row) in candidate_ids


def _should_expire_pending(row: Mapping[str, Any], *, now: int, turn_index: int) -> bool:
    if is_cleanup_protected(row, now=now):
        return False
    if is_strictly_traceable(row):
        return False
    if _created_at(row) and now - _created_at(row) >= STALE_PENDING_SECONDS:
        return True
    if _pending_turn_grace_expired(row, turn_index=turn_index):
        return True
    if _pending_idle_expectation_expired(row, now=now):
        return True
    return not is_strictly_traceable(row)


def _should_mark_open_diagnostic(row: Mapping[str, Any], *, now: int) -> bool:
    if is_cleanup_protected(row, now=now):
        return False
    if is_strictly_traceable(row) and not (_created_at(row) and now - _created_at(row) >= STALE_OPEN_ITEM_SECONDS):
        return False
    return not is_strictly_traceable(row) or (
        _created_at(row) and now - _created_at(row) >= STALE_OPEN_ITEM_SECONDS
    )


class CleanupOwner:
    """Single owner for M15.3 cleanup mutations."""

    @staticmethod
    def apply_intents(
        state: dict[str, Any],
        *,
        now: int,
        turn_index: int,
        source: str,
    ) -> CleanupRunResult:
        if bool(state.get("m13_ui_turn_in_progress")):
            return CleanupRunResult(events=[], ops={})
        if not _cleanup_apply_enabled(state):
            return CleanupRunResult(events=[], ops={})
        m13 = _mapping(state.get("m13_drive_state"))
        meta = _meta_state(m13)
        active = [CleanupInterventionIntent.from_mapping(row) for row in meta.get("cleanup_active", []) or []]
        if not active:
            return CleanupRunResult(events=[], ops={})
        open_items, pending = _row_kind_lists(state)
        consumed = list(meta.get("cleanup_consumed", []))
        events: list[dict[str, Any]] = []
        ops = {
            "merged_duplicates": 0,
            "expired_pending_expectations": 0,
            "diagnostic_open_items": 0,
            "recall_deprioritized": 0,
        }
        run_id = _new_id("m15_cleanup_run")

        for intent in active:
            payload = _mapping(intent.payload)
            candidate_ids = set(_string_list(payload.get("candidate_ids"), limit=MAX_CLEANUP_ROWS_PER_RUN))
            before_ops = dict(ops)

            for rows, kind in ((open_items, "open_item"), (pending, "pending_expectation")):
                by_sig: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
                for row in rows:
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if bool(row.get("pin")):
                        continue
                    if str(row.get("scheduled_intent_id") or row.get("intent_id") or "").strip():
                        continue
                    if _status(row).startswith("merged_into:"):
                        continue
                    sig = _structural_signature(row, kind=kind)
                    if sig is None:
                        continue
                    by_sig.setdefault(sig, []).append(row)
                for group in by_sig.values():
                    if len(group) < 2:
                        continue
                    group.sort(key=lambda row: (_created_at(row) or now, _row_id(row)))
                    canonical = group[0]
                    canonical_id = _row_id(canonical)
                    if not canonical_id:
                        continue
                    for duplicate in group[1:]:
                        if ops["merged_duplicates"] >= MAX_MERGES_PER_RUN:
                            break
                        duplicate_id = _row_id(duplicate)
                        duplicate["status"] = f"merged_into:{canonical_id}"
                        duplicate["merged_at"] = now
                        duplicate["merged_by"] = "m15_3_cleanup_owner"
                        canonical.setdefault("merged_from", [])
                        if isinstance(canonical["merged_from"], list) and duplicate_id not in canonical["merged_from"]:
                            canonical["merged_from"].append(duplicate_id)
                        canonical["evidence_refs"] = list(
                            dict.fromkeys([*strict_evidence_refs(canonical), *strict_evidence_refs(duplicate)])
                        )[:8]
                        canonical["bound_memory_ids"] = list(
                            dict.fromkeys([*strict_bound_memory_ids(canonical), *strict_bound_memory_ids(duplicate)])
                        )[:8]
                        ops["merged_duplicates"] += 1
                        op_name = "merge_open_item" if kind == "open_item" else "merge_pending_expectation"
                        _append_cleanup_op_event(
                            events,
                            run_id=run_id,
                            intent_id=intent.intent_id,
                            at=now,
                            turn_index=turn_index,
                            op=op_name,
                            source_ids=[duplicate_id],
                            retained_id=canonical_id,
                            new_status=duplicate["status"],
                            reason_code="structural_duplicate",
                        )

            if intent.intent_kind in {"cleanup_pending_expectation_backlog", "deprioritize_low_traceability_recall_burden"}:
                for row in pending:
                    if ops["expired_pending_expectations"] >= MAX_EXPIRES_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if _status(row, default="pending") not in {"", "pending", "active", "uncertain", "due"}:
                        continue
                    if not _should_expire_pending(row, now=now, turn_index=turn_index):
                        continue
                    row_id = _row_id(row)
                    row["status"] = "expired"
                    row["expired_at"] = now
                    row["expired_reason_code"] = "m15_3_low_traceability_or_stale"
                    ops["expired_pending_expectations"] += 1
                    _append_cleanup_op_event(
                        events,
                        run_id=run_id,
                        intent_id=intent.intent_id,
                        at=now,
                        turn_index=turn_index,
                        op="expire_pending_expectation",
                        source_ids=[row_id],
                        retained_id=row_id,
                        new_status="expired",
                        reason_code=str(row["expired_reason_code"]),
                    )

            if intent.intent_kind in {"cleanup_open_item_backlog", "deprioritize_low_traceability_recall_burden"}:
                for row in open_items:
                    if ops["diagnostic_open_items"] >= MAX_DIAGNOSTIC_MARKS_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if _status(row, default="open") not in {"", "open", "active", "pending"}:
                        continue
                    if not _should_mark_open_diagnostic(row, now=now):
                        continue
                    row_id = _row_id(row)
                    row["status"] = "diagnostic_only"
                    row["diagnostic_only_at"] = now
                    row["diagnostic_only_reason_code"] = "m15_3_low_traceability_or_stale"
                    ops["diagnostic_open_items"] += 1
                    _append_cleanup_op_event(
                        events,
                        run_id=run_id,
                        intent_id=intent.intent_id,
                        at=now,
                        turn_index=turn_index,
                        op="mark_open_item_diagnostic_only",
                        source_ids=[row_id],
                        retained_id=row_id,
                        new_status="diagnostic_only",
                        reason_code=str(row["diagnostic_only_reason_code"]),
                    )

            if intent.intent_kind == "deprioritize_low_traceability_recall_burden":
                ttl = min(
                    7 * 86400,
                    max(3600, int(payload.get("ttl_seconds", DEFAULT_DEPRIORITIZE_TTL_SECONDS) or DEFAULT_DEPRIORITIZE_TTL_SECONDS)),
                )
                for row in [*open_items, *pending]:
                    if ops["recall_deprioritized"] >= MAX_RECALL_DEPRIORITIZE_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if is_cleanup_protected(row, now=now) or is_strictly_traceable(row):
                        continue
                    row_id = _row_id(row)
                    row["recall_deprioritized_until"] = now + ttl
                    row["recall_deprioritized_reason_code"] = "m15_3_low_traceability_recall_burden"
                    ops["recall_deprioritized"] += 1
                    _append_cleanup_op_event(
                        events,
                        run_id=run_id,
                        intent_id=intent.intent_id,
                        at=now,
                        turn_index=turn_index,
                        op="mark_recall_deprioritized",
                        source_ids=[row_id],
                        retained_id=row_id,
                        new_status="recall_deprioritized",
                        reason_code=str(row["recall_deprioritized_reason_code"]),
                    )

            consumed_row = intent.to_dict()
            consumed_row["consumed_at"] = now
            consumed_row["source"] = source
            consumed_row["ops_delta"] = {key: ops[key] - before_ops.get(key, 0) for key in ops}
            consumed.append(consumed_row)
            events.append(
                {
                    "type": "CleanupIntentConsumedEvent",
                    "intent_id": intent.intent_id,
                    "intent_kind": intent.intent_kind,
                    "at": now,
                    "turn_index": turn_index,
                    "ops_delta": consumed_row["ops_delta"],
                    "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
                }
            )

        meta["cleanup_active"] = []
        meta["cleanup_consumed"] = consumed[-MAX_CONSUMED_CLEANUP_INTENTS:]
        m13["meta_control_intents"] = meta
        m13["m15_cleanup"] = {
            "last_run_id": run_id,
            "last_run_at": now,
            "last_source": source,
            "last_ops": dict(ops),
            "cleanup_active_count": len(meta["cleanup_active"]),
            "cleanup_consumed_count": len(meta["cleanup_consumed"]),
            "strict_traceability": summarize_strict_traceability(state),
        }
        state["m13_drive_state"] = m13
        events.append(
            {
                "type": "CleanupRunEvent",
                "run_id": run_id,
                "at": now,
                "turn_index": turn_index,
                "source": source,
                "ops": dict(ops),
                "row_deletion_count": 0,
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
        return CleanupRunResult(events=events, ops=dict(ops))
