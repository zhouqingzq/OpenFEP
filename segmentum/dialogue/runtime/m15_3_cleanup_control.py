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
MAX_CLEANUP_OPS_PER_RUN = 12
DEFAULT_DEPRIORITIZE_TTL_SECONDS = 24 * 3600

OPEN_BACKLOG_MIN_ROWS = 6
PENDING_BACKLOG_MIN_ROWS = 6
LOW_TRACEABILITY_RATIO_FLOOR = 0.60
STALE_OPEN_ITEM_SECONDS = 7 * 86400
STALE_PENDING_SECONDS = 3 * 86400

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
    refs.extend(_string_list(row.get("source_intent_id") or row.get("scheduled_intent_id") or row.get("intent_id"), limit=limit))
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
    }


def _cleanup_apply_enabled(state: Mapping[str, Any]) -> bool:
    if str(os.environ.get("SEGMENTUM_CLEANUP_CONTROL_APPLY", "") or "").strip() == "1":
        return True
    if str(os.environ.get("SEGMENTUM_META_CONTROL_APPLY", "") or "").strip() == "1":
        return True
    initiative = _mapping(_mapping(state.get("m13_drive_state")).get("initiative"))
    return str(initiative.get("proactive_policy_profile", "") or "") == "streamlit_open_chat"


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
                    "type": "CleanupInterventionExpiredEvent",
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


def _low_traceable_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if not is_strictly_traceable(row)]


def _duplicate_local_ids(rows: list[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        row_id = _row_id(row)
        if row_id:
            counts[row_id] = counts.get(row_id, 0) + 1
    return sorted(row_id for row_id, count in counts.items() if count > 1)


def _detection_event_base(now: int, turn_index: int, source: str) -> dict[str, Any]:
    return {
        "at": now,
        "turn_index": turn_index,
        "source": source,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def _open_item_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in state.get("open_items", []) or [] if isinstance(row, Mapping) and _status(row, default="open") in {"", "open", "active", "pending"}]


def _pending_expectation_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in state.get("pending_expectations", []) or [] if isinstance(row, Mapping) and _status(row, default="pending") in {"", "pending", "active", "uncertain", "due"}]


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
    low = _low_traceable_rows(rows)
    dup_ids = _duplicate_local_ids(rows)
    stale = [
        _row_id(row)
        for row in rows
        if _created_at(row) and now - _created_at(row) >= STALE_OPEN_ITEM_SECONDS
    ]
    ratio = len(low) / max(1, len(rows))
    if len(rows) < OPEN_BACKLOG_MIN_ROWS and ratio < LOW_TRACEABILITY_RATIO_FLOOR and not dup_ids:
        return [], []
    candidate_ids = list(dict.fromkeys([_row_id(row) for row in low if _row_id(row)] + stale + dup_ids))[:MAX_CLEANUP_OPS_PER_RUN]
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
    low = _low_traceable_rows(rows)
    dup_ids = _duplicate_local_ids(rows)
    stale = [
        _row_id(row)
        for row in rows
        if _created_at(row) and now - _created_at(row) >= STALE_PENDING_SECONDS
    ]
    ratio = len(low) / max(1, len(rows))
    if len(rows) < PENDING_BACKLOG_MIN_ROWS and ratio < LOW_TRACEABILITY_RATIO_FLOOR and not dup_ids:
        return [], []
    candidate_ids = list(dict.fromkeys([_row_id(row) for row in low if _row_id(row)] + stale + dup_ids))[:MAX_CLEANUP_OPS_PER_RUN]
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
        "emitted_intent_id": intent.intent_id if apply_intents else "",
    }
    if apply_intents:
        _store_cleanup_intent(state, intent)
        return [event], [intent]
    return [event], []


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
    open_rows = _open_item_rows(state)
    pending_rows = _pending_expectation_rows(state)
    low_ids = [_row_id(row) for row in [*open_rows, *pending_rows] if _row_id(row) and not is_strictly_traceable(row)]
    low_ids = list(dict.fromkeys(low_ids))
    if len(low_ids) < 4:
        return [], []
    reject_reason = str(_mapping(current_idle_tick_event).get("reject_reason", "") or "")
    if source == "idle_cognitive_tick" and reject_reason not in {"no_high_value_target", "no_eligible_expectation", "cleanup_filtered_low_traceability_candidates"}:
        return [], []
    intent = CleanupInterventionIntent(
        intent_id=_new_id("m15_cleanup_intent"),
        at=now,
        turn_index=turn_index,
        detector="LowTraceabilityRecallBurdenDetector",
        intent_kind="deprioritize_low_traceability_recall_burden",
        payload={"candidate_ids": low_ids[:MAX_CLEANUP_OPS_PER_RUN], "ttl_seconds": DEFAULT_DEPRIORITIZE_TTL_SECONDS},
        evidence_refs=[],
        expires_at=now + 3600,
    )
    event = {
        **_detection_event_base(now, turn_index, source),
        "type": "LowTraceabilityRecallBurdenDetectedEvent",
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
            events.append({**_detection_event_base(now, turn_index, source), "type": "CleanupDetectorErrorEvent", "detector": name, "error_type": type(exc).__name__})
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
        events.append({**_detection_event_base(now, turn_index, source), "type": "CleanupDetectorErrorEvent", "detector": "LowTraceabilityRecallBurdenDetector", "error_type": type(exc).__name__})
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
        m13 = _mapping(state.get("m13_drive_state"))
        meta = _meta_state(m13)
        active = [CleanupInterventionIntent.from_mapping(row) for row in meta.get("cleanup_active", []) or []]
        if not active:
            return CleanupRunResult(events=[], ops={})
        open_items, pending = _row_kind_lists(state)
        consumed = list(meta.get("cleanup_consumed", []))
        kept: list[dict[str, Any]] = []
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
            candidate_ids = set(_string_list(payload.get("candidate_ids"), limit=MAX_CLEANUP_OPS_PER_RUN))
            before_ops = dict(ops)

            # 1. merge_structural_duplicates
            for rows, kind in ((open_items, "open_item"), (pending, "pending_expectation")):
                by_sig: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
                for row in rows:
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
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
                        if ops["merged_duplicates"] >= MAX_CLEANUP_OPS_PER_RUN:
                            break
                        duplicate["status"] = f"merged_into:{canonical_id}"
                        duplicate["merged_at"] = now
                        duplicate["merged_by"] = "m15_3_cleanup_owner"
                        ops["merged_duplicates"] += 1

            # 2. expire_stale_pending_expectations
            if intent.intent_kind in {"cleanup_pending_expectation_backlog", "deprioritize_low_traceability_recall_burden"}:
                for row in pending:
                    if ops["expired_pending_expectations"] >= MAX_CLEANUP_OPS_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if _status(row, default="pending") not in {"", "pending", "active", "uncertain", "due"}:
                        continue
                    if is_strictly_traceable(row) and not (_created_at(row) and now - _created_at(row) >= STALE_PENDING_SECONDS):
                        continue
                    row["status"] = "expired"
                    row["expired_at"] = now
                    row["expired_reason_code"] = "m15_3_low_traceability_or_stale"
                    ops["expired_pending_expectations"] += 1

            # 3. mark_stale_open_items_diagnostic_only
            if intent.intent_kind in {"cleanup_open_item_backlog", "deprioritize_low_traceability_recall_burden"}:
                for row in open_items:
                    if ops["diagnostic_open_items"] >= MAX_CLEANUP_OPS_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if _status(row, default="open") not in {"", "open", "active", "pending"}:
                        continue
                    if is_strictly_traceable(row) and not (_created_at(row) and now - _created_at(row) >= STALE_OPEN_ITEM_SECONDS):
                        continue
                    row["status"] = "diagnostic_only"
                    row["diagnostic_only_at"] = now
                    row["diagnostic_only_reason_code"] = "m15_3_low_traceability_or_stale"
                    ops["diagnostic_open_items"] += 1

            # 4. mark_low_traceability_recall_deprioritized
            if intent.intent_kind == "deprioritize_low_traceability_recall_burden":
                ttl = min(7 * 86400, max(3600, int(payload.get("ttl_seconds", DEFAULT_DEPRIORITIZE_TTL_SECONDS) or DEFAULT_DEPRIORITIZE_TTL_SECONDS)))
                for row in [*open_items, *pending]:
                    if ops["recall_deprioritized"] >= MAX_CLEANUP_OPS_PER_RUN:
                        break
                    if not isinstance(row, dict) or not _matches_candidate(row, candidate_ids):
                        continue
                    if is_strictly_traceable(row):
                        continue
                    row["recall_deprioritized_until"] = now + ttl
                    row["recall_deprioritized_reason_code"] = "m15_3_low_traceability_recall_burden"
                    ops["recall_deprioritized"] += 1

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

        meta["cleanup_active"] = kept[-MAX_ACTIVE_CLEANUP_INTENTS:]
        meta["cleanup_consumed"] = consumed[-MAX_CONSUMED_CLEANUP_INTENTS:]
        m13["meta_control_intents"] = meta
        m13["m15_cleanup"] = {
            "last_run_id": run_id,
            "last_run_at": now,
            "last_source": source,
            "last_ops": dict(ops),
            "cleanup_active_count": len(meta["cleanup_active"]),
            "cleanup_consumed_count": len(meta["cleanup_consumed"]),
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
