"""M13.6 memory-backed expected-free-energy bridge for Path B.

This is a bounded engineering proxy. It reads traceable runtime expectations
and memory evidence, then produces advisory policy guidance. It does not write
initiative, outbox, memory, self-cognition, or visible text.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime
import uuid
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_drive import (
    _bounded_float,
    _mapping,
    _string_list,
    normalize_m13_drive_state,
)
from segmentum.dialogue.runtime.m14_7_recall_scoring import MEMORY_EFE_RECALL_FLOOR, score_recall_candidate
from segmentum.dialogue.runtime.m15_3_cleanup_control import (
    cleanup_ineligibility_reason,
    explicit_scheduled_anchor_refs,
    is_strictly_traceable,
    strict_bound_memory_ids,
    strict_evidence_refs,
)
from segmentum.dialogue.runtime.m15_episode_ledger import EpisodeLedger, outreach_margin_history_adjustment

ENGINEERING_PROXY_LABEL = "mvp_local_memory_efe_bridge"

ACTIVE_GRACE_SECONDS = 15
NEXT_USER_TURN_EXPECTED_WINDOW_SECONDS = 900
DUE_AT_EXPECTED_WINDOW_SECONDS = 3600
SCHEDULED_OUTREACH_DEFAULT_WINDOW_SECONDS = 86400
F_MEMORY_CAP = 1.5
F_MEMORY_TOP_K = 5
M13_6_OUTREACH_RESETTLE_WINDOW_SECONDS = 3600
MINIMUM_OUTREACH_MARGIN = 0.08
MAX_OUTREACH_RESOLUTION_PRIOR = 0.45
MAX_PENDING_MEMORY_EFE_SETTLEMENTS = 8
MEMORY_EFE_SETTLEMENT_TTL_TURNS = 5
MEMORY_EFE_SETTLEMENT_TTL_SECONDS = 86400

_ELIGIBLE_NEXT_USER = {"next_user_turn", "next_turn"}
_SETTLED_STATUSES = {"confirmed", "settled", "closed", "resolved"}
_UNRESOLVED_STATUSES = {"pending", "open", "due", "uncertain", ""}
_ACTIVE_SCHEDULED_STATUSES = {"preparing", "prepared", "awaiting_delivery"}
_VAGUE_NEXT_CHECKS = {
    "",
    "later",
    "next_user_turn",
    "next_turn",
    "someday",
    "eventually",
    "regular",
    "routine",
    "ongoing",
    "tbd",
    "pending",
}
_BOUNDARY_HIGH = {"hard", "blocked", "forbidden"}
_PENDING_SETTLEMENT_SUPPRESSION = "pending_memory_efe_settlement_active"


class M13MemoryEfeWeights:
    """Single place for M13.6 proxy weights; no scattered magic numbers."""

    confidence: float = 0.42
    recall_salience: float = 0.22
    future_prediction_value: float = 0.12
    stale_memory_penalty: float = 0.18
    epistemic: float = 0.35
    repetition: float = 0.20
    recall_failure: float = 0.55


M13_MEMORY_EFE_WEIGHTS = M13MemoryEfeWeights()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float) -> float:
    return round(float(value), 6)


def _looks_like_concrete_due(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.casefold() in _VAGUE_NEXT_CHECKS:
        return False
    if _epoch(stripped) > 0:
        return True
    return any(ch.isdigit() for ch in stripped) and (
        "-" in stripped or "T" in stripped or stripped.isdigit()
    )


def _epoch(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip()
    if not text:
        return 0
    try:
        return max(0, int(float(text)))
    except ValueError:
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        return max(0, int(datetime.fromisoformat(normalized).timestamp()))
    except ValueError:
        return 0


def _created_at(row: Mapping[str, Any]) -> int:
    return _epoch(
        row.get("created_at_epoch")
        or row.get("created_at")
        or row.get("created_turn_at")
        or row.get("at")
    )


def _content_summary(row: Mapping[str, Any]) -> str:
    for key in ("content_summary", "summary", "title", "content", "text", "next_check"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return " ".join(value.split())[:160]
    return ""


def _evidence_refs(row: Mapping[str, Any], *, source_kind: str = "") -> list[str]:
    if source_kind == "scheduled_outreach":
        return explicit_scheduled_anchor_refs(row, limit=8)
    return strict_evidence_refs(row, limit=8)


def _bound_memory_ids(row: Mapping[str, Any]) -> list[str]:
    return strict_bound_memory_ids(row, limit=8)


def _relationship_precision(state: Mapping[str, Any]) -> float:
    m13 = normalize_m13_drive_state(state.get("m13_drive_state"))
    rel = _mapping(m13.get("relation_path_precision"))
    if not rel:
        return 0.0
    return max(_bounded_float(value) for value in rel.values())


def default_memory_efe_state() -> dict[str, Any]:
    return {
        "phase": "",
        "eligible_for_efe": [],
        "diagnostic_only": [],
        "social_prediction_error": 0.0,
        "epistemic_prediction_error": 0.0,
        "repetition_tension": 0.0,
        "f_memory": 0.0,
        "efe_by_policy": {},
        "policy_costs": {},
        "selected_policy": "",
        "reply_angle_bias": "none",
        "should_outreach": False,
        "outreach_margin": 0.0,
        "traceable_expectation_id": "",
        "suppression_reasons": [],
        "evidence_refs": [],
        "reason_codes": [],
        "last_memory_efe_outreach_at": 0,
        "pending_settlements": [],
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }


def normalize_memory_efe_state(raw: Any) -> dict[str, Any]:
    base = default_memory_efe_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    for key in ("eligible_for_efe", "diagnostic_only", "pending_settlements"):
        rows = merged.get(key)
        merged[key] = [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    for key in ("efe_by_policy", "policy_costs"):
        value = merged.get(key)
        merged[key] = dict(value) if isinstance(value, Mapping) else {}
    for key in ("suppression_reasons", "evidence_refs", "reason_codes"):
        merged[key] = _string_list(merged.get(key), limit=12)
    for key in (
        "social_prediction_error",
        "epistemic_prediction_error",
        "repetition_tension",
        "f_memory",
    ):
        merged[key] = _round(max(0.0, float(merged.get(key, 0.0) or 0.0)))
    merged["outreach_margin"] = _round(float(merged.get("outreach_margin", 0.0) or 0.0))
    merged["last_memory_efe_outreach_at"] = max(0, int(merged.get("last_memory_efe_outreach_at", 0) or 0))
    merged["should_outreach"] = bool(merged.get("should_outreach"))
    merged["reply_angle_bias"] = str(merged.get("reply_angle_bias", "none") or "none")
    merged["selected_policy"] = str(merged.get("selected_policy", "") or "")
    merged["traceable_expectation_id"] = str(merged.get("traceable_expectation_id", "") or "")
    merged["phase"] = str(merged.get("phase", "") or "")
    merged["engineering_proxy_label"] = ENGINEERING_PROXY_LABEL
    return merged


@dataclass(frozen=True)
class NormalizedExpectation:
    expectation_id: str
    source_kind: str
    content_summary: str
    status: str
    confidence: float
    precision: float
    due_at: int
    expected_window_seconds: int
    elapsed_since_due_seconds: int
    required_for_action: float
    bound_memory_ids: list[str]
    evidence_refs: list[str]
    relationship_weight: float
    boundary_cost_hint: str
    scheduled_intent_id: str
    eligible: bool
    ineligibility_reason: str = ""
    precision_approx: dict[str, float] = field(default_factory=dict)
    recall_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "source_kind": self.source_kind,
            "content_summary": self.content_summary,
            "status": self.status,
            "confidence": _round(self.confidence),
            "precision": _round(self.precision),
            "due_at": self.due_at,
            "expected_window_seconds": self.expected_window_seconds,
            "elapsed_since_due_seconds": self.elapsed_since_due_seconds,
            "required_for_action": _round(self.required_for_action),
            "bound_memory_ids": list(self.bound_memory_ids[:8]),
            "evidence_refs": list(self.evidence_refs[:8]),
            "relationship_weight": _round(self.relationship_weight),
            "boundary_cost_hint": self.boundary_cost_hint,
            "scheduled_intent_id": self.scheduled_intent_id,
            "eligible": self.eligible,
            "ineligibility_reason": self.ineligibility_reason,
            "precision_approx": dict(self.precision_approx),
            "recall_keys": list(self.recall_keys[:8]),
        }


@dataclass(frozen=True)
class NormalizedExpectationSet:
    eligible_for_efe: list[NormalizedExpectation] = field(default_factory=list)
    diagnostic_only: list[NormalizedExpectation] = field(default_factory=list)
    bound_recall_seed_ids: list[str] = field(default_factory=list)
    bound_recall_floor_bypassed_ids: list[str] = field(default_factory=list)


@dataclass
class M13MemoryEfeEvaluationResult:
    event_id: str
    phase: str
    eligible_for_efe: list[NormalizedExpectation]
    diagnostic_only: list[NormalizedExpectation]
    social_prediction_error: float
    epistemic_prediction_error: float
    repetition_tension: float
    f_memory: float
    efe_by_policy: dict[str, float]
    policy_costs: dict[str, float]
    selected_policy: str
    reply_angle_bias: str
    should_outreach: bool
    outreach_margin: float
    traceable_expectation_id: str
    suppression_reasons: list[str]
    reason_codes: list[str]
    evidence_refs: list[str]
    bound_recall_seed_ids: list[str] = field(default_factory=list)
    bound_recall_floor_bypassed_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def _precision_components(
    row: Mapping[str, Any],
    *,
    now: int,
    due_at: int,
    evidence_refs: list[str],
    bound_memory_ids: list[str],
) -> tuple[float, dict[str, float]]:
    confidence = _bounded_float(row.get("confidence"), default=0.45)
    salience_candidates = [
        _bounded_float(row.get("recall_salience"), default=0.0),
        _bounded_float(row.get("salience"), default=0.0),
        _bounded_float(row.get("memory_salience"), default=0.0),
    ]
    if evidence_refs or bound_memory_ids:
        salience_candidates.append(0.55)
    recall_salience = max(salience_candidates)
    future_value = max(
        _bounded_float(row.get("future_prediction_value"), default=0.0),
        _bounded_float(row.get("prediction_value"), default=0.0),
        0.45 if due_at else 0.0,
    )
    last_recalled = _epoch(row.get("last_recalled_at") or row.get("last_seen_at"))
    stale = 0.0
    if last_recalled > 0:
        stale = _clamp((now - last_recalled) / float(30 * 86400))
    elif due_at > 0 and now - due_at > 7 * 86400:
        stale = 0.35
    precision = (
        M13_MEMORY_EFE_WEIGHTS.confidence * confidence
        + M13_MEMORY_EFE_WEIGHTS.recall_salience * recall_salience
        + M13_MEMORY_EFE_WEIGHTS.future_prediction_value * future_value
        - M13_MEMORY_EFE_WEIGHTS.stale_memory_penalty * stale
    )
    return _clamp(precision), {
        "confidence": _round(confidence),
        "recall_salience_approx": _round(recall_salience),
        "future_prediction_value_approx": _round(future_value),
        "stale_memory_penalty_approx": _round(stale),
    }


def _make_expectation(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    now: int,
    phase: str,
    source_kind: str,
    due_at: int,
    expected_window_seconds: int,
    eligible: bool,
    ineligibility_reason: str = "",
) -> NormalizedExpectation:
    expectation_id = str(
        row.get("expectation_id")
        or row.get("id")
        or row.get("open_item_id")
        or row.get("intent_id")
        or ""
    ).strip()
    evidence_refs = _evidence_refs(row, source_kind=source_kind)
    bound_ids = _bound_memory_ids(row)
    status = str(row.get("status", "pending") or "pending").strip().lower()
    if source_kind == "open_item" and status in {"", "active"}:
        status = "open"
    cleanup_reason = cleanup_ineligibility_reason(row, now=now, phase=phase, expectation=True)
    if cleanup_reason:
        eligible = False
        ineligibility_reason = cleanup_reason
    elif source_kind in {"pending_expectation", "memory_dynamics_expectation", "open_item"} and eligible and not is_strictly_traceable(row):
        eligible = False
        ineligibility_reason = "not_traceable_or_testable"
    relationship_weight = _clamp(0.75 + 0.25 * _relationship_precision(state), 0.75, 1.0)
    recall_keys = _string_list(row.get("recall_keys"), limit=8)
    precision, precision_approx = _precision_components(
        row,
        now=now,
        due_at=due_at,
        evidence_refs=evidence_refs,
        bound_memory_ids=bound_ids,
    )
    elapsed = max(0, now - due_at) if due_at > 0 else 0
    required = 1.0 if (evidence_refs or bound_ids) and status in _UNRESOLVED_STATUSES | {"due"} else 0.0
    if not expectation_id or (not evidence_refs and not bound_ids and not due_at and source_kind != "open_item"):
        eligible = False
        ineligibility_reason = ineligibility_reason or "not_traceable_or_testable"
    if phase == "idle" and status in {"confirmed", "settled"}:
        eligible = False
        ineligibility_reason = ineligibility_reason or "already_settled"
    return NormalizedExpectation(
        expectation_id=expectation_id[:120],
        source_kind=source_kind,
        content_summary=_content_summary(row),
        status=status,
        confidence=_bounded_float(row.get("confidence"), default=0.45),
        precision=precision,
        due_at=due_at,
        expected_window_seconds=max(1, int(expected_window_seconds or DUE_AT_EXPECTED_WINDOW_SECONDS)),
        elapsed_since_due_seconds=elapsed,
        required_for_action=required,
        bound_memory_ids=bound_ids,
        evidence_refs=evidence_refs,
        relationship_weight=relationship_weight,
        boundary_cost_hint=str(row.get("boundary_strength", row.get("boundary_cost_hint", "")) or "").lower(),
        scheduled_intent_id=str(row.get("scheduled_intent_id", row.get("intent_id", "")) or "")[:120],
        eligible=eligible,
        ineligibility_reason=ineligibility_reason,
        precision_approx=precision_approx,
        recall_keys=recall_keys,
    )


def _normalize_pending_expectation(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    now: int,
    phase: str,
) -> NormalizedExpectation:
    source_kind = (
        "memory_dynamics_expectation"
        if str(row.get("source", "") or "").strip() == "memory_dynamics_adapter"
        else "pending_expectation"
    )
    verify_on = str(row.get("verify_on", row.get("verify", "")) or "").strip().lower()
    due_at = _epoch(row.get("due_at_epoch") or row.get("due_at"))
    scheduled_id = str(row.get("scheduled_intent_id", "") or "").strip()
    window = int(row.get("expected_window_seconds", 0) or 0)
    eligible = False
    reason = ""
    if source_kind == "memory_dynamics_expectation" and (
        verify_on in _VAGUE_NEXT_CHECKS or verify_on in _ELIGIBLE_NEXT_USER or verify_on == "memory_dynamics_idle"
    ):
        created = _created_at(row)
        due_at = due_at or (created + ACTIVE_GRACE_SECONDS if created else 0)
        window = window or ACTIVE_GRACE_SECONDS
        if phase == "in_turn":
            eligible = bool(due_at)
        elif phase == "idle" and due_at:
            eligible = now >= due_at + window
            reason = "" if eligible else "memory_dynamics_tension_not_idle_long_enough"
        else:
            reason = "missing_memory_dynamics_anchor"
    elif verify_on in _ELIGIBLE_NEXT_USER:
        if phase == "in_turn":
            due_at = due_at or _created_at(row)
            window = window or NEXT_USER_TURN_EXPECTED_WINDOW_SECONDS
            eligible = True
        elif phase == "idle":
            created = _created_at(row)
            due_at = due_at or (created + ACTIVE_GRACE_SECONDS if created else 0)
            window = window or NEXT_USER_TURN_EXPECTED_WINDOW_SECONDS
            last_user = _epoch(_mapping(state.get("temporal_state")).get("last_user_turn_at"))
            newer_user_turn_arrived = bool(created and last_user and last_user > created)
            eligible = bool(due_at and now >= due_at + window and not newer_user_turn_arrived)
            reason = "" if eligible else "next_user_turn_not_overdue_or_newer_user_turn_seen"
    elif due_at or scheduled_id:
        window = window or DUE_AT_EXPECTED_WINDOW_SECONDS
        eligible = True
    else:
        reason = "verify_later_without_concrete_due" if verify_on == "later" else "missing_concrete_due"
    return _make_expectation(
        row,
        state=state,
        now=now,
        phase=phase,
        source_kind=source_kind,
        due_at=due_at,
        expected_window_seconds=window or DUE_AT_EXPECTED_WINDOW_SECONDS,
        eligible=eligible,
        ineligibility_reason=reason,
    )


def _open_item_next_check_kind(next_check: str) -> str:
    lowered = str(next_check or "").strip().casefold()
    if lowered in _ELIGIBLE_NEXT_USER:
        return "next_user_turn"
    if lowered in _VAGUE_NEXT_CHECKS:
        return "vague"
    if _looks_like_concrete_due(next_check):
        return "explicit_due_text"
    if lowered:
        return "other"
    return "empty"


def _normalize_open_item(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    now: int,
    phase: str,
) -> NormalizedExpectation:
    next_check = str(row.get("next_check", row.get("next_step", "")) or "").strip()
    next_kind = _open_item_next_check_kind(next_check)
    scheduled_id = str(row.get("scheduled_intent_id", row.get("intent_id", "")) or "").strip()
    due_at = _epoch(row.get("due_at_epoch") or row.get("due_at"))
    explicit_window = int(row.get("expected_window_seconds", row.get("due_window_seconds", 0)) or 0)
    window = explicit_window or DUE_AT_EXPECTED_WINDOW_SECONDS
    status = str(row.get("status", "open") or "open").strip().lower()
    temporal = _mapping(state.get("temporal_state"))
    last_user = _epoch(temporal.get("last_user_turn_at"))
    eligible = False
    reason = ""

    if status != "open":
        reason = "vague_or_not_open"
    elif scheduled_id and due_at > 0:
        # M14.2-linked wall-clock anchor only; not a generic open-item alarm.
        eligible = True
    elif next_kind == "next_user_turn":
        created = _created_at(row) or last_user
        window = explicit_window or NEXT_USER_TURN_EXPECTED_WINDOW_SECONDS
        due_at = due_at or (created + ACTIVE_GRACE_SECONDS if created else 0)
        if phase == "in_turn":
            eligible = bool(due_at)
        elif phase == "idle" and due_at:
            newer_user_turn = bool(created and last_user and last_user > created)
            if not newer_user_turn:
                eligible = now >= due_at + window
                reason = "" if eligible else "next_user_turn_not_overdue_or_newer_user_turn_seen"
            else:
                # User spoke since the open loop was raised but did not close it.
                closure_due = last_user + ACTIVE_GRACE_SECONDS
                due_at = closure_due
                eligible = now >= closure_due
                reason = "" if eligible else "open_item_awaiting_closure_after_user_turn"
        else:
            reason = "missing_observation_anchor"
    elif next_kind == "vague" or next_kind == "empty":
        reason = "vague_or_missing_traceable_next_check"
    elif due_at > 0 and not scheduled_id:
        reason = "wall_clock_open_item_requires_scheduled_intent"
    else:
        reason = "open_item_not_traceable"

    return _make_expectation(
        row,
        state=state,
        now=now,
        phase=phase,
        source_kind="open_item",
        due_at=due_at,
        expected_window_seconds=window,
        eligible=eligible,
        ineligibility_reason=reason,
    )


def _normalize_scheduled_intent(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    now: int,
    phase: str,
) -> NormalizedExpectation:
    due_at = _epoch(row.get("due_at_epoch") or row.get("due_at"))
    window = int(row.get("due_window_seconds", 0) or SCHEDULED_OUTREACH_DEFAULT_WINDOW_SECONDS)
    eligible = str(row.get("kind", "")) == "scheduled_outreach" and bool(due_at)
    return _make_expectation(
        row,
        state=state,
        now=now,
        phase=phase,
        source_kind="scheduled_outreach",
        due_at=due_at,
        expected_window_seconds=window,
        eligible=eligible,
        ineligibility_reason="" if eligible else "scheduled_outreach_without_due_at",
    )


def normalize_expectations_for_efe(
    state: Mapping[str, Any],
    *,
    now: int,
    phase: str,
    structural_signals: Mapping[str, Any] | None = None,
) -> NormalizedExpectationSet:
    eligible: list[NormalizedExpectation] = []
    diagnostic: list[NormalizedExpectation] = []
    bound_recall_seed_ids: list[str] = []
    bound_recall_floor_bypassed_ids: list[str] = []
    memory_by_id: dict[str, Mapping[str, Any]] = {}
    for key in ("short_term_memory", "long_term_memory"):
        rows = state.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                row_id = str(row.get("id", "") or "").strip()
                if row_id:
                    memory_by_id[row_id] = row

    def add(expectation: NormalizedExpectation) -> None:
        if expectation.eligible and expectation.bound_memory_ids:
            bound_rows = [memory_by_id[item] for item in expectation.bound_memory_ids if item in memory_by_id]
            if bound_rows and all(str(row.get("status", "") or "") == "archived" for row in bound_rows):
                diagnostic.append(replace(expectation, eligible=False, ineligibility_reason="bound_memories_archived"))
                return
            if bound_rows:
                scores = [
                    score_recall_candidate(
                        row,
                        query=[expectation.content_summary, *expectation.recall_keys, *expectation.evidence_refs],
                        now=now,
                        retrieved_context={"phase": "memory_efe"},
                    )
                    for row in bound_rows
                ]
                if scores and max(scores) < MEMORY_EFE_RECALL_FLOOR:
                    if phase == "idle" and (expectation.evidence_refs or expectation.bound_memory_ids):
                        for bound_id in expectation.bound_memory_ids:
                            if bound_id in memory_by_id and bound_id not in bound_recall_seed_ids:
                                bound_recall_seed_ids.append(bound_id)
                            if bound_id in memory_by_id and bound_id not in bound_recall_floor_bypassed_ids:
                                bound_recall_floor_bypassed_ids.append(bound_id)
                        eligible.append(expectation)
                        return
                    diagnostic.append(replace(expectation, eligible=False, ineligibility_reason="bound_memory_recall_score_below_floor"))
                    return
        if expectation.eligible:
            eligible.append(expectation)
        else:
            diagnostic.append(expectation)

    for row in state.get("pending_expectations", []) or []:
        if isinstance(row, Mapping):
            add(_normalize_pending_expectation(row, state=state, now=now, phase=phase))
    for row in state.get("open_items", []) or []:
        if isinstance(row, Mapping):
            add(_normalize_open_item(row, state=state, now=now, phase=phase))
    sig = _mapping(structural_signals)
    for row in sig.get("scheduled_intents", []) or []:
        if isinstance(row, Mapping):
            add(_normalize_scheduled_intent(row, state=state, now=now, phase=phase))

    return NormalizedExpectationSet(
        eligible_for_efe=eligible,
        diagnostic_only=diagnostic,
        bound_recall_seed_ids=bound_recall_seed_ids[:12],
        bound_recall_floor_bypassed_ids=bound_recall_floor_bypassed_ids[:12],
    )


def _due_pressure(e: NormalizedExpectation) -> float:
    if e.due_at <= 0 or e.elapsed_since_due_seconds <= 0:
        return 0.0
    return _clamp(e.elapsed_since_due_seconds / float(max(1, e.expected_window_seconds)))


def _unresolved_pressure(e: NormalizedExpectation, *, phase: str, user_active: bool) -> float:
    if e.status in _SETTLED_STATUSES:
        return 0.0
    if e.status == "violated" and phase == "in_turn":
        return 0.0
    if e.status == "violated" and phase != "in_turn" and not user_active:
        return 1.0
    if e.status in _UNRESOLVED_STATUSES or e.status == "due":
        return 1.0
    return 0.0


def _observation_gap(e: NormalizedExpectation, *, phase: str, user_active: bool) -> float:
    if e.status in _SETTLED_STATUSES:
        return 0.0
    if e.status == "violated" and phase == "in_turn":
        return 0.0
    if e.status == "violated" and not user_active:
        return 0.90
    due = _due_pressure(e)
    if due >= 1.0:
        return 0.65
    if due > 0.0:
        return 0.35
    return 0.15


def _retrieved_memory_ids(retrieved_memories: list[Mapping[str, Any]] | None) -> set[str]:
    if not retrieved_memories:
        return set()
    return {str(item.get("id", "")).strip() for item in retrieved_memories if str(item.get("id", "")).strip()}


def _recall_failure_active(
    expectation: NormalizedExpectation,
    retrieved_memories: list[Mapping[str, Any]] | None,
) -> float:
    """1.0 only when linked memories were expected but not present in retrieved set."""
    if expectation.required_for_action <= 0.0:
        return 0.0
    targets = list(
        dict.fromkeys(
            [
                *expectation.bound_memory_ids,
                *expectation.evidence_refs,
                *expectation.recall_keys,
            ]
        )
    )
    if not targets:
        return 0.0
    if retrieved_memories is None:
        return 0.0
    retrieved_ids = _retrieved_memory_ids(retrieved_memories)
    return 1.0 if not any(target in retrieved_ids for target in targets) else 0.0


def _top_k_error_sum(
    expectations: list[NormalizedExpectation],
    errors_by_id: Mapping[str, float],
    *,
    k: int,
) -> float:
    ranked = sorted(
        expectations,
        key=lambda e: (-float(errors_by_id.get(e.expectation_id, 0.0) or 0.0), e.expectation_id),
    )
    return _round(sum(float(errors_by_id.get(e.expectation_id, 0.0) or 0.0) for e in ranked[:k]))


def _initiative_suppression_reasons(
    state: Mapping[str, Any],
    *,
    now: int,
    turn_index: int,
) -> list[str]:
    from segmentum.dialogue.runtime.m13_initiative import (
        DEFAULT_COOLDOWN_TURNS,
        DEFAULT_MAX_PROACTIVE_PER_SESSION,
        normalize_initiative_state,
    )

    initiative = normalize_initiative_state(
        normalize_m13_drive_state(state.get("m13_drive_state")).get("initiative")
    )
    reasons: list[str] = []
    if not initiative.get("user_opt_in"):
        reasons.append("not_opted_in")
    if not initiative.get("enabled"):
        reasons.append("initiative_disabled")
    if int(initiative.get("proactive_count_this_session", 0) or 0) >= int(
        initiative.get("max_proactive_per_session", DEFAULT_MAX_PROACTIVE_PER_SESSION) or 1
    ):
        reasons.append("session_limit_reached")
    if int(initiative.get("cooldown_until_timestamp", 0) or 0) > now:
        reasons.append("cooldown_active")
    last_turn = int(initiative.get("last_proactive_turn_index", -1) or -1)
    cooldown_turns = int(initiative.get("cooldown_turns", DEFAULT_COOLDOWN_TURNS) or DEFAULT_COOLDOWN_TURNS)
    if last_turn >= 0 and turn_index - last_turn <= cooldown_turns:
        reasons.append("cooldown_active")
    if not initiative.get("implicit_idle_delivery"):
        reasons.append("delivery_channel_unavailable")
    return reasons


def _prediction_components(
    expectations: list[NormalizedExpectation],
    *,
    phase: str,
    user_active: bool,
    retrieved_memories: list[Mapping[str, Any]] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    social: dict[str, float] = {}
    epistemic: dict[str, float] = {}
    for e in expectations:
        pe_social = (
            e.precision
            * _due_pressure(e)
            * _unresolved_pressure(e, phase=phase, user_active=user_active)
            * e.relationship_weight
            * _observation_gap(e, phase=phase, user_active=user_active)
        )
        recall_active = _recall_failure_active(e, retrieved_memories)
        pe_epi = (
            e.precision
            * e.required_for_action
            * recall_active
            * M13_MEMORY_EFE_WEIGHTS.recall_failure
        )
        social[e.expectation_id] = _round(pe_social)
        epistemic[e.expectation_id] = _round(pe_epi)
    return social, epistemic


def _repetition_tension(
    m13_boredom_evaluation: Any | None,
    m13_reward_evaluation: Any | None,
    state: Mapping[str, Any],
) -> float:
    rep = _bounded_float(getattr(m13_boredom_evaluation, "repetition_pressure", 0.0), default=0.0)
    reward = m13_reward_evaluation
    stale = bool(getattr(reward, "path_feels_stale_proxy", False))
    if reward is None:
        reward_state = _mapping(normalize_m13_drive_state(state.get("m13_drive_state")).get("affective_reward_proxy"))
        stale = bool(reward_state.get("path_feels_stale_proxy"))
    return _clamp(rep + (0.08 if stale else 0.0))


def _traceable_expectation_id(
    expectations: list[NormalizedExpectation],
    social: Mapping[str, float],
) -> str:
    if not expectations:
        return ""
    ranked = sorted(
        expectations,
        key=lambda e: (
            -float(social.get(e.expectation_id, 0.0) or 0.0),
            e.due_at if e.due_at > 0 else 2**62,
            e.expectation_id,
        ),
    )
    return ranked[0].expectation_id if ranked else ""


def _boundary_cost(expectation: NormalizedExpectation | None, state: Mapping[str, Any]) -> float:
    hint = str(expectation.boundary_cost_hint if expectation else "").lower()
    if hint in _BOUNDARY_HIGH:
        return 0.45
    if hint == "soft":
        return 0.20
    sharing = _mapping(_mapping(state.get("social_sharing_policy")))
    learned = sharing.get("learned_boundaries", [])
    if isinstance(learned, list) and learned:
        return 0.20
    return 0.05


def _risk_penalty(state: Mapping[str, Any]) -> float:
    m13 = normalize_m13_drive_state(state.get("m13_drive_state"))
    reward = _mapping(m13.get("affective_reward_proxy"))
    if _bounded_float(reward.get("opponent_strength"), default=0.0) >= 0.5:
        return 0.30
    return 0.0


def _duplicate_outreach_reasons(
    state: Mapping[str, Any],
    *,
    traceable_expectation_id: str,
    structural_signals: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if not traceable_expectation_id:
        return reasons
    sig = _mapping(structural_signals)
    for row in sig.get("scheduled_intents", []) or []:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "") or "")
        keys = {
            str(row.get("traceable_expectation_id", "")),
            str(row.get("open_item_id", "")),
            str(row.get("intent_id", "")),
        }
        if status in _ACTIVE_SCHEDULED_STATUSES and traceable_expectation_id in keys:
            reasons.append("scheduled_outreach_already_active")
            break
    for row in sig.get("queued_outreach", []) or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status", "")) != "pending":
            continue
        keys = {
            str(row.get("traceable_expectation_id", "")),
            str(row.get("source_intent_id", "")),
            str(row.get("open_item_id", "")),
        }
        evidence = {str(x) for x in row.get("evidence_refs", []) or []}
        if traceable_expectation_id in keys or traceable_expectation_id in evidence:
            reasons.append("queued_outreach_already_pending")
            break
    memory_efe = normalize_memory_efe_state(
        normalize_m13_drive_state(state.get("m13_drive_state")).get("memory_efe")
    )
    for row in memory_efe.get("pending_settlements", []):
        if (
            isinstance(row, Mapping)
            and str(row.get("traceable_expectation_id", "")) == traceable_expectation_id
            and str(row.get("status", "pending")) == "pending"
        ):
            reasons.append(_PENDING_SETTLEMENT_SUPPRESSION)
            break
    return reasons


def _policy_costs(
    *,
    expectation: NormalizedExpectation | None,
    state: Mapping[str, Any],
) -> dict[str, float]:
    boundary = _boundary_cost(expectation, state)
    regret_bias = _bounded_float(_mapping(state.get("social_sharing_policy")).get("regret_bias"), default=0.0)
    relationship_cost = 0.10 + regret_bias
    return {
        "wait_decay_risk": 0.05,
        "low_control_cost": 0.02,
        "recall_cost": 0.08,
        "control_cost": 0.10,
        "boundary_cost": _round(boundary),
        "relationship_cost": _round(relationship_cost),
        "risk_penalty": _round(_risk_penalty(state)),
        "boundary_cost_high": boundary >= 0.30,
        "relationship_cost_high": relationship_cost >= 0.30,
    }


def _compute_idle_efe(
    *,
    f_memory: float,
    eligible_count: int,
    expectation: NormalizedExpectation | None,
    state: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], float]:
    costs = _policy_costs(expectation=expectation, state=state)
    expected_internal_resolution = min(0.15, 0.04 * eligible_count)
    expected_epistemic_gain = min(0.25, 0.08 * eligible_count)
    traceable = 1.0 if expectation and expectation.expectation_id else 0.0
    evidence_strength = 1.0 if expectation and (expectation.evidence_refs or expectation.bound_memory_ids) else 0.65
    boundary_allowance = 0.0 if costs["boundary_cost_high"] else 1.0
    expected_outreach_resolution = min(
        MAX_OUTREACH_RESOLUTION_PRIOR,
        0.38 * traceable * evidence_strength * boundary_allowance,
    )
    expected_information_gain = min(0.25, f_memory * 0.16)
    if expectation and expectation.source_kind in {"open_item", "scheduled_outreach", "memory_dynamics_expectation"}:
        expected_information_gain = max(expected_information_gain, 0.25)
    efe = {
        "wait": f_memory + float(costs["wait_decay_risk"]) + float(costs["low_control_cost"]),
        "reflect": f_memory * (1.0 - expected_internal_resolution)
        + float(costs["recall_cost"])
        - expected_epistemic_gain,
        "outreach": f_memory * (1.0 - expected_outreach_resolution)
        + float(costs["boundary_cost"])
        + float(costs["relationship_cost"])
        + float(costs["control_cost"])
        - expected_information_gain
        + float(costs["risk_penalty"]),
    }
    efe = {key: _round(value) for key, value in efe.items()}
    costs["expected_internal_resolution"] = _round(expected_internal_resolution)
    costs["expected_epistemic_gain"] = _round(expected_epistemic_gain)
    costs["expected_information_gain"] = _round(expected_information_gain)
    costs["expected_outreach_resolution"] = _round(expected_outreach_resolution)
    return efe, costs, expected_outreach_resolution


def _continue_reply_efe(
    *,
    repetition_tension: float,
    information_gain_proxy: float,
    repair_pressure: float,
    reply_angle_bias: str,
) -> tuple[dict[str, float], dict[str, float]]:
    gain_by_bias = {
        "new_angle": 0.18,
        "clarify_open_loop": 0.14,
        "repair_expectation": 0.12,
        "summarize_then_advance": 0.10,
        "none": 0.0,
    }
    expected_reply_gain = gain_by_bias.get(reply_angle_bias, 0.0)
    low_info = 1.0 - _bounded_float(information_gain_proxy, default=0.0)
    efe = repetition_tension + low_info + repair_pressure - expected_reply_gain
    return (
        {"continue_reply": _round(efe)},
        {
            "low_expected_information_gain": _round(low_info),
            "repair_pressure_if_violated": _round(repair_pressure),
            "expected_reply_information_gain": _round(expected_reply_gain),
        },
    )


def _choose_reply_angle_bias(
    *,
    conscious_plan: Mapping[str, Any],
    epistemic_prediction_error: float,
    eligible: list[NormalizedExpectation],
    repetition_tension: float,
    progress_signal: float,
) -> str:
    for item in conscious_plan.get("expectation_results", []) or []:
        if isinstance(item, Mapping) and str(item.get("status", "")).strip().lower() == "violated":
            return "repair_expectation"
    if epistemic_prediction_error >= 0.20 and any(e.required_for_action > 0 for e in eligible):
        return "clarify_open_loop"
    if repetition_tension >= 0.35:
        return "new_angle"
    if progress_signal >= 0.35:
        return "summarize_then_advance"
    return "none"


def evaluate_memory_efe(
    state: Mapping[str, Any],
    *,
    phase: str,
    now: int,
    turn_index: int,
    user_active: bool,
    memory_dynamics: Mapping[str, Any] | None = None,
    retrieved_memories: list[Mapping[str, Any]] | None = None,
    m13_boredom_evaluation: Any | None = None,
    m13_reward_evaluation: Any | None = None,
    structural_signals: Mapping[str, Any] | None = None,
    conscious_plan: Mapping[str, Any] | None = None,
    episode_ledger: EpisodeLedger | None = None,
) -> M13MemoryEfeEvaluationResult:
    phase = "idle" if phase == "idle" else "in_turn"
    temporal = _mapping(state.get("temporal_state"))
    idle_seconds = max(0, now - _epoch(temporal.get("last_user_turn_at"))) if _epoch(temporal.get("last_user_turn_at")) else 0
    active = bool(user_active or phase == "in_turn" or idle_seconds < ACTIVE_GRACE_SECONDS)
    expectations = normalize_expectations_for_efe(
        state,
        now=now,
        phase=phase,
        structural_signals=structural_signals,
    )
    social_by_id, epistemic_by_id = _prediction_components(
        expectations.eligible_for_efe,
        phase=phase,
        user_active=active,
        retrieved_memories=retrieved_memories,
    )
    social_error = _top_k_error_sum(expectations.eligible_for_efe, social_by_id, k=F_MEMORY_TOP_K)
    epistemic_error = _top_k_error_sum(expectations.eligible_for_efe, epistemic_by_id, k=F_MEMORY_TOP_K)
    f_memory = _round(min(F_MEMORY_CAP, social_error + M13_MEMORY_EFE_WEIGHTS.epistemic * epistemic_error))
    repetition = _round(_repetition_tension(m13_boredom_evaluation, m13_reward_evaluation, state))
    traceable_id = _traceable_expectation_id(expectations.eligible_for_efe, social_by_id)
    traceable = next((e for e in expectations.eligible_for_efe if e.expectation_id == traceable_id), None)
    evidence_refs = list(dict.fromkeys((traceable.evidence_refs if traceable else [])[:8]))
    conscious = _mapping(conscious_plan)
    control = _mapping(_mapping(memory_dynamics).get("control_guidance"))
    repair_pressure = _bounded_float(control.get("repair_bias"), default=0.0)
    progress_signal = _bounded_float(getattr(m13_boredom_evaluation, "progress_signal", 0.0), default=0.0)
    info_gain = _bounded_float(getattr(m13_boredom_evaluation, "information_gain_proxy", 0.0), default=0.0)

    suppression: list[str] = []
    should_outreach = False
    outreach_margin = 0.0
    ledger_margin_requirement_delta = 0.0
    reply_angle_bias = "none"
    selected_policy = ""
    expected_resolution = 0.0

    if phase == "in_turn":
        reply_angle_bias = _choose_reply_angle_bias(
            conscious_plan=conscious,
            epistemic_prediction_error=epistemic_error,
            eligible=expectations.eligible_for_efe,
            repetition_tension=repetition,
            progress_signal=progress_signal,
        )
        efe_by_policy, policy_costs = _continue_reply_efe(
            repetition_tension=repetition,
            information_gain_proxy=info_gain,
            repair_pressure=repair_pressure,
            reply_angle_bias=reply_angle_bias,
        )
        selected_policy = "continue_reply"
        suppression.append("phase_not_idle")
        if active:
            suppression.append("user_active")
    else:
        efe_by_policy, policy_costs, expected_resolution = _compute_idle_efe(
            f_memory=f_memory,
            eligible_count=len(expectations.eligible_for_efe),
            expectation=traceable,
            state=state,
        )
        selected_policy = min(efe_by_policy, key=lambda key: (efe_by_policy[key], key)) if efe_by_policy else "wait"
        outreach_margin = _round(efe_by_policy.get("wait", 0.0) - efe_by_policy.get("outreach", 0.0))
        if episode_ledger is not None:
            ledger_margin_requirement_delta = outreach_margin_history_adjustment(
                episode_ledger.search("memory_efe_outreach", limit=8)
            )
            if ledger_margin_requirement_delta > 0:
                policy_costs["ledger_outreach_margin_requirement_delta"] = ledger_margin_requirement_delta
        if active:
            suppression.append("user_active")
        if not expectations.eligible_for_efe:
            suppression.append("no_efe_eligible_expectation")
        if not traceable_id:
            suppression.append("no_traceable_expectation")
        if bool(policy_costs.get("boundary_cost_high")):
            suppression.append("boundary_cost_high")
        if bool(policy_costs.get("relationship_cost_high")):
            suppression.append("relationship_cost_high")
        if float(policy_costs.get("risk_penalty", 0.0) or 0.0) >= 0.30:
            suppression.append("memory_efe_opponent_risk")
        if expected_resolution <= 0.0:
            suppression.append("insufficient_expected_resolution")
        required_margin = MINIMUM_OUTREACH_MARGIN + ledger_margin_requirement_delta
        if outreach_margin < required_margin:
            suppression.append("outreach_margin_too_small")
        memory_efe_state = normalize_memory_efe_state(
            normalize_m13_drive_state(state.get("m13_drive_state")).get("memory_efe")
        )
        last_at = int(memory_efe_state.get("last_memory_efe_outreach_at", 0) or 0)
        if last_at > 0 and now - last_at < M13_6_OUTREACH_RESETTLE_WINDOW_SECONDS:
            suppression.append("recently_outreached")
        suppression.extend(_duplicate_outreach_reasons(state, traceable_expectation_id=traceable_id, structural_signals=structural_signals))
        suppression.extend(_initiative_suppression_reasons(state, now=now, turn_index=turn_index))
        should_outreach = selected_policy == "outreach" and not suppression

    reason_codes = []
    if social_error > 0:
        reason_codes.append("memory_backed_social_prediction_error")
    if epistemic_error > 0:
        reason_codes.append("memory_backed_epistemic_prediction_error")
    if repetition > 0:
        reason_codes.append("repetition_tension_read_only")
    if reply_angle_bias != "none":
        reason_codes.append(f"reply_angle_bias:{reply_angle_bias}")
    reason_codes.extend(suppression)
    diagnostic_reasons = list(
        dict.fromkeys(
            row.ineligibility_reason
            for row in expectations.diagnostic_only
            if row.ineligibility_reason
        )
    )
    cleanup_diagnostic_reasons = {
        "expectation_expired",
        "expectation_merged",
        "low_traceability_cleanup_deprioritized",
        "self_referential_evidence_only",
    }
    if any(reason in cleanup_diagnostic_reasons for reason in diagnostic_reasons):
        reason_codes.append("cleanup_filtered_low_traceability_candidates")
    reason_codes.extend(diagnostic_reasons[:4])
    reason_codes = list(dict.fromkeys(reason_codes))[:12]

    event_id = _new_id("m13_memory_efe")
    event = {
        "type": "MemoryEfeEvaluationEvent",
        "event_id": event_id,
        "phase": phase,
        "turn_index": turn_index,
        "at": now,
        "eligible_count": len(expectations.eligible_for_efe),
        "diagnostic_only_count": len(expectations.diagnostic_only),
        "social_prediction_error": social_error,
        "epistemic_prediction_error": epistemic_error,
        "repetition_tension": repetition,
        "f_memory": f_memory,
        "efe_by_policy": dict(efe_by_policy),
        "policy_costs": dict(policy_costs),
        "selected_policy": selected_policy,
        "should_outreach": should_outreach,
        "traceable_expectation_id": traceable_id,
        "bound_recall_seed_ids": list(expectations.bound_recall_seed_ids[:12]),
        "bound_recall_floor_bypassed_ids": list(expectations.bound_recall_floor_bypassed_ids[:12]),
        "suppression_reasons": list(dict.fromkeys(suppression))[:12],
        "ledger_outreach_margin_requirement_delta": ledger_margin_requirement_delta,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
    if traceable and traceable.precision_approx:
        event["precision_approx"] = dict(traceable.precision_approx)
    events = [event]
    if suppression:
        events.append(
            {
                "type": "MemoryEfeSuppressionEvent",
                "event_id": _new_id("m13_memory_efe_supp"),
                "source_event_id": event_id,
                "phase": phase,
                "turn_index": turn_index,
                "at": now,
                "traceable_expectation_id": traceable_id,
                "suppression_reasons": list(dict.fromkeys(suppression))[:12],
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
    if should_outreach:
        events.append(
            {
                "type": "MemoryEfeOutreachRecommendationEvent",
                "event_id": _new_id("m13_memory_efe_outreach"),
                "source_event_id": event_id,
                "turn_index": turn_index,
                "at": now,
                "traceable_expectation_id": traceable_id,
                "outreach_margin": outreach_margin,
                "expected_resolution_prior": _round(expected_resolution),
                "evidence_refs": evidence_refs,
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )

    return M13MemoryEfeEvaluationResult(
        event_id=event_id,
        phase=phase,
        eligible_for_efe=expectations.eligible_for_efe,
        diagnostic_only=expectations.diagnostic_only,
        social_prediction_error=social_error,
        epistemic_prediction_error=epistemic_error,
        repetition_tension=repetition,
        f_memory=f_memory,
        efe_by_policy=dict(efe_by_policy),
        policy_costs=dict(policy_costs),
        selected_policy=selected_policy,
        reply_angle_bias=reply_angle_bias,
        should_outreach=should_outreach,
        outreach_margin=outreach_margin,
        traceable_expectation_id=traceable_id,
        suppression_reasons=list(dict.fromkeys(suppression))[:12],
        reason_codes=reason_codes,
        evidence_refs=evidence_refs,
        bound_recall_seed_ids=list(expectations.bound_recall_seed_ids[:12]),
        bound_recall_floor_bypassed_ids=list(expectations.bound_recall_floor_bypassed_ids[:12]),
        events=events,
    )


def apply_memory_efe_state(
    m13_state: Mapping[str, Any],
    evaluation: M13MemoryEfeEvaluationResult,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_m13_drive_state(m13_state)
    prior = normalize_memory_efe_state(state.get("memory_efe"))
    durable = {
        "last_memory_efe_outreach_at": prior.get("last_memory_efe_outreach_at", 0),
        "pending_settlements": list(prior.get("pending_settlements", []))[-MAX_PENDING_MEMORY_EFE_SETTLEMENTS:],
    }
    state["memory_efe"] = {
        "phase": evaluation.phase,
        "eligible_for_efe": [e.to_dict() for e in evaluation.eligible_for_efe[:F_MEMORY_TOP_K]],
        "diagnostic_only": [e.to_dict() for e in evaluation.diagnostic_only[:F_MEMORY_TOP_K]],
        "social_prediction_error": evaluation.social_prediction_error,
        "epistemic_prediction_error": evaluation.epistemic_prediction_error,
        "repetition_tension": evaluation.repetition_tension,
        "f_memory": evaluation.f_memory,
        "efe_by_policy": dict(evaluation.efe_by_policy),
        "policy_costs": dict(evaluation.policy_costs),
        "selected_policy": evaluation.selected_policy,
        "reply_angle_bias": evaluation.reply_angle_bias,
        "should_outreach": evaluation.should_outreach,
        "outreach_margin": evaluation.outreach_margin,
        "traceable_expectation_id": evaluation.traceable_expectation_id,
        "suppression_reasons": list(evaluation.suppression_reasons[:12]),
        "evidence_refs": list(evaluation.evidence_refs[:8]),
        "reason_codes": list(evaluation.reason_codes[:12]),
        **durable,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
    return state, []


def apply_memory_efe_state_with_store_lock(
    store: Any,
    evaluation: M13MemoryEfeEvaluationResult,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Persist M13.6 snapshot through the existing session file lock."""
    from segmentum.dialogue.runtime.m14_1_background_continuity import session_file_lock

    with session_file_lock(store.root):
        state = store.load()
        m13_state, events = apply_memory_efe_state(state.get("m13_drive_state", {}), evaluation)
        state["m13_drive_state"] = m13_state
        store.save(state)
        return state, events


def prompt_safe_memory_efe_guidance(evaluation: M13MemoryEfeEvaluationResult) -> dict[str, Any]:
    return {
        "reply_angle_bias": evaluation.reply_angle_bias,
        "reason_codes": list(evaluation.reason_codes[:6]),
        "evidence_refs": list(evaluation.evidence_refs[:6]),
        "advisory_only": True,
    }


def merge_memory_efe_guidance_into_control(
    memory_dynamics: dict[str, Any],
    evaluation: M13MemoryEfeEvaluationResult,
) -> None:
    control = _mapping(memory_dynamics.get("control_guidance"))
    control["memory_efe_guidance"] = prompt_safe_memory_efe_guidance(evaluation)
    memory_dynamics["control_guidance"] = control


def build_memory_efe_outreach_proposal_input(
    evaluation: M13MemoryEfeEvaluationResult,
) -> dict[str, Any] | None:
    if not evaluation.should_outreach or not evaluation.traceable_expectation_id:
        return None
    expectation = next(
        (
            row
            for row in evaluation.eligible_for_efe
            if row.expectation_id == evaluation.traceable_expectation_id
        ),
        None,
    )
    summary = str(getattr(expectation, "content_summary", "") or "").strip()
    intent = (
        f"Follow up on the unresolved expectation: {summary[:180]}"
        if summary
        else "Offer one short follow-up tied to the unresolved traceable expectation."
    )
    return {
        "trigger": "memory_efe_outreach",
        "source": "memory_efe",
        "trigger_evidence_refs": list(evaluation.evidence_refs[:8]),
        "traceable_expectation_id": evaluation.traceable_expectation_id,
        "ordinary_language_intent": intent,
        "expected_resolution_prior": _round(evaluation.policy_costs.get("expected_outreach_resolution", 0.0)),
        "efe_by_policy": dict(evaluation.efe_by_policy),
        "suppression_reasons": list(evaluation.suppression_reasons[:12]),
        "source_kind": str(getattr(expectation, "source_kind", "") or ""),
    }


def build_memory_efe_outreach_proposal(
    evaluation: M13MemoryEfeEvaluationResult,
    *,
    now: int,
    initiative: Mapping[str, Any],
):
    from segmentum.dialogue.runtime.m13_initiative import ProactiveTurnProposal

    proposal_input = build_memory_efe_outreach_proposal_input(evaluation)
    if proposal_input is None:
        return None
    return ProactiveTurnProposal(
        proposal_id=_new_id("m13_memefe_prop"),
        created_at=now,
        source="memory_efe",
        trigger="memory_efe_outreach",
        trigger_evidence_refs=list(proposal_input["trigger_evidence_refs"]),
        urgency_band="medium",
        expected_user_value_band="medium",
        risk_band="low",
        proposed_action="answer",
        proposed_topic=str(evaluation.traceable_expectation_id)[:120],
        ordinary_language_intent=str(proposal_input["ordinary_language_intent"])[:240],
        expires_at=now + 300,
        cooldown_cost=int(initiative.get("cooldown_turns", 2) or 2),
        traceable_expectation_id=evaluation.traceable_expectation_id,
        expected_resolution_prior=float(proposal_input["expected_resolution_prior"]),
        efe_by_policy=dict(evaluation.efe_by_policy),
        suppression_reasons=list(evaluation.suppression_reasons[:12]),
        source_kind=str(proposal_input.get("source_kind", "")),
        selection_reason_codes=["memory_efe_should_outreach"],
    )


def register_memory_efe_outreach_settlement(
    m13_state: Mapping[str, Any],
    *,
    evaluation: M13MemoryEfeEvaluationResult,
    proposal_id: str,
    delivery_status: str,
    now: int,
    turn_index: int | None = None,
    m15_episode_id: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_m13_drive_state(m13_state)
    memory_efe = normalize_memory_efe_state(state.get("memory_efe"))
    if not evaluation.traceable_expectation_id:
        return state, []
    pending = list(memory_efe.get("pending_settlements", []))
    row = {
        "pending_id": _new_id("m13_memory_efe_pending"),
        "source": "memory_efe_outreach",
        "traceable_expectation_id": evaluation.traceable_expectation_id,
        "prior_f_memory": evaluation.f_memory,
        "expected_resolution": _round(evaluation.policy_costs.get("expected_outreach_resolution", 0.0)),
        "proposal_id": str(proposal_id),
        "delivery_status": str(delivery_status or "unknown"),
        "created_turn_index": int(turn_index) if turn_index is not None else 0,
        "m15_episode_id": str(m15_episode_id or ""),
        "created_at": now,
        "expires_at": now + MEMORY_EFE_SETTLEMENT_TTL_SECONDS,
        "expires_after_turns": MEMORY_EFE_SETTLEMENT_TTL_TURNS,
        "expires_after_seconds": MEMORY_EFE_SETTLEMENT_TTL_SECONDS,
        "evidence_refs": list(evaluation.evidence_refs[:8]),
        "status": "pending",
    }
    pending.append(row)
    memory_efe["pending_settlements"] = pending[-MAX_PENDING_MEMORY_EFE_SETTLEMENTS:]
    if delivery_status == "delivered":
        memory_efe["last_memory_efe_outreach_at"] = now
    state["memory_efe"] = memory_efe
    return state, [
        {
            "type": "MemoryEfeSettlementEvent",
            "pending_id": row["pending_id"],
            "traceable_expectation_id": row["traceable_expectation_id"],
            "delivery_status": row["delivery_status"],
            "created_turn_index": row["created_turn_index"],
            "m15_episode_id": row["m15_episode_id"],
            "created_at": now,
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
        }
    ]


def settle_memory_efe_outreach(
    m13_state: Mapping[str, Any],
    *,
    conscious_plan: Mapping[str, Any] | None = None,
    turn_index: int,
    now: int,
    delivery_failures: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_m13_drive_state(m13_state)
    memory_efe = normalize_memory_efe_state(state.get("memory_efe"))
    pending = [dict(row) for row in memory_efe.get("pending_settlements", []) if isinstance(row, Mapping)]
    if not pending:
        return state, []
    results_by_id: dict[str, str] = {}
    for row in _mapping(conscious_plan).get("expectation_results", []) or []:
        if not isinstance(row, Mapping):
            continue
        eid = str(row.get("expectation_id", row.get("id", "")) or "")
        if eid:
            results_by_id[eid] = str(row.get("status", "") or "").lower()
    failures = {str(k): str(v) for k, v in (delivery_failures or {}).items()}
    remaining: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row in pending:
        pending_id = str(row.get("pending_id", ""))
        traceable_id = str(row.get("traceable_expectation_id", ""))
        created_turn = int(row.get("created_turn_index", turn_index) or turn_index)
        expires_turns = int(row.get("expires_after_turns", MEMORY_EFE_SETTLEMENT_TTL_TURNS) or MEMORY_EFE_SETTLEMENT_TTL_TURNS)
        expired = now >= int(row.get("expires_at", 0) or 0) or turn_index > created_turn + expires_turns
        delivery_status = str(row.get("delivery_status", "unknown"))
        observed = 0.0
        outcome = ""
        reasons: list[str] = []
        if pending_id in failures or delivery_status in {"failed", "suppressed"}:
            outcome = "unresolved"
            reasons.append("delivery_not_completed")
        elif traceable_id in results_by_id:
            status = results_by_id[traceable_id]
            if status in _SETTLED_STATUSES:
                observed = min(1.0, float(row.get("expected_resolution", 0.0) or 0.0))
                outcome = "resolved"
                reasons.append("later_user_observation_confirmed")
            elif status == "violated":
                outcome = "unresolved"
                reasons.append("later_user_observation_violated")
            else:
                outcome = "uncertain"
                reasons.append("later_user_observation_uncertain")
        elif expired:
            outcome = "uncertain"
            reasons.append("ttl_expired_without_observation")
        else:
            remaining.append(row)
            continue
        remaining_error = _round(max(0.0, float(row.get("prior_f_memory", 0.0) or 0.0) - observed))
        settlement_id = _new_id("m13_memory_efe_settle")
        events.append(
            {
                "type": "MemoryEfeSettlementEvent",
                "settlement_id": settlement_id,
                "pending_id": pending_id,
                "m15_episode_id": str(row.get("m15_episode_id", "")),
                "prior_turn_index": int(row.get("created_turn_index", 0) or 0),
                "observed_resolution": _round(observed),
                "remaining_prediction_error": remaining_error,
                "outcome_band": outcome,
                "reason_codes": reasons[:6],
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            }
        )
    memory_efe["pending_settlements"] = remaining[-MAX_PENDING_MEMORY_EFE_SETTLEMENTS:]
    state["memory_efe"] = memory_efe
    return state, events


def prompt_safe_m13_memory_efe_diagnostics(evaluation: M13MemoryEfeEvaluationResult) -> dict[str, Any]:
    return {
        "phase": evaluation.phase,
        "eligible_count": len(evaluation.eligible_for_efe),
        "diagnostic_only_count": len(evaluation.diagnostic_only),
        "selected_policy": evaluation.selected_policy,
        "reply_angle_bias": evaluation.reply_angle_bias,
        "should_outreach": evaluation.should_outreach,
        "suppression_reasons": list(evaluation.suppression_reasons[:8]),
        "traceable_expectation_id": evaluation.traceable_expectation_id,
        "social_prediction_error": evaluation.social_prediction_error,
        "epistemic_prediction_error": evaluation.epistemic_prediction_error,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
