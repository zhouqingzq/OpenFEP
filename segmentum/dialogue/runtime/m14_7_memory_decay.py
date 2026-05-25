"""M14.7 bounded LTM decay for Path B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.memory_dynamics import compute_retention_pressure


ENGINEERING_PROXY_LABEL = "mvp_local_memory_decay"
MAX_DECAY_DELTA_PER_TICK = 0.05
MAX_ARCHIVE_FLIPS_PER_TICK = 3
LTM_ARCHIVE_THRESHOLD = 0.05
IDENTITY_RELEVANCE_FLOOR = 0.65
CONSOLIDATION_DECAY_DELTA_PER_RUN = 0.02
LAST_RECALL_GRACE_SECONDS = 86400


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


@dataclass(frozen=True)
class DecayResult:
    rows_touched: int = 0
    rows_archived: int = 0
    rows_exempted: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


def _decay_rate(row: Mapping[str, Any]) -> float:
    try:
        pressure = compute_retention_pressure(
            identity_continuity_value=_bounded_float(row.get("identity_relevance")),
            relationship_continuity_value=_bounded_float(row.get("relationship_relevance")),
            future_prediction_value=_bounded_float(row.get("future_prediction_value")),
            affective_salience=_bounded_float(row.get("salience")),
            confidence=_bounded_float(row.get("confidence"), default=0.5),
            memory_type=str(row.get("kind", "") or ""),
        )
        return max(0.0, min(MAX_DECAY_DELTA_PER_TICK, (1.0 - float(pressure.total_pressure)) * 0.02))
    except Exception:
        salience = _bounded_float(row.get("salience"), default=0.2)
        return max(0.0, min(MAX_DECAY_DELTA_PER_TICK, (1.0 - salience) * 0.01))


def apply_memory_decay_tick(state: dict[str, Any], *, now: int, turn_index: int) -> DecayResult:
    rows = state.get("long_term_memory", [])
    if not isinstance(rows, list):
        rows = []
        state["long_term_memory"] = rows
    touched = 0
    archived = 0
    exempted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        row.setdefault("last_decay_at", now)
        if bool(row.get("pin")) or _bounded_float(row.get("identity_relevance")) >= IDENTITY_RELEVANCE_FLOOR:
            exempted += 1
            row["last_decay_at"] = now
            continue
        if str(row.get("status", "") or "") == "archived":
            continue
        elapsed = max(0, now - int(row.get("last_decay_at", row.get("created_at", now)) or now))
        if elapsed <= 0:
            continue
        rate = _decay_rate(row)
        delta = min(MAX_DECAY_DELTA_PER_TICK, rate * min(1.0, elapsed / 86400.0))
        if delta <= 0:
            row["last_decay_at"] = now
            continue
        salience = _bounded_float(row.get("salience"), default=0.2)
        row["salience"] = round(max(0.0, salience - delta), 6)
        row["last_decay_at"] = now
        touched += 1
        if row["salience"] < LTM_ARCHIVE_THRESHOLD and archived < MAX_ARCHIVE_FLIPS_PER_TICK:
            row["status"] = "archived"
            archived += 1
    event = {
        "type": "MemoryDecayTickEvent",
        "at": now,
        "turn_index": turn_index,
        "rows_touched": touched,
        "rows_archived": archived,
        "rows_exempted": exempted,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
    return DecayResult(rows_touched=touched, rows_archived=archived, rows_exempted=exempted, events=[event])


def apply_consolidation_decay_extension(state: dict[str, Any], *, now: int, turn_index: int) -> DecayResult:
    rows = state.get("long_term_memory", [])
    if not isinstance(rows, list):
        rows = []
        state["long_term_memory"] = rows
    touched = 0
    archived = 0
    exempted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("pin")) or _bounded_float(row.get("identity_relevance")) >= IDENTITY_RELEVANCE_FLOOR:
            exempted += 1
            continue
        if str(row.get("status", "") or "").startswith("merged_into:"):
            continue
        if str(row.get("status", "") or "") == "archived":
            continue
        if int(row.get("recall_count_session", 0) or 0) != 0:
            continue
        last_recalled = int(row.get("last_recalled_at", 0) or 0)
        if last_recalled and now - last_recalled < LAST_RECALL_GRACE_SECONDS:
            continue
        salience = _bounded_float(row.get("salience"), default=0.2)
        delta = min(MAX_DECAY_DELTA_PER_TICK, CONSOLIDATION_DECAY_DELTA_PER_RUN)
        if delta <= 0:
            continue
        row["salience"] = round(max(0.0, salience - delta), 6)
        row["last_consolidation_decay_at"] = now
        touched += 1
        if row["salience"] < LTM_ARCHIVE_THRESHOLD and archived < MAX_ARCHIVE_FLIPS_PER_TICK:
            row["status"] = "archived"
            archived += 1
    event = {
        "type": "ConsolidationDecayExtensionEvent",
        "at": now,
        "turn_index": turn_index,
        "rows_touched": touched,
        "rows_archived": archived,
        "rows_exempted": exempted,
        "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
    }
    return DecayResult(rows_touched=touched, rows_archived=archived, rows_exempted=exempted, events=[event])
