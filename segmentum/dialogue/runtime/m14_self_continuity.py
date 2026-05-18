"""M14.1 operational self-continuity (stored under self_cognition)."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_drive import _bounded_float, _new_id, _string_list
from segmentum.dialogue.runtime.m14_1_background_continuity import M14_1_ENGINEERING_PROXY_LABEL

K_STABLE_PROMOTION = 3
K_DRIFT_KNOWN_LIMIT = 4
K_SELF_REVIEW_TICKS = 6
MIN_BASELINE_UPDATE_CONFIDENCE = 0.7
MAX_DRIFT_WINDOW = 20
MAX_STABLE_CANDIDATES = 24


def default_self_continuity_state() -> dict[str, Any]:
    return {
        "baseline_summary": "",
        "baseline_stable_values": [],
        "baseline_known_limits": [],
        "baseline_updated_at": 0,
        "drift_window": [],
        "identity_tension_history": [],
        "stable_value_candidates": [],
        "last_self_review_at": 0,
        "self_review_count_today": 0,
        "idle_ticks_since_review": 0,
        "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
    }


def normalize_self_continuity_state(raw: Any) -> dict[str, Any]:
    base = default_self_continuity_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["baseline_summary"] = str(merged.get("baseline_summary", "") or "")[:800]
    merged["baseline_stable_values"] = [
        str(item)[:200] for item in merged.get("baseline_stable_values", []) if str(item).strip()
    ][:32]
    known_limits: list[Any] = []
    for item in merged.get("baseline_known_limits", []) or []:
        if isinstance(item, Mapping):
            known_limits.append(
                {
                    "limit": str(item.get("limit", ""))[:240],
                    "source": str(item.get("source", ""))[:80],
                    "patch_ids": str(item.get("patch_ids", ""))[:160],
                    "review_count": max(0, int(item.get("review_count", 0) or 0)),
                }
            )
        elif str(item).strip():
            known_limits.append(str(item)[:240])
    merged["baseline_known_limits"] = known_limits[:32]
    merged["baseline_updated_at"] = int(merged.get("baseline_updated_at", 0) or 0)
    for list_key in ("drift_window", "identity_tension_history", "stable_value_candidates"):
        rows = merged.get(list_key)
        merged[list_key] = [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    merged["last_self_review_at"] = int(merged.get("last_self_review_at", 0) or 0)
    merged["self_review_count_today"] = max(0, int(merged.get("self_review_count_today", 0) or 0))
    merged["idle_ticks_since_review"] = max(0, int(merged.get("idle_ticks_since_review", 0) or 0))
    return merged


def get_self_continuity_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    cognition = state.get("self_cognition", {})
    if not isinstance(cognition, Mapping):
        return default_self_continuity_state()
    return normalize_self_continuity_state(cognition.get("self_continuity"))


def attach_self_continuity(state: dict[str, Any], continuity: dict[str, Any]) -> None:
    cognition = state.setdefault("self_cognition", {})
    if not isinstance(cognition, dict):
        cognition = {}
        state["self_cognition"] = cognition
    cognition["self_continuity"] = normalize_self_continuity_state(continuity)


def build_self_continuity_snapshot(continuity: Mapping[str, Any]) -> dict[str, Any]:
    drift = continuity.get("drift_window", [])
    recent_drift = [dict(item) for item in drift[-5:]] if isinstance(drift, list) else []
    return {
        "baseline_summary": str(continuity.get("baseline_summary", "") or "")[:400],
        "baseline_stable_values": list(continuity.get("baseline_stable_values", []))[:8],
        "baseline_known_limits": list(continuity.get("baseline_known_limits", []))[:8],
        "recent_drift": recent_drift,
        "stable_value_candidates": [
            dict(item) for item in continuity.get("stable_value_candidates", [])[:6] if isinstance(item, Mapping)
        ],
    }


def should_run_self_review(continuity: Mapping[str, Any]) -> bool:
    ticks = int(continuity.get("idle_ticks_since_review", 0) or 0)
    return ticks >= K_SELF_REVIEW_TICKS


def note_idle_tick(continuity: dict[str, Any]) -> dict[str, Any]:
    merged = dict(continuity)
    merged["idle_ticks_since_review"] = int(merged.get("idle_ticks_since_review", 0) or 0) + 1
    return merged


def apply_self_cognition_patch_to_continuity(
    continuity: dict[str, Any],
    proposal: Mapping[str, Any],
    *,
    now: int,
    retrieved_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministic post-process after SelfCognitionPatchOwner commit path."""
    events: list[dict[str, Any]] = []
    merged = dict(continuity)
    if not bool(proposal.get("apply")):
        return merged, events

    confidence = _bounded_float(proposal.get("confidence"))
    refs = _string_list(proposal.get("evidence_refs"), limit=8)
    delta = str(proposal.get("summary_delta", "") or "").strip()
    patch_id = _new_id("sc_drift")

    if confidence >= MIN_BASELINE_UPDATE_CONFIDENCE and refs and (set(refs) & retrieved_ids):
        if delta:
            summary = str(merged.get("baseline_summary", "") or "").strip()
            merged["baseline_summary"] = (summary + " " + delta).strip()[:800] if summary else delta[:800]
            merged["baseline_updated_at"] = now
            events.append(
                {
                    "type": "SelfBaselineUpdateEvent",
                    "at": now,
                    "patch_id": patch_id,
                    "confidence": confidence,
                    "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
                }
            )
    else:
        drift = list(merged.get("drift_window", []))
        drift.append(
            {
                "patch_id": patch_id,
                "at": now,
                "magnitude": confidence,
                "direction": "pending",
                "kept_into_baseline": False,
                "summary_delta": delta[:240],
            }
        )
        merged["drift_window"] = drift[-MAX_DRIFT_WINDOW:]
        events.append(
            {
                "type": "SelfBaselineUpdateRejectedEvent",
                "at": now,
                "patch_id": patch_id,
                "violation_codes": ["low_confidence"] if confidence < MIN_BASELINE_UPDATE_CONFIDENCE else ["missing_evidence_refs"],
                "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
            }
        )

    for tension in proposal.get("new_identity_tensions", []) or []:
        text = str(tension).strip()
        if not text:
            continue
        history = list(merged.get("identity_tension_history", []))
        history.append({"tension": text[:200], "opened_at": now, "resolved_at": None, "resolution_kind": ""})
        merged["identity_tension_history"] = history[-MAX_DRIFT_WINDOW:]

    for limit in proposal.get("new_known_limits", []) or []:
        text = str(limit).strip()
        if text and text not in merged.get("baseline_known_limits", []):
            merged.setdefault("baseline_known_limits", []).append(text[:200])

    candidate_values: list[str] = []
    for key in ("stable_value_candidates", "baseline_stable_value_candidates", "new_stable_value_candidates"):
        for value in proposal.get(key, []) or []:
            candidate_values.append(str(value))
    for value in candidate_values:
        _bump_stable_candidate(merged, value, now=now)

    return merged, events


def _bump_stable_candidate(continuity: dict[str, Any], value: str, *, now: int) -> None:
    if not value.strip():
        return
    candidates = list(continuity.get("stable_value_candidates", []))
    for row in candidates:
        if str(row.get("value", "")) == value:
            row["count"] = int(row.get("count", 0) or 0) + 1
            row["last_seen_at"] = now
            continuity["stable_value_candidates"] = candidates[-MAX_STABLE_CANDIDATES:]
            if int(row["count"]) >= K_STABLE_PROMOTION:
                stable = list(continuity.get("baseline_stable_values", []))
                if value not in stable:
                    stable.append(value[:200])
                    continuity["baseline_stable_values"] = stable[-32:]
            return
    candidates.append({"value": value[:200], "count": 1, "last_seen_at": now})
    continuity["stable_value_candidates"] = candidates[-MAX_STABLE_CANDIDATES:]


def run_self_review_tick(
    continuity: dict[str, Any],
    *,
    now: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministic promotion / drift absorption without extra LLM."""
    events: list[dict[str, Any]] = []
    merged = dict(continuity)
    merged["idle_ticks_since_review"] = 0
    merged["last_self_review_at"] = now
    merged["self_review_count_today"] = int(merged.get("self_review_count_today", 0) or 0) + 1

    drift = list(merged.get("drift_window", []))
    pending = [d for d in drift if not bool(d.get("kept_into_baseline"))]
    if len(pending) >= K_DRIFT_KNOWN_LIMIT:
        selected = pending[:K_DRIFT_KNOWN_LIMIT]
        oldest = selected[0]
        text = str(oldest.get("summary_delta", "") or "").strip()
        if text:
            limits = list(merged.get("baseline_known_limits", []))
            entry = f"{text} (self_continuity_drift)"
            if entry not in limits:
                ids = ",".join(str(row.get("patch_id", "")) for row in selected if row.get("patch_id"))[:120]
                review_count = int(merged.get("self_review_count_today", 0) or 0)
                limits.append(
                    {
                        "limit": entry[:240],
                        "source": "self_continuity_drift",
                        "patch_ids": ids,
                        "review_count": review_count,
                    }
                )
                merged["baseline_known_limits"] = limits[-32:]
        for row in selected:
            row["kept_into_baseline"] = True
        merged["drift_window"] = drift[-MAX_DRIFT_WINDOW:]

    events.append(
        {
            "type": "SelfReviewEvent",
            "at": now,
            "drift_pending": len(pending),
            "engineering_proxy_label": M14_1_ENGINEERING_PROXY_LABEL,
        }
    )
    return merged, events
