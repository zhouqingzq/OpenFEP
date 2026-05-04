"""M9.0 helpers: payloads for MemoryRecallEvent / MemoryInterferenceEvent on the bus."""

from __future__ import annotations

from typing import Mapping, Sequence

from .memory_dynamics import detect_memory_interference, memory_overdominance_detected_from_retrieval
from .memory_evidence import unify_evidence


def _retrieval_maps_from_diagnostics(diagnostics: object | None) -> list[dict[str, object]]:
    if diagnostics is None:
        return []
    raw = list(getattr(diagnostics, "retrieved_memories", []) or [])
    out: list[dict[str, object]] = []
    for item in raw[:12]:
        if isinstance(item, Mapping):
            out.append(dict(item))
        else:
            out.append({"content": str(item)})
    return out


def build_unified_evidence_dicts(
    *,
    anchored_items: Sequence[object] | None,
    diagnostics: object | None,
    current_cue: str,
) -> list[dict[str, object]]:
    """Flatten MemoryEvidence records to JSON-safe dicts for bus / contract."""
    unified = unify_evidence(
        anchored_items=list(anchored_items or []),
        retrieval_results=_retrieval_maps_from_diagnostics(diagnostics),
        hypotheses=(),
        current_cue=current_cue,
    )
    return [e.to_dict() for e in unified]


def build_memory_recall_event_payload(
    *,
    cue: str,
    diagnostics: object | None,
    anchored_items: Sequence[object] | None,
    turn_id: str,
) -> dict[str, object]:
    cue_text = (cue or "").strip()
    unified_dicts = build_unified_evidence_dicts(
        anchored_items=anchored_items,
        diagnostics=diagnostics,
        current_cue=cue_text,
    )
    episode_ids = [
        str(x)
        for x in (getattr(diagnostics, "retrieved_episode_ids", []) or [])[:8]
    ]
    memory_hit = bool(getattr(diagnostics, "memory_hit", False))
    stance = (
        f"cued_recall:{len(unified_dicts)} evidence items"
        if cue_text
        else "unknown: empty cue, episodic detail not asserted from LTM"
    )
    return {
        "turn_id": turn_id,
        "cue": cue_text,
        "cue_stance": stance,
        "memory_hit": memory_hit,
        "episode_ids": episode_ids,
        "unified_evidence": unified_dicts,
        "recall_count": len(unified_dicts),
    }


def build_memory_interference_event_payload(
    *,
    diagnostics: object | None,
    last_retrieval_result: Mapping[str, object] | None,
) -> dict[str, object]:
    """Mirror agent retrieval interference diagnostics onto the cognitive bus."""
    memories = _retrieval_maps_from_diagnostics(diagnostics)
    prediction_delta = dict(getattr(diagnostics, "prediction_delta", {}) or {})
    signal = detect_memory_interference(
        diagnostics=diagnostics,
        retrieved_memories=memories,
        prediction_delta=prediction_delta,
    )
    raw_bus = dict(last_retrieval_result or {})
    bus_interference = raw_bus.get("memory_interference")
    if isinstance(bus_interference, Mapping) and bus_interference.get("detected"):
        merged = dict(bus_interference)
        merged.setdefault("kind", str(signal.kind or "memory_interference"))
        merged["severity"] = max(
            float(merged.get("severity", 0.0) or 0.0),
            float(signal.severity or 0.0),
        )
        interference_dict = merged
    else:
        interference_dict = signal.to_dict()
    overdominance = bool(
        memory_overdominance_detected_from_retrieval(raw_bus)
        if raw_bus
        else False
    )
    return {
        "interference": interference_dict,
        "overdominance": overdominance,
        "retrieved_sample_size": len(memories),
    }