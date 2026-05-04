from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .memory_decay import (
    DecayReport,
    LONG_DORMANT_ACCESS_THRESHOLD,
    LONG_DORMANT_TRACE_THRESHOLD,
    SHORT_CLEANUP_TRACE_THRESHOLD,
    access_decay_rate,
    decay_accessibility,
    decay_accessibility_for_level,
    decay_trace_strength,
    decay_trace_strength_for_level,
    trace_decay_rate,
)
from .memory_anchored import AnchoredMemoryItem
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel


_LOGGER = logging.getLogger(__name__)
_PROMOTION_METADATA_WARNING_EMITTED = False


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _style_value(cognitive_style: dict[str, object] | object | None, key: str, default: float = 0.0) -> float:
    if cognitive_style is not None and hasattr(cognitive_style, key):
        return _coerce_float(getattr(cognitive_style, key, default), default)
    if isinstance(cognitive_style, dict):
        return _coerce_float(cognitive_style.get(key, default), default)
    return default


def _internal_metadata(entry: MemoryEntry) -> dict[str, object]:
    if entry.compression_metadata is None:
        entry.compression_metadata = {}
    internal = entry.compression_metadata.get("m45_internal")
    if not isinstance(internal, dict):
        internal = {}
        entry.compression_metadata["m45_internal"] = internal
    return internal


def _warn_missing_promotion_evidence(entry: MemoryEntry, missing_sections: list[str]) -> None:
    global _PROMOTION_METADATA_WARNING_EMITTED
    if _PROMOTION_METADATA_WARNING_EMITTED:
        return
    _LOGGER.warning(
        "Promotion is running in fallback mode because encoding evidence is missing "
        "(entry_id=%s, missing=%s); identity signal degraded.",
        entry.id,
        ",".join(missing_sections),
    )
    _PROMOTION_METADATA_WARNING_EMITTED = True


def _effective_cycle(entry: MemoryEntry, baseline_cycle: int) -> int:
    return max(entry.created_at, entry.last_accessed, baseline_cycle)


def _ensure_decay_baseline(entry: MemoryEntry) -> tuple[dict[str, object], int, float, float]:
    internal = _internal_metadata(entry)
    baseline_cycle = _coerce_int(
        internal.get("last_decay_cycle", max(entry.created_at, entry.last_accessed)),
        max(entry.created_at, entry.last_accessed),
    )
    base_trace = _coerce_float(internal.get("decay_base_trace_strength", entry.trace_strength), entry.trace_strength)
    base_access = _coerce_float(
        internal.get("decay_base_accessibility", entry.accessibility),
        entry.accessibility,
    )
    internal["last_decay_cycle"] = baseline_cycle
    internal["decay_base_trace_strength"] = base_trace
    internal["decay_base_accessibility"] = base_access
    return internal, baseline_cycle, base_trace, base_access


def _refresh_decay_baseline(entry: MemoryEntry, *, cycle: int) -> dict[str, object]:
    internal = _internal_metadata(entry)
    internal["last_decay_cycle"] = cycle
    internal["decay_base_trace_strength"] = entry.trace_strength
    internal["decay_base_accessibility"] = entry.accessibility
    return internal


LEGACY_MAPPED_KEYS = {
    "episode_id",
    "timestamp",
    "cycle",
    "state_vector",
    "state_snapshot",
    "action_taken",
    "action",
    "outcome_state",
    "outcome",
    "predicted_outcome",
    "prediction_error",
    "risk",
    "value_score",
    "total_surprise",
    "weighted_surprise",
    "embedding",
    "value_label",
    "preferred_probability",
    "preference_log_value",
    "observation",
    "prediction",
    "errors",
    "body_state",
    "support_count",
    "support",
    "last_seen_cycle",
    "identity_critical",
    "continuity_tags",
    "gating_reasons",
    "lifecycle_stage",
    "episode_family",
    "content",
    "memory_class",
    "source_type",
    "store_level",
    "abstractness",
    "encoding_source",
    "encoding_strength",
    "fep_prediction_error",
    "surprise",
    "attention_budget_total",
    "attention_budget_requested",
    "attention_budget_granted",
    "attention_budget_denied",
    "centroid",
    "residual_norm_mean",
    "residual_norm_var",
    "support_ids",
    "consolidation_source",
    "semantic_reconstruction_error",
    "replay_second_pass_error",
    "salience_delta",
    "retention_adjustment",
    "compression_metadata",
}


def _legacy_unmapped(entry: MemoryEntry) -> dict[str, object]:
    if not isinstance(entry.compression_metadata, dict):
        return {}
    payload = entry.compression_metadata.get("legacy_unmapped")
    if not isinstance(payload, dict):
        return {}
    return deepcopy(payload)


def _legacy_template(entry: MemoryEntry, *, copy: bool = True) -> dict[str, object]:
    if not isinstance(entry.compression_metadata, dict):
        return {}
    payload = entry.compression_metadata.get("legacy_template")
    if not isinstance(payload, dict):
        return {}
    return deepcopy(payload) if copy else payload


def _legacy_episode_id(payload: dict[str, object], index: int = 0) -> str:
    timestamp = _coerce_int(payload.get("timestamp", payload.get("cycle", 0)))
    return str(payload.get("episode_id", f"legacy-{timestamp}-{index}"))


def _legacy_payload_matches(entry: MemoryEntry, payload: dict[str, object]) -> bool:
    return _legacy_template(entry, copy=False) == payload


def _sanitize_legacy_compression_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    sanitized = deepcopy(metadata)
    sanitized.pop("legacy_template", None)
    sanitized.pop("legacy_unmapped", None)
    return sanitized


def _payload_without_recursive_legacy(payload: dict[str, object]) -> dict[str, object]:
    sanitized = deepcopy(payload)
    had_compression_metadata = "compression_metadata" in sanitized
    sanitized_metadata = _sanitize_legacy_compression_metadata(
        sanitized.get("compression_metadata")
    )
    if had_compression_metadata or sanitized_metadata:
        sanitized["compression_metadata"] = sanitized_metadata
    else:
        sanitized.pop("compression_metadata", None)
    return sanitized


def _legacy_entry_from_payload(payload: dict[str, object], index: int = 0) -> MemoryEntry:
    timestamp = _coerce_int(payload.get("timestamp", payload.get("cycle", 0)))
    action = str(payload.get("action", "unknown"))
    outcome = str(
        payload.get(
            "predicted_outcome",
            payload.get("value_label", _outcome_summary(payload.get("outcome", "neutral"))),
        )
    )
    content = str(
        payload.get(
            "content",
            f"Legacy episode at cycle {timestamp}: {action} -> {outcome}",
        )
    )
    total_surprise = _coerce_float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0)))
    encoding_strength = _coerce_float(payload.get("encoding_strength"), total_surprise)
    encoded_arousal = _coerce_float(payload.get("arousal"), total_surprise)
    support_count = _coerce_int(payload.get("support_count", payload.get("support", 1)), 1)
    store_level = StoreLevel.SHORT
    if bool(payload.get("identity_critical", False)) or total_surprise >= 0.9:
        store_level = StoreLevel.LONG
    elif support_count >= 2 or total_surprise >= 0.5:
        store_level = StoreLevel.MID
    metadata = _sanitize_legacy_compression_metadata(payload.get("compression_metadata"))
    sanitized_payload = _payload_without_recursive_legacy(payload)
    metadata.update({
        "legacy_template": sanitized_payload,
        "legacy_unmapped": {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in LEGACY_MAPPED_KEYS
        },
        "m45_internal": {"last_decay_cycle": timestamp},
    })
    m410_keys = (
        "encoding_source",
        "encoding_strength",
        "fep_prediction_error",
        "surprise",
        "attention_budget_total",
        "attention_budget_requested",
        "attention_budget_granted",
        "attention_budget_denied",
    )
    for key in m410_keys:
        if key in payload:
            metadata[key] = deepcopy(payload[key])
    memory_class = MemoryClass(str(payload.get("memory_class", MemoryClass.EPISODIC.value)))
    source_type = SourceType(str(payload.get("source_type", SourceType.EXPERIENCE.value)))
    store_level_hint = payload.get("store_level")
    return MemoryEntry(
        id=_legacy_episode_id(payload, index=index),
        content=content,
        memory_class=memory_class,
        store_level=StoreLevel(str(store_level_hint)) if store_level_hint is not None else store_level,
        source_type=source_type,
        created_at=timestamp,
        last_accessed=_coerce_int(payload.get("last_seen_cycle", timestamp)),
        valence=_coerce_float(payload.get("value_score", 0.0)),
        arousal=_clamp(encoded_arousal),
        encoding_attention=_clamp(_coerce_float(payload.get("attention_budget_granted"), encoding_strength)),
        novelty=_clamp(_coerce_float(payload.get("fep_prediction_error", payload.get("prediction_error", 0.0)))),
        relevance_goal=_clamp(abs(_coerce_float(payload.get("value_relevance", 0.0)))),
        relevance_threat=_clamp(_coerce_float(payload.get("threat_significance", 0.0))),
        relevance_self=1.0 if bool(payload.get("identity_critical", False)) else 0.0,
        relevance_social=_clamp(0.6 if str(payload.get("episode_family", "")) == "social_signal" else 0.0),
        relevance_reward=_clamp(abs(_coerce_float(payload.get("value_score", 0.0)))),
        relevance=_clamp(
            max(
                _coerce_float(payload.get("value_relevance", 0.0)),
                _coerce_float(payload.get("threat_significance", 0.0)),
                1.0 if bool(payload.get("identity_critical", False)) else 0.0,
            )
        ),
        salience=_clamp(encoding_strength),
        trace_strength=max(0.05, _clamp(encoding_strength)),
        accessibility=max(0.05, _clamp(encoding_strength)),
        abstractness=_coerce_float(payload.get("abstractness"), 0.25),
        source_confidence=0.9,
        reality_confidence=0.85,
        semantic_tags=sorted(
            {
                action,
                outcome,
                *[str(item) for item in payload.get("continuity_tags", []) if str(item)],
                str(payload.get("episode_family", "")),
            }
            - {""}
        ),
        context_tags=sorted(
            {
                *[str(item) for item in payload.get("gating_reasons", []) if str(item)],
                str(payload.get("lifecycle_stage", "")),
            }
            - {""}
        ),
        anchor_slots={
            "time": str(timestamp),
            "place": None,
            "agents": None,
            "action": action,
            "outcome": outcome,
        },
        anchor_strengths={
            "time": "weak",
            "place": "weak",
            "agents": "strong",
            "action": "strong",
            "outcome": "strong",
        },
        mood_context=str(payload.get("predicted_outcome", "")),
        retrieval_count=0,
        support_count=max(1, support_count),
        counterevidence_count=0,
        competing_interpretations=None,
        compression_metadata=metadata,
        derived_from=[],
        is_dormant=bool(payload.get("lifecycle_stage") == "archived_summary"),
        state_vector=list(payload.get("embedding", [])) if isinstance(payload.get("embedding"), list) else None,
        centroid=list(payload.get("centroid", [])) if isinstance(payload.get("centroid"), list) else None,
        residual_norm_mean=payload.get("residual_norm_mean"),
        residual_norm_var=payload.get("residual_norm_var"),
        support_ids=payload.get("support_ids") if isinstance(payload.get("support_ids"), list) else None,
        consolidation_source=(
            str(payload.get("consolidation_source"))
            if payload.get("consolidation_source") is not None
            else None
        ),
        semantic_reconstruction_error=payload.get("semantic_reconstruction_error"),
        replay_second_pass_error=payload.get("replay_second_pass_error"),
        salience_delta=payload.get("salience_delta"),
        retention_adjustment=payload.get("retention_adjustment"),
    )


def _outcome_summary(value: Any) -> str:
    if isinstance(value, dict):
        summary = value.get("summary")
        if summary is not None:
            return str(summary)
    return str(value or "")


def _update_legacy_outcome_payload(value: object, outcome_text: str) -> object:
    if isinstance(value, dict):
        updated = deepcopy(value)
        updated["summary"] = outcome_text
        return updated
    if value is None:
        return {"summary": outcome_text}
    return outcome_text


def _update_legacy_action_payload(value: object, action: str) -> object:
    if isinstance(value, dict):
        updated = deepcopy(value)
        updated["name"] = action
        return updated
    if value is None:
        return action
    return action


def _has_source_conflict(entry: MemoryEntry) -> bool:
    if not isinstance(entry.compression_metadata, dict):
        return False
    metadata = entry.compression_metadata
    if metadata.get("source_conflict") or metadata.get("provenance_conflict"):
        return True
    conflict_type = str(metadata.get("conflict_type", "")).lower()
    if conflict_type == "source":
        return True
    conflicts = metadata.get("source_conflicts")
    return isinstance(conflicts, list) and bool(conflicts)


def _has_reality_conflict(entry: MemoryEntry) -> bool:
    if entry.counterevidence_count > 0 or bool(entry.competing_interpretations):
        return True
    if not isinstance(entry.compression_metadata, dict):
        return False
    metadata = entry.compression_metadata
    if metadata.get("factual_conflict") or metadata.get("counterevidence_present"):
        return True
    conflict_type = str(metadata.get("conflict_type", "")).lower()
    if conflict_type == "factual":
        return True
    conflicts = metadata.get("reality_conflicts")
    return isinstance(conflicts, list) and bool(conflicts)


@dataclass
class MemoryStore:
    entries: list[MemoryEntry] = field(default_factory=list)
    short_to_mid_salience_threshold: float = 0.55
    short_to_mid_retrieval_threshold: int = 2
    mid_to_long_salience_threshold: float = 0.80
    mid_to_long_retrieval_threshold: int = 3
    identity_priority_threshold: float = 0.75
    identity_long_threshold: float = 0.85
    state_window_size: int = 30
    agent_state_vector: dict[str, object] | None = None
    last_cleanup_report: dict[str, object] = field(default_factory=dict)
    semantic_schemas: list[dict[str, object]] = field(default_factory=list)
    latest_schema_update: dict[str, object] = field(default_factory=dict)
    anchored_items: list[AnchoredMemoryItem] = field(default_factory=list)

    def _find_index(self, entry_id: str) -> int | None:
        for index, entry in enumerate(self.entries):
            if entry.id == entry_id:
                return index
        return None

    def _promote_store_level(
        self,
        entry: MemoryEntry,
        *,
        current_state: dict[str, object] | None = None,
        cognitive_style: dict[str, object] | object | None = None,
    ) -> None:
        from .memory_state import identity_match_ratio_for_entry, normalize_agent_state

        internal, baseline_cycle, base_trace, base_access = _ensure_decay_baseline(entry)
        effective_cycle = _effective_cycle(entry, baseline_cycle)
        old_level = entry.store_level
        new_level = old_level
        state_vector = normalize_agent_state(current_state or self.agent_state_vector)
        selectivity = _style_value(cognitive_style, "attention_selectivity", 0.0)
        rigidity = _style_value(cognitive_style, "update_rigidity", 0.0)
        identity_active = 1.0 if state_vector.identity_active_themes else 0.0
        encoding_audit = dict(dict(entry.compression_metadata or {}).get("encoding_audit", {}))
        dynamic_salience_audit = dict(
            dict(entry.compression_metadata or {}).get("dynamic_salience_audit", {})
        )
        missing_evidence: list[str] = []
        if not encoding_audit:
            missing_evidence.append("encoding_audit")
        if not dynamic_salience_audit:
            missing_evidence.append("dynamic_salience_audit")
        consolidation_entry = (
            entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}
            and entry.consolidation_source == "dynamics"
            and bool(entry.centroid)
        )
        if missing_evidence and not consolidation_entry:
            _warn_missing_promotion_evidence(entry, missing_evidence)
        retained_identity_priority = "identity_continuity_priority" in list(
            dict(encoding_audit.get("retention_priority", {})).get("reasons", [])
        )
        encoded_identity_match_ratio = _coerce_float(
            dynamic_salience_audit.get("identity_match_ratio"),
            0.0,
        )
        identity_match_ratio = max(
            identity_match_ratio_for_entry(entry, state_vector),
            encoded_identity_match_ratio,
        )
        identity_context_available = bool(state_vector.identity_active_themes)
        identity_fallback_active = entry.relevance_self >= self.identity_priority_threshold
        identity_link_strength = _clamp((entry.relevance_self * 0.6) + (identity_match_ratio * 0.4))
        identity_link_active = (
            (identity_context_available or retained_identity_priority or identity_fallback_active)
            and identity_link_strength >= self.identity_priority_threshold
        )
        threat_snapshot_bonus = (
            0.14
            if (
                state_vector.threat_level >= 0.60
                and entry.arousal >= 0.55
                and entry.memory_class is MemoryClass.EPISODIC
            )
            else 0.0
        )
        novelty_noise_penalty = (
            0.16 + (selectivity * 0.04)
            if (
                entry.novelty >= 0.75
                and entry.relevance_self < 0.20
                and identity_active > 0.0
            )
            else 0.0
        )
        retrieval_short = min(1.0, entry.retrieval_count / max(1, self.short_to_mid_retrieval_threshold))
        retrieval_long = min(1.0, entry.retrieval_count / max(1, self.mid_to_long_retrieval_threshold))
        identity_bonus = entry.relevance_self * (0.18 + (0.10 * identity_active))
        abstraction_bonus = (
            entry.abstractness * 0.16
            if entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}
            else 0.0
        )
        score_breakdown = {
            "salience_signal": round(entry.salience * 0.52, 6),
            "retrieval_short_signal": round(retrieval_short * 0.16, 6),
            "retrieval_long_signal": round(retrieval_long * 0.22, 6),
            "identity_signal": round(identity_bonus, 6),
            "abstraction_bonus": round(abstraction_bonus, 6),
            "threat_snapshot_bonus": round(threat_snapshot_bonus, 6),
            "novelty_noise_penalty": round(novelty_noise_penalty, 6),
            "rigidity_penalty": round(rigidity * 0.08, 6),
            "selectivity_bias": round(selectivity * 0.03 * max(0.0, entry.relevance_self - 0.20), 6),
        }
        short_to_mid_score = (
            score_breakdown["salience_signal"]
            + score_breakdown["retrieval_short_signal"]
            + score_breakdown["identity_signal"]
            + score_breakdown["threat_snapshot_bonus"]
            + score_breakdown["selectivity_bias"]
            - score_breakdown["novelty_noise_penalty"]
            - score_breakdown["rigidity_penalty"]
        )
        self_relevance_multiplier = 1.0 + (0.35 * identity_link_strength) if identity_link_active else 1.0
        unbounded_short_to_mid_score = short_to_mid_score * self_relevance_multiplier
        boosted_short_to_mid_score = (
            min(0.95, unbounded_short_to_mid_score)
            if identity_link_active
            else short_to_mid_score
        )
        score_cap_applied = identity_link_active and boosted_short_to_mid_score < unbounded_short_to_mid_score
        mid_to_long_score = (
            (entry.salience * 0.44)
            + score_breakdown["retrieval_long_signal"]
            + score_breakdown["abstraction_bonus"]
            + (entry.relevance_self * (0.22 + (0.10 * identity_active)))
            + (threat_snapshot_bonus * 0.85)
            - novelty_noise_penalty
            - (rigidity * 0.10)
        )
        score_thresholds = {"short_to_mid": 0.55, "mid_to_long": 0.72}
        promotion_reasons = [
            label
            for label, value in (
                ("salience_signal", entry.salience),
                ("retrieval_signal", max(retrieval_short, retrieval_long)),
                ("identity_alignment", entry.relevance_self),
                ("threat_snapshot", threat_snapshot_bonus),
            )
            if float(value) > 0.0
        ]
        if entry.compression_metadata is None:
            entry.compression_metadata = {}
        entry.compression_metadata["m47_promotion_audit"] = {
            "short_to_mid_score": round(boosted_short_to_mid_score, 6),
            "mid_to_long_score": round(mid_to_long_score, 6),
            "score_thresholds": dict(score_thresholds),
            "score_breakdown": dict(score_breakdown),
            "state_identity_active": bool(identity_active),
            "identity_match_ratio": round(identity_match_ratio, 6),
            "identity_link_strength": round(identity_link_strength, 6),
            "identity_link_active": bool(identity_link_active),
            "identity_context_available": bool(identity_context_available),
            "identity_fallback_active": bool(identity_fallback_active),
            "retained_identity_priority": bool(retained_identity_priority),
            "self_relevance_multiplier": round(self_relevance_multiplier, 6),
            "base_short_to_mid_score": round(short_to_mid_score, 6),
            "boosted_short_to_mid_score": round(boosted_short_to_mid_score, 6),
            "score_cap_applied": bool(score_cap_applied),
        }

        if new_level is StoreLevel.SHORT and boosted_short_to_mid_score >= score_thresholds["short_to_mid"]:
            new_level = StoreLevel.MID
        if new_level is StoreLevel.MID and mid_to_long_score >= score_thresholds["mid_to_long"]:
            new_level = StoreLevel.LONG

        if new_level is old_level:
            return

        self.promote_entry(
            entry,
            new_level=new_level,
            reasons=promotion_reasons,
            effective_cycle=effective_cycle,
            promotion_context={
                "short_to_mid_threshold": score_thresholds["short_to_mid"],
                "mid_to_long_threshold": score_thresholds["mid_to_long"],
                "promotion_score_short_to_mid": round(boosted_short_to_mid_score, 6),
                "promotion_score_mid_to_long": round(mid_to_long_score, 6),
                "promotion_score_breakdown": dict(score_breakdown),
                "identity_match_ratio": round(identity_match_ratio, 6),
                "identity_link_strength": round(identity_link_strength, 6),
                "identity_link_active": bool(identity_link_active),
                "identity_context_available": bool(identity_context_available),
                "identity_fallback_active": bool(identity_fallback_active),
                "retained_identity_priority": bool(retained_identity_priority),
                "self_relevance_multiplier": round(self_relevance_multiplier, 6),
                "base_short_to_mid_score": round(short_to_mid_score, 6),
                "boosted_short_to_mid_score": round(boosted_short_to_mid_score, 6),
                "score_cap_applied": bool(score_cap_applied),
                "promotion_channel": "memory_store_add",
            },
        )

    def promote_entry(
        self,
        entry: MemoryEntry,
        *,
        new_level: StoreLevel,
        reasons: list[str] | None = None,
        effective_cycle: int | None = None,
        promotion_context: dict[str, object] | None = None,
    ) -> bool:
        old_level = entry.store_level
        if new_level is old_level:
            return False

        _, baseline_cycle, base_trace, base_access = _ensure_decay_baseline(entry)
        resolved_cycle = _effective_cycle(entry, baseline_cycle) if effective_cycle is None else max(
            int(effective_cycle),
            baseline_cycle,
            entry.created_at,
            entry.last_accessed,
        )
        elapsed = max(0, resolved_cycle - baseline_cycle)
        old_trace = decay_trace_strength_for_level(
            base_trace,
            old_level,
            elapsed,
            memory_class=entry.memory_class,
        )
        old_access = decay_accessibility_for_level(base_access, old_level, elapsed)
        new_trace = decay_trace_strength_for_level(
            base_trace,
            new_level,
            elapsed,
            memory_class=entry.memory_class,
        )
        new_access = decay_accessibility_for_level(base_access, new_level, elapsed)

        entry.store_level = new_level
        entry.trace_strength = new_trace
        entry.accessibility = new_access
        entry.last_accessed = max(entry.last_accessed, resolved_cycle)
        internal = _refresh_decay_baseline(entry, cycle=entry.last_accessed)
        promotion_record = {
            "reason": "+".join(reasons or []) if reasons else "promotion",
            "effective_cycle": entry.last_accessed,
            "elapsed_since_baseline": elapsed,
            "old_level": old_level.value,
            "new_level": new_level.value,
            "trace_before": old_trace,
            "trace_after": new_trace,
            "accessibility_before": old_access,
            "accessibility_after": new_access,
            "trace_rate_before": trace_decay_rate(old_level, memory_class=entry.memory_class),
            "trace_rate_after": trace_decay_rate(new_level, memory_class=entry.memory_class),
            "access_rate_before": access_decay_rate(old_level),
            "access_rate_after": access_decay_rate(new_level),
        }
        if promotion_context:
            promotion_record.update(deepcopy(promotion_context))
        history = internal.get("promotion_history")
        if not isinstance(history, list):
            history = []
            internal["promotion_history"] = history
        history.append(promotion_record)
        internal["last_promotion"] = promotion_record
        return True

    def _audit_m9_retention_for_entry(self, entry: MemoryEntry, *, cycle: int) -> dict[str, object]:
        """M9.0: log value-based retention pressure and decay state on write."""
        from .memory_dynamics import compute_decay_state, compute_retention_pressure

        has_conflict = _has_reality_conflict(entry) or _has_source_conflict(entry)
        tag_tuple = tuple(str(t) for t in (entry.semantic_tags or ())[:8])
        conf = _clamp(
            0.35
            + 0.65
            * _clamp((entry.reality_confidence + entry.source_confidence) / 2.0),
        )
        maint = 0.14 if entry.store_level is StoreLevel.SHORT else 0.06
        rp = compute_retention_pressure(
            identity_continuity_value=_clamp(entry.relevance_self),
            relationship_continuity_value=_clamp(
                entry.relevance_social * 0.65 + entry.valence * 0.25
            ),
            future_prediction_value=_clamp(entry.salience * 0.55 + entry.relevance_goal * 0.35),
            affective_salience=_clamp(entry.arousal),
            user_emphasis=_clamp(min(1.0, entry.retrieval_count / 10.0)),
            maintenance_cost=maint,
            confidence=conf,
            has_conflict=has_conflict,
            tags=tag_tuple,
            memory_type=str(getattr(entry.memory_class, "value", entry.memory_class)),
        )
        decay_state = compute_decay_state(
            retention_pressure=rp,
            cycle=cycle,
            created_at_cycle=entry.created_at,
            last_access_cycles_ago=max(0, cycle - int(entry.last_accessed or 0)),
            access_frequency=max(1, int(entry.retrieval_count or 0)),
        )
        snapshot: dict[str, object] = {
            "logged_cycle": int(cycle),
            "retention_pressure": rp.to_dict(),
            "decay_state": decay_state,
            "summary_reason": rp.decay_reason,
        }
        if entry.compression_metadata is None:
            entry.compression_metadata = {}
        history = entry.compression_metadata.get("m9_retention_history")
        if not isinstance(history, list):
            history = []
        history.append(snapshot)
        entry.compression_metadata["m9_retention_history"] = history
        entry.compression_metadata["m9_retention"] = snapshot
        return snapshot

    def add(
        self,
        entry: MemoryEntry,
        *,
        current_state: dict[str, object] | None = None,
        cognitive_style: dict[str, object] | object | None = None,
    ) -> str:
        from .memory_state import update_agent_state_vector

        entry.sync_content_hash()
        self._promote_store_level(
            entry,
            current_state=current_state,
            cognitive_style=cognitive_style,
        )
        _, baseline_cycle, _, _ = _ensure_decay_baseline(entry)
        effective_cycle = _effective_cycle(entry, baseline_cycle)
        _refresh_decay_baseline(entry, cycle=effective_cycle)
        existing_index = self._find_index(entry.id)
        if existing_index is None:
            self.entries.append(entry)
        else:
            self.entries[existing_index] = entry
        self.agent_state_vector = update_agent_state_vector(
            self,
            window_size=self.state_window_size,
            cycle=max(entry.created_at, entry.last_accessed),
            active_goals=_string_list(
                current_state.get("active_goals") if isinstance(current_state, dict) else []
            ),
        ).to_dict()
        self._audit_m9_retention_for_entry(
            entry,
            cycle=max(entry.created_at, entry.last_accessed),
        )
        return entry.id

    def upsert_legacy_episode(self, payload: dict[str, object], *, index: int = 0) -> str:
        entry_id = _legacy_episode_id(payload, index=index)
        existing_index = self._find_index(entry_id)
        if existing_index is not None and _legacy_payload_matches(self.entries[existing_index], payload):
            return self.entries[existing_index].id
        entry = _legacy_entry_from_payload(payload, index=index)
        if existing_index is None:
            self.entries.append(entry)
        else:
            self.entries[existing_index] = entry
        return entry.id

    def remove_legacy_episode(self, episode_id: str) -> bool:
        existing_index = self._find_index(str(episode_id))
        if existing_index is None:
            return False
        self.entries.pop(existing_index)
        return True

    def replace_legacy_group(self, episodes: list[dict[str, object]]) -> dict[str, list[str]]:
        existing_by_id = {entry.id: entry for entry in self.entries}
        reused_ids: list[str] = []
        upserted_ids: list[str] = []
        new_entries: list[MemoryEntry] = []
        for index, payload in enumerate(episodes):
            entry_id = _legacy_episode_id(payload, index=index)
            existing = existing_by_id.get(entry_id)
            if existing is not None and _legacy_payload_matches(existing, payload):
                reused_ids.append(entry_id)
                new_entries.append(existing)
                continue
            upserted_ids.append(entry_id)
            new_entries.append(_legacy_entry_from_payload(payload, index=index))
        removed_ids = [entry.id for entry in self.entries if entry.id not in {item.id for item in new_entries}]
        self.entries = new_entries
        return {
            "reused_ids": reused_ids,
            "upserted_ids": upserted_ids,
            "removed_ids": removed_ids,
        }

    def get(self, entry_id: str) -> MemoryEntry | None:
        index = self._find_index(entry_id)
        return None if index is None else self.entries[index]

    def query_by_tags(self, tags: list[str], k: int = 5) -> list[MemoryEntry]:
        normalized = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        scored: list[tuple[float, float, MemoryEntry]] = []
        for entry in self.entries:
            entry_tags = {tag.lower() for tag in [*entry.semantic_tags, *entry.context_tags]}
            overlap = len(normalized & entry_tags)
            if not overlap and normalized:
                continue
            score = float(overlap) + (0.25 * entry.accessibility) + (0.15 * entry.salience)
            scored.append((score, entry.trace_strength, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:k]]

    def get_latest_episodic_entry(self) -> MemoryEntry | None:
        episodic = [e for e in self.entries if e.memory_class is MemoryClass.EPISODIC]
        if not episodic:
            return None
        episodic.sort(key=lambda e: e.created_at, reverse=True)
        return episodic[0]

    def episodic_entries(self) -> list[MemoryEntry]:
        result = [e for e in self.entries if e.memory_class is MemoryClass.EPISODIC]
        result.sort(key=lambda e: e.created_at)
        return result

    def episodic_count(self) -> int:
        return sum(1 for e in self.entries if e.memory_class is MemoryClass.EPISODIC)

    def retrieve(
        self,
        query,
        *,
        current_mood: str | None = None,
        k: int = 5,
        agent_state=None,
        cognitive_style=None,
    ):
        from .memory_retrieval import retrieve

        return retrieve(
            query,
            self,
            current_mood=current_mood,
            k=k,
            agent_state=agent_state or self.agent_state_vector,
            cognitive_style=cognitive_style,
        )

    def reconsolidate_entry(
        self,
        entry_id: str,
        *,
        current_mood: str | None = None,
        current_context_tags: list[str] | None = None,
        current_cycle: int | None = None,
        current_state: dict[str, object] | None = None,
        recall_artifact=None,
        conflict_type=None,
        cognitive_style=None,
    ):
        from .memory_consolidation import reconsolidate

        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        return reconsolidate(
            entry,
            current_mood,
            current_context_tags,
            store=self,
            current_cycle=current_cycle,
            current_state=current_state,
            recall_artifact=recall_artifact,
            conflict_type=conflict_type,
            cognitive_style=cognitive_style,
        )

    def run_consolidation_cycle(
        self,
        current_cycle: int,
        rng: random.Random,
        current_state: dict[str, object] | None = None,
        cognitive_style=None,
    ):
        from .memory_consolidation import run_consolidation_cycle

        report = run_consolidation_cycle(
            self,
            current_cycle=current_cycle,
            rng=rng,
            current_state=current_state,
            cognitive_style=cognitive_style,
        )
        return report

    def mark_dormant(self, entry_id: str) -> None:
        entry = self.get(entry_id)
        if entry is not None:
            entry.is_dormant = True

    def cleanup_short(self, threshold: float = SHORT_CLEANUP_TRACE_THRESHOLD) -> int:
        deleted: list[str] = []
        retained_low_value: list[str] = []
        keep: list[MemoryEntry] = []
        for entry in self.entries:
            if entry.store_level is StoreLevel.SHORT and entry.trace_strength <= threshold:
                deleted.append(entry.id)
                continue
            if entry.trace_strength <= threshold:
                retained_low_value.append(entry.id)
            keep.append(entry)
        self.entries = keep
        self.last_cleanup_report = {
            "threshold": threshold,
            "deleted_short_residue": deleted,
            "retained_non_short_low_trace": retained_low_value,
        }
        return len(deleted)

    # ── Anchored Memory Item methods (M8) ──────────────────────────────

    def add_anchored_item(self, item: AnchoredMemoryItem) -> str:
        """Add an AnchoredMemoryItem to the store. Returns its memory_id."""
        self.anchored_items.append(item)
        return item.memory_id

    def commit_write_intent(self, intent) -> tuple[str, str]:
        """Commit a MemoryWriteIntent to the store. Returns (memory_id, operation).

        Records write-intent provenance tags on the anchored item so its
        source turn, utterance, speaker, and extraction cycle are auditable.

        Backward compat: this is an explicit commit path that wraps
        add_anchored_item().  Direct add_anchored_item() still works for
        callers that do not have a bus or intent infrastructure.
        """
        from segmentum.memory_anchored import MemoryWriteIntent  # late import

        if not isinstance(intent, MemoryWriteIntent):
            raise TypeError("commit_write_intent requires a MemoryWriteIntent")
        if intent.operation == "reject":
            return intent.item.memory_id if intent.item else "", "reject"
        if intent.item is None:
            raise ValueError("MemoryWriteIntent has no item to commit")

        item = intent.item
        if intent.trace is not None:
            item.tags = list(item.tags or [])
            item.tags.append(f"write_intent_id:{intent.intent_id}")
            item.tags.append(f"source_turn:{intent.trace.source_turn_id}")
            item.tags.append(f"source_speaker:{intent.trace.source_speaker}")
            item.tags.append(f"committed_at_cycle:{item.created_at_cycle or 0}")

        mid = self.add_anchored_item(item)
        return mid, intent.operation

    def get_anchored_items(
        self,
        *,
        status_filter: str | None = None,
        visibility_filter: str | None = None,
        memory_type_filter: str | None = None,
    ) -> list[AnchoredMemoryItem]:
        """Retrieve anchored items with optional filters."""
        result = self.anchored_items
        if status_filter is not None:
            result = [it for it in result if it.status == status_filter]
        if visibility_filter is not None:
            result = [it for it in result if it.visibility == visibility_filter]
        if memory_type_filter is not None:
            result = [it for it in result if it.memory_type == memory_type_filter]
        return result

    def retract_anchored_item(self, memory_id: str) -> bool:
        """Mark an anchored item as retracted. Returns True if found."""
        for item in self.anchored_items:
            if item.memory_id == memory_id:
                item.status = "retracted"
                return True
        return False

    def find_anchored_by_proposition(self, substring: str) -> list[AnchoredMemoryItem]:
        """Find anchored items whose proposition contains *substring*."""
        sub = substring.strip().lower()
        if not sub:
            return []
        return [it for it in self.anchored_items if sub in it.proposition.lower()]

    def prune_anchored(
        self, *, max_count: int = 50, current_cycle: int = 0,
    ) -> int:
        """Prune anchored items to at most *max_count*, preferring newer items.

        Retracted items are evicted first. Items with ``created_at_cycle == 0``
        (pre-existing, no cycle info) are treated as oldest. Returns number
        removed.
        """
        if len(self.anchored_items) <= max_count:
            return 0
        sorted_items = sorted(
            self.anchored_items,
            key=lambda it: (
                0 if it.status == 'retracted' else 1,
                it.created_at_cycle,
            ),
        )
        to_remove = sorted_items[:len(self.anchored_items) - max_count]
        remove_ids = {it.memory_id for it in to_remove}
        self.anchored_items = [
            it for it in self.anchored_items if it.memory_id not in remove_ids
        ]
        return len(remove_ids)

    def _should_abstract_entry(self, entry: MemoryEntry, elapsed: int) -> bool:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return False
        return bool(
            elapsed >= 8
            and (
                entry.retrieval_count <= 1
                or entry.support_count <= 1
                or entry.abstractness >= 0.60
                or entry.is_dormant
            )
        )

    def _source_confidence_drift(self, entry: MemoryEntry, elapsed: int) -> float:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return 0.0
        pressure = 0.0
        if entry.source_type is SourceType.RECONSTRUCTION:
            pressure += 0.0009 * elapsed
        if entry.support_count <= 1:
            pressure += 0.0005 * elapsed
        if _has_source_conflict(entry):
            pressure += 0.0012 * elapsed
        return min(0.18, pressure)

    def _reality_confidence_drift(self, entry: MemoryEntry, elapsed: int) -> float:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return 0.0
        pressure = 0.0
        if entry.counterevidence_count > 0:
            pressure += min(0.12, 0.01 * entry.counterevidence_count) + (0.001 * elapsed)
        if bool(entry.competing_interpretations):
            pressure += 0.0008 * elapsed
        if _has_reality_conflict(entry):
            pressure += 0.0012 * elapsed
        if entry.abstractness >= 0.75 and entry.retrieval_count == 0:
            pressure += 0.0005 * elapsed
        return min(0.22, pressure)

    def apply_decay(self, current_cycle: int) -> DecayReport:
        report = DecayReport(current_cycle=current_cycle)
        for entry in self.entries:
            report.processed_entries += 1
            internal, baseline_cycle, base_trace, base_access = _ensure_decay_baseline(entry)
            elapsed = max(0, current_cycle - baseline_cycle)
            if elapsed <= 0:
                continue
            entry.trace_strength = decay_trace_strength_for_level(
                base_trace,
                entry.store_level,
                elapsed,
                memory_class=entry.memory_class,
            )
            entry.accessibility = decay_accessibility_for_level(base_access, entry.store_level, elapsed)
            if self._should_abstract_entry(entry, elapsed):
                new_abstractness = _clamp(entry.abstractness + min(0.16, 0.002 * elapsed))
                if new_abstractness > entry.abstractness:
                    entry.abstractness = new_abstractness
                    report.abstracted_entries.append(entry.id)
            source_before = entry.source_confidence
            reality_before = entry.reality_confidence
            source_drift = self._source_confidence_drift(entry, elapsed)
            reality_drift = self._reality_confidence_drift(entry, elapsed)
            if source_drift > 0.0:
                entry.source_confidence = _clamp(entry.source_confidence - source_drift)
            if reality_drift > 0.0:
                entry.reality_confidence = _clamp(entry.reality_confidence - reality_drift)
            if entry.source_confidence != source_before:
                report.source_confidence_drifted.append(entry.id)
            if entry.reality_confidence != reality_before:
                report.reality_confidence_drifted.append(entry.id)
            if entry.source_confidence != source_before or entry.reality_confidence != reality_before:
                report.confidence_drifted.append(entry.id)
            if (
                entry.store_level is StoreLevel.LONG
                and entry.trace_strength <= LONG_DORMANT_TRACE_THRESHOLD
                and entry.accessibility <= LONG_DORMANT_ACCESS_THRESHOLD
            ):
                entry.is_dormant = True
                report.dormant_marked.append(entry.id)
            _refresh_decay_baseline(entry, cycle=current_cycle)
        before_cleanup = {entry.id for entry in self.entries}
        self.cleanup_short()
        after_cleanup = {entry.id for entry in self.entries}
        report.deleted_short_residue = sorted(before_cleanup - after_cleanup)
        # M8.5: prune anchored items during decay cycle
        anchored_before = len(self.anchored_items)
        report.anchored_pruned = self.prune_anchored(current_cycle=current_cycle)
        return report

    def delete_entry(self, entry_id: str) -> bool:
        index = self._find_index(entry_id)
        if index is None:
            return False
        self.entries.pop(index)
        return True

    def archive_entry(self, entry_id: str, *, archive_cycle: int, reason: str = "") -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.store_level = StoreLevel.DORMANT
        cm = dict(entry.compression_metadata or {})
        cm["archived_at_cycle"] = archive_cycle
        cm["archive_reason"] = reason
        entry.compression_metadata = cm
        return True

    def replay_sample(
        self,
        rng,
        *,
        batch_size: int | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        if not self.entries:
            return []
        size = batch_size or 6
        latest_cycle = max(e.created_at for e in self.entries)
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self.entries:
            recency_weight = 1.0 / (1.0 + max(0.0, latest_cycle - entry.created_at))
            priority = entry.salience + entry.novelty + recency_weight
            scored.append((priority, entry))
        scored.sort(key=lambda item: -item[0])
        sample_size = min(size, len(scored))
        high_priority_count = min(len(scored), int(round(sample_size * 0.8)))
        high_priority = [entry for _, entry in scored[:high_priority_count]]
        remaining = [entry for _, entry in scored[high_priority_count:]]
        random_count = sample_size - len(high_priority)
        if remaining and random_count > 0:
            random_replay = rng.sample(remaining, min(random_count, len(remaining)))
        else:
            random_replay = []
        result = high_priority + random_replay
        if limit is not None:
            result = result[: max(0, int(limit))]
        return result

    def assign_clusters(self, cluster_distance_threshold: float = 0.65) -> int:
        if not self.entries:
            return 0
        created = 0
        centroids: list[list[float]] = []
        for entry in sorted(self.entries, key=lambda e: e.created_at):
            if entry.cluster_id is not None:
                continue
            embedding = list(entry.embedding) if entry.embedding else [0.0] * 32
            best_distance = float("inf")
            best_cluster = -1
            for cid, centroid in enumerate(centroids):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, centroid)))
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = cid
            if best_distance < cluster_distance_threshold and best_cluster >= 0:
                entry.cluster_id = best_cluster
            else:
                entry.cluster_id = len(centroids)
                centroids.append(list(embedding))
                created += 1
        return created

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "short_to_mid_salience_threshold": self.short_to_mid_salience_threshold,
            "short_to_mid_retrieval_threshold": self.short_to_mid_retrieval_threshold,
            "mid_to_long_salience_threshold": self.mid_to_long_salience_threshold,
            "mid_to_long_retrieval_threshold": self.mid_to_long_retrieval_threshold,
            "identity_priority_threshold": self.identity_priority_threshold,
            "identity_long_threshold": self.identity_long_threshold,
            "state_window_size": self.state_window_size,
            "agent_state_vector": dict(self.agent_state_vector or {}),
            "last_cleanup_report": dict(self.last_cleanup_report),
            "semantic_schemas": [dict(s) for s in self.semantic_schemas if isinstance(s, dict)],
            "latest_schema_update": dict(self.latest_schema_update),
            "anchored_items": [it.to_dict() for it in self.anchored_items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MemoryStore:
        return cls(
            entries=[
                MemoryEntry.from_dict(item)
                for item in payload.get("entries", [])
                if isinstance(item, dict)
            ],
            short_to_mid_salience_threshold=_coerce_float(
                payload.get("short_to_mid_salience_threshold"), 0.55
            ),
            short_to_mid_retrieval_threshold=_coerce_int(
                payload.get("short_to_mid_retrieval_threshold"), 2
            ),
            mid_to_long_salience_threshold=_coerce_float(
                payload.get("mid_to_long_salience_threshold"), 0.80
            ),
            mid_to_long_retrieval_threshold=_coerce_int(
                payload.get("mid_to_long_retrieval_threshold"), 3
            ),
            identity_priority_threshold=_coerce_float(
                payload.get("identity_priority_threshold"), 0.75
            ),
            identity_long_threshold=_coerce_float(
                payload.get("identity_long_threshold"), 0.85
            ),
            state_window_size=_coerce_int(payload.get("state_window_size"), 30),
            agent_state_vector=dict(payload.get("agent_state_vector", {}))
            if isinstance(payload.get("agent_state_vector"), dict)
            else None,
            last_cleanup_report=dict(payload.get("last_cleanup_report", {}))
            if isinstance(payload.get("last_cleanup_report"), dict)
            else {},
            semantic_schemas=[
                dict(s) for s in payload.get("semantic_schemas", [])
                if isinstance(s, dict)
            ],
            latest_schema_update=dict(payload.get("latest_schema_update", {}))
            if isinstance(payload.get("latest_schema_update"), dict)
            else {},
            anchored_items=[
                AnchoredMemoryItem.from_dict(it)
                for it in payload.get("anchored_items", [])
                if isinstance(it, dict)
            ],
        )

    @classmethod
    def from_legacy_episodes(cls, episodes: list[dict[str, object]]) -> MemoryStore:
        store = cls()
        store.replace_legacy_group(episodes)
        return store

    def _entry_to_legacy_payload(self, entry: MemoryEntry) -> dict[str, object]:
        template = _legacy_template(entry)
        extras = _legacy_unmapped(entry)
        action = entry.anchor_slots.get("action") or str(
            template.get("action", template.get("action_taken", "unknown"))
        )
        outcome_text = entry.anchor_slots.get("outcome") or str(
            template.get("predicted_outcome", template.get("value_label", entry.content))
        )
        base_payload = {
            "episode_id": entry.id,
            "timestamp": entry.created_at,
            "cycle": entry.created_at,
            "state_vector": deepcopy(template.get("state_vector"))
            if isinstance(template.get("state_vector"), dict)
            else {},
            "state_snapshot": deepcopy(template.get("state_snapshot"))
            if isinstance(template.get("state_snapshot"), dict)
            else {},
            "action_taken": _update_legacy_action_payload(template.get("action_taken"), action),
            "action": action,
            "outcome_state": _update_legacy_outcome_payload(template.get("outcome_state"), outcome_text),
            "outcome": _update_legacy_outcome_payload(template.get("outcome"), outcome_text),
            "predicted_outcome": outcome_text,
            "prediction_error": entry.novelty,
            "risk": _coerce_float(template.get("risk", 0.0)),
            "value_score": entry.valence,
            "total_surprise": entry.salience,
            "weighted_surprise": entry.salience,
            "embedding": deepcopy(template.get("embedding"))
            if isinstance(template.get("embedding"), list)
            else (list(entry.state_vector) if entry.state_vector is not None else []),
            "value_label": outcome_text,
            "preferred_probability": _coerce_float(template.get("preferred_probability", 0.0)),
            "preference_log_value": _coerce_float(template.get("preference_log_value", 0.0)),
            "observation": deepcopy(template.get("observation"))
            if isinstance(template.get("observation"), dict)
            else {},
            "prediction": deepcopy(template.get("prediction"))
            if isinstance(template.get("prediction"), dict)
            else {},
            "errors": deepcopy(template.get("errors"))
            if isinstance(template.get("errors"), dict)
            else {},
            "body_state": deepcopy(template.get("body_state"))
            if isinstance(template.get("body_state"), dict)
            else {},
            "support_count": entry.support_count,
            "support": entry.support_count,
            "last_seen_cycle": entry.last_accessed,
            "identity_critical": entry.relevance_self >= self.identity_priority_threshold,
            "continuity_tags": list(entry.semantic_tags),
            "gating_reasons": list(entry.context_tags),
            "lifecycle_stage": template.get("lifecycle_stage", "validated_episode"),
            "episode_family": template.get("episode_family", ""),
            "content": entry.content,
            "memory_class": entry.memory_class.value,
            "source_type": entry.source_type.value,
            "store_level": entry.store_level.value,
            "abstractness": entry.abstractness,
            "encoding_source": dict(entry.compression_metadata or {}).get("encoding_source"),
            "encoding_strength": dict(entry.compression_metadata or {}).get("encoding_strength", entry.salience),
            "fep_prediction_error": dict(entry.compression_metadata or {}).get(
                "fep_prediction_error",
                entry.novelty,
            ),
            "surprise": dict(entry.compression_metadata or {}).get("surprise", entry.salience),
            "attention_budget_total": dict(entry.compression_metadata or {}).get("attention_budget_total"),
            "attention_budget_requested": dict(entry.compression_metadata or {}).get("attention_budget_requested"),
            "attention_budget_granted": dict(entry.compression_metadata or {}).get("attention_budget_granted"),
            "attention_budget_denied": dict(entry.compression_metadata or {}).get("attention_budget_denied"),
            "centroid": list(entry.centroid) if entry.centroid is not None else None,
            "residual_norm_mean": entry.residual_norm_mean,
            "residual_norm_var": entry.residual_norm_var,
            "support_ids": list(entry.support_ids) if entry.support_ids is not None else None,
            "consolidation_source": entry.consolidation_source,
            "semantic_reconstruction_error": entry.semantic_reconstruction_error,
            "replay_second_pass_error": entry.replay_second_pass_error,
            "salience_delta": entry.salience_delta,
            "retention_adjustment": entry.retention_adjustment,
            "compression_metadata": _sanitize_legacy_compression_metadata(entry.compression_metadata),
        }
        for key, value in extras.items():
            if key in LEGACY_MAPPED_KEYS:
                continue
            base_payload[key] = deepcopy(value)
        return base_payload

    def to_legacy_episodes(self, *, entry_ids: set[str] | None = None) -> list[dict[str, object]]:
        entries = self.entries if entry_ids is None else [e for e in self.entries if e.id in entry_ids]
        return [self._entry_to_legacy_payload(entry) for entry in entries]

    def legacy_payloads_for_entries(self, entries: list[MemoryEntry]) -> dict[str, dict[str, object]]:
        """Build legacy payloads only for the given entries, keyed by entry id."""
        return {entry.id: self._entry_to_legacy_payload(entry) for entry in entries}
