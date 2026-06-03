from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
from math import sqrt
from typing import TYPE_CHECKING, Any, Mapping

from .memory_encoding import EncodingDynamics
from .memory_model import (
    AnchorStrength,
    MemoryClass,
    MemoryEntry,
    MemoryPath,
    PathCueSignature,
    PathOutcomeProfile,
    PathRiskProfile,
    SourceType,
    StoreLevel,
)
from .memory_state import identity_match_ratio_for_entry, normalize_agent_state
from .memory_retrieval import RecallArtifact

if TYPE_CHECKING:
    from .memory_store import MemoryStore


RECONSTRUCTION_ABSTRACT_THRESHOLD = 0.70
RECONSTRUCTION_CONTENT_MIN_LENGTH = 50
RECONSTRUCTION_CONFIDENCE_THRESHOLD = 0.40
BOOST_ACCESS = 0.20
BOOST_TRACE = 0.03
ABSTRACTNESS_INCREMENT = 0.008
DEFAULT_MINIMUM_SUPPORT = 5
DEFAULT_RUNTIME_MINIMUM_SUPPORT = 3
DEFAULT_SMOOTHING = 2.0
PATTERN_BRIDGE_MIN_SCORE = 0.42
PATTERN_SEMANTIC_MIN_OVERLAP = 0.20
PATTERN_CONTEXT_MIN_OVERLAP = 0.15
PATTERN_VECTOR_DISTANCE_MAX = 1.05
LOW_SURPRISE_MAX = 0.20
MEDIUM_SURPRISE_MAX = 0.45
HIGH_SURPRISE_MIN = 0.45
SOURCE_REWRITE_FLOOR = 0.70
IDENTITY_REWRITE_FLOOR = 0.80


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _style_value(current_state: dict[str, object] | None, key: str, default: float = 0.0) -> float:
    if not current_state:
        return default
    candidate = current_state.get("cognitive_style")
    if hasattr(candidate, key):
        try:
            return float(getattr(candidate, key))
        except (TypeError, ValueError):
            return default
    if isinstance(candidate, dict):
        try:
            return float(candidate.get(key, default))
        except (TypeError, ValueError):
            return default
    try:
        return float(current_state.get(key, default))
    except (TypeError, ValueError):
        return default


def _shared_semantic_overlap(left: MemoryEntry, right: MemoryEntry) -> float:
    left_tags = {item.lower() for item in left.semantic_tags}
    right_tags = {item.lower() for item in right.semantic_tags}
    if not left_tags or not right_tags:
        return 0.0
    return len(left_tags & right_tags) / max(len(left_tags | right_tags), 1)


def _shared_context_overlap(left: MemoryEntry, right: MemoryEntry) -> float:
    left_tags = {item.lower() for item in left.context_tags}
    right_tags = {item.lower() for item in right.context_tags}
    if not left_tags or not right_tags:
        return 0.0
    return len(left_tags & right_tags) / max(len(left_tags | right_tags), 1)


def _entry_distance(left: MemoryEntry, right: MemoryEntry) -> float:
    left_vector = _entry_vector(left)
    right_vector = _entry_vector(right)
    width = min(len(left_vector), len(right_vector))
    if width <= 0:
        return float("inf")
    return _residual_norm(left_vector[:width], right_vector[:width])


def _entry_similarity(left: MemoryEntry, right: MemoryEntry) -> float:
    distance = _entry_distance(left, right)
    if distance == float("inf"):
        return 0.0
    return 1.0 / (1.0 + distance)


def _bridge_score(left: MemoryEntry, right: MemoryEntry) -> float:
    semantic = _shared_semantic_overlap(left, right)
    context = _shared_context_overlap(left, right)
    vector_similarity = _entry_similarity(left, right)
    return (semantic * 0.55) + (context * 0.15) + (vector_similarity * 0.30)


def _pattern_neighbor(left: MemoryEntry, right: MemoryEntry) -> bool:
    semantic = _shared_semantic_overlap(left, right)
    context = _shared_context_overlap(left, right)
    distance = _entry_distance(left, right)
    vector_similarity = _entry_similarity(left, right)
    if semantic >= 0.50:
        return True
    if semantic >= PATTERN_SEMANTIC_MIN_OVERLAP and distance <= PATTERN_VECTOR_DISTANCE_MAX:
        return True
    if semantic >= PATTERN_SEMANTIC_MIN_OVERLAP and context >= PATTERN_CONTEXT_MIN_OVERLAP:
        return True
    return bool(
        _bridge_score(left, right) >= PATTERN_BRIDGE_MIN_SCORE
        and semantic > 0.0
        and vector_similarity >= 0.45
    )


def _copy_entry(entry: MemoryEntry) -> MemoryEntry:
    return MemoryEntry.from_dict(entry.to_dict())


def _semantic_vector_with_dynamic_axes(vector: list[float], entry: MemoryEntry) -> list[float]:
    return [
        *[float(item) for item in vector],
        float(entry.salience),
        float(entry.trace_strength),
        float(entry.accessibility),
    ]


def _entry_vector(entry: MemoryEntry) -> list[float]:
    if entry.centroid:
        return _semantic_vector_with_dynamic_axes([float(item) for item in entry.centroid], entry)
    if entry.state_vector:
        return _semantic_vector_with_dynamic_axes([float(item) for item in entry.state_vector], entry)
    metadata = dict(entry.compression_metadata or {})
    explicit = metadata.get("state_vector")
    if isinstance(explicit, list):
        return _semantic_vector_with_dynamic_axes([float(item) for item in explicit], entry)
    legacy = metadata.get("legacy_template")
    if isinstance(legacy, dict):
        embedding = legacy.get("embedding")
        if isinstance(embedding, list):
            return _semantic_vector_with_dynamic_axes([float(item) for item in embedding], entry)
        state_vector = legacy.get("state_vector")
        if isinstance(state_vector, dict):
            return _semantic_vector_with_dynamic_axes(
                [float(value) for _, value in sorted(state_vector.items()) if isinstance(value, (int, float))],
                entry,
            )
    values = [
        entry.valence,
        entry.arousal,
        entry.encoding_attention,
        entry.novelty,
        entry.relevance_goal,
        entry.relevance_threat,
        entry.relevance_self,
        entry.relevance_social,
        entry.relevance_reward,
        entry.relevance,
        entry.salience,
    ]
    return [float(value) for value in values]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors if vector)
    if width <= 0:
        return []
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def _residual_norm(vector: list[float], centroid: list[float]) -> float:
    width = min(len(vector), len(centroid))
    if width <= 0:
        return 0.0
    return sqrt(sum((float(vector[index]) - float(centroid[index])) ** 2 for index in range(width)))


def _vector_semantic_stats(entries: list[MemoryEntry]) -> dict[str, object]:
    vectors = [_entry_vector(entry) for entry in entries]
    vectors = [vector for vector in vectors if vector]
    centroid = _mean_vector(vectors)
    residuals = [_residual_norm(vector, centroid) for vector in vectors]
    residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
    residual_var = (
        sum((value - residual_mean) ** 2 for value in residuals) / len(residuals)
        if residuals
        else 0.0
    )
    return {
        "centroid": centroid,
        "residual_norm_mean": residual_mean,
        "residual_norm_var": residual_var,
        "support_ids": [entry.id for entry in entries],
        "residual_by_id": {
            entry.id: _residual_norm(_entry_vector(entry), centroid)
            for entry in entries
        },
    }


def _fold_residuals_into_sources(entries: list[MemoryEntry], stats: dict[str, object]) -> None:
    residual_by_id = dict(stats.get("residual_by_id", {}))
    for entry in entries:
        entry.semantic_reconstruction_error = float(residual_by_id.get(entry.id, 0.0))
        metadata = dict(entry.compression_metadata or {})
        metadata["semantic_reconstruction_error"] = entry.semantic_reconstruction_error
        entry.compression_metadata = metadata


def _apply_vector_semantic_fields(
    entry: MemoryEntry,
    sources: list[MemoryEntry],
    *,
    lineage_type: str,
) -> MemoryEntry:
    stats = _vector_semantic_stats(sources)
    _fold_residuals_into_sources(sources, stats)
    entry.centroid = list(stats["centroid"])
    entry.residual_norm_mean = float(stats["residual_norm_mean"])
    entry.residual_norm_var = float(stats["residual_norm_var"])
    entry.support_ids = list(stats["support_ids"])
    entry.consolidation_source = "dynamics"
    metadata = dict(entry.compression_metadata or {})
    metadata.update(
        {
            "centroid": list(entry.centroid or []),
            "residual_norm_mean": entry.residual_norm_mean,
            "residual_norm_var": entry.residual_norm_var,
            "support_ids": list(entry.support_ids or []),
            "consolidation_source": "dynamics",
            "lineage_type": lineage_type,
        }
    )
    entry.compression_metadata = metadata
    return entry


class ConflictType(str, Enum):
    FACTUAL = "factual"
    SOURCE = "source"
    INTERPRETIVE = "interpretive"


class ReconsolidationUpdateType(str, Enum):
    REINFORCEMENT_ONLY = "reinforcement_only"
    CONTEXTUAL_REBINDING = "contextual_rebinding"
    STRUCTURAL_RECONSTRUCTION = "structural_reconstruction"
    CONFLICT_MARKING = "conflict_marking"


@dataclass(frozen=True)
class MemoryReuseEvent:
    reuse_event_id: str
    memory_id: str
    path_id: str = ""
    prediction_before_reuse: dict[str, float] | None = None
    observation_after_reuse: dict[str, float] | None = None
    reuse_prediction_error: float = 0.0
    reuse_free_energy_delta: float = 0.0
    recall_confidence: float = 0.0
    contradiction_detected: bool = False
    live_reuse: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "reuse_event_id": self.reuse_event_id,
            "memory_id": self.memory_id,
            "path_id": self.path_id,
            "prediction_before_reuse": dict(self.prediction_before_reuse or {}),
            "observation_after_reuse": dict(self.observation_after_reuse or {}),
            "reuse_prediction_error": round(self.reuse_prediction_error, 6),
            "reuse_free_energy_delta": round(self.reuse_free_energy_delta, 6),
            "recall_confidence": round(self.recall_confidence, 6),
            "contradiction_detected": bool(self.contradiction_detected),
            "live_reuse": bool(self.live_reuse),
        }


@dataclass(frozen=True)
class ReconstructionConfig:
    abstract_threshold: float = RECONSTRUCTION_ABSTRACT_THRESHOLD
    content_min_length: int = RECONSTRUCTION_CONTENT_MIN_LENGTH
    confidence_threshold: float = RECONSTRUCTION_CONFIDENCE_THRESHOLD
    maximum_borrow_sources: int = 2
    current_cycle: int = 0
    current_state: dict[str, object] | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    entry: MemoryEntry
    triggered: bool
    trigger_reason: str | None
    borrowed_source_ids: list[str]
    reconstructed_fields: list[str]
    protected_fields: list[str]
    reconstruction_trace: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry.id,
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "borrowed_source_ids": list(self.borrowed_source_ids),
            "reconstructed_fields": list(self.reconstructed_fields),
            "protected_fields": list(self.protected_fields),
            "reconstruction_trace": deepcopy(self.reconstruction_trace),
        }


@dataclass(frozen=True)
class ReconsolidationReport:
    entry_id: str
    update_type: str
    fields_strengthened: list[str]
    fields_rebound: list[str]
    fields_reconstructed: list[str]
    conflict_flags: list[str]
    confidence_delta: dict[str, float]
    version_changed: bool
    reuse_event_id: str = ""
    reuse_surprise_level: str = ""
    reason_code: str = ""
    suppressed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "update_type": self.update_type,
            "fields_strengthened": list(self.fields_strengthened),
            "fields_rebound": list(self.fields_rebound),
            "fields_reconstructed": list(self.fields_reconstructed),
            "conflict_flags": list(self.conflict_flags),
            "confidence_delta": dict(self.confidence_delta),
            "version_changed": self.version_changed,
            "reuse_event_id": self.reuse_event_id,
            "reuse_surprise_level": self.reuse_surprise_level,
            "reason_code": self.reason_code,
            "suppressed": self.suppressed,
        }


@dataclass(frozen=True)
class ConflictResolution:
    conflict_type: str
    source_confidence_delta: float
    reality_confidence_delta: float
    counterevidence_delta: int
    competing_interpretations_added: list[str]
    dormant_shadow_created: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "conflict_type": self.conflict_type,
            "source_confidence_delta": self.source_confidence_delta,
            "reality_confidence_delta": self.reality_confidence_delta,
            "counterevidence_delta": self.counterevidence_delta,
            "competing_interpretations_added": list(self.competing_interpretations_added),
            "dormant_shadow_created": self.dormant_shadow_created,
        }


@dataclass(frozen=True)
class ValidationResult:
    entry_id: str
    score: float
    threshold: float
    passed: bool
    validation_status: str
    validation_discount: float

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "validation_status": self.validation_status,
            "validation_discount": self.validation_discount,
        }


@dataclass(frozen=True)
class UpgradeReport:
    promoted_ids: list[str]
    promotion_reasons: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted_ids": list(self.promoted_ids),
            "promotion_reasons": dict(self.promotion_reasons),
        }


@dataclass(frozen=True)
class CleanupReport:
    deleted_ids: list[str]
    dormant_ids: list[str]
    absorbed_ids: list[str]
    confidence_drift_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "deleted_ids": list(self.deleted_ids),
            "dormant_ids": list(self.dormant_ids),
            "absorbed_ids": list(self.absorbed_ids),
            "confidence_drift_ids": list(self.confidence_drift_ids),
        }


@dataclass(frozen=True)
class ConsolidationReport:
    upgrade: UpgradeReport
    extracted_patterns: list[str]
    replay_reencoded_ids: list[str]
    validated_inference_ids: list[str]
    cleanup: CleanupReport

    def to_dict(self) -> dict[str, object]:
        return {
            "upgrade": self.upgrade.to_dict(),
            "extracted_patterns": list(self.extracted_patterns),
            "replay_reencoded_ids": list(self.replay_reencoded_ids),
            "validated_inference_ids": list(self.validated_inference_ids),
            "cleanup": self.cleanup.to_dict(),
        }


def _protected_fields(entry: MemoryEntry, current_state: dict[str, object] | None = None) -> list[str]:
    uncertainty_sensitivity = _style_value(current_state, "uncertainty_sensitivity", 0.0)
    protected: list[str] = []
    for key, strength in entry.anchor_strengths.items():
        if strength in {AnchorStrength.LOCKED, AnchorStrength.STRONG}:
            protected.append(key)
        elif uncertainty_sensitivity >= 0.8 and key in {"time", "place"} and strength is AnchorStrength.WEAK:
            protected.append(key)
    if entry.memory_class is MemoryClass.PROCEDURAL:
        protected.append("procedure_steps")
    return sorted(set(protected))


def _trigger_reason(entry: MemoryEntry, config: ReconstructionConfig) -> str | None:
    rigidity_penalty = _style_value(config.current_state, "update_rigidity", 0.0) * 0.10
    abstract_threshold = min(0.95, config.abstract_threshold + rigidity_penalty)
    if entry.abstractness > abstract_threshold and entry.memory_class is MemoryClass.SEMANTIC:
        return "semantic_abstractness"
    if entry.abstractness > abstract_threshold and len(entry.content) < config.content_min_length:
        return "abstract_short_content"
    if entry.reality_confidence < max(0.10, config.confidence_threshold - (rigidity_penalty * 0.5)) and entry.retrieval_count > 0:
        return "low_reality_after_retrieval"
    return None


def _borrow_candidates(
    primary: MemoryEntry,
    candidates: list[MemoryEntry],
    store: "MemoryStore",
    config: ReconstructionConfig,
) -> list[MemoryEntry]:
    derived_ids = set(primary.derived_from)
    ranked: list[tuple[tuple[float, float, float, float, float, float], MemoryEntry]] = []
    for entry in candidates:
        if entry.id == primary.id:
            continue
        validation_status = str(dict(entry.compression_metadata or {}).get("validation_status", "validated"))
        if entry.memory_class is MemoryClass.INFERRED and validation_status in {"unvalidated", "contradicted"}:
            continue
        derived_score = 1.0 if entry.id in derived_ids else 0.0
        semantic_score = _shared_semantic_overlap(primary, entry)
        bridge_score = _bridge_score(primary, entry)
        vector_score = _entry_similarity(primary, entry)
        mood_score = 1.0 if primary.mood_context and primary.mood_context == entry.mood_context else 0.0
        context_score = _shared_context_overlap(primary, entry)
        if derived_score <= 0.0 and semantic_score < 0.15 and vector_score < 0.45 and context_score < 0.15:
            continue
        ranked.append(
            (
                (
                    derived_score,
                    bridge_score,
                    semantic_score,
                    vector_score,
                    mood_score,
                    context_score + min(1.0, entry.support_count / 6.0),
                ),
                entry,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked[: config.maximum_borrow_sources]]


def _apply_reconstruction(
    primary: MemoryEntry,
    borrow_sources: list[MemoryEntry],
    config: ReconstructionConfig,
) -> tuple[MemoryEntry, list[str], list[str]]:
    reconstructed = _copy_entry(primary)
    protected_fields = _protected_fields(primary, config.current_state)
    reconstructed_fields: list[str] = []
    if primary.memory_class is MemoryClass.PROCEDURAL:
        reconstructed_fields.append("procedure_outline")
        for source in borrow_sources:
            for context in source.execution_contexts:
                if context not in reconstructed.execution_contexts:
                    reconstructed.execution_contexts.append(context)
        if borrow_sources:
            reconstructed.content = (
                f"{primary.content} | reconstructed with {len(borrow_sources)} procedural supports"
            )
    else:
        for source in borrow_sources:
            for slot, value in source.anchor_slots.items():
                if not value or slot in protected_fields or reconstructed.anchor_slots.get(slot):
                    continue
                reconstructed.anchor_slots[slot] = value
                reconstructed_fields.append(slot)
        if borrow_sources:
            borrowed_summaries = ", ".join(source.id for source in borrow_sources)
            reconstructed.content = f"{primary.content} | reconstructed from {borrowed_summaries}"
            reconstructed_fields.append("content")
        if primary.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED} and borrow_sources:
            reconstructed.abstractness = _clamp(primary.abstractness + 0.03)
    reconstructed.source_type = SourceType.RECONSTRUCTION
    reconstructed.reality_confidence = _clamp(primary.reality_confidence - 0.10)
    reconstructed.last_accessed = max(reconstructed.last_accessed, config.current_cycle)
    reconstructed.sync_content_hash()
    return reconstructed, reconstructed_fields, protected_fields


def maybe_reconstruct(
    primary: MemoryEntry,
    candidates: list[MemoryEntry],
    store: "MemoryStore",
    config: ReconstructionConfig,
) -> ReconstructionResult:
    trigger_reason = _trigger_reason(primary, config)
    if trigger_reason is None:
        return ReconstructionResult(
            entry=primary,
            triggered=False,
            trigger_reason=None,
            borrowed_source_ids=[],
            reconstructed_fields=[],
            protected_fields=_protected_fields(primary, config.current_state),
            reconstruction_trace={"triggered": False},
        )
    borrow_sources = _borrow_candidates(primary, candidates, store, config)
    reconstructed, reconstructed_fields, protected_fields = _apply_reconstruction(
        primary,
        borrow_sources,
        config,
    )
    reconstruction_trace = {
        "triggered": True,
        "trigger_reason": trigger_reason,
        "primary_id": primary.id,
        "borrowed_source_ids": [entry.id for entry in borrow_sources],
        "reconstructed_fields": list(reconstructed_fields),
        "protected_fields": list(protected_fields),
    }
    metadata = dict(reconstructed.compression_metadata or {})
    metadata["reconstruction_trace"] = reconstruction_trace
    reconstructed.compression_metadata = metadata
    return ReconstructionResult(
        entry=reconstructed,
        triggered=True,
        trigger_reason=trigger_reason,
        borrowed_source_ids=[entry.id for entry in borrow_sources],
        reconstructed_fields=reconstructed_fields,
        protected_fields=protected_fields,
        reconstruction_trace=reconstruction_trace,
    )


def resolve_conflict(
    existing: MemoryEntry,
    incoming: MemoryEntry | RecallArtifact | Mapping[str, object] | None,
    conflict_type: ConflictType,
) -> ConflictResolution:
    if isinstance(incoming, RecallArtifact):
        interpretation = f"recall:{incoming.primary_entry_id}"
    elif isinstance(incoming, Mapping):
        interpretation = f"recall:{str(incoming.get('primary_entry_id', 'unknown'))}"
    elif incoming is not None:
        interpretation = f"entry:{incoming.id}"
    else:
        interpretation = "recall:unknown"
    if conflict_type is ConflictType.FACTUAL:
        return ConflictResolution(
            conflict_type=conflict_type.value,
            source_confidence_delta=0.0,
            reality_confidence_delta=-0.18,
            counterevidence_delta=1,
            competing_interpretations_added=[interpretation],
            dormant_shadow_created=False,
        )
    if conflict_type is ConflictType.SOURCE:
        return ConflictResolution(
            conflict_type=conflict_type.value,
            source_confidence_delta=-0.20,
            reality_confidence_delta=-0.04,
            counterevidence_delta=0,
            competing_interpretations_added=[interpretation],
            dormant_shadow_created=False,
        )
    return ConflictResolution(
        conflict_type=conflict_type.value,
        source_confidence_delta=0.0,
        reality_confidence_delta=0.0,
        counterevidence_delta=0,
        competing_interpretations_added=[interpretation],
        dormant_shadow_created=False,
    )


def _entry_is_identity_critical(entry: MemoryEntry) -> bool:
    if entry.relevance_self >= 0.60:
        return True
    metadata = dict(entry.compression_metadata or {})
    legacy = metadata.get("legacy_template")
    if isinstance(legacy, dict) and bool(legacy.get("identity_critical", False)):
        return True
    return False


def _entry_is_source_sensitive(entry: MemoryEntry) -> bool:
    metadata = dict(entry.compression_metadata or {})
    if bool(metadata.get("source_conflict")) or bool(metadata.get("source_sensitive")):
        return True
    if entry.source_type in {SourceType.HEARSAY, SourceType.RECONSTRUCTION}:
        return True
    return False


def _reuse_surprise_score(event: MemoryReuseEvent) -> float:
    mismatch = max(0.0, float(event.reuse_prediction_error))
    destabilization = max(0.0, -float(event.reuse_free_energy_delta))
    confidence_penalty = max(0.0, 1.0 - float(event.recall_confidence)) * 0.15
    contradiction_bonus = 0.35 if event.contradiction_detected else 0.0
    return _clamp(max(mismatch, destabilization) + confidence_penalty + contradiction_bonus)


def classify_reuse_surprise(event: MemoryReuseEvent) -> str:
    score = _reuse_surprise_score(event)
    if event.contradiction_detected:
        return "conflict"
    if score < LOW_SURPRISE_MAX:
        return "low"
    if score < MEDIUM_SURPRISE_MAX:
        return "medium"
    return "high"


def reconsolidate(
    entry: MemoryEntry,
    current_mood: str | None,
    current_context_tags: list[str] | None,
    *,
    store: "MemoryStore | None" = None,
    current_cycle: int | None = None,
    current_state: dict[str, object] | None = None,
    recall_artifact: RecallArtifact | None = None,
    conflict_type: ConflictType | None = None,
    cognitive_style=None,
    reuse_event: MemoryReuseEvent | None = None,
) -> ReconsolidationReport:
    before_version = entry.version
    before_source_confidence = entry.source_confidence
    before_reality_confidence = entry.reality_confidence
    fields_strengthened = ["accessibility", "trace_strength", "retrieval_count", "last_accessed", "abstractness"]
    fields_rebound: list[str] = []
    fields_reconstructed: list[str] = []
    conflict_flags: list[str] = []
    update_rigidity = _clamp(_style_value(current_state, "update_rigidity", 0.0))
    error_aversion = _clamp(_style_value(current_state, "error_aversion", 0.0))
    normalized_state = normalize_agent_state(current_state)
    identity_match = identity_match_ratio_for_entry(entry, normalized_state)
    effective_boost_access = BOOST_ACCESS * (1.0 - (update_rigidity * 0.3))
    if identity_match > 0.0 and entry.relevance_self >= 0.35:
        effective_boost_access *= max(0.75, 1.0 - (identity_match * 0.25))
    if error_aversion >= 0.60 and entry.valence < 0.0:
        effective_boost_access *= 1.05

    update_type = ReconsolidationUpdateType.REINFORCEMENT_ONLY
    reuse_surprise_level = ""
    reason_code = ""
    suppressed = False
    if reuse_event is not None:
        metadata = dict(entry.compression_metadata or {})
        reuse_gate = dict(metadata.get("m17_reuse_gate", {}))
        processed_ids = [
            str(item) for item in reuse_gate.get("processed_event_ids", []) if str(item)
        ]
        if not reuse_event.live_reuse:
            suppressed = True
            reason_code = "background_access_not_live_reuse"
        elif reuse_event.reuse_event_id in processed_ids:
            suppressed = True
            reason_code = "duplicate_reuse_event"
        reuse_surprise_level = classify_reuse_surprise(reuse_event)
        if suppressed:
            return ReconsolidationReport(
                entry_id=entry.id,
                update_type=update_type.value,
                fields_strengthened=[],
                fields_rebound=[],
                fields_reconstructed=[],
                conflict_flags=[],
                confidence_delta={"source_confidence": 0.0, "reality_confidence": 0.0},
                version_changed=False,
                reuse_event_id=reuse_event.reuse_event_id,
                reuse_surprise_level=reuse_surprise_level,
                reason_code=reason_code,
                suppressed=True,
            )
        reason_code = {
            "low": "low_surprise_reinforcement",
            "medium": "medium_surprise_context_rebinding",
            "high": "high_surprise_candidate_reconstruction",
            "conflict": "explicit_conflict_marking",
        }.get(reuse_surprise_level, "")
        reuse_gate["last_event"] = reuse_event.to_dict()
        reuse_gate["last_surprise_level"] = reuse_surprise_level
        reuse_gate["processed_event_ids"] = [*processed_ids, reuse_event.reuse_event_id][-64:]
        metadata["m17_reuse_gate"] = reuse_gate
        entry.compression_metadata = metadata

    entry.accessibility = _clamp(entry.accessibility + effective_boost_access)
    entry.trace_strength = _clamp(entry.trace_strength + BOOST_TRACE)
    entry.retrieval_count += 1
    entry.abstractness = _clamp(entry.abstractness + ABSTRACTNESS_INCREMENT)
    if current_cycle is not None:
        entry.last_accessed = max(entry.last_accessed, int(current_cycle))

    effective_conflict_type = conflict_type
    if reuse_event is not None and reuse_event.contradiction_detected and effective_conflict_type is None:
        effective_conflict_type = ConflictType.FACTUAL

    if effective_conflict_type is not None:
        resolution = resolve_conflict(entry, recall_artifact, effective_conflict_type)
        entry.source_confidence = _clamp(entry.source_confidence + resolution.source_confidence_delta)
        entry.reality_confidence = _clamp(entry.reality_confidence + resolution.reality_confidence_delta)
        entry.counterevidence_count += resolution.counterevidence_delta
        interpretations = list(entry.competing_interpretations or [])
        for interpretation in resolution.competing_interpretations_added:
            if interpretation not in interpretations:
                interpretations.append(interpretation)
        entry.competing_interpretations = interpretations or None
        conflict_flags.append(effective_conflict_type.value)
        update_type = ReconsolidationUpdateType.CONFLICT_MARKING
    elif reuse_event is not None and reuse_surprise_level == "medium":
        if current_mood and current_mood != entry.mood_context:
            entry.mood_context = current_mood
            fields_rebound.append("mood_context")
        if current_context_tags:
            merged_contexts = list(dict.fromkeys([*entry.context_tags, *_string_list(current_context_tags)]))
            if merged_contexts != entry.context_tags:
                entry.context_tags = merged_contexts
                fields_rebound.append("context_tags")
        if fields_rebound:
            update_type = ReconsolidationUpdateType.CONTEXTUAL_REBINDING
    elif reuse_event is None and current_mood and current_mood != entry.mood_context:
        entry.mood_context = current_mood
        fields_rebound.append("mood_context")
        update_type = ReconsolidationUpdateType.CONTEXTUAL_REBINDING
    if reuse_event is None and current_context_tags:
        merged_contexts = list(dict.fromkeys([*entry.context_tags, *_string_list(current_context_tags)]))
        if merged_contexts != entry.context_tags:
            entry.context_tags = merged_contexts
            fields_rebound.append("context_tags")
            if update_type is ReconsolidationUpdateType.REINFORCEMENT_ONLY:
                update_type = ReconsolidationUpdateType.CONTEXTUAL_REBINDING

    if store is not None and effective_conflict_type is None:
        config = ReconstructionConfig(
            current_cycle=current_cycle or entry.last_accessed,
            current_state=current_state,
        )
        reconstruction_blocked = update_rigidity >= 0.85
        if identity_match >= 0.72 and entry.relevance_self >= 0.60:
            reconstruction_blocked = True
        if error_aversion >= 0.60 and entry.valence < 0.0:
            reconstruction_blocked = True
        if reuse_event is not None:
            surprise_score = _reuse_surprise_score(reuse_event)
            if reuse_surprise_level != "high":
                reconstruction_blocked = True
            if _entry_is_identity_critical(entry) and surprise_score < IDENTITY_REWRITE_FLOOR:
                reconstruction_blocked = True
                reason_code = "identity_rewrite_floor_not_met"
            if _entry_is_source_sensitive(entry) and surprise_score < SOURCE_REWRITE_FLOOR:
                reconstruction_blocked = True
                reason_code = "source_rewrite_floor_not_met"
        if not reconstruction_blocked:
            reconstruction = maybe_reconstruct(entry, store.entries, store, config)
            if reconstruction.triggered:
                reconstructed_entry = reconstruction.entry
                entry.content = reconstructed_entry.content
                entry.anchor_slots = reconstructed_entry.anchor_slots
                entry.execution_contexts = reconstructed_entry.execution_contexts
                entry.source_type = reconstructed_entry.source_type
                entry.reality_confidence = reconstructed_entry.reality_confidence
                entry.compression_metadata = reconstructed_entry.compression_metadata
                entry.content_hash = reconstructed_entry.content_hash
                entry.version = reconstructed_entry.version
                fields_reconstructed.extend(reconstruction.reconstructed_fields)
                update_type = ReconsolidationUpdateType.STRUCTURAL_RECONSTRUCTION
                if reuse_event is not None:
                    metadata = dict(entry.compression_metadata or {})
                    metadata["m17_reuse_gate"] = {
                        **dict(metadata.get("m17_reuse_gate", {})),
                        "last_reconstruction_event_id": reuse_event.reuse_event_id,
                        "last_surprise_level": reuse_surprise_level,
                    }
                    entry.compression_metadata = metadata
            elif reuse_event is not None and reuse_surprise_level == "high":
                reason_code = reason_code or "high_surprise_reconstruction_not_triggered"
        elif reuse_event is not None and reuse_surprise_level == "high" and not reason_code:
            reason_code = "high_surprise_reconstruction_blocked"
    if entry.memory_class is MemoryClass.PROCEDURAL:
        fields_reconstructed = [field_name for field_name in fields_reconstructed if field_name != "procedure_steps"]

    return ReconsolidationReport(
        entry_id=entry.id,
        update_type=update_type.value,
        fields_strengthened=fields_strengthened,
        fields_rebound=fields_rebound,
        fields_reconstructed=fields_reconstructed,
        conflict_flags=conflict_flags,
        confidence_delta={
            "source_confidence": round(entry.source_confidence - before_source_confidence, 6),
            "reality_confidence": round(entry.reality_confidence - before_reality_confidence, 6),
        },
        version_changed=entry.version != before_version,
        reuse_event_id=reuse_event.reuse_event_id if reuse_event is not None else "",
        reuse_surprise_level=reuse_surprise_level,
        reason_code=reason_code,
        suppressed=suppressed,
    )


def compress_episodic_cluster_to_semantic_skeleton(entries: list[MemoryEntry]) -> MemoryEntry:
    if not entries:
        raise ValueError("episodic cluster compression requires entries")
    support_ids = [entry.id for entry in entries]
    shared_semantic_tags = sorted(set(entries[0].semantic_tags).intersection(*[set(entry.semantic_tags) for entry in entries[1:]]))
    if not shared_semantic_tags:
        tag_counts: dict[str, int] = {}
        for entry in entries:
            for tag in entry.semantic_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        minimum_shared = max(2, int(round(len(entries) * 0.5)))
        shared_semantic_tags = [
            tag
            for tag, _count in sorted(
                tag_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if tag_counts[tag] >= minimum_shared
        ][:4]
    if not shared_semantic_tags:
        shared_semantic_tags = sorted({tag for entry in entries for tag in entry.semantic_tags[:3]})[:4]
    shared_context_tags = sorted(set(entries[0].context_tags).intersection(*[set(entry.context_tags) for entry in entries[1:]]))
    action_values = sorted({entry.anchor_slots.get("action") for entry in entries if entry.anchor_slots.get("action")})
    outcome_values = sorted({entry.anchor_slots.get("outcome") for entry in entries if entry.anchor_slots.get("outcome")})
    stable_structure = {
        "semantic_tags": list(shared_semantic_tags),
        "context_tags": list(shared_context_tags),
        "actions": action_values,
        "outcomes": outcome_values,
    }
    trigger_summary = ", ".join(shared_semantic_tags[:3] or shared_context_tags[:3] or ["recurring context"])
    action_summary = ", ".join(str(item) for item in action_values[:3] if item) or "reuse prior action pattern"
    outcome_summary = ", ".join(str(item) for item in outcome_values[:3] if item) or "future expectation shaping"
    skeleton_content = (
        f"Reusable action structure: when cues match [{trigger_summary}], "
        f"prefer actions [{action_summary}] to support [{outcome_summary}] within observed bounds."
    )
    identity_cluster = any(entry.relevance_self >= 0.6 for entry in entries)
    lineage_type = "identity_consolidation" if identity_cluster else "episodic_compression"
    semantic = MemoryEntry(
        content=skeleton_content,
        memory_class=MemoryClass.SEMANTIC,
        store_level=StoreLevel.MID,
        source_type=SourceType.EXPERIENCE,
        created_at=min(entry.created_at for entry in entries),
        last_accessed=max(entry.last_accessed for entry in entries),
        valence=sum(entry.valence for entry in entries) / len(entries),
        arousal=max(entry.arousal for entry in entries),
        encoding_attention=max(entry.encoding_attention for entry in entries),
        novelty=sum(entry.novelty for entry in entries) / len(entries),
        relevance_goal=max(entry.relevance_goal for entry in entries),
        relevance_threat=max(entry.relevance_threat for entry in entries),
        relevance_self=max(entry.relevance_self for entry in entries),
        relevance_social=max(entry.relevance_social for entry in entries),
        relevance_reward=max(entry.relevance_reward for entry in entries),
        relevance=max(entry.relevance for entry in entries),
        salience=max(entry.salience for entry in entries),
        trace_strength=0.78,
        accessibility=0.62,
        abstractness=0.84,
        source_confidence=0.88,
        reality_confidence=0.80,
        semantic_tags=shared_semantic_tags,
        context_tags=shared_context_tags,
        mood_context=entries[0].mood_context,
        retrieval_count=sum(entry.retrieval_count for entry in entries),
        support_count=len(entries),
        compression_metadata={
            "support_entry_ids": support_ids,
            "discarded_detail_types": ["time", "place", "single_episode_detail"],
            "stable_structure": stable_structure,
            "abstraction_reason": "stabilized pattern across episodic cluster",
            "predictive_use_cases": ["pattern-guided recall", "future expectation shaping"],
            "lineage_type": lineage_type,
            "display_content": skeleton_content,
            "content_role": "reusable_action_structure_summary",
            "behavior_inputs": ["centroid", "residual_norm_mean", "residual_norm_var", "support_ids"],
        },
        derived_from=support_ids,
    )
    return _apply_vector_semantic_fields(
        semantic,
        entries,
        lineage_type=lineage_type,
    )


def _group_pattern_candidates(
    store: "MemoryStore",
    *,
    minimum_support: int,
) -> list[list[MemoryEntry]]:
    eligible = [
        entry
        for entry in store.entries
        if entry.memory_class is not MemoryClass.PROCEDURAL and len(entry.semantic_tags) >= 2
    ]
    eligible.sort(key=lambda entry: (entry.created_at, entry.id))
    groups: list[list[MemoryEntry]] = []
    visited: set[str] = set()
    for seed in eligible:
        if seed.id in visited:
            continue
        component: list[MemoryEntry] = []
        frontier = [seed]
        visited.add(seed.id)
        while frontier:
            current = frontier.pop()
            component.append(current)
            for candidate in eligible:
                if candidate.id in visited:
                    continue
                if _pattern_neighbor(current, candidate):
                    visited.add(candidate.id)
                    frontier.append(candidate)
        if len(component) >= minimum_support:
            component.sort(key=lambda entry: (entry.created_at, entry.id))
            groups.append(component)
    groups.sort(
        key=lambda entries: (
            min(entry.created_at for entry in entries),
            "|".join(entry.id for entry in entries[:2]),
        )
    )
    return groups


def extract_patterns(
    store: "MemoryStore",
    *,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    smoothing: float = DEFAULT_SMOOTHING,
) -> list[MemoryEntry]:
    results: list[MemoryEntry] = []
    for group in _group_pattern_candidates(store, minimum_support=minimum_support):
        episodic_group = [entry for entry in group if entry.memory_class is MemoryClass.EPISODIC]
        if episodic_group:
            skeleton = compress_episodic_cluster_to_semantic_skeleton(episodic_group)
            results.append(skeleton)
            for source in episodic_group:
                metadata = dict(source.compression_metadata or {})
                metadata["absorbed_by"] = skeleton.id
                source.compression_metadata = metadata
        contradiction_count = sum(entry.counterevidence_count for entry in group)
        support_count = len(group)
        inferred = MemoryEntry(
            content="inferred centroid display",
            memory_class=MemoryClass.INFERRED,
            store_level=StoreLevel.MID,
            source_type=SourceType.INFERENCE,
            created_at=min(entry.created_at for entry in group),
            last_accessed=max(entry.last_accessed for entry in group),
            valence=sum(entry.valence for entry in group) / support_count,
            arousal=max(entry.arousal for entry in group),
            encoding_attention=max(entry.encoding_attention for entry in group),
            novelty=sum(entry.novelty for entry in group) / support_count,
            relevance_goal=max(entry.relevance_goal for entry in group),
            relevance_threat=max(entry.relevance_threat for entry in group),
            relevance_self=max(entry.relevance_self for entry in group),
            relevance_social=max(entry.relevance_social for entry in group),
            relevance_reward=max(entry.relevance_reward for entry in group),
            relevance=max(entry.relevance for entry in group),
            salience=max(entry.salience for entry in group),
            trace_strength=0.58,
            accessibility=0.50,
            abstractness=0.87,
            source_confidence=0.90,
            reality_confidence=_clamp(support_count / (support_count + contradiction_count + smoothing)),
            semantic_tags=sorted({tag for entry in group for tag in entry.semantic_tags})[:6],
            context_tags=sorted({tag for entry in group for tag in entry.context_tags})[:4],
            mood_context=group[0].mood_context,
            support_count=support_count,
            counterevidence_count=contradiction_count,
            competing_interpretations=[f"hypothesis:{entry.id}" for entry in group[:2]],
            compression_metadata={
                "support_entry_ids": [entry.id for entry in group],
                "stable_structure": {"semantic_tags": sorted({tag for entry in group for tag in entry.semantic_tags})[:6]},
                "discarded_detail_types": ["single_episode_detail"],
                "abstraction_reason": "candidate pattern inferred from repeated support",
                "predictive_use_cases": ["hypothesis candidate", "low-confidence planning hint"],
                "lineage_type": "pattern_extraction",
                "validation_status": "unvalidated",
                "validation_discount": 0.35,
                "display_content": "dynamics inferred centroid",
                "content_role": "metadata_display_only",
                "behavior_inputs": ["centroid", "residual_norm_mean", "residual_norm_var", "support_ids"],
            },
            derived_from=[entry.id for entry in group],
        )
        inferred = _apply_vector_semantic_fields(
            inferred,
            group,
            lineage_type="pattern_extraction",
        )
        results.append(inferred)
        break
    return results


def _path_legacy_template(entry: MemoryEntry) -> dict[str, object]:
    metadata = dict(entry.compression_metadata or {})
    payload = metadata.get("legacy_template")
    return dict(payload) if isinstance(payload, dict) else {}


def _path_action(entry: MemoryEntry) -> str:
    action = str(entry.anchor_slots.get("action") or "").strip().lower()
    if action:
        return action
    legacy = _path_legacy_template(entry)
    return str(legacy.get("action", legacy.get("action_taken", ""))).strip().lower()


def _path_outcome(entry: MemoryEntry) -> str:
    outcome = str(entry.anchor_slots.get("outcome") or "").strip().lower()
    if outcome:
        return outcome
    legacy = _path_legacy_template(entry)
    return str(
        legacy.get("predicted_outcome", legacy.get("value_label", entry.content))
    ).strip().lower()


def _path_effect_profile(entry: MemoryEntry) -> dict[str, float]:
    legacy = _path_legacy_template(entry)
    outcome = legacy.get("outcome")
    if isinstance(outcome, dict):
        return {
            str(key): float(value)
            for key, value in outcome.items()
            if isinstance(value, (int, float))
        }
    outcome_state = legacy.get("outcome_state")
    if isinstance(outcome_state, dict):
        return {
            str(key): float(value)
            for key, value in outcome_state.items()
            if isinstance(value, (int, float))
        }
    return {}


def _path_credit(entry: MemoryEntry) -> dict[str, object]:
    metadata = dict(entry.compression_metadata or {})
    payload = metadata.get("m17_memory_credit")
    return dict(payload) if isinstance(payload, dict) else {}


def _path_quality(
    *,
    support_count: int,
    confirmation_count: int,
    violation_count: int,
    future_path_utility: float,
    contradiction_burden: float,
    maintenance_cost: float,
    error_avoidance_gain: float,
) -> float:
    support_signal = min(1.0, float(support_count) / 4.0) * 0.28
    confirmation_total = max(1.0, float(confirmation_count + violation_count))
    confirmation_signal = (float(confirmation_count) / confirmation_total) * 0.28
    utility_signal = max(0.0, float(future_path_utility)) * 0.20
    avoidance_signal = max(0.0, float(error_avoidance_gain)) * 0.12
    contradiction_penalty = max(0.0, float(contradiction_burden)) * 0.35
    maintenance_penalty = max(0.0, float(maintenance_cost)) * 0.10
    raw = (
        0.12
        + support_signal
        + confirmation_signal
        + utility_signal
        + avoidance_signal
        - contradiction_penalty
        - maintenance_penalty
    )
    if violation_count > confirmation_count:
        raw -= min(0.18, (violation_count - confirmation_count) * 0.06)
    return _clamp(raw)


def _path_sensitive_channels(entries: list[MemoryEntry]) -> list[str]:
    totals: dict[str, float] = {}
    for entry in entries:
        legacy = _path_legacy_template(entry)
        for bucket_name in ("observation", "errors"):
            bucket = legacy.get(bucket_name)
            if not isinstance(bucket, dict):
                continue
            for key, value in bucket.items():
                if not isinstance(value, (int, float)):
                    continue
                if abs(float(value)) < 0.12:
                    continue
                totals[str(key)] = totals.get(str(key), 0.0) + abs(float(value))
    return [
        key
        for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ][:4]


def _path_cue_signature(entries: list[MemoryEntry]) -> PathCueSignature:
    semantic_counter: dict[str, int] = {}
    context_counter: dict[str, int] = {}
    for entry in entries:
        for tag in entry.semantic_tags:
            token = str(tag).strip().lower()
            if not token or token == _path_action(entry):
                continue
            semantic_counter[token] = semantic_counter.get(token, 0) + 1
        for tag in entry.context_tags:
            token = str(tag).strip().lower()
            if not token:
                continue
            context_counter[token] = context_counter.get(token, 0) + 1
    semantic_tags = [
        key
        for key, _ in sorted(semantic_counter.items(), key=lambda item: (-item[1], item[0]))
    ][:4]
    context_tags = [
        key
        for key, _ in sorted(context_counter.items(), key=lambda item: (-item[1], item[0]))
    ][:4]
    return PathCueSignature(
        semantic_tags=semantic_tags,
        context_tags=context_tags,
        sensitive_channels=_path_sensitive_channels(entries),
    )


def _path_component_id(action: str, cue_signature: PathCueSignature) -> str:
    seed = "|".join(
        [
            action,
            ",".join(cue_signature.semantic_tags[:3]),
            ",".join(cue_signature.context_tags[:3]),
            ",".join(cue_signature.sensitive_channels[:3]),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"path:{action}:{digest}"


def _path_polarity(
    *,
    future_path_utility: float,
    confirmation_count: int,
    violation_count: int,
    mean_risk: float,
    contradiction_burden: float,
    expected_surprise: float,
) -> str:
    if future_path_utility < -0.10 or violation_count > confirmation_count + 1:
        return "negative"
    if contradiction_burden >= 0.20 or mean_risk >= 0.45 or expected_surprise >= 0.45:
        return "cautionary"
    return "positive"


def derive_memory_paths(
    store: "MemoryStore",
    *,
    current_cycle: int,
) -> list[MemoryPath]:
    grouped_by_action: dict[str, list[MemoryEntry]] = {}
    for entry in store.entries:
        if entry.memory_class is not MemoryClass.EPISODIC or entry.is_dormant:
            continue
        metadata = dict(entry.compression_metadata or {})
        value_memory = metadata.get("value_memory")
        candidate_kind = value_memory.get("candidate_kind") if isinstance(value_memory, dict) else None
        if candidate_kind in {"quarantined_candidate", "rejected_candidate"}:
            continue
        action = _path_action(entry)
        if not action:
            continue
        grouped_by_action.setdefault(action, []).append(entry)

    paths: list[MemoryPath] = []
    for action, action_entries in sorted(grouped_by_action.items()):
        action_entries = sorted(action_entries, key=lambda item: (item.created_at, item.id))
        visited: set[str] = set()
        for seed in action_entries:
            if seed.id in visited:
                continue
            component: list[MemoryEntry] = []
            frontier = [seed]
            visited.add(seed.id)
            while frontier:
                current = frontier.pop()
                component.append(current)
                for candidate in action_entries:
                    if candidate.id in visited:
                        continue
                    if _pattern_neighbor(current, candidate):
                        visited.add(candidate.id)
                        frontier.append(candidate)
            if len(component) < 2:
                continue
            component.sort(key=lambda item: (item.created_at, item.id))
            cue_signature = _path_cue_signature(component)
            path_id = _path_component_id(action, cue_signature)
            outcome_counts: dict[str, float] = {}
            effect_totals: dict[str, float] = {}
            preferred_probability_total = 0.0
            mean_risk_total = 0.0
            max_risk = 0.0
            future_path_utility_total = 0.0
            contradiction_total = 0.0
            error_avoidance_total = 0.0
            novelty_total = 0.0
            free_energy_delta_total = 0.0
            confirmation_count = 0
            violation_count = 0
            for entry in component:
                legacy = _path_legacy_template(entry)
                outcome = _path_outcome(entry)
                if outcome:
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0.0) + 1.0
                for key, value in _path_effect_profile(entry).items():
                    effect_totals[key] = effect_totals.get(key, 0.0) + float(value)
                preferred_probability_total += float(legacy.get("preferred_probability", 0.0) or 0.0)
                risk = float(legacy.get("risk", 0.0) or 0.0)
                mean_risk_total += risk
                max_risk = max(max_risk, risk)
                novelty_total += float(entry.novelty)
                credit = _path_credit(entry)
                future_path_utility_total += float(credit.get("future_path_utility", 0.0) or 0.0)
                contradiction_total += float(credit.get("contradiction_burden", 0.0) or 0.0)
                error_avoidance_total += float(credit.get("error_avoidance_gain", 0.0) or 0.0)
                confirmation_count += int(credit.get("confirmed_count", 0) or 0)
                violation_count += (
                    int(credit.get("violated_count", 0) or 0) + int(entry.counterevidence_count or 0)
                )
                last_signal = credit.get("last_signal")
                if isinstance(last_signal, dict):
                    free_energy_delta_total += float(last_signal.get("free_energy_delta", 0.0) or 0.0)
            support_count = len(component)
            effect_profile = {
                key: round(float(value) / support_count, 6)
                for key, value in sorted(effect_totals.items())
            }
            total_outcomes = sum(outcome_counts.values()) or 1.0
            outcome_distribution = {
                key: round(float(value) / total_outcomes, 6)
                for key, value in sorted(outcome_counts.items())
            }
            future_path_utility = future_path_utility_total / support_count
            contradiction_burden = contradiction_total / support_count
            error_avoidance_gain = error_avoidance_total / support_count
            maintenance_cost = min(
                0.60,
                0.08
                + (0.03 * max(0, len(cue_signature.semantic_tags) - 1))
                + (0.02 * max(0, len(cue_signature.context_tags) - 1)),
            )
            mean_risk = mean_risk_total / support_count
            expected_surprise = novelty_total / support_count
            polarity = _path_polarity(
                future_path_utility=future_path_utility,
                confirmation_count=confirmation_count,
                violation_count=violation_count,
                mean_risk=mean_risk,
                contradiction_burden=contradiction_burden,
                expected_surprise=expected_surprise,
            )
            path_quality = _path_quality(
                support_count=support_count,
                confirmation_count=confirmation_count,
                violation_count=violation_count,
                future_path_utility=future_path_utility,
                contradiction_burden=contradiction_burden,
                maintenance_cost=maintenance_cost,
                error_avoidance_gain=error_avoidance_gain,
            )
            paths.append(
                MemoryPath(
                    path_id=path_id,
                    source_episode_ids=[entry.id for entry in component],
                    source_memory_ids=[entry.id for entry in component],
                    dominant_action=action,
                    cue_signature=cue_signature,
                    outcome_profile=PathOutcomeProfile(
                        outcome_distribution=outcome_distribution,
                        predicted_effects=effect_profile,
                        preferred_probability=preferred_probability_total / support_count,
                        future_path_utility=future_path_utility,
                    ),
                    risk_profile=PathRiskProfile(
                        mean_risk=mean_risk,
                        max_risk=max_risk,
                        contradiction_burden=contradiction_burden,
                        maintenance_cost=maintenance_cost,
                        caution_score=max(mean_risk, contradiction_burden, expected_surprise * 0.70),
                    ),
                    expected_surprise_profile={
                        "mean_prediction_error": round(expected_surprise, 6),
                        "mean_free_energy_delta": round(free_energy_delta_total / support_count, 6),
                        "error_avoidance_gain": round(error_avoidance_gain, 6),
                    },
                    support_count=support_count,
                    confirmation_count=confirmation_count,
                    violation_count=violation_count,
                    path_quality=path_quality,
                    path_polarity=polarity,
                    last_updated_cycle=int(current_cycle),
                )
            )
    paths.sort(key=lambda item: (-float(item.path_quality), -int(item.support_count), item.path_id))
    return paths


def _support_ids(entry: MemoryEntry) -> set[str]:
    metadata = dict(entry.compression_metadata or {})
    support_ids = entry.support_ids or metadata.get("support_ids") or metadata.get("support_entry_ids") or []
    return {str(item) for item in support_ids if str(item)}


def _refresh_parent_semantics(store: "MemoryStore", touched_source_ids: set[str]) -> None:
    for semantic in store.entries:
        if semantic.memory_class not in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}:
            continue
        support_ids = _support_ids(semantic)
        if not support_ids or not (support_ids & touched_source_ids):
            continue
        sources = [
            entry
            for entry in store.entries
            if entry.id in support_ids and entry.id != semantic.id
        ]
        if not sources:
            continue
        lineage_type = str(dict(semantic.compression_metadata or {}).get("lineage_type", "pattern_extraction"))
        before_centroid = list(semantic.centroid or [])
        before_residual_mean = semantic.residual_norm_mean
        before_residual_var = semantic.residual_norm_var
        _apply_vector_semantic_fields(semantic, sources, lineage_type=lineage_type)
        metadata = dict(semantic.compression_metadata or {})
        metadata["m410_replay_refresh"] = {
            "touched_source_ids": sorted(support_ids & touched_source_ids),
            "centroid_before": before_centroid,
            "centroid_after": list(semantic.centroid or []),
            "residual_norm_mean_before": before_residual_mean,
            "residual_norm_mean_after": semantic.residual_norm_mean,
            "residual_norm_var_before": before_residual_var,
            "residual_norm_var_after": semantic.residual_norm_var,
        }
        semantic.compression_metadata = metadata


def constrained_replay(
    store: "MemoryStore",
    rng: random.Random,
    batch_size: int = 32,
) -> list[MemoryEntry]:
    """Re-encode and return existing source entries touched by replay."""
    weighted = sorted(
        store.entries,
        key=lambda entry: (
            entry.salience + (0.15 * entry.arousal) + (0.10 * entry.retrieval_count)
            + (0.10 if entry.counterevidence_count > 0 else 0.0)
        ),
        reverse=True,
    )
    sampled = weighted[: max(1, min(batch_size, 3))]
    replay_entries: list[MemoryEntry] = []
    for source in sampled:
        second_pass_error = _clamp(
            source.novelty
            + (float(source.semantic_reconstruction_error or 0.0) * 0.25)
            + (0.03 if source.counterevidence_count else 0.0)
        )
        result, salience_delta, retention_adjustment = EncodingDynamics.reencode(
            first_pass_strength=source.salience,
            prediction_error=second_pass_error,
            surprise=_clamp(max(source.salience, second_pass_error)),
            arousal=_clamp(source.arousal),
            attention_budget=max(0.05, source.encoding_attention),
            requested_budget=max(0.05, source.encoding_attention),
        )
        source.replay_second_pass_error = result.prediction_error
        source.salience_delta = salience_delta
        source.retention_adjustment = retention_adjustment
        source.salience = _clamp(source.salience + retention_adjustment)
        source.trace_strength = _clamp(source.trace_strength + (retention_adjustment * 0.5))
        source.accessibility = _clamp(source.accessibility + (retention_adjustment * 0.5))
        metadata = dict(source.compression_metadata or {})
        metadata["m410_replay"] = {
            **result.to_dict(),
            "replay_second_pass_error": source.replay_second_pass_error,
            "salience_delta": source.salience_delta,
            "retention_adjustment": source.retention_adjustment,
        }
        source.compression_metadata = metadata
        replay_entries.append(source)
    _refresh_parent_semantics(store, {entry.id for entry in replay_entries})
    rng.shuffle(replay_entries)
    return replay_entries


def validate_inference(entry: MemoryEntry) -> ValidationResult:
    metadata = dict(entry.compression_metadata or {})
    replay_persistence = _clamp(float(metadata.get("replay_persistence", min(1.0, entry.retrieval_count / 4.0))))
    support_score = _clamp(min(1.0, entry.support_count / DEFAULT_MINIMUM_SUPPORT))
    cross_context_consistency = _clamp(float(metadata.get("cross_context_consistency", min(1.0, len(set(entry.context_tags)) / 3.0))))
    predictive_gain = _clamp(float(metadata.get("predictive_gain", min(1.0, entry.relevance + 0.1))))
    contradiction_penalty = _clamp(float(metadata.get("contradiction_penalty", min(1.0, entry.counterevidence_count / 4.0))))
    score = _clamp(
        (0.25 * replay_persistence)
        + (0.30 * support_score)
        + (0.20 * cross_context_consistency)
        + (0.25 * predictive_gain)
        - (0.35 * contradiction_penalty)
    )
    threshold = 0.55
    if contradiction_penalty >= 0.75:
        validation_status = "contradicted"
        validation_discount = 0.15
    elif score >= threshold:
        validation_status = "validated"
        validation_discount = 1.0
    elif score >= 0.40:
        validation_status = "partially_supported"
        validation_discount = 0.70
    else:
        validation_status = "unvalidated"
        validation_discount = 0.35
    metadata["validation_status"] = validation_status
    metadata["validation_discount"] = validation_discount
    metadata["inference_write_score"] = score
    entry.compression_metadata = metadata
    if validation_status == "validated":
        entry.reality_confidence = _clamp(max(entry.reality_confidence, 0.68))
        if entry.store_level is StoreLevel.MID:
            entry.store_level = StoreLevel.LONG
    return ValidationResult(
        entry_id=entry.id,
        score=round(score, 6),
        threshold=threshold,
        passed=validation_status == "validated",
        validation_status=validation_status,
        validation_discount=validation_discount,
    )


def consolidate_upgrade(
    store: "MemoryStore",
    current_cycle: int,
    *,
    current_state: dict[str, object] | None = None,
    cognitive_style=None,
) -> UpgradeReport:
    state_vector = normalize_agent_state(current_state or getattr(store, "agent_state_vector", None))
    update_rigidity = _clamp(_style_value(current_state, "update_rigidity", 0.0))
    promoted_ids: list[str] = []
    reasons: dict[str, str] = {}
    identity_bias = 0.10 if state_vector.identity_active_themes else 0.0
    threat_bias = 0.08 if state_vector.threat_level >= 0.6 else 0.0
    group_support: dict[str, int] = {}
    for entry in store.entries:
        if len(entry.semantic_tags) < 2:
            continue
        signature = "|".join(sorted(tag.lower() for tag in entry.semantic_tags[:2]))
        group_support[signature] = group_support.get(signature, 0) + 1
    for entry in store.entries:
        redundancy = 0.20 if dict(entry.compression_metadata or {}).get("absorbed_by") else 0.0
        signature = "|".join(sorted(tag.lower() for tag in entry.semantic_tags[:2])) if len(entry.semantic_tags) >= 2 else ""
        cluster_support_count = group_support.get(signature, 0)
        pattern_support = max(
            min(1.0, entry.support_count / DEFAULT_MINIMUM_SUPPORT),
            min(1.0, cluster_support_count / DEFAULT_MINIMUM_SUPPORT),
        )
        retrieval_norm = min(1.0, entry.retrieval_count / 4.0)
        identity_alignment = identity_match_ratio_for_entry(entry, state_vector)
        novelty_noise_penalty = 0.24 if entry.novelty >= 0.75 and entry.relevance_self < 0.20 else 0.0
        priority = (
            (0.35 * entry.salience)
            + (0.25 * retrieval_norm)
            + (0.25 * pattern_support)
            - (0.15 * redundancy)
            + identity_bias
            + threat_bias
            + (identity_alignment * 0.15)
            - novelty_noise_penalty
        )
        if entry.relevance_self >= 0.35 and identity_alignment > 0.0:
            priority += 0.08 + (identity_alignment * 0.12)
        if update_rigidity >= 0.70 and entry.relevance_self >= 0.35:
            priority += 0.04
        old_level = entry.store_level
        new_level = old_level
        promotion_reasons: list[str] = []
        if old_level is StoreLevel.SHORT and priority > 0.45:
            new_level = StoreLevel.MID
            promotion_reasons.append("short_to_mid_priority")
            if cluster_support_count >= DEFAULT_MINIMUM_SUPPORT:
                promotion_reasons.append("cluster_support")
        elif old_level is StoreLevel.MID and (
            priority > 0.68
            or (entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED} and entry.support_count >= 3)
        ):
            new_level = StoreLevel.LONG
            promotion_reasons.append("mid_to_long_stability")
            if entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED} and entry.support_count >= 3:
                promotion_reasons.append("stable_abstraction_support")
        if store.promote_entry(
            entry,
            new_level=new_level,
            reasons=promotion_reasons,
            effective_cycle=current_cycle,
            promotion_context={
                "promotion_channel": "consolidation_cycle",
                "consolidation_priority": round(priority, 6),
                "pattern_support": round(pattern_support, 6),
                "cluster_support_count": cluster_support_count,
                "retrieval_norm": round(retrieval_norm, 6),
                "identity_alignment": round(identity_alignment, 6),
                "redundancy_penalty": round(redundancy, 6),
            },
        ):
            promoted_ids.append(entry.id)
            reasons[entry.id] = "+".join(promotion_reasons) if promotion_reasons else "promotion"
    return UpgradeReport(promoted_ids=promoted_ids, promotion_reasons=reasons)


def consolidation_cleanup(store: "MemoryStore", current_cycle: int) -> CleanupReport:
    deleted_ids: list[str] = []
    dormant_ids: list[str] = []
    absorbed_ids: list[str] = []
    confidence_drift_ids: list[str] = []
    retained: list[MemoryEntry] = []
    for entry in store.entries:
        metadata = dict(entry.compression_metadata or {})
        if entry.store_level is StoreLevel.SHORT and entry.trace_strength < 0.05:
            deleted_ids.append(entry.id)
            continue
        if metadata.get("absorbed_by"):
            absorbed_ids.append(entry.id)
            entry.accessibility = _clamp(entry.accessibility * 0.65)
            entry.abstractness = _clamp(entry.abstractness + 0.10)
        if entry.store_level in {StoreLevel.MID, StoreLevel.LONG} and entry.counterevidence_count > 0:
            before = (entry.source_confidence, entry.reality_confidence)
            entry.source_confidence = _clamp(entry.source_confidence - 0.03)
            entry.reality_confidence = _clamp(entry.reality_confidence - 0.05)
            if before != (entry.source_confidence, entry.reality_confidence):
                confidence_drift_ids.append(entry.id)
        if entry.store_level is StoreLevel.LONG and entry.accessibility < 0.08 and entry.trace_strength < 0.08:
            entry.is_dormant = True
            dormant_ids.append(entry.id)
        retained.append(entry)
    store.entries = retained
    return CleanupReport(
        deleted_ids=deleted_ids,
        dormant_ids=dormant_ids,
        absorbed_ids=absorbed_ids,
        confidence_drift_ids=confidence_drift_ids,
    )


def run_consolidation_cycle(
    store: "MemoryStore",
    *,
    current_cycle: int,
    rng: random.Random,
    current_state: dict[str, object] | None = None,
    cognitive_style=None,
) -> ConsolidationReport:
    upgrade = consolidate_upgrade(
        store,
        current_cycle,
        current_state=current_state,
        cognitive_style=cognitive_style,
    )
    extracted = extract_patterns(store, minimum_support=DEFAULT_RUNTIME_MINIMUM_SUPPORT)
    extracted_ids: list[str] = []
    for entry in extracted:
        store.add(entry, current_state=current_state, cognitive_style=cognitive_style)
        extracted_ids.append(entry.id)
    replay_created = constrained_replay(store, rng=rng)
    replay_reencoded_ids: list[str] = []
    validated_ids: list[str] = []
    for entry in replay_created:
        replay_reencoded_ids.append(entry.id)
    for entry in extracted:
        if entry.memory_class is MemoryClass.INFERRED:
            validation = validate_inference(entry)
            if validation.passed:
                validated_ids.append(entry.id)
    cleanup = consolidation_cleanup(store, current_cycle)
    store.refresh_memory_paths(current_cycle=current_cycle)
    return ConsolidationReport(
        upgrade=upgrade,
        extracted_patterns=extracted_ids,
        replay_reencoded_ids=replay_reencoded_ids,
        validated_inference_ids=validated_ids,
        cleanup=cleanup,
    )
