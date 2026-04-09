from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .memory_model import AnchorStrength, MemoryClass, MemoryEntry, SourceType, StoreLevel
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
DEFAULT_SMOOTHING = 2.0


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


def _copy_entry(entry: MemoryEntry) -> MemoryEntry:
    return MemoryEntry.from_dict(entry.to_dict())


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
    replay_created_ids: list[str]
    validated_inference_ids: list[str]
    cleanup: CleanupReport

    def to_dict(self) -> dict[str, object]:
        return {
            "upgrade": self.upgrade.to_dict(),
            "extracted_patterns": list(self.extracted_patterns),
            "replay_created_ids": list(self.replay_created_ids),
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
    ranked: list[tuple[tuple[float, float, float, float], MemoryEntry]] = []
    for entry in candidates:
        if entry.id == primary.id:
            continue
        validation_status = str(dict(entry.compression_metadata or {}).get("validation_status", "validated"))
        if entry.memory_class is MemoryClass.INFERRED and validation_status in {"unvalidated", "contradicted"}:
            continue
        derived_score = 1.0 if entry.id in derived_ids else 0.0
        semantic_score = _shared_semantic_overlap(primary, entry)
        mood_score = 1.0 if primary.mood_context and primary.mood_context == entry.mood_context else 0.0
        context_score = _shared_context_overlap(primary, entry)
        ranked.append(((derived_score, semantic_score, mood_score, context_score), entry))
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
    incoming: MemoryEntry | RecallArtifact,
    conflict_type: ConflictType,
) -> ConflictResolution:
    if isinstance(incoming, RecallArtifact):
        interpretation = f"recall:{incoming.primary_entry_id}"
    else:
        interpretation = f"entry:{incoming.id}"
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

    entry.accessibility = _clamp(entry.accessibility + effective_boost_access)
    entry.trace_strength = _clamp(entry.trace_strength + BOOST_TRACE)
    entry.retrieval_count += 1
    entry.abstractness = _clamp(entry.abstractness + ABSTRACTNESS_INCREMENT)
    if current_cycle is not None:
        entry.last_accessed = max(entry.last_accessed, int(current_cycle))

    update_type = ReconsolidationUpdateType.REINFORCEMENT_ONLY
    if conflict_type is not None and recall_artifact is not None:
        resolution = resolve_conflict(entry, recall_artifact, conflict_type)
        entry.source_confidence = _clamp(entry.source_confidence + resolution.source_confidence_delta)
        entry.reality_confidence = _clamp(entry.reality_confidence + resolution.reality_confidence_delta)
        entry.counterevidence_count += resolution.counterevidence_delta
        interpretations = list(entry.competing_interpretations or [])
        for interpretation in resolution.competing_interpretations_added:
            if interpretation not in interpretations:
                interpretations.append(interpretation)
        entry.competing_interpretations = interpretations or None
        conflict_flags.append(conflict_type.value)
        update_type = ReconsolidationUpdateType.CONFLICT_MARKING
    elif current_mood and current_mood != entry.mood_context:
        entry.mood_context = current_mood
        fields_rebound.append("mood_context")
        update_type = ReconsolidationUpdateType.CONTEXTUAL_REBINDING
    if current_context_tags:
        merged_contexts = list(dict.fromkeys([*entry.context_tags, *_string_list(current_context_tags)]))
        if merged_contexts != entry.context_tags:
            entry.context_tags = merged_contexts
            fields_rebound.append("context_tags")
            if update_type is ReconsolidationUpdateType.REINFORCEMENT_ONLY:
                update_type = ReconsolidationUpdateType.CONTEXTUAL_REBINDING

    if store is not None and conflict_type is None:
        config = ReconstructionConfig(
            current_cycle=current_cycle or entry.last_accessed,
            current_state=current_state,
        )
        reconstruction_blocked = update_rigidity >= 0.85
        if identity_match >= 0.50 and entry.relevance_self >= 0.45:
            reconstruction_blocked = True
        if error_aversion >= 0.60 and entry.valence < 0.0:
            reconstruction_blocked = True
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
    )


def compress_episodic_cluster_to_semantic_skeleton(entries: list[MemoryEntry]) -> MemoryEntry:
    if not entries:
        raise ValueError("episodic cluster compression requires entries")
    support_ids = [entry.id for entry in entries]
    shared_semantic_tags = sorted(set(entries[0].semantic_tags).intersection(*[set(entry.semantic_tags) for entry in entries[1:]]))
    if not shared_semantic_tags:
        shared_semantic_tags = sorted({tag for entry in entries for tag in entry.semantic_tags[:2]})[:4]
    shared_context_tags = sorted(set(entries[0].context_tags).intersection(*[set(entry.context_tags) for entry in entries[1:]]))
    action_values = sorted({entry.anchor_slots.get("action") for entry in entries if entry.anchor_slots.get("action")})
    outcome_values = sorted({entry.anchor_slots.get("outcome") for entry in entries if entry.anchor_slots.get("outcome")})
    stable_structure = {
        "semantic_tags": list(shared_semantic_tags),
        "context_tags": list(shared_context_tags),
        "actions": action_values,
        "outcomes": outcome_values,
    }
    identity_cluster = any(entry.relevance_self >= 0.6 for entry in entries)
    lineage_type = "identity_consolidation" if identity_cluster else "episodic_compression"
    content = (
        f"Semantic skeleton from {len(entries)} episodes: "
        f"{', '.join(shared_semantic_tags[:3] or ['stable pattern'])}"
    )
    return MemoryEntry(
        content=content,
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
        },
        derived_from=support_ids,
    )


def _group_pattern_candidates(
    store: "MemoryStore",
    *,
    minimum_support: int,
) -> list[list[MemoryEntry]]:
    buckets: dict[str, list[MemoryEntry]] = {}
    for entry in store.entries:
        if entry.memory_class is MemoryClass.PROCEDURAL or len(entry.semantic_tags) < 2:
            continue
        key = "|".join(sorted(entry.semantic_tags[:2]))
        buckets.setdefault(key, []).append(entry)
    return [entries for entries in buckets.values() if len(entries) >= minimum_support]


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
            content=f"Inferred pattern from {support_count} related memories",
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
            },
            derived_from=[entry.id for entry in group],
        )
        results.append(inferred)
        break
    return results


def constrained_replay(
    store: "MemoryStore",
    rng: random.Random,
    batch_size: int = 32,
) -> list[MemoryEntry]:
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
        tags = list(dict.fromkeys(source.semantic_tags[:3] or source.context_tags[:2]))
        replay_entries.append(
            MemoryEntry(
                content=f"Replay hypothesis from {source.id}: {' / '.join(tags or ['memory pattern'])}",
                memory_class=MemoryClass.INFERRED,
                store_level=StoreLevel.MID,
                source_type=SourceType.INFERENCE,
                created_at=source.last_accessed,
                last_accessed=source.last_accessed,
                valence=source.valence,
                arousal=source.arousal,
                encoding_attention=source.encoding_attention,
                novelty=min(1.0, source.novelty + 0.05),
                relevance_goal=source.relevance_goal,
                relevance_threat=source.relevance_threat,
                relevance_self=source.relevance_self,
                relevance_social=source.relevance_social,
                relevance_reward=source.relevance_reward,
                relevance=source.relevance,
                salience=min(1.0, source.salience + 0.05),
                trace_strength=0.42,
                accessibility=0.38,
                abstractness=0.82,
                source_confidence=0.90,
                reality_confidence=0.32,
                semantic_tags=tags or list(source.semantic_tags),
                context_tags=list(source.context_tags),
                mood_context=source.mood_context,
                support_count=max(1, source.support_count),
                competing_interpretations=[f"replay:{source.id}"],
                compression_metadata={
                    "support_entry_ids": [source.id],
                    "abstraction_reason": "constrained replay candidate",
                    "predictive_use_cases": ["hypothesis candidate"],
                    "lineage_type": "pattern_extraction",
                    "validation_status": "unvalidated",
                    "validation_discount": 0.35,
                },
                derived_from=[source.id],
            )
        )
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
    extracted = extract_patterns(store)
    extracted_ids: list[str] = []
    for entry in extracted:
        store.add(entry, current_state=current_state, cognitive_style=cognitive_style)
        extracted_ids.append(entry.id)
    replay_created = constrained_replay(store, rng=rng)
    replay_created_ids: list[str] = []
    validated_ids: list[str] = []
    for entry in replay_created:
        store.add(entry, current_state=current_state, cognitive_style=cognitive_style)
        replay_created_ids.append(entry.id)
        validation = validate_inference(entry)
        if validation.passed:
            validated_ids.append(entry.id)
    cleanup = consolidation_cleanup(store, current_cycle)
    return ConsolidationReport(
        upgrade=upgrade,
        extracted_patterns=extracted_ids,
        replay_created_ids=replay_created_ids,
        validated_inference_ids=validated_ids,
        cleanup=cleanup,
    )
