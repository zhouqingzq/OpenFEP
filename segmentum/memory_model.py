from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any
from uuid import uuid4


REQUIRED_ANCHOR_KEYS = ("time", "place", "agents", "action", "outcome")


class MemoryClass(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    INFERRED = "inferred"


class StoreLevel(str, Enum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"


class SourceType(str, Enum):
    EXPERIENCE = "experience"
    REHEARSAL = "rehearsal"
    INFERENCE = "inference"
    HEARSAY = "hearsay"
    RECONSTRUCTION = "reconstruction"

    @classmethod
    def _missing_(cls, value: object) -> "SourceType" | None:
        """Backward-compatible coercion for legacy dataset/source labels."""
        text = str(value or "").strip().lower()
        if not text:
            return cls.EXPERIENCE
        alias_map = {
            # Legacy narrative/text-source labels used by older milestones.
            "user_story": cls.EXPERIENCE,
            "user_diary": cls.EXPERIENCE,
            "open_text": cls.EXPERIENCE,
            "narrative": cls.EXPERIENCE,
            "test": cls.EXPERIENCE,
            # World/scenario ids sometimes flowed into source_type fields.
            "foraging_valley": cls.EXPERIENCE,
            "predator_river": cls.EXPERIENCE,
            # Common near-synonyms.
            "observed": cls.EXPERIENCE,
            "witnessed": cls.EXPERIENCE,
            "heard": cls.HEARSAY,
            "inferred": cls.INFERENCE,
            "reconstructed": cls.RECONSTRUCTION,
        }
        if text in alias_map:
            return alias_map[text]
        # Default unknown legacy values to experiential source for robustness.
        return cls.EXPERIENCE


class AnchorStrength(str, Enum):
    LOCKED = "locked"
    STRONG = "strong"
    WEAK = "weak"


def compute_content_hash(content: str) -> str:
    normalized = (content or "").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


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


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return []


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_coerce_float(item) for item in value]


def _coerce_str_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    return _coerce_str_list(value)


def _coerce_float_list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    return _coerce_float_list(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_anchor_slots(payload: dict[str, Any] | None) -> dict[str, str | None]:
    base = {key: None for key in REQUIRED_ANCHOR_KEYS}
    if isinstance(payload, dict):
        for key in REQUIRED_ANCHOR_KEYS:
            raw = payload.get(key)
            base[key] = None if raw is None else str(raw)
    return base


def _default_anchor_strengths(memory_class: MemoryClass) -> dict[str, AnchorStrength]:
    if memory_class is MemoryClass.EPISODIC:
        return {
            "time": AnchorStrength.WEAK,
            "place": AnchorStrength.WEAK,
            "agents": AnchorStrength.STRONG,
            "action": AnchorStrength.STRONG,
            "outcome": AnchorStrength.STRONG,
        }
    return {key: AnchorStrength.WEAK for key in REQUIRED_ANCHOR_KEYS}


def _normalize_anchor_strengths(
    payload: dict[str, Any] | None,
    *,
    memory_class: MemoryClass,
) -> dict[str, AnchorStrength]:
    normalized = _default_anchor_strengths(memory_class)
    if isinstance(payload, dict):
        for key in REQUIRED_ANCHOR_KEYS:
            raw = payload.get(key)
            if raw is None:
                continue
            normalized[key] = AnchorStrength(str(raw))
    if memory_class is MemoryClass.EPISODIC:
        protected = [normalized["agents"], normalized["action"], normalized["outcome"]]
        strong_count = sum(
            1 for item in protected if item in {AnchorStrength.STRONG, AnchorStrength.LOCKED}
        )
        has_locked = any(item is AnchorStrength.LOCKED for item in protected)
        if strong_count < 2 and not has_locked:
            raise ValueError(
                "episodic memories must retain protected anchors; agents/action/outcome cannot all be weak"
            )
        if all(normalized[key] is AnchorStrength.WEAK for key in REQUIRED_ANCHOR_KEYS):
            raise ValueError("episodic memories cannot set all anchor strengths to weak")
    return normalized


def _normalize_step_confidence(steps: list[str], values: list[float]) -> list[float]:
    if not steps:
        return []
    if not values:
        return [0.8 for _ in steps]
    if len(values) != len(steps):
        raise ValueError("step_confidence must align with procedure_steps")
    return [max(0.0, min(1.0, _coerce_float(item, 0.8))) for item in values]


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    content_hash: str = ""
    memory_class: MemoryClass = MemoryClass.EPISODIC
    store_level: StoreLevel = StoreLevel.SHORT
    source_type: SourceType = SourceType.EXPERIENCE
    created_at: int = 0
    last_accessed: int = 0
    valence: float = 0.0
    arousal: float = 0.0
    encoding_attention: float = 0.0
    novelty: float = 0.0
    relevance_goal: float = 0.0
    relevance_threat: float = 0.0
    relevance_self: float = 0.0
    relevance_social: float = 0.0
    relevance_reward: float = 0.0
    relevance: float = 0.0
    salience: float = 0.0
    trace_strength: float = 0.0
    accessibility: float = 0.0
    abstractness: float = 0.0
    source_confidence: float = 0.0
    reality_confidence: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    anchor_slots: dict[str, str | None] = field(default_factory=dict)
    anchor_strengths: dict[str, AnchorStrength] = field(default_factory=dict)
    procedure_steps: list[str] = field(default_factory=list)
    step_confidence: list[float] = field(default_factory=list)
    execution_contexts: list[str] = field(default_factory=list)
    mood_context: str = ""
    retrieval_count: int = 0
    support_count: int = 1
    counterevidence_count: int = 0
    competing_interpretations: list[str] | None = None
    compression_metadata: dict[str, object] | None = None
    derived_from: list[str] = field(default_factory=list)
    version: int = 1
    is_dormant: bool = False
    state_vector: list[float] | None = None
    centroid: list[float] | None = None
    residual_norm_mean: float | None = None
    residual_norm_var: float | None = None
    support_ids: list[str] | None = None
    consolidation_source: str | None = None
    semantic_reconstruction_error: float | None = None
    replay_second_pass_error: float | None = None
    salience_delta: float | None = None
    retention_adjustment: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid4())
        self.memory_class = MemoryClass(self.memory_class)
        self.store_level = StoreLevel(self.store_level)
        self.source_type = SourceType(self.source_type)
        self.content = str(self.content or "")
        self.content_hash = self.content_hash or compute_content_hash(self.content)
        self.anchor_slots = _normalize_anchor_slots(self.anchor_slots)
        self.anchor_strengths = _normalize_anchor_strengths(
            self.anchor_strengths,
            memory_class=self.memory_class,
        )
        self.semantic_tags = _coerce_str_list(self.semantic_tags)
        self.context_tags = _coerce_str_list(self.context_tags)
        self.procedure_steps = _coerce_str_list(self.procedure_steps)
        self.step_confidence = _normalize_step_confidence(
            self.procedure_steps,
            _coerce_float_list(self.step_confidence),
        )
        self.execution_contexts = _coerce_str_list(self.execution_contexts)
        self.derived_from = _coerce_str_list(self.derived_from)
        self.state_vector = _coerce_float_list_or_none(self.state_vector)
        self.centroid = _coerce_float_list_or_none(self.centroid)
        self.support_ids = _coerce_str_list_or_none(self.support_ids)
        self.residual_norm_mean = _coerce_optional_float(self.residual_norm_mean)
        self.residual_norm_var = _coerce_optional_float(self.residual_norm_var)
        self.semantic_reconstruction_error = _coerce_optional_float(self.semantic_reconstruction_error)
        self.replay_second_pass_error = _coerce_optional_float(self.replay_second_pass_error)
        self.salience_delta = _coerce_optional_float(self.salience_delta)
        self.retention_adjustment = _coerce_optional_float(self.retention_adjustment)
        if self.consolidation_source is not None:
            self.consolidation_source = str(self.consolidation_source)
        if self.competing_interpretations is not None:
            self.competing_interpretations = _coerce_str_list(self.competing_interpretations)
        if self.compression_metadata is not None and not isinstance(self.compression_metadata, dict):
            self.compression_metadata = {}
        self.created_at = _coerce_int(self.created_at)
        self.last_accessed = _coerce_int(self.last_accessed, self.created_at)
        self.version = max(1, _coerce_int(self.version, 1))
        self.retrieval_count = max(0, _coerce_int(self.retrieval_count))
        self.support_count = max(0, _coerce_int(self.support_count, 1))
        self.counterevidence_count = max(0, _coerce_int(self.counterevidence_count))
        self.mood_context = str(self.mood_context or "")
        if self.memory_class is MemoryClass.PROCEDURAL and not self.procedure_steps:
            raise ValueError("procedural memories require explicit procedure_steps")

    def sync_content_hash(self) -> None:
        new_hash = compute_content_hash(self.content)
        if not self.content_hash:
            self.content_hash = new_hash
            return
        if new_hash != self.content_hash:
            self.content_hash = new_hash
            self.version += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_class": self.memory_class.value,
            "store_level": self.store_level.value,
            "source_type": self.source_type.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "valence": self.valence,
            "arousal": self.arousal,
            "encoding_attention": self.encoding_attention,
            "novelty": self.novelty,
            "relevance_goal": self.relevance_goal,
            "relevance_threat": self.relevance_threat,
            "relevance_self": self.relevance_self,
            "relevance_social": self.relevance_social,
            "relevance_reward": self.relevance_reward,
            "relevance": self.relevance,
            "salience": self.salience,
            "trace_strength": self.trace_strength,
            "accessibility": self.accessibility,
            "abstractness": self.abstractness,
            "source_confidence": self.source_confidence,
            "reality_confidence": self.reality_confidence,
            "semantic_tags": list(self.semantic_tags),
            "context_tags": list(self.context_tags),
            "anchor_slots": dict(self.anchor_slots),
            "anchor_strengths": {
                key: value.value for key, value in self.anchor_strengths.items()
            },
            "procedure_steps": list(self.procedure_steps),
            "step_confidence": list(self.step_confidence),
            "execution_contexts": list(self.execution_contexts),
            "mood_context": self.mood_context,
            "retrieval_count": self.retrieval_count,
            "support_count": self.support_count,
            "counterevidence_count": self.counterevidence_count,
            "competing_interpretations": (
                list(self.competing_interpretations)
                if self.competing_interpretations is not None
                else None
            ),
            "compression_metadata": (
                dict(self.compression_metadata) if self.compression_metadata is not None else None
            ),
            "derived_from": list(self.derived_from),
            "version": self.version,
            "is_dormant": bool(self.is_dormant),
            "state_vector": list(self.state_vector) if self.state_vector is not None else None,
            "centroid": list(self.centroid) if self.centroid is not None else None,
            "residual_norm_mean": self.residual_norm_mean,
            "residual_norm_var": self.residual_norm_var,
            "support_ids": list(self.support_ids) if self.support_ids is not None else None,
            "consolidation_source": self.consolidation_source,
            "semantic_reconstruction_error": self.semantic_reconstruction_error,
            "replay_second_pass_error": self.replay_second_pass_error,
            "salience_delta": self.salience_delta,
            "retention_adjustment": self.retention_adjustment,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MemoryEntry:
        metadata = payload.get("compression_metadata")
        return cls(
            id=str(payload.get("id", "")) or str(uuid4()),
            content=str(payload.get("content", "")),
            content_hash=str(payload.get("content_hash", "")),
            memory_class=MemoryClass(str(payload.get("memory_class", MemoryClass.EPISODIC.value))),
            store_level=StoreLevel(str(payload.get("store_level", StoreLevel.SHORT.value))),
            source_type=SourceType(str(payload.get("source_type", SourceType.EXPERIENCE.value))),
            created_at=_coerce_int(payload.get("created_at", 0)),
            last_accessed=_coerce_int(payload.get("last_accessed", payload.get("created_at", 0))),
            valence=_coerce_float(payload.get("valence", 0.0)),
            arousal=_coerce_float(payload.get("arousal", 0.0)),
            encoding_attention=_coerce_float(payload.get("encoding_attention", 0.0)),
            novelty=_coerce_float(payload.get("novelty", 0.0)),
            relevance_goal=_coerce_float(payload.get("relevance_goal", 0.0)),
            relevance_threat=_coerce_float(payload.get("relevance_threat", 0.0)),
            relevance_self=_coerce_float(payload.get("relevance_self", 0.0)),
            relevance_social=_coerce_float(payload.get("relevance_social", 0.0)),
            relevance_reward=_coerce_float(payload.get("relevance_reward", 0.0)),
            relevance=_coerce_float(payload.get("relevance", 0.0)),
            salience=_coerce_float(payload.get("salience", 0.0)),
            trace_strength=_coerce_float(payload.get("trace_strength", 0.0)),
            accessibility=_coerce_float(payload.get("accessibility", 0.0)),
            abstractness=_coerce_float(payload.get("abstractness", 0.0)),
            source_confidence=_coerce_float(payload.get("source_confidence", 0.0)),
            reality_confidence=_coerce_float(payload.get("reality_confidence", 0.0)),
            semantic_tags=_coerce_str_list(payload.get("semantic_tags", [])),
            context_tags=_coerce_str_list(payload.get("context_tags", [])),
            anchor_slots=dict(payload.get("anchor_slots", {}))
            if isinstance(payload.get("anchor_slots"), dict)
            else {},
            anchor_strengths=dict(payload.get("anchor_strengths", {}))
            if isinstance(payload.get("anchor_strengths"), dict)
            else {},
            procedure_steps=_coerce_str_list(payload.get("procedure_steps", [])),
            step_confidence=_coerce_float_list(payload.get("step_confidence", [])),
            execution_contexts=_coerce_str_list(payload.get("execution_contexts", [])),
            mood_context=str(payload.get("mood_context", "")),
            retrieval_count=_coerce_int(payload.get("retrieval_count", 0)),
            support_count=_coerce_int(payload.get("support_count", 1)),
            counterevidence_count=_coerce_int(payload.get("counterevidence_count", 0)),
            competing_interpretations=payload.get("competing_interpretations")
            if isinstance(payload.get("competing_interpretations"), list)
            else None,
            compression_metadata=dict(metadata) if isinstance(metadata, dict) else None,
            derived_from=_coerce_str_list(payload.get("derived_from", [])),
            version=_coerce_int(payload.get("version", 1)),
            is_dormant=bool(payload.get("is_dormant", False)),
            state_vector=_coerce_float_list_or_none(payload.get("state_vector")),
            centroid=_coerce_float_list_or_none(payload.get("centroid")),
            residual_norm_mean=_coerce_optional_float(payload.get("residual_norm_mean")),
            residual_norm_var=_coerce_optional_float(payload.get("residual_norm_var")),
            support_ids=_coerce_str_list_or_none(payload.get("support_ids")),
            consolidation_source=(
                str(payload.get("consolidation_source"))
                if payload.get("consolidation_source") is not None
                else None
            ),
            semantic_reconstruction_error=_coerce_optional_float(
                payload.get("semantic_reconstruction_error")
            ),
            replay_second_pass_error=_coerce_optional_float(
                payload.get("replay_second_pass_error")
            ),
            salience_delta=_coerce_optional_float(payload.get("salience_delta")),
            retention_adjustment=_coerce_optional_float(payload.get("retention_adjustment")),
        )


@dataclass
class PathCueSignature:
    semantic_tags: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    sensitive_channels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.semantic_tags = _coerce_str_list(self.semantic_tags)
        self.context_tags = _coerce_str_list(self.context_tags)
        self.sensitive_channels = _coerce_str_list(self.sensitive_channels)

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_tags": list(self.semantic_tags),
            "context_tags": list(self.context_tags),
            "sensitive_channels": list(self.sensitive_channels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "PathCueSignature":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            semantic_tags=_coerce_str_list(payload.get("semantic_tags", [])),
            context_tags=_coerce_str_list(payload.get("context_tags", [])),
            sensitive_channels=_coerce_str_list(payload.get("sensitive_channels", [])),
        )


@dataclass
class PathOutcomeProfile:
    outcome_distribution: dict[str, float] = field(default_factory=dict)
    predicted_effects: dict[str, float] = field(default_factory=dict)
    preferred_probability: float = 0.0
    future_path_utility: float = 0.0

    def __post_init__(self) -> None:
        self.outcome_distribution = {
            str(key): _coerce_float(value)
            for key, value in dict(self.outcome_distribution or {}).items()
            if str(key)
        }
        self.predicted_effects = {
            str(key): _coerce_float(value)
            for key, value in dict(self.predicted_effects or {}).items()
            if str(key)
        }
        self.preferred_probability = _coerce_float(self.preferred_probability)
        self.future_path_utility = _coerce_float(self.future_path_utility)

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_distribution": dict(self.outcome_distribution),
            "predicted_effects": dict(self.predicted_effects),
            "preferred_probability": self.preferred_probability,
            "future_path_utility": self.future_path_utility,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "PathOutcomeProfile":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            outcome_distribution=(
                dict(payload.get("outcome_distribution", {}))
                if isinstance(payload.get("outcome_distribution"), dict)
                else {}
            ),
            predicted_effects=(
                dict(payload.get("predicted_effects", {}))
                if isinstance(payload.get("predicted_effects"), dict)
                else {}
            ),
            preferred_probability=_coerce_float(payload.get("preferred_probability", 0.0)),
            future_path_utility=_coerce_float(payload.get("future_path_utility", 0.0)),
        )


@dataclass
class PathRiskProfile:
    mean_risk: float = 0.0
    max_risk: float = 0.0
    contradiction_burden: float = 0.0
    maintenance_cost: float = 0.0
    caution_score: float = 0.0

    def __post_init__(self) -> None:
        self.mean_risk = _coerce_float(self.mean_risk)
        self.max_risk = _coerce_float(self.max_risk)
        self.contradiction_burden = _coerce_float(self.contradiction_burden)
        self.maintenance_cost = _coerce_float(self.maintenance_cost)
        self.caution_score = _coerce_float(self.caution_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_risk": self.mean_risk,
            "max_risk": self.max_risk,
            "contradiction_burden": self.contradiction_burden,
            "maintenance_cost": self.maintenance_cost,
            "caution_score": self.caution_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "PathRiskProfile":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            mean_risk=_coerce_float(payload.get("mean_risk", 0.0)),
            max_risk=_coerce_float(payload.get("max_risk", 0.0)),
            contradiction_burden=_coerce_float(payload.get("contradiction_burden", 0.0)),
            maintenance_cost=_coerce_float(payload.get("maintenance_cost", 0.0)),
            caution_score=_coerce_float(payload.get("caution_score", 0.0)),
        )


@dataclass
class MemoryPath:
    path_id: str
    source_episode_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)
    dominant_action: str = ""
    cue_signature: PathCueSignature = field(default_factory=PathCueSignature)
    outcome_profile: PathOutcomeProfile = field(default_factory=PathOutcomeProfile)
    risk_profile: PathRiskProfile = field(default_factory=PathRiskProfile)
    expected_surprise_profile: dict[str, float] = field(default_factory=dict)
    support_count: int = 0
    confirmation_count: int = 0
    violation_count: int = 0
    path_quality: float = 0.0
    path_polarity: str = "positive"
    last_updated_cycle: int = 0

    def __post_init__(self) -> None:
        self.path_id = str(self.path_id or "")
        self.source_episode_ids = _coerce_str_list(self.source_episode_ids)
        self.source_memory_ids = _coerce_str_list(self.source_memory_ids)
        self.dominant_action = str(self.dominant_action or "")
        if not isinstance(self.cue_signature, PathCueSignature):
            self.cue_signature = PathCueSignature.from_dict(
                dict(self.cue_signature) if isinstance(self.cue_signature, dict) else {}
            )
        if not isinstance(self.outcome_profile, PathOutcomeProfile):
            self.outcome_profile = PathOutcomeProfile.from_dict(
                dict(self.outcome_profile) if isinstance(self.outcome_profile, dict) else {}
            )
        if not isinstance(self.risk_profile, PathRiskProfile):
            self.risk_profile = PathRiskProfile.from_dict(
                dict(self.risk_profile) if isinstance(self.risk_profile, dict) else {}
            )
        self.expected_surprise_profile = {
            str(key): _coerce_float(value)
            for key, value in dict(self.expected_surprise_profile or {}).items()
            if str(key)
        }
        self.support_count = max(0, _coerce_int(self.support_count))
        self.confirmation_count = max(0, _coerce_int(self.confirmation_count))
        self.violation_count = max(0, _coerce_int(self.violation_count))
        self.path_quality = _coerce_float(self.path_quality)
        self.path_polarity = str(self.path_polarity or "positive")
        self.last_updated_cycle = max(0, _coerce_int(self.last_updated_cycle))

    def to_dict(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "source_episode_ids": list(self.source_episode_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "dominant_action": self.dominant_action,
            "cue_signature": self.cue_signature.to_dict(),
            "outcome_profile": self.outcome_profile.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "expected_surprise_profile": dict(self.expected_surprise_profile),
            "support_count": self.support_count,
            "confirmation_count": self.confirmation_count,
            "violation_count": self.violation_count,
            "path_quality": self.path_quality,
            "path_polarity": self.path_polarity,
            "last_updated_cycle": self.last_updated_cycle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MemoryPath":
        return cls(
            path_id=str(payload.get("path_id", "")),
            source_episode_ids=_coerce_str_list(payload.get("source_episode_ids", [])),
            source_memory_ids=_coerce_str_list(payload.get("source_memory_ids", [])),
            dominant_action=str(payload.get("dominant_action", "")),
            cue_signature=PathCueSignature.from_dict(
                dict(payload.get("cue_signature", {}))
                if isinstance(payload.get("cue_signature"), dict)
                else {}
            ),
            outcome_profile=PathOutcomeProfile.from_dict(
                dict(payload.get("outcome_profile", {}))
                if isinstance(payload.get("outcome_profile"), dict)
                else {}
            ),
            risk_profile=PathRiskProfile.from_dict(
                dict(payload.get("risk_profile", {}))
                if isinstance(payload.get("risk_profile"), dict)
                else {}
            ),
            expected_surprise_profile=(
                dict(payload.get("expected_surprise_profile", {}))
                if isinstance(payload.get("expected_surprise_profile"), dict)
                else {}
            ),
            support_count=_coerce_int(payload.get("support_count", 0)),
            confirmation_count=_coerce_int(payload.get("confirmation_count", 0)),
            violation_count=_coerce_int(payload.get("violation_count", 0)),
            path_quality=_coerce_float(payload.get("path_quality", 0.0)),
            path_polarity=str(payload.get("path_polarity", "positive")),
            last_updated_cycle=_coerce_int(payload.get("last_updated_cycle", 0)),
        )
