from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _coerce_str_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _coerce_float_dict(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    payload: dict[str, float] = {}
    for key, value in values.items():
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


def _coerce_object_dict(values: object) -> dict[str, object]:
    if not isinstance(values, dict):
        return {}
    return {str(key): value for key, value in values.items()}


@dataclass(slots=True)
class NarrativeEpisode:
    episode_id: str
    timestamp: int
    source: str
    raw_text: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NarrativeEpisode":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            timestamp=int(payload.get("timestamp", 0)),
            source=str(payload.get("source", "unknown")),
            raw_text=str(payload.get("raw_text", "")),
            tags=_coerce_str_list(payload.get("tags")),
            metadata=_coerce_object_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class CompiledNarrativeEvent:
    event_id: str
    timestamp: int
    setting: str
    actors: list[str]
    subject_role: str
    event_type: str
    outcome_type: str
    self_involvement: float
    witnessed: bool
    direct_harm: bool
    controllability_hint: float
    annotations: dict[str, object]
    source_episode_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CompiledNarrativeEvent":
        return cls(
            event_id=str(payload.get("event_id", "")),
            timestamp=int(payload.get("timestamp", 0)),
            setting=str(payload.get("setting", "unknown")),
            actors=_coerce_str_list(payload.get("actors")),
            subject_role=str(payload.get("subject_role", "observer")),
            event_type=str(payload.get("event_type", "unknown_event")),
            outcome_type=str(payload.get("outcome_type", "neutral")),
            self_involvement=float(payload.get("self_involvement", 0.0)),
            witnessed=bool(payload.get("witnessed", False)),
            direct_harm=bool(payload.get("direct_harm", False)),
            controllability_hint=float(payload.get("controllability_hint", 0.0)),
            annotations=_coerce_object_dict(payload.get("annotations")),
            source_episode_id=str(payload.get("source_episode_id", "")),
        )


@dataclass(slots=True)
class AppraisalVector:
    physical_threat: float = 0.0
    social_threat: float = 0.0
    uncertainty: float = 0.0
    controllability: float = 0.0
    novelty: float = 0.0
    loss: float = 0.0
    moral_salience: float = 0.0
    contamination: float = 0.0
    attachment_signal: float = 0.0
    trust_impact: float = 0.0
    self_efficacy_impact: float = 0.0
    meaning_violation: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(max(-1.0, min(1.0, value)))
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AppraisalVector":
        return cls(**_coerce_float_dict(payload))


@dataclass(slots=True)
class SemanticEvidence:
    evidence_id: str
    source_type: str
    label: str
    strength: float
    matched_text: list[str] = field(default_factory=list)
    direction: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "label": self.label,
            "strength": float(self.strength),
            "matched_text": list(self.matched_text),
            "direction": self.direction,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "SemanticEvidence":
        if not payload:
            return cls(evidence_id="", source_type="", label="", strength=0.0)
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            source_type=str(payload.get("source_type", "")),
            label=str(payload.get("label", "")),
            strength=float(payload.get("strength", 0.0)),
            matched_text=_coerce_str_list(payload.get("matched_text")),
            direction=str(payload.get("direction", "")),
            metadata=_coerce_object_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class SemanticGrounding:
    episode_id: str
    motifs: list[str] = field(default_factory=list)
    evidence: list[SemanticEvidence] = field(default_factory=list)
    semantic_direction_scores: dict[str, float] = field(default_factory=dict)
    lexical_surface_hits: dict[str, int] = field(default_factory=dict)
    paraphrase_hits: dict[str, int] = field(default_factory=dict)
    implicit_hits: dict[str, int] = field(default_factory=dict)
    supporting_segments: list[str] = field(default_factory=list)
    provenance: dict[str, object] = field(default_factory=dict)
    low_signal: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "motifs": list(self.motifs),
            "evidence": [item.to_dict() for item in self.evidence],
            "semantic_direction_scores": dict(self.semantic_direction_scores),
            "lexical_surface_hits": dict(self.lexical_surface_hits),
            "paraphrase_hits": dict(self.paraphrase_hits),
            "implicit_hits": dict(self.implicit_hits),
            "supporting_segments": list(self.supporting_segments),
            "provenance": dict(self.provenance),
            "low_signal": bool(self.low_signal),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "SemanticGrounding":
        if not payload:
            return cls(episode_id="")
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            motifs=_coerce_str_list(payload.get("motifs")),
            evidence=[
                SemanticEvidence.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
            semantic_direction_scores=_coerce_float_dict(payload.get("semantic_direction_scores")),
            lexical_surface_hits={
                str(key): int(value)
                for key, value in dict(payload.get("lexical_surface_hits", {})).items()
                if isinstance(value, (int, float))
            },
            paraphrase_hits={
                str(key): int(value)
                for key, value in dict(payload.get("paraphrase_hits", {})).items()
                if isinstance(value, (int, float))
            },
            implicit_hits={
                str(key): int(value)
                for key, value in dict(payload.get("implicit_hits", {})).items()
                if isinstance(value, (int, float))
            },
            supporting_segments=_coerce_str_list(payload.get("supporting_segments")),
            provenance=_coerce_object_dict(payload.get("provenance")),
            low_signal=bool(payload.get("low_signal", False)),
        )


@dataclass(slots=True)
class EmbodiedNarrativeEpisode:
    episode_id: str
    timestamp: int
    observation: dict[str, float]
    appraisal: dict[str, float]
    body_state: dict[str, float]
    predicted_outcome: str
    value_tags: list[str]
    narrative_tags: list[str]
    compiler_confidence: float
    provenance: dict[str, object]
    uncertainty_decomposition: dict[str, object] = field(default_factory=dict)
    semantic_grounding: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "observation": dict(self.observation),
            "appraisal": dict(self.appraisal),
            "body_state": dict(self.body_state),
            "predicted_outcome": self.predicted_outcome,
            "value_tags": list(self.value_tags),
            "narrative_tags": list(self.narrative_tags),
            "compiler_confidence": float(self.compiler_confidence),
            "provenance": dict(self.provenance),
            "uncertainty_decomposition": dict(self.uncertainty_decomposition),
            "semantic_grounding": dict(self.semantic_grounding),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EmbodiedNarrativeEpisode":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            timestamp=int(payload.get("timestamp", 0)),
            observation=_coerce_float_dict(payload.get("observation")),
            appraisal=_coerce_float_dict(payload.get("appraisal")),
            body_state=_coerce_float_dict(payload.get("body_state")),
            predicted_outcome=str(payload.get("predicted_outcome", "neutral")),
            value_tags=_coerce_str_list(payload.get("value_tags")),
            narrative_tags=_coerce_str_list(payload.get("narrative_tags")),
            compiler_confidence=float(payload.get("compiler_confidence", 0.0)),
            provenance=_coerce_object_dict(payload.get("provenance")),
            uncertainty_decomposition=_coerce_object_dict(
                payload.get("uncertainty_decomposition")
            ),
            semantic_grounding=_coerce_object_dict(payload.get("semantic_grounding")),
        )
