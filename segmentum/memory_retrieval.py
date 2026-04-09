from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Any

from .m4_cognitive_style import CognitiveStyleParameters
from .memory_model import AnchorStrength, MemoryClass, MemoryEntry
from .memory_state import identity_match_ratio_for_entry, normalize_agent_state

if TYPE_CHECKING:
    from .memory_store import MemoryStore


RETRIEVAL_W1_TAG = 0.35
RETRIEVAL_W2_CONTEXT = 0.15
RETRIEVAL_W3_MOOD = 0.15
RETRIEVAL_W4_ACCESSIBILITY = 0.20
RETRIEVAL_W5_RECENCY = 0.15
DEFAULT_DOMINANCE_THRESHOLD = 0.15

NEGATIVE_MOOD_TOKENS = {
    "afraid",
    "anxious",
    "fearful",
    "sad",
    "stressed",
    "threatened",
    "upset",
    "worried",
}
POSITIVE_MOOD_TOKENS = {
    "calm",
    "confident",
    "content",
    "happy",
    "hopeful",
    "safe",
    "steady",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_token(value: str) -> str:
    return str(value).strip().lower()


def _style_value(
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None,
    key: str,
    default: float = 0.0,
) -> float:
    if cognitive_style is not None and hasattr(cognitive_style, key):
        try:
            return float(getattr(cognitive_style, key))
        except (TypeError, ValueError):
            return default
    if isinstance(cognitive_style, dict):
        try:
            return float(cognitive_style.get(key, default))
        except (TypeError, ValueError):
            return default
    return default


def _token_set(values: list[str]) -> set[str]:
    return {_normalize_token(item) for item in values if _normalize_token(item)}


def _content_tokens(text: str) -> set[str]:
    return {
        chunk.lower()
        for chunk in str(text or "").replace(",", " ").replace(".", " ").replace(";", " ").split()
        if chunk.strip()
    }


def _jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _entry_state_vector(entry: MemoryEntry) -> list[float]:
    metadata = dict(entry.compression_metadata or {})
    explicit = metadata.get("state_vector")
    if isinstance(explicit, list):
        return [float(item) for item in explicit]
    legacy = metadata.get("legacy_template")
    if isinstance(legacy, dict):
        vector = legacy.get("embedding")
        if isinstance(vector, list):
            return [float(item) for item in vector]
    return []


def _keyword_overlap(query_keywords: list[str], entry: MemoryEntry) -> float:
    keywords = _token_set(query_keywords)
    if not keywords:
        return 0.0
    return _jaccard_overlap(keywords, _content_tokens(entry.content))


def _tag_overlap(query: "RetrievalQuery", entry: MemoryEntry) -> float:
    semantic_overlap = _jaccard_overlap(_token_set(query.semantic_tags), _token_set(entry.semantic_tags))
    keyword_overlap = _keyword_overlap(query.content_keywords, entry)
    if semantic_overlap <= 0.0:
        return keyword_overlap * 0.6
    if keyword_overlap <= 0.0:
        return semantic_overlap
    return _clamp((semantic_overlap * 0.75) + (keyword_overlap * 0.25))


def _context_overlap(query: "RetrievalQuery", entry: MemoryEntry) -> float:
    tag_overlap = _jaccard_overlap(_token_set(query.context_tags), _token_set(entry.context_tags))
    vector_overlap = _cosine_similarity(query.state_vector, _entry_state_vector(entry))
    if not query.context_tags:
        return _clamp(vector_overlap)
    if not query.state_vector:
        return _clamp(tag_overlap)
    return _clamp((tag_overlap * 0.6) + (vector_overlap * 0.4))


def _mood_polarity(text: str) -> str:
    tokens = _content_tokens(text)
    if tokens & NEGATIVE_MOOD_TOKENS:
        return "negative"
    if tokens & POSITIVE_MOOD_TOKENS:
        return "positive"
    return "neutral"


def _mood_match(current_mood: str | None, entry: MemoryEntry) -> float:
    if not current_mood:
        return 0.0
    current_polarity = _mood_polarity(current_mood)
    entry_polarity = _mood_polarity(entry.mood_context)
    if current_polarity == "negative":
        if entry.valence < -0.1:
            return 1.0
        if entry_polarity == "negative":
            return 0.8
        return 0.0
    if current_polarity == "positive":
        if entry.valence > 0.1:
            return 1.0
        if entry_polarity == "positive":
            return 0.8
        return 0.0
    if _normalize_token(current_mood) == _normalize_token(entry.mood_context):
        return 0.8
    return 0.2 if entry_polarity == "neutral" else 0.0


def _recency_bonus(entry: MemoryEntry, reference_cycle: int) -> float:
    if reference_cycle <= 0:
        return 0.0
    distance = max(0, int(reference_cycle) - max(entry.last_accessed, entry.created_at))
    return _clamp(1.0 / (1.0 + (distance / 20.0)))


def _entry_anchor_summary(entry: MemoryEntry) -> str:
    parts: list[str] = []
    for key in ("time", "place", "agents", "action", "outcome"):
        value = entry.anchor_slots.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _primary_recall_backbone(entry: MemoryEntry) -> str:
    protected = [
        key
        for key, strength in entry.anchor_strengths.items()
        if strength in {AnchorStrength.LOCKED, AnchorStrength.STRONG} and entry.anchor_slots.get(key)
    ]
    stable_tags = ", ".join(entry.semantic_tags[:3])
    protected_summary = ", ".join(
        f"{key}={entry.anchor_slots.get(key)}" for key in protected[:3]
    )
    if entry.memory_class is MemoryClass.SEMANTIC:
        return f"semantic pattern [{stable_tags or entry.memory_class.value}]"
    if entry.memory_class is MemoryClass.INFERRED:
        return f"candidate explanation [{stable_tags or entry.memory_class.value}]"
    if protected_summary:
        return f"episodic backbone [{protected_summary}]"
    if stable_tags:
        return f"{entry.memory_class.value} backbone [{stable_tags}]"
    return f"{entry.memory_class.value} backbone"


@dataclass(frozen=True)
class RetrievalQuery:
    semantic_tags: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    content_keywords: list[str] = field(default_factory=list)
    state_vector: list[float] = field(default_factory=list)
    reference_cycle: int = 0
    target_memory_class: MemoryClass | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_tags", _string_list(self.semantic_tags))
        object.__setattr__(self, "context_tags", _string_list(self.context_tags))
        object.__setattr__(self, "content_keywords", _string_list(self.content_keywords))
        object.__setattr__(self, "state_vector", [float(item) for item in self.state_vector])
        target = self.target_memory_class
        if target is not None and not isinstance(target, MemoryClass):
            object.__setattr__(self, "target_memory_class", MemoryClass(str(target)))
        object.__setattr__(self, "reference_cycle", int(self.reference_cycle))


@dataclass(frozen=True)
class ScoredCandidate:
    entry_id: str
    memory_class: str
    retrieval_score: float
    content_preview: str
    score_breakdown: dict[str, float]
    source_type: str
    validation_status: str
    entry: MemoryEntry = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "memory_class": self.memory_class,
            "retrieval_score": self.retrieval_score,
            "content_preview": self.content_preview,
            "score_breakdown": dict(self.score_breakdown),
            "source_type": self.source_type,
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class CompetitionResult:
    primary: MemoryEntry | None
    competitors: list[MemoryEntry]
    confidence: str
    interference_risk: bool
    competing_interpretations: list[str]
    dominance_margin: float

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_id": self.primary.id if self.primary is not None else None,
            "competitor_ids": [entry.id for entry in self.competitors],
            "confidence": self.confidence,
            "interference_risk": self.interference_risk,
            "competing_interpretations": list(self.competing_interpretations),
            "dominance_margin": self.dominance_margin,
        }


@dataclass(frozen=True)
class RecallArtifact:
    content: str
    primary_entry_id: str
    auxiliary_entry_ids: list[str]
    confidence: float
    source_trace: list[str]
    reconstructed_fields: list[str]
    protected_fields: list[str]
    competing_interpretations: list[str] | None = None
    procedure_step_outline: list[str] | None = None
    source_statuses: dict[str, str] | None = None
    candidate_ids: list[str] | None = None
    anchor_contributions: dict[str, list[dict[str, str]]] | None = None
    competition_snapshot: dict[str, object] | None = None
    donor_blocks: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "primary_entry_id": self.primary_entry_id,
            "auxiliary_entry_ids": list(self.auxiliary_entry_ids),
            "confidence": self.confidence,
            "source_trace": list(self.source_trace),
            "reconstructed_fields": list(self.reconstructed_fields),
            "protected_fields": list(self.protected_fields),
            "competing_interpretations": (
                list(self.competing_interpretations)
                if self.competing_interpretations is not None
                else None
            ),
            "procedure_step_outline": (
                list(self.procedure_step_outline) if self.procedure_step_outline is not None else None
            ),
            "source_statuses": dict(self.source_statuses or {}),
            "candidate_ids": list(self.candidate_ids or []),
            "anchor_contributions": deepcopy(self.anchor_contributions or {}),
            "competition_snapshot": deepcopy(self.competition_snapshot or {}),
            "donor_blocks": deepcopy(self.donor_blocks or []),
        }


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[ScoredCandidate]
    recall_hypothesis: RecallArtifact | None
    recall_confidence: float
    source_trace: list[str]
    reconstruction_trace: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "recall_hypothesis": (
                self.recall_hypothesis.to_dict() if self.recall_hypothesis is not None else None
            ),
            "recall_confidence": self.recall_confidence,
            "source_trace": list(self.source_trace),
            "reconstruction_trace": deepcopy(self.reconstruction_trace),
        }


def compete_candidates(
    candidates: list[ScoredCandidate],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> CompetitionResult:
    if not candidates:
        return CompetitionResult(
            primary=None,
            competitors=[],
            confidence="none",
            interference_risk=False,
            competing_interpretations=[],
            dominance_margin=0.0,
        )
    ranked = sorted(candidates, key=lambda item: item.retrieval_score, reverse=True)
    primary = ranked[0].entry
    if len(ranked) == 1:
        return CompetitionResult(
            primary=primary,
            competitors=[],
            confidence="high",
            interference_risk=False,
            competing_interpretations=[],
            dominance_margin=1.0,
        )
    margin = ranked[0].retrieval_score - ranked[1].retrieval_score
    if margin > dominance_threshold:
        return CompetitionResult(
            primary=primary,
            competitors=[],
            confidence="high",
            interference_risk=False,
            competing_interpretations=[],
            dominance_margin=margin,
        )
    competitors = [
        candidate.entry
        for candidate in ranked[1:]
        if (ranked[0].retrieval_score - candidate.retrieval_score) < dominance_threshold
    ]
    interpretations = [
        f"{entry.id}: alternative recall via {entry.content[:80]}"
        for entry in competitors
    ]
    return CompetitionResult(
        primary=primary,
        competitors=competitors,
        confidence="low",
        interference_risk=bool(competitors),
        competing_interpretations=interpretations,
        dominance_margin=margin,
    )


def build_recall_artifact(
    primary: MemoryEntry,
    auxiliaries: list[MemoryEntry],
    competition: CompetitionResult,
    query: RetrievalQuery,
    *,
    candidate_ids: list[str] | None = None,
    donor_blocks: list[dict[str, str]] | None = None,
) -> RecallArtifact:
    protected_fields = [
        key
        for key, strength in primary.anchor_strengths.items()
        if strength in {AnchorStrength.LOCKED, AnchorStrength.STRONG}
    ]
    reconstructed_fields: list[str] = []
    if auxiliaries:
        reconstructed_fields.append("auxiliary_detail_blend")
    if query.content_keywords:
        reconstructed_fields.append("query_focus")
    if query.context_tags:
        reconstructed_fields.append("context_focus")
    source_trace = [f"primary:{primary.id}"]
    source_trace.extend(f"auxiliary:{entry.id}" for entry in auxiliaries)
    source_statuses = {
        entry.id: str(dict(entry.compression_metadata or {}).get("validation_status", "validated"))
        for entry in [primary, *auxiliaries]
    }
    query_summary = ", ".join([*query.semantic_tags[:2], *query.context_tags[:2], *query.content_keywords[:2]])
    anchor_contributions: dict[str, list[dict[str, str]]] = {}
    for key in ("time", "place", "agents", "action", "outcome"):
        value = primary.anchor_slots.get(key)
        if value:
            anchor_contributions.setdefault(key, []).append(
                {"entry_id": primary.id, "role": "primary", "value": str(value)}
            )
    for auxiliary in auxiliaries:
        for key in ("time", "place", "agents", "action", "outcome"):
            value = auxiliary.anchor_slots.get(key)
            if not value:
                continue
            if primary.anchor_strengths.get(key) in {AnchorStrength.LOCKED, AnchorStrength.STRONG}:
                continue
            if any(item.get("value") == str(value) for item in anchor_contributions.get(key, [])):
                continue
            anchor_contributions.setdefault(key, []).append(
                {"entry_id": auxiliary.id, "role": "auxiliary", "value": str(value)}
            )
    competition_snapshot = competition.to_dict()
    if primary.memory_class is MemoryClass.PROCEDURAL:
        outline = list(primary.procedure_steps)
        for auxiliary in auxiliaries:
            for step in auxiliary.procedure_steps:
                if step not in outline:
                    outline.append(step)
        content = (
            f"Procedure recall reconstructed from {primary.id}"
            f" with {len(outline)} anchored steps"
        )
        if query_summary:
            content += f" for cues [{query_summary}]"
        return RecallArtifact(
            content=content,
            primary_entry_id=primary.id,
            auxiliary_entry_ids=[entry.id for entry in auxiliaries],
            confidence=0.85 if competition.confidence == "high" else 0.45,
            source_trace=source_trace,
            reconstructed_fields=reconstructed_fields or ["procedure_step_outline"],
            protected_fields=protected_fields or ["procedure_steps"],
            competing_interpretations=competition.competing_interpretations or None,
            procedure_step_outline=outline,
            source_statuses=source_statuses,
            candidate_ids=list(candidate_ids or []),
            anchor_contributions=anchor_contributions,
            competition_snapshot=competition_snapshot,
            donor_blocks=list(donor_blocks or []),
        )
    protected_summary = ", ".join(
        f"{field}={primary.anchor_slots.get(field)}"
        for field in protected_fields
        if primary.anchor_slots.get(field)
    )
    weak_donor_summary = ", ".join(
        f"{field}<-{items[-1]['entry_id']}"
        for field, items in anchor_contributions.items()
        if any(item.get("role") == "auxiliary" for item in items)
    )
    detail_fragments = [
        f"candidate_set:{', '.join(candidate_ids or [primary.id])}",
        f"primary_memory_class:{primary.memory_class.value}",
    ]
    if protected_summary:
        detail_fragments.append(f"protected_anchors:{protected_summary}")
    if weak_donor_summary:
        detail_fragments.append(f"weak_field_donors:{weak_donor_summary}")
    if query_summary:
        detail_fragments.append(f"cue_reconstruction:{query_summary}")
    if competition.competing_interpretations:
        detail_fragments.append("competition_preserved:alternative interpretations remain active")
    if donor_blocks:
        detail_fragments.append(
            "donor_blocks:" + ", ".join(f"{item['entry_id']}:{item['reason']}" for item in donor_blocks)
        )
    content = " | ".join(detail_fragments)
    return RecallArtifact(
        content=content,
        primary_entry_id=primary.id,
        auxiliary_entry_ids=[entry.id for entry in auxiliaries],
        confidence=0.85 if competition.confidence == "high" else 0.45,
        source_trace=source_trace,
        reconstructed_fields=reconstructed_fields or ["cue_guided_reconstruction"],
        protected_fields=protected_fields,
        competing_interpretations=competition.competing_interpretations or None,
        procedure_step_outline=None,
        source_statuses=source_statuses,
        candidate_ids=list(candidate_ids or []),
        anchor_contributions=anchor_contributions,
        competition_snapshot=competition_snapshot,
        donor_blocks=list(donor_blocks or []),
    )


def _filter_entries(query: RetrievalQuery, store: "MemoryStore") -> list[MemoryEntry]:
    candidates: list[MemoryEntry] = []
    for entry in store.entries:
        if entry.is_dormant:
            continue
        if query.target_memory_class is not None and entry.memory_class is not query.target_memory_class:
            continue
        candidates.append(entry)
    return candidates


def _score_entry(
    query: RetrievalQuery,
    entry: MemoryEntry,
    current_mood: str | None,
    *,
    agent_state=None,
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
) -> ScoredCandidate:
    tag_score = _tag_overlap(query, entry)
    context_score = _context_overlap(query, entry)
    mood_score = _mood_match(current_mood, entry)
    accessibility_score = _clamp(entry.accessibility)
    recency_score = _recency_bonus(entry, query.reference_cycle)
    exploration_bias = _clamp(_style_value(cognitive_style, "exploration_bias", 0.0))
    attention_selectivity = _clamp(_style_value(cognitive_style, "attention_selectivity", 0.0))
    novelty_bonus = exploration_bias * 0.2 * (1.0 / (1.0 + max(0, entry.retrieval_count)))
    specificity_bonus = attention_selectivity * 0.12 * tag_score
    identity_alignment = identity_match_ratio_for_entry(entry, agent_state) * 0.15
    validation_status = str(dict(entry.compression_metadata or {}).get("validation_status", "validated"))
    validation_discount = 1.0
    if entry.memory_class is MemoryClass.INFERRED:
        validation_discount = _clamp(
            float(dict(entry.compression_metadata or {}).get("validation_discount", 0.35)),
            0.0,
            1.0,
        )
    retrieval_score = (
        (RETRIEVAL_W1_TAG * tag_score)
        + (RETRIEVAL_W2_CONTEXT * context_score)
        + (RETRIEVAL_W3_MOOD * mood_score)
        + (RETRIEVAL_W4_ACCESSIBILITY * accessibility_score)
        + (RETRIEVAL_W5_RECENCY * recency_score)
        + novelty_bonus
        + specificity_bonus
        + identity_alignment
    )
    retrieval_score *= validation_discount
    return ScoredCandidate(
        entry_id=entry.id,
        memory_class=entry.memory_class.value,
        retrieval_score=round(retrieval_score, 6),
        content_preview=entry.content[:120],
        score_breakdown={
            "tag_overlap": round(tag_score, 6),
            "context_overlap": round(context_score, 6),
            "mood_match": round(mood_score, 6),
            "accessibility": round(accessibility_score, 6),
            "recency": round(recency_score, 6),
            **(
                {"novelty_bonus": round(novelty_bonus, 6)}
                if novelty_bonus > 0.0
                else {}
            ),
            **(
                {"specificity_bonus": round(specificity_bonus, 6)}
                if specificity_bonus > 0.0
                else {}
            ),
            **(
                {"identity_alignment": round(identity_alignment, 6)}
                if identity_alignment > 0.0
                else {}
            ),
            **(
                {"validation_discount": round(validation_discount, 6)}
                if validation_discount < 0.999999 or entry.memory_class is MemoryClass.INFERRED
                else {}
            ),
        },
        source_type=entry.source_type.value,
        validation_status=validation_status,
        entry=entry,
    )


def retrieve(
    query: RetrievalQuery,
    store: "MemoryStore",
    current_mood: str | None = None,
    k: int = 5,
    *,
    agent_state=None,
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
) -> RetrievalResult:
    normalized_state = normalize_agent_state(agent_state or getattr(store, "agent_state_vector", None))
    attention_selectivity = _clamp(_style_value(cognitive_style, "attention_selectivity", 0.0))
    ranked = sorted(
        [
            _score_entry(
                query,
                entry,
                current_mood,
                agent_state=normalized_state,
                cognitive_style=cognitive_style,
            )
            for entry in _filter_entries(query, store)
        ],
        key=lambda item: (item.retrieval_score, item.entry.trace_strength, item.entry.id),
        reverse=True,
    )
    top_candidates = ranked[: max(0, int(k))]
    dominance_threshold = max(0.03, DEFAULT_DOMINANCE_THRESHOLD - (attention_selectivity * 0.15))
    competition = compete_candidates(top_candidates, dominance_threshold=dominance_threshold)
    recall_hypothesis: RecallArtifact | None = None
    reconstruction_trace: dict[str, object] = {
        "query": {
            "semantic_tags": list(query.semantic_tags),
            "context_tags": list(query.context_tags),
            "content_keywords": list(query.content_keywords),
            "reference_cycle": query.reference_cycle,
            "target_memory_class": (
                query.target_memory_class.value if query.target_memory_class is not None else None
            ),
            "agent_state_vector": normalized_state.to_dict(),
            "dominance_threshold": round(dominance_threshold, 6),
        },
        "competition": competition.to_dict(),
    }
    source_trace: list[str] = []
    if competition.primary is not None:
        auxiliaries: list[MemoryEntry] = []
        donor_blocks: list[dict[str, str]] = []
        for candidate in top_candidates[1:]:
            if candidate.entry.memory_class is MemoryClass.INFERRED:
                metadata = dict(candidate.entry.compression_metadata or {})
                if str(metadata.get("validation_status", "unvalidated")) in {"unvalidated", "contradicted"}:
                    donor_blocks.append(
                        {
                            "entry_id": candidate.entry.id,
                            "reason": str(metadata.get("validation_status", "unvalidated")),
                        }
                    )
                    continue
            if len(auxiliaries) >= 2:
                break
            auxiliaries.append(candidate.entry)
        recall_hypothesis = build_recall_artifact(
            competition.primary,
            auxiliaries,
            competition,
            query,
            candidate_ids=[candidate.entry_id for candidate in top_candidates],
            donor_blocks=donor_blocks,
        )
        source_trace = list(recall_hypothesis.source_trace)
        reconstruction_trace["selected_auxiliaries"] = [entry.id for entry in auxiliaries]
        reconstruction_trace["validation_statuses"] = dict(recall_hypothesis.source_statuses or {})
        reconstruction_trace["candidate_ids"] = list(recall_hypothesis.candidate_ids or [])
        reconstruction_trace["anchor_contributions"] = deepcopy(recall_hypothesis.anchor_contributions or {})
        reconstruction_trace["competition_snapshot"] = deepcopy(recall_hypothesis.competition_snapshot or {})
        reconstruction_trace["donor_blocks"] = deepcopy(recall_hypothesis.donor_blocks or [])
        if competition.primary.compression_metadata is None:
            competition.primary.compression_metadata = {}
        if isinstance(competition.primary.compression_metadata, dict):
            competition.primary.compression_metadata["m47_recall_audit"] = {
                "candidate_ids": list(recall_hypothesis.candidate_ids or []),
                "anchor_contributions": deepcopy(recall_hypothesis.anchor_contributions or {}),
                "competition_snapshot": deepcopy(recall_hypothesis.competition_snapshot or {}),
                "donor_blocks": deepcopy(recall_hypothesis.donor_blocks or []),
            }
    return RetrievalResult(
        candidates=top_candidates,
        recall_hypothesis=recall_hypothesis,
        recall_confidence=recall_hypothesis.confidence if recall_hypothesis is not None else 0.0,
        source_trace=source_trace,
        reconstruction_trace=reconstruction_trace,
    )
