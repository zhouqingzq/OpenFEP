from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from .narrative_types import AppraisalVector, CompiledNarrativeEvent, NarrativeEpisode


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _to_str_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value) for value in values if str(value))


def _to_int_tuple(values: object) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _to_float_dict(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    payload: dict[str, float] = {}
    for key, value in values.items():
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


class NarrativeUnknownType(StrEnum):
    TRUST = "trust"
    INTENT = "intent"
    THREAT_PERSISTENCE = "threat_persistence"
    SOCIAL_RUPTURE_CAUSE = "social_rupture_cause"
    ENVIRONMENT_RELIABILITY = "environment_reliability"
    COMMUNICATION = "communication"
    MOTIVE = "motive"
    GENERAL = "general"


class LatentCauseType(StrEnum):
    PERSISTENT_BETRAYAL = "persistent_betrayal"
    TEMPORARY_CONSTRAINT = "temporary_constraint"
    MISCOMMUNICATION = "miscommunication"
    PERSISTENT_THREAT = "persistent_threat"
    LOCAL_ACCIDENT = "local_accident"
    ENVIRONMENTAL_INTERFERENCE = "environmental_interference"
    PROTECTIVE_SUPPORT = "protective_support"
    UNKNOWN = "unknown"


class SurfaceCueType(StrEnum):
    RHETORICAL_CONTRAST = "rhetorical_contrast"
    EMOTIONAL_INTENSIFIER = "emotional_intensifier"
    LOW_SIGNAL_NOISE = "low_signal_noise"
    DECORATIVE_DETAIL = "decorative_detail"


@dataclass(frozen=True)
class DecisionRelevanceMap:
    action_choice: float = 0.0
    social_stance: float = 0.0
    risk_level: float = 0.0
    verification_urgency: float = 0.0
    memory_protection: float = 0.0
    continuity_impact: float = 0.0
    downstream_prediction_delta: float = 0.0
    total_score: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "action_choice": round(self.action_choice, 6),
            "social_stance": round(self.social_stance, 6),
            "risk_level": round(self.risk_level, 6),
            "verification_urgency": round(self.verification_urgency, 6),
            "memory_protection": round(self.memory_protection, 6),
            "continuity_impact": round(self.continuity_impact, 6),
            "downstream_prediction_delta": round(self.downstream_prediction_delta, 6),
            "total_score": round(self.total_score, 6),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DecisionRelevanceMap":
        if not payload:
            return cls()
        return cls(
            action_choice=float(payload.get("action_choice", 0.0)),
            social_stance=float(payload.get("social_stance", 0.0)),
            risk_level=float(payload.get("risk_level", 0.0)),
            verification_urgency=float(payload.get("verification_urgency", 0.0)),
            memory_protection=float(payload.get("memory_protection", 0.0)),
            continuity_impact=float(payload.get("continuity_impact", 0.0)),
            downstream_prediction_delta=float(payload.get("downstream_prediction_delta", 0.0)),
            total_score=float(payload.get("total_score", 0.0)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class HypothesisSupport:
    support_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    support_score: float = 0.0
    contradiction_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "support_evidence": list(self.support_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "support_score": round(self.support_score, 6),
            "contradiction_score": round(self.contradiction_score, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "HypothesisSupport":
        if not payload:
            return cls()
        return cls(
            support_evidence=_to_str_tuple(payload.get("support_evidence", [])),
            contradicting_evidence=_to_str_tuple(payload.get("contradicting_evidence", [])),
            support_score=float(payload.get("support_score", 0.0)),
            contradiction_score=float(payload.get("contradiction_score", 0.0)),
        )


@dataclass(frozen=True)
class NarrativeUnknown:
    unknown_id: str
    unknown_type: str
    source_episode_id: str
    source_span: str
    unresolved_reason: str
    uncertainty_level: float
    action_relevant: bool
    linked_entities: tuple[str, ...] = ()
    linked_chapters: tuple[int, ...] = ()
    evidence_links: tuple[str, ...] = ()
    decision_relevance: DecisionRelevanceMap = field(default_factory=DecisionRelevanceMap)
    competing_hypothesis_ids: tuple[str, ...] = ()
    latent_cause_ids: tuple[str, ...] = ()
    surface_cue_ids: tuple[str, ...] = ()
    promotion_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "unknown_id": self.unknown_id,
            "unknown_type": self.unknown_type,
            "source_episode_id": self.source_episode_id,
            "source_span": self.source_span,
            "unresolved_reason": self.unresolved_reason,
            "uncertainty_level": round(self.uncertainty_level, 6),
            "action_relevant": bool(self.action_relevant),
            "linked_entities": list(self.linked_entities),
            "linked_chapters": list(self.linked_chapters),
            "evidence_links": list(self.evidence_links),
            "decision_relevance": self.decision_relevance.to_dict(),
            "competing_hypothesis_ids": list(self.competing_hypothesis_ids),
            "latent_cause_ids": list(self.latent_cause_ids),
            "surface_cue_ids": list(self.surface_cue_ids),
            "promotion_reason": self.promotion_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NarrativeUnknown":
        if not payload:
            return cls(
                unknown_id="",
                unknown_type=NarrativeUnknownType.GENERAL.value,
                source_episode_id="",
                source_span="",
                unresolved_reason="",
                uncertainty_level=0.0,
                action_relevant=False,
            )
        return cls(
            unknown_id=str(payload.get("unknown_id", "")),
            unknown_type=str(payload.get("unknown_type", NarrativeUnknownType.GENERAL.value)),
            source_episode_id=str(payload.get("source_episode_id", "")),
            source_span=str(payload.get("source_span", "")),
            unresolved_reason=str(payload.get("unresolved_reason", "")),
            uncertainty_level=float(payload.get("uncertainty_level", 0.0)),
            action_relevant=bool(payload.get("action_relevant", False)),
            linked_entities=_to_str_tuple(payload.get("linked_entities", [])),
            linked_chapters=_to_int_tuple(payload.get("linked_chapters", [])),
            evidence_links=_to_str_tuple(payload.get("evidence_links", [])),
            decision_relevance=DecisionRelevanceMap.from_dict(
                payload.get("decision_relevance")
                if isinstance(payload.get("decision_relevance"), Mapping)
                else None
            ),
            competing_hypothesis_ids=_to_str_tuple(payload.get("competing_hypothesis_ids", [])),
            latent_cause_ids=_to_str_tuple(payload.get("latent_cause_ids", [])),
            surface_cue_ids=_to_str_tuple(payload.get("surface_cue_ids", [])),
            promotion_reason=str(payload.get("promotion_reason", "")),
        )


@dataclass(frozen=True)
class CompetingHypothesis:
    hypothesis_id: str
    parent_unknown_id: str
    statement: str
    prior_plausibility: float
    support: HypothesisSupport = field(default_factory=HypothesisSupport)
    implied_consequences: tuple[str, ...] = ()
    expected_state_shift: dict[str, float] = field(default_factory=dict)
    decision_relevance: DecisionRelevanceMap = field(default_factory=DecisionRelevanceMap)
    latent_cause_type: str = LatentCauseType.UNKNOWN.value

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "parent_unknown_id": self.parent_unknown_id,
            "statement": self.statement,
            "prior_plausibility": round(self.prior_plausibility, 6),
            "support": self.support.to_dict(),
            "implied_consequences": list(self.implied_consequences),
            "expected_state_shift": {
                str(key): round(float(value), 6)
                for key, value in self.expected_state_shift.items()
            },
            "decision_relevance": self.decision_relevance.to_dict(),
            "latent_cause_type": self.latent_cause_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "CompetingHypothesis":
        if not payload:
            return cls(
                hypothesis_id="",
                parent_unknown_id="",
                statement="",
                prior_plausibility=0.0,
            )
        return cls(
            hypothesis_id=str(payload.get("hypothesis_id", "")),
            parent_unknown_id=str(payload.get("parent_unknown_id", "")),
            statement=str(payload.get("statement", "")),
            prior_plausibility=float(payload.get("prior_plausibility", 0.0)),
            support=HypothesisSupport.from_dict(
                payload.get("support") if isinstance(payload.get("support"), Mapping) else None
            ),
            implied_consequences=_to_str_tuple(payload.get("implied_consequences", [])),
            expected_state_shift=_to_float_dict(payload.get("expected_state_shift", {})),
            decision_relevance=DecisionRelevanceMap.from_dict(
                payload.get("decision_relevance")
                if isinstance(payload.get("decision_relevance"), Mapping)
                else None
            ),
            latent_cause_type=str(payload.get("latent_cause_type", LatentCauseType.UNKNOWN.value)),
        )


@dataclass(frozen=True)
class LatentCauseCandidate:
    cause_id: str
    parent_unknown_id: str
    cause_type: str
    statement: str
    confidence: float
    evidence_links: tuple[str, ...] = ()
    decision_relevance: DecisionRelevanceMap = field(default_factory=DecisionRelevanceMap)

    def to_dict(self) -> dict[str, object]:
        return {
            "cause_id": self.cause_id,
            "parent_unknown_id": self.parent_unknown_id,
            "cause_type": self.cause_type,
            "statement": self.statement,
            "confidence": round(self.confidence, 6),
            "evidence_links": list(self.evidence_links),
            "decision_relevance": self.decision_relevance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "LatentCauseCandidate":
        if not payload:
            return cls(
                cause_id="",
                parent_unknown_id="",
                cause_type=LatentCauseType.UNKNOWN.value,
                statement="",
                confidence=0.0,
            )
        return cls(
            cause_id=str(payload.get("cause_id", "")),
            parent_unknown_id=str(payload.get("parent_unknown_id", "")),
            cause_type=str(payload.get("cause_type", LatentCauseType.UNKNOWN.value)),
            statement=str(payload.get("statement", "")),
            confidence=float(payload.get("confidence", 0.0)),
            evidence_links=_to_str_tuple(payload.get("evidence_links", [])),
            decision_relevance=DecisionRelevanceMap.from_dict(
                payload.get("decision_relevance")
                if isinstance(payload.get("decision_relevance"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class SurfaceCue:
    cue_id: str
    source_episode_id: str
    cue_type: str
    cue_text: str
    salience: float
    decision_relevance: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "source_episode_id": self.source_episode_id,
            "cue_type": self.cue_type,
            "cue_text": self.cue_text,
            "salience": round(self.salience, 6),
            "decision_relevance": round(self.decision_relevance, 6),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SurfaceCue":
        if not payload:
            return cls(
                cue_id="",
                source_episode_id="",
                cue_type=SurfaceCueType.DECORATIVE_DETAIL.value,
                cue_text="",
                salience=0.0,
                decision_relevance=0.0,
                rationale="",
            )
        return cls(
            cue_id=str(payload.get("cue_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            cue_type=str(payload.get("cue_type", SurfaceCueType.DECORATIVE_DETAIL.value)),
            cue_text=str(payload.get("cue_text", "")),
            salience=float(payload.get("salience", 0.0)),
            decision_relevance=float(payload.get("decision_relevance", 0.0)),
            rationale=str(payload.get("rationale", "")),
        )


@dataclass(frozen=True)
class NarrativeAmbiguityProfile:
    total_unknown_count: int = 0
    decision_relevant_unknown_count: int = 0
    competing_hypothesis_count: int = 0
    latent_cause_count: int = 0
    surface_cue_count: int = 0
    interpretive_competition: float = 0.0
    latent_cause_uncertainty_burden: float = 0.0
    social_ambiguity_burden: float = 0.0
    trust_ambiguity_burden: float = 0.0
    environment_ambiguity_burden: float = 0.0
    retained_uncertainty_burden: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_unknown_count": int(self.total_unknown_count),
            "decision_relevant_unknown_count": int(self.decision_relevant_unknown_count),
            "competing_hypothesis_count": int(self.competing_hypothesis_count),
            "latent_cause_count": int(self.latent_cause_count),
            "surface_cue_count": int(self.surface_cue_count),
            "interpretive_competition": round(self.interpretive_competition, 6),
            "latent_cause_uncertainty_burden": round(self.latent_cause_uncertainty_burden, 6),
            "social_ambiguity_burden": round(self.social_ambiguity_burden, 6),
            "trust_ambiguity_burden": round(self.trust_ambiguity_burden, 6),
            "environment_ambiguity_burden": round(self.environment_ambiguity_burden, 6),
            "retained_uncertainty_burden": round(self.retained_uncertainty_burden, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NarrativeAmbiguityProfile":
        if not payload:
            return cls()
        return cls(
            total_unknown_count=int(payload.get("total_unknown_count", 0)),
            decision_relevant_unknown_count=int(payload.get("decision_relevant_unknown_count", 0)),
            competing_hypothesis_count=int(payload.get("competing_hypothesis_count", 0)),
            latent_cause_count=int(payload.get("latent_cause_count", 0)),
            surface_cue_count=int(payload.get("surface_cue_count", 0)),
            interpretive_competition=float(payload.get("interpretive_competition", 0.0)),
            latent_cause_uncertainty_burden=float(
                payload.get("latent_cause_uncertainty_burden", 0.0)
            ),
            social_ambiguity_burden=float(payload.get("social_ambiguity_burden", 0.0)),
            trust_ambiguity_burden=float(payload.get("trust_ambiguity_burden", 0.0)),
            environment_ambiguity_burden=float(
                payload.get("environment_ambiguity_burden", 0.0)
            ),
            retained_uncertainty_burden=float(payload.get("retained_uncertainty_burden", 0.0)),
        )


@dataclass(frozen=True)
class UncertaintyDecompositionResult:
    episode_id: str = ""
    source_episode_id: str = ""
    unknowns: tuple[NarrativeUnknown, ...] = ()
    competing_hypotheses: tuple[CompetingHypothesis, ...] = ()
    latent_causes: tuple[LatentCauseCandidate, ...] = ()
    surface_cues: tuple[SurfaceCue, ...] = ()
    profile: NarrativeAmbiguityProfile = field(default_factory=NarrativeAmbiguityProfile)
    promoted_unknown_ids: tuple[str, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "source_episode_id": self.source_episode_id,
            "unknowns": [item.to_dict() for item in self.unknowns],
            "competing_hypotheses": [item.to_dict() for item in self.competing_hypotheses],
            "latent_causes": [item.to_dict() for item in self.latent_causes],
            "surface_cues": [item.to_dict() for item in self.surface_cues],
            "profile": self.profile.to_dict(),
            "promoted_unknown_ids": list(self.promoted_unknown_ids),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "UncertaintyDecompositionResult":
        if not payload:
            return cls()
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            unknowns=tuple(
                NarrativeUnknown.from_dict(item)
                for item in payload.get("unknowns", [])
                if isinstance(item, Mapping)
            ),
            competing_hypotheses=tuple(
                CompetingHypothesis.from_dict(item)
                for item in payload.get("competing_hypotheses", [])
                if isinstance(item, Mapping)
            ),
            latent_causes=tuple(
                LatentCauseCandidate.from_dict(item)
                for item in payload.get("latent_causes", [])
                if isinstance(item, Mapping)
            ),
            surface_cues=tuple(
                SurfaceCue.from_dict(item)
                for item in payload.get("surface_cues", [])
                if isinstance(item, Mapping)
            ),
            profile=NarrativeAmbiguityProfile.from_dict(
                payload.get("profile") if isinstance(payload.get("profile"), Mapping) else None
            ),
            promoted_unknown_ids=_to_str_tuple(payload.get("promoted_unknown_ids", [])),
            summary=str(payload.get("summary", "")),
        )

    def explanation_payload(self) -> dict[str, object]:
        top_unknowns = [item.to_dict() for item in self.unknowns[:3]]
        hypotheses_by_unknown: dict[str, list[dict[str, object]]] = {}
        for hypothesis in self.competing_hypotheses:
            hypotheses_by_unknown.setdefault(hypothesis.parent_unknown_id, []).append(
                hypothesis.to_dict()
            )
        return {
            "summary": self.summary,
            "profile": self.profile.to_dict(),
            "top_unknowns": top_unknowns,
            "hypotheses_by_unknown": hypotheses_by_unknown,
            "surface_cues": [item.to_dict() for item in self.surface_cues[:4]],
            "latent_causes": [item.to_dict() for item in self.latent_causes[:4]],
        }


class NarrativeUncertaintyDecomposer:
    def __init__(
        self,
        *,
        max_unknowns: int = 3,
        max_hypotheses_per_unknown: int = 3,
        promotion_threshold: float = 0.34,
        surface_relevance_threshold: float = 0.24,
    ) -> None:
        self.max_unknowns = max(1, int(max_unknowns))
        self.max_hypotheses_per_unknown = max(2, int(max_hypotheses_per_unknown))
        self.promotion_threshold = float(promotion_threshold)
        self.surface_relevance_threshold = float(surface_relevance_threshold)

    def decompose(
        self,
        *,
        episode: NarrativeEpisode,
        compiled_event: CompiledNarrativeEvent,
        appraisal: AppraisalVector,
    ) -> UncertaintyDecompositionResult:
        text = episode.raw_text.casefold()
        metadata = dict(episode.metadata)
        chapter_ids = self._linked_chapters(metadata)
        entities = self._linked_entities(metadata)
        surface_cues = self._surface_cues(episode=episode, text=text, compiled_event=compiled_event)
        unknowns = self._candidate_unknowns(
            episode=episode,
            text=text,
            compiled_event=compiled_event,
            appraisal=appraisal,
            linked_entities=entities,
            linked_chapters=chapter_ids,
            surface_cues=surface_cues,
        )
        hypotheses: list[CompetingHypothesis] = []
        latent_causes: list[LatentCauseCandidate] = []
        finalized_unknowns: list[NarrativeUnknown] = []
        promoted_ids: list[str] = []
        for unknown in unknowns[: self.max_unknowns]:
            generated = self._hypotheses_for_unknown(
                unknown=unknown,
                compiled_event=compiled_event,
                appraisal=appraisal,
                text=text,
            )[: self.max_hypotheses_per_unknown]
            cause_ids: list[str] = []
            for hypothesis in generated:
                hypotheses.append(hypothesis)
                if hypothesis.decision_relevance.total_score >= self.promotion_threshold:
                    cause_id = f"{hypothesis.hypothesis_id}:cause"
                    latent_causes.append(
                        LatentCauseCandidate(
                            cause_id=cause_id,
                            parent_unknown_id=unknown.unknown_id,
                            cause_type=hypothesis.latent_cause_type,
                            statement=hypothesis.statement,
                            confidence=_clamp(
                                hypothesis.prior_plausibility * 0.55
                                + hypothesis.support.support_score * 0.35
                                - hypothesis.support.contradiction_score * 0.18
                            ),
                            evidence_links=hypothesis.support.support_evidence,
                            decision_relevance=hypothesis.decision_relevance,
                        )
                    )
                    cause_ids.append(cause_id)
            hypothesis_ids = tuple(item.hypothesis_id for item in generated)
            action_relevant = bool(unknown.decision_relevance.total_score >= self.promotion_threshold)
            if action_relevant:
                promoted_ids.append(unknown.unknown_id)
            finalized_unknowns.append(
                NarrativeUnknown(
                    **{
                        **unknown.__dict__,
                        "action_relevant": action_relevant,
                        "competing_hypothesis_ids": hypothesis_ids,
                        "latent_cause_ids": tuple(cause_ids),
                    }
                )
            )
        profile = self._profile(finalized_unknowns, hypotheses, latent_causes, surface_cues)
        summary = self._summary(finalized_unknowns, hypotheses, surface_cues, profile)
        return UncertaintyDecompositionResult(
            episode_id=episode.episode_id,
            source_episode_id=episode.episode_id,
            unknowns=tuple(finalized_unknowns),
            competing_hypotheses=tuple(hypotheses),
            latent_causes=tuple(latent_causes),
            surface_cues=tuple(surface_cues),
            profile=profile,
            promoted_unknown_ids=tuple(promoted_ids),
            summary=summary,
        )

    def _linked_entities(self, metadata: Mapping[str, object]) -> tuple[str, ...]:
        entities: list[str] = []
        for key in ("counterpart_id", "environment_id", "commitment_id", "thread_id"):
            value = metadata.get(key)
            if value:
                entities.append(str(value))
        return tuple(dict.fromkeys(entities))

    def _linked_chapters(self, metadata: Mapping[str, object]) -> tuple[int, ...]:
        values = metadata.get("chapter_ids")
        chapter_ids = list(_to_int_tuple(values))
        if not chapter_ids and "chapter_id" in metadata:
            try:
                chapter_ids.append(int(metadata["chapter_id"]))
            except (TypeError, ValueError):
                pass
        return tuple(dict.fromkeys(chapter_ids))

    def _surface_cues(
        self,
        *,
        episode: NarrativeEpisode,
        text: str,
        compiled_event: CompiledNarrativeEvent,
    ) -> list[SurfaceCue]:
        cues: list[SurfaceCue] = []
        if any(token in text for token in ("but", "however", "yet", "although", "但是", "可是")):
            cues.append(
                SurfaceCue(
                    cue_id=f"{episode.episode_id}:surface:contrast",
                    source_episode_id=episode.episode_id,
                    cue_type=SurfaceCueType.RHETORICAL_CONTRAST.value,
                    cue_text="contrastive connector",
                    salience=0.46,
                    decision_relevance=0.22,
                    rationale="Contrast increases ambiguity but is not itself a latent cause.",
                )
            )
        if any(token in text for token in ("very", "extremely", "always", "never", "so much", "!")):
            cues.append(
                SurfaceCue(
                    cue_id=f"{episode.episode_id}:surface:intense",
                    source_episode_id=episode.episode_id,
                    cue_type=SurfaceCueType.EMOTIONAL_INTENSIFIER.value,
                    cue_text="emotional intensifier",
                    salience=0.42,
                    decision_relevance=0.18,
                    rationale="Emphasis raises salience but usually does not decide policy by itself.",
                )
            )
        annotations = dict(compiled_event.annotations)
        if bool(annotations.get("low_signal", False)):
            cues.append(
                SurfaceCue(
                    cue_id=f"{episode.episode_id}:surface:low_signal",
                    source_episode_id=episode.episode_id,
                    cue_type=SurfaceCueType.LOW_SIGNAL_NOISE.value,
                    cue_text="weak semantic support",
                    salience=0.30,
                    decision_relevance=0.12,
                    rationale="Low-signal detail should be discounted unless tied to later action impact.",
                )
            )
        return cues[:3]

    def _candidate_unknowns(
        self,
        *,
        episode: NarrativeEpisode,
        text: str,
        compiled_event: CompiledNarrativeEvent,
        appraisal: AppraisalVector,
        linked_entities: tuple[str, ...],
        linked_chapters: tuple[int, ...],
        surface_cues: list[SurfaceCue],
    ) -> list[NarrativeUnknown]:
        candidates: list[NarrativeUnknown] = []
        social_threat = max(0.0, float(appraisal.social_threat))
        physical_threat = max(0.0, float(appraisal.physical_threat))
        uncertainty = max(0.0, float(appraisal.uncertainty))
        trust_impact = float(appraisal.trust_impact)
        conflict_cues = list(compiled_event.annotations.get("conflict_cues", []))
        surface_ids = tuple(
            item.cue_id
            for item in surface_cues
            if item.decision_relevance <= self.surface_relevance_threshold
        )
        if compiled_event.event_type == "social_exclusion" or trust_impact <= -0.25:
            relevance = self._decision_relevance(
                unknown_type=NarrativeUnknownType.TRUST.value,
                risk=max(0.18, social_threat * 0.65),
                social=max(0.38, abs(trust_impact) * 0.95),
                action=0.44,
                verification=max(0.42, uncertainty * 0.70),
                continuity=0.26 if linked_entities else 0.18,
            )
            candidates.append(
                NarrativeUnknown(
                    unknown_id=f"{episode.episode_id}:unknown:trust",
                    unknown_type=NarrativeUnknownType.TRUST.value,
                    source_episode_id=episode.episode_id,
                    source_span=episode.raw_text[:180],
                    unresolved_reason="The narrative indicates social rupture, but the counterpart's trustworthiness is not settled.",
                    uncertainty_level=_clamp(max(uncertainty, 0.48)),
                    action_relevant=False,
                    linked_entities=linked_entities,
                    linked_chapters=linked_chapters,
                    evidence_links=(compiled_event.event_type, *tuple(conflict_cues[:2])),
                    decision_relevance=relevance,
                    surface_cue_ids=surface_ids,
                    promotion_reason="Trust ambiguity changes social stance and later verification policy.",
                )
            )
        if compiled_event.event_type in {"predator_attack", "witnessed_death"} or physical_threat >= 0.35:
            relevance = self._decision_relevance(
                unknown_type=NarrativeUnknownType.THREAT_PERSISTENCE.value,
                risk=max(0.50, physical_threat * 0.95),
                social=0.08,
                action=0.48,
                verification=max(0.40, uncertainty * 0.72),
                continuity=0.22,
            )
            candidates.append(
                NarrativeUnknown(
                    unknown_id=f"{episode.episode_id}:unknown:threat_persistence",
                    unknown_type=NarrativeUnknownType.THREAT_PERSISTENCE.value,
                    source_episode_id=episode.episode_id,
                    source_span=episode.raw_text[:180],
                    unresolved_reason="The narrative shows danger, but it is unclear whether the cause is stable or accidental.",
                    uncertainty_level=_clamp(max(uncertainty, 0.42)),
                    action_relevant=False,
                    linked_entities=linked_entities,
                    linked_chapters=linked_chapters,
                    evidence_links=(compiled_event.event_type, *tuple(conflict_cues[:2])),
                    decision_relevance=relevance,
                    surface_cue_ids=surface_ids,
                    promotion_reason="Threat persistence changes risk posture, memory protection, and verification urgency.",
                )
            )
        if conflict_cues or any(
            token in text
            for token in (
                "unclear",
                "uncertain",
                "didn't know",
                "not sure",
                "mixed signals",
                "why",
                "whether",
                "不确定",
                "不知道",
                "为什么",
            )
        ):
            relevance = self._decision_relevance(
                unknown_type=NarrativeUnknownType.ENVIRONMENT_RELIABILITY.value,
                risk=max(0.16, physical_threat * 0.48),
                social=max(0.12, social_threat * 0.32),
                action=0.26,
                verification=max(0.34, uncertainty * 0.76),
                continuity=0.12,
            )
            candidates.append(
                NarrativeUnknown(
                    unknown_id=f"{episode.episode_id}:unknown:environment",
                    unknown_type=NarrativeUnknownType.ENVIRONMENT_RELIABILITY.value,
                    source_episode_id=episode.episode_id,
                    source_span=episode.raw_text[:180],
                    unresolved_reason="The narrative contains contrast or explicit uncertainty, so the environment or evidence source may be misleading.",
                    uncertainty_level=_clamp(max(uncertainty, 0.34)),
                    action_relevant=False,
                    linked_entities=linked_entities,
                    linked_chapters=linked_chapters,
                    evidence_links=tuple(conflict_cues[:3] or ["explicit_uncertainty"]),
                    decision_relevance=relevance,
                    surface_cue_ids=surface_ids,
                    promotion_reason="Evidence reliability matters when later predictions depend on the same narrative frame.",
                )
            )
        if compiled_event.event_type == "rescue" and conflict_cues:
            relevance = self._decision_relevance(
                unknown_type=NarrativeUnknownType.MOTIVE.value,
                risk=0.14,
                social=max(0.32, abs(trust_impact) * 0.65),
                action=0.30,
                verification=max(0.36, uncertainty * 0.64),
                continuity=0.20 if linked_entities else 0.10,
            )
            candidates.append(
                NarrativeUnknown(
                    unknown_id=f"{episode.episode_id}:unknown:motive",
                    unknown_type=NarrativeUnknownType.MOTIVE.value,
                    source_episode_id=episode.episode_id,
                    source_span=episode.raw_text[:180],
                    unresolved_reason="Support is present, but mixed cues leave counterpart motive underdetermined.",
                    uncertainty_level=_clamp(max(uncertainty, 0.38)),
                    action_relevant=False,
                    linked_entities=linked_entities,
                    linked_chapters=linked_chapters,
                    evidence_links=tuple(conflict_cues[:2] or ["support_with_contrast"]),
                    decision_relevance=relevance,
                    surface_cue_ids=surface_ids,
                    promotion_reason="Counterpart motive affects trust calibration and social repair decisions.",
                )
            )
        merged: dict[tuple[str, tuple[str, ...]], NarrativeUnknown] = {}
        for unknown in candidates:
            key = (unknown.unknown_type, unknown.linked_entities)
            current = merged.get(key)
            if current is None or unknown.decision_relevance.total_score > current.decision_relevance.total_score:
                merged[key] = unknown
        ordered = sorted(
            merged.values(),
            key=lambda item: (item.decision_relevance.total_score, item.uncertainty_level),
            reverse=True,
        )
        return ordered[: self.max_unknowns]

    def _decision_relevance(
        self,
        *,
        unknown_type: str,
        risk: float,
        social: float,
        action: float,
        verification: float,
        continuity: float,
    ) -> DecisionRelevanceMap:
        memory = max(0.10, continuity * 0.72 + risk * 0.18)
        downstream = max(action, risk, social, verification)
        summary = {
            NarrativeUnknownType.TRUST.value: "This uncertainty changes trust stance and social action.",
            NarrativeUnknownType.THREAT_PERSISTENCE.value: "This uncertainty changes danger prediction and caution.",
            NarrativeUnknownType.ENVIRONMENT_RELIABILITY.value: "This uncertainty changes how strongly to trust current evidence.",
            NarrativeUnknownType.MOTIVE.value: "This uncertainty changes social repair and trust calibration.",
        }.get(unknown_type, "This uncertainty could change later prediction or policy.")
        total = _clamp(
            action * 0.22
            + social * 0.16
            + risk * 0.22
            + verification * 0.18
            + memory * 0.10
            + continuity * 0.06
            + downstream * 0.06
        )
        return DecisionRelevanceMap(
            action_choice=_clamp(action),
            social_stance=_clamp(social),
            risk_level=_clamp(risk),
            verification_urgency=_clamp(verification),
            memory_protection=_clamp(memory),
            continuity_impact=_clamp(continuity),
            downstream_prediction_delta=_clamp(downstream),
            total_score=total,
            summary=summary,
        )

    def _hypotheses_for_unknown(
        self,
        *,
        unknown: NarrativeUnknown,
        compiled_event: CompiledNarrativeEvent,
        appraisal: AppraisalVector,
        text: str,
    ) -> list[CompetingHypothesis]:
        del appraisal
        if unknown.unknown_type == NarrativeUnknownType.TRUST.value:
            return [
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="betrayal",
                    statement="The rupture reflects a stable betrayal or low-trust counterpart.",
                    prior=0.52 if compiled_event.event_type == "social_exclusion" else 0.40,
                    support=("social_rupture", "trust_break"),
                    contradict=("support_received",) if compiled_event.event_type == "rescue" else (),
                    cause_type=LatentCauseType.PERSISTENT_BETRAYAL.value,
                    expected_state={"social": 0.24, "danger": 0.56},
                    consequences=("prefer caution with counterpart", "increase verification urgency"),
                    relevance_boost=0.12,
                ),
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="constraint",
                    statement="The rupture was driven by temporary constraint or resource scarcity rather than durable betrayal.",
                    prior=0.34,
                    support=("resource_loss",) if compiled_event.outcome_type == "resource_loss" else ("uncertainty",),
                    contradict=("direct_harm",) if compiled_event.direct_harm else (),
                    cause_type=LatentCauseType.TEMPORARY_CONSTRAINT.value,
                    expected_state={"social": 0.46, "danger": 0.34},
                    consequences=("preserve cautious openness", "verify situational limits"),
                    relevance_boost=0.05,
                ),
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="miscommunication",
                    statement="The apparent rupture may reflect miscommunication or incomplete information.",
                    prior=0.28 if any(token in text for token in ("mixed signals", "misread", "message", "误会")) else 0.20,
                    support=("contrastive_connector",) if "contrastive_connector" in unknown.evidence_links else ("explicit_uncertainty",),
                    contradict=("trust_break",) if compiled_event.event_type == "social_exclusion" else (),
                    cause_type=LatentCauseType.MISCOMMUNICATION.value,
                    expected_state={"social": 0.52, "danger": 0.28},
                    consequences=("seek clarification before closure", "avoid overcommitting to threat interpretation"),
                    relevance_boost=0.03,
                ),
            ]
        if unknown.unknown_type == NarrativeUnknownType.THREAT_PERSISTENCE.value:
            return [
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="persistent",
                    statement="There is a stable threat source that will continue shaping future danger.",
                    prior=0.50,
                    support=("direct_threat", "physical_exposure"),
                    contradict=("novelty_engagement",) if compiled_event.event_type == "exploration" else (),
                    cause_type=LatentCauseType.PERSISTENT_THREAT.value,
                    expected_state={"danger": 0.72, "shelter": 0.58},
                    consequences=("favor hide or scan", "protect memory of threat cues"),
                    relevance_boost=0.14,
                ),
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="accident",
                    statement="The danger was a local accident rather than a persistent source.",
                    prior=0.32,
                    support=("temporal_shift",) if "temporal_shift" in unknown.evidence_links else ("uncertainty",),
                    contradict=("direct_threat",) if compiled_event.event_type == "predator_attack" else (),
                    cause_type=LatentCauseType.LOCAL_ACCIDENT.value,
                    expected_state={"danger": 0.38, "novelty": 0.34},
                    consequences=("resume exploration sooner", "verify recurrence before escalating"),
                    relevance_boost=0.06,
                ),
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="environment",
                    statement="Environmental interference or misleading conditions created the danger appearance.",
                    prior=0.24 if "contrastive_connector" in unknown.evidence_links else 0.18,
                    support=("contrastive_connector",) if "contrastive_connector" in unknown.evidence_links else ("explicit_uncertainty",),
                    contradict=("physical_exposure",) if "physical_exposure" in compiled_event.annotations.get("event_structure_signals", []) else (),
                    cause_type=LatentCauseType.ENVIRONMENTAL_INTERFERENCE.value,
                    expected_state={"danger": 0.44, "novelty": 0.26},
                    consequences=("verify environment before attributing stable threat", "avoid overweighting one scene"),
                    relevance_boost=0.04,
                ),
            ]
        if unknown.unknown_type == NarrativeUnknownType.ENVIRONMENT_RELIABILITY.value:
            return [
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="misleading",
                    statement="The environment or evidence source is misleading enough to distort immediate interpretation.",
                    prior=0.40,
                    support=tuple(unknown.evidence_links[:2] or ("explicit_uncertainty",)),
                    contradict=("support_received",) if compiled_event.event_type == "rescue" else (),
                    cause_type=LatentCauseType.ENVIRONMENTAL_INTERFERENCE.value,
                    expected_state={"danger": 0.48, "social": 0.36},
                    consequences=("delay strong updates", "verify before policy lock-in"),
                    relevance_boost=0.08,
                ),
                self._make_hypothesis(
                    unknown=unknown,
                    suffix="stable_signal",
                    statement="The evidence is noisy but still points to a stable underlying cause.",
                    prior=0.33,
                    support=("event_structure",),
                    contradict=("low_signal",) if "low_signal" in unknown.surface_cue_ids else (),
                    cause_type=LatentCauseType.UNKNOWN.value,
                    expected_state={"danger": 0.56, "social": 0.30},
                    consequences=("preserve uncertainty but still act cautiously",),
                    relevance_boost=0.05,
                ),
            ]
        return [
            self._make_hypothesis(
                unknown=unknown,
                suffix="supportive",
                statement="The counterpart's motive is protective and should be tested rather than assumed.",
                prior=0.42 if compiled_event.event_type == "rescue" else 0.25,
                support=("support_received",) if compiled_event.event_type == "rescue" else ("uncertainty",),
                contradict=("trust_break",) if compiled_event.event_type == "social_exclusion" else (),
                cause_type=LatentCauseType.PROTECTIVE_SUPPORT.value,
                expected_state={"social": 0.58},
                consequences=("allow calibrated contact", "seek confirming evidence"),
                relevance_boost=0.06,
            ),
            self._make_hypothesis(
                unknown=unknown,
                suffix="guarded",
                statement="The motive is mixed or strategic, so trust should remain provisional.",
                prior=0.38,
                support=tuple(unknown.evidence_links[:2] or ("contrast",)),
                contradict=(),
                cause_type=LatentCauseType.UNKNOWN.value,
                expected_state={"social": 0.40, "danger": 0.34},
                consequences=("maintain bounded contact", "preserve verification budget"),
                relevance_boost=0.08,
            ),
        ]

    def _make_hypothesis(
        self,
        *,
        unknown: NarrativeUnknown,
        suffix: str,
        statement: str,
        prior: float,
        support: tuple[str, ...],
        contradict: tuple[str, ...],
        cause_type: str,
        expected_state: dict[str, float],
        consequences: tuple[str, ...],
        relevance_boost: float,
    ) -> CompetingHypothesis:
        support_score = _clamp(0.34 + len([item for item in support if item]) * 0.12)
        contradiction_score = _clamp(len([item for item in contradict if item]) * 0.12)
        relevance = DecisionRelevanceMap(
            action_choice=_clamp(unknown.decision_relevance.action_choice + relevance_boost * 0.70),
            social_stance=_clamp(unknown.decision_relevance.social_stance + relevance_boost * 0.35),
            risk_level=_clamp(
                unknown.decision_relevance.risk_level
                + (0.10 if "danger" in expected_state else 0.0)
                - (0.08 if expected_state.get("social", 0.0) >= 0.50 else 0.0)
            ),
            verification_urgency=_clamp(unknown.decision_relevance.verification_urgency + relevance_boost),
            memory_protection=_clamp(unknown.decision_relevance.memory_protection + relevance_boost * 0.45),
            continuity_impact=unknown.decision_relevance.continuity_impact,
            downstream_prediction_delta=_clamp(max(abs(value - 0.5) for value in expected_state.values()) if expected_state else 0.0),
            total_score=_clamp(unknown.decision_relevance.total_score + relevance_boost),
            summary=unknown.decision_relevance.summary,
        )
        return CompetingHypothesis(
            hypothesis_id=f"{unknown.unknown_id}:hyp:{suffix}",
            parent_unknown_id=unknown.unknown_id,
            statement=statement,
            prior_plausibility=_clamp(prior),
            support=HypothesisSupport(
                support_evidence=tuple(item for item in support if item),
                contradicting_evidence=tuple(item for item in contradict if item),
                support_score=support_score,
                contradiction_score=contradiction_score,
            ),
            implied_consequences=consequences,
            expected_state_shift=expected_state,
            decision_relevance=relevance,
            latent_cause_type=cause_type,
        )

    def _profile(
        self,
        unknowns: list[NarrativeUnknown],
        hypotheses: list[CompetingHypothesis],
        latent_causes: list[LatentCauseCandidate],
        surface_cues: list[SurfaceCue],
    ) -> NarrativeAmbiguityProfile:
        social = sum(
            1
            for item in unknowns
            if item.unknown_type in {NarrativeUnknownType.TRUST.value, NarrativeUnknownType.MOTIVE.value}
        )
        trust = sum(1 for item in unknowns if item.unknown_type == NarrativeUnknownType.TRUST.value)
        environment = sum(
            1
            for item in unknowns
            if item.unknown_type
            in {
                NarrativeUnknownType.ENVIRONMENT_RELIABILITY.value,
                NarrativeUnknownType.THREAT_PERSISTENCE.value,
            }
        )
        retained = sum(item.decision_relevance.total_score for item in unknowns if item.action_relevant)
        return NarrativeAmbiguityProfile(
            total_unknown_count=len(unknowns),
            decision_relevant_unknown_count=sum(1 for item in unknowns if item.action_relevant),
            competing_hypothesis_count=len(hypotheses),
            latent_cause_count=len(latent_causes),
            surface_cue_count=len(surface_cues),
            interpretive_competition=_clamp(len(hypotheses) / max(1.0, len(unknowns) * 3.0)),
            latent_cause_uncertainty_burden=_clamp(len(latent_causes) / max(1.0, len(hypotheses))),
            social_ambiguity_burden=_clamp(social / max(1.0, len(unknowns))),
            trust_ambiguity_burden=_clamp(trust / max(1.0, len(unknowns))),
            environment_ambiguity_burden=_clamp(environment / max(1.0, len(unknowns))),
            retained_uncertainty_burden=_clamp(retained / max(1.0, len(unknowns))),
        )

    def _summary(
        self,
        unknowns: list[NarrativeUnknown],
        hypotheses: list[CompetingHypothesis],
        surface_cues: list[SurfaceCue],
        profile: NarrativeAmbiguityProfile,
    ) -> str:
        if not unknowns:
            if surface_cues:
                return "The narrative contains salience cues, but none are important enough to preserve as action-relevant uncertainty."
            return "The narrative does not currently expose a retained action-relevant uncertainty."
        top = unknowns[0]
        hypothesis_count = len([item for item in hypotheses if item.parent_unknown_id == top.unknown_id])
        if top.unknown_type == NarrativeUnknownType.TRUST.value and top.linked_entities:
            return (
                f"The narrative leaves trust unresolved for {top.linked_entities[0]}, "
                f"with {hypothesis_count} competing explanations that imply different social actions."
            )
        if top.unknown_type == NarrativeUnknownType.THREAT_PERSISTENCE.value:
            return (
                "The key unresolved latent cause is whether the threat is persistent or accidental, "
                "which materially changes future caution and verification."
            )
        if profile.decision_relevant_unknown_count == 0:
            return "Some ambiguity remains, but it currently looks more like surface texture than decision-relevant cause."
        return (
            f"{profile.decision_relevant_unknown_count} narrative uncertainty item(s) were retained because "
            "they would materially change later prediction, verification, or action."
        )
