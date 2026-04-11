from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters
from .memory_model import (
    AnchorStrength,
    MemoryClass,
    MemoryEntry,
    SourceType,
    StoreLevel,
)
from .memory_state import merge_state_context, resolve_dynamic_salience


EMERGENCY_AROUSAL_THRESHOLD = 0.9
EMERGENCY_SALIENCE_THRESHOLD = 0.92

SOURCE_CONFIDENCE_DEFAULTS: dict[SourceType, tuple[float, float]] = {
    SourceType.EXPERIENCE: (0.9, 0.85),
    SourceType.REHEARSAL: (0.8, 0.8),
    SourceType.HEARSAY: (0.7, 0.5),
    SourceType.INFERENCE: (0.9, 0.35),
    SourceType.RECONSTRUCTION: (0.4, 0.5),
}


@dataclass(frozen=True)
class EncodingDynamicsInput:
    prediction_error: float
    surprise: float
    arousal: float
    attention_budget: float
    requested_budget: float = 1.0
    raw_drive_total: float | None = None


@dataclass(frozen=True)
class EncodingDynamicsResult:
    encoding_source: str
    encoding_strength: float
    prediction_error: float
    surprise: float
    arousal: float
    raw_drive: float
    raw_strength: float
    attention_budget_total: float
    attention_budget_requested: float
    attention_budget_granted: float
    attention_budget_denied: float
    fallback_evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "encoding_source": self.encoding_source,
            "encoding_strength": round(self.encoding_strength, 6),
            "fep_prediction_error": round(self.prediction_error, 6),
            "surprise": round(self.surprise, 6),
            "arousal": round(self.arousal, 6),
            "raw_drive": round(self.raw_drive, 6),
            "raw_strength": round(self.raw_strength, 6),
            "attention_budget_total": round(self.attention_budget_total, 6),
            "attention_budget_requested": round(self.attention_budget_requested, 6),
            "attention_budget_granted": round(self.attention_budget_granted, 6),
            "attention_budget_denied": round(self.attention_budget_denied, 6),
            "fallback_evidence": dict(self.fallback_evidence),
        }


class EncodingDynamics:
    @staticmethod
    def raw_drive(prediction_error: float, surprise: float, arousal: float) -> float:
        return max(0.0, float(prediction_error)) * max(0.0, float(surprise)) * max(0.0, float(arousal))

    @staticmethod
    def raw_strength(prediction_error: float, surprise: float, arousal: float) -> float:
        return _clamp(
            (0.45 * _clamp(prediction_error))
            + (0.35 * _clamp(surprise))
            + (0.20 * _clamp(arousal))
        )

    @classmethod
    def score(cls, signal: EncodingDynamicsInput) -> EncodingDynamicsResult:
        raw_drive = cls.raw_drive(signal.prediction_error, signal.surprise, signal.arousal)
        raw_strength = cls.raw_strength(signal.prediction_error, signal.surprise, signal.arousal)
        requested = max(1e-9, float(signal.requested_budget))
        budget_total = max(0.0, float(signal.attention_budget))
        total_raw_drive = max(raw_drive, float(signal.raw_drive_total if signal.raw_drive_total is not None else raw_drive), 1e-9)
        granted = min(requested, budget_total * raw_drive / total_raw_drive)
        denied = max(0.0, requested - granted)
        encoding_strength = _clamp(raw_strength * (granted / requested))
        return EncodingDynamicsResult(
            encoding_source="dynamics",
            encoding_strength=encoding_strength,
            prediction_error=_clamp(signal.prediction_error),
            surprise=_clamp(signal.surprise),
            arousal=_clamp(signal.arousal),
            raw_drive=raw_drive,
            raw_strength=raw_strength,
            attention_budget_total=budget_total,
            attention_budget_requested=requested,
            attention_budget_granted=granted,
            attention_budget_denied=denied,
        )

    @classmethod
    def score_many(cls, signals: list[EncodingDynamicsInput]) -> list[EncodingDynamicsResult]:
        drives = [cls.raw_drive(item.prediction_error, item.surprise, item.arousal) for item in signals]
        total = sum(drives)
        return [
            cls.score(
                EncodingDynamicsInput(
                    prediction_error=signal.prediction_error,
                    surprise=signal.surprise,
                    arousal=signal.arousal,
                    attention_budget=signal.attention_budget,
                    requested_budget=signal.requested_budget,
                    raw_drive_total=total,
                )
            )
            for signal in signals
        ]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> EncodingDynamicsResult | None:
        if "fep_prediction_error" in payload:
            prediction_error_raw = payload.get("fep_prediction_error")
        else:
            prediction_error_raw = payload.get("prediction_error")
        surprise_raw = payload.get("surprise", payload.get("total_surprise", payload.get("weighted_surprise")))
        arousal_raw = payload.get("arousal")
        budget_raw = payload.get("available_attention_budget", payload.get("attention_budget"))
        if any(value is None for value in (prediction_error_raw, surprise_raw, arousal_raw, budget_raw)):
            return None
        requested_raw = payload.get("attention_budget_requested", payload.get("encoding_attention", payload.get("attention", 1.0)))
        total_raw = payload.get("attention_budget_raw_drive_total")
        return cls.score(
            EncodingDynamicsInput(
                prediction_error=_coerce_float(prediction_error_raw, 0.0),
                surprise=_coerce_float(surprise_raw, 0.0),
                arousal=_coerce_float(arousal_raw, 0.0),
                attention_budget=_coerce_float(budget_raw, 0.0),
                requested_budget=max(1e-9, _coerce_float(requested_raw, 1.0)),
                raw_drive_total=(
                    _coerce_float(total_raw, 0.0)
                    if total_raw is not None
                    else None
                ),
            )
        )

    @classmethod
    def reencode(
        cls,
        *,
        first_pass_strength: float,
        prediction_error: float,
        surprise: float,
        arousal: float,
        attention_budget: float,
        requested_budget: float = 1.0,
    ) -> tuple[EncodingDynamicsResult, float, float]:
        result = cls.score(
            EncodingDynamicsInput(
                prediction_error=prediction_error,
                surprise=surprise,
                arousal=arousal,
                attention_budget=attention_budget,
                requested_budget=requested_budget,
            )
        )
        salience_delta = result.encoding_strength - _clamp(first_pass_strength)
        retention_adjustment = _clamp(salience_delta, -0.35, 0.35)
        return result, salience_delta, retention_adjustment


class FallbackHeuristic:
    THREAT_KEYWORDS = {
        "alarm", "attack", "cost", "danger", "failure", "hazard", "injury",
        "loss", "penalty", "punishment", "risk", "threat", "unsafe",
    }
    REWARD_KEYWORDS = {
        "achievement", "benefit", "bonus", "gain", "progress", "reward",
        "success", "win",
    }
    SOCIAL_KEYWORDS = {
        "ally", "caregiver", "family", "friend", "mentor", "partner",
        "relationship", "social", "team",
    }
    FIRST_PERSON_TOKENS = {"i", "me", "my", "mine", "myself"}
    TASK_ONLY_KEYWORDS = {"task", "todo", "current", "assignment", "ticket", "corridor", "light"}
    OUTCOME_NEGATIVE_KEYWORDS = {"loss", "penalty", "danger", "failure", "alarm", "unsafe"}
    OUTCOME_POSITIVE_KEYWORDS = {"reward", "success", "gain", "progress", "stabilized", "stable"}
    usage_histogram: dict[str, int] = {"heuristic": 0, "dynamics": 0}

    @classmethod
    def keyword_signal(cls, tokens: set[str], keywords: set[str], *, scale: float = 3.0) -> tuple[float, list[str]]:
        return _keyword_signal(tokens, keywords, scale=scale)

    @classmethod
    def threat_evidence(cls, tokens: set[str]) -> tuple[float, list[str]]:
        keyword_score, keyword_evidence = cls.keyword_signal(tokens, cls.THREAT_KEYWORDS)
        outcome_score, outcome_evidence = cls.keyword_signal(tokens, cls.OUTCOME_NEGATIVE_KEYWORDS, scale=2.0)
        return max(keyword_score, outcome_score), [
            *[f"threat_keyword:{item}" for item in keyword_evidence],
            *[f"threat_outcome:{item}" for item in outcome_evidence],
        ]

    @classmethod
    def social_evidence(cls, tokens: set[str]) -> tuple[float, list[str]]:
        score, evidence = cls.keyword_signal(tokens, cls.SOCIAL_KEYWORDS)
        return score, [f"social_keyword:{item}" for item in evidence]

    @classmethod
    def reward_evidence(cls, tokens: set[str]) -> tuple[float, list[str]]:
        keyword_score, keyword_evidence = cls.keyword_signal(tokens, cls.REWARD_KEYWORDS)
        outcome_score, outcome_evidence = cls.keyword_signal(tokens, cls.OUTCOME_POSITIVE_KEYWORDS, scale=2.0)
        return max(keyword_score, outcome_score), [
            *[f"reward_keyword:{item}" for item in keyword_evidence],
            *[f"reward_outcome:{item}" for item in outcome_evidence],
        ]

    @classmethod
    def guardrails(cls, tokens: set[str]) -> list[str]:
        guardrails: list[str] = []
        if tokens & cls.FIRST_PERSON_TOKENS:
            guardrails.append("guard:first_person_not_identity")
        if tokens & cls.TASK_ONLY_KEYWORDS:
            guardrails.append("guard:task_relevance_not_identity")
        return guardrails

    @classmethod
    def score(cls, raw_input: dict[str, Any], computed_strength: float, *, arousal: float, attention: float) -> EncodingDynamicsResult:
        tokens = _tokenize(
            raw_input.get("content"),
            raw_input.get("semantic_tags"),
            raw_input.get("context_tags"),
            raw_input.get("outcome"),
            raw_input.get("predicted_outcome"),
        )
        threat_score, threat_terms = cls.keyword_signal(tokens, cls.THREAT_KEYWORDS)
        negative_score, negative_terms = cls.keyword_signal(tokens, cls.OUTCOME_NEGATIVE_KEYWORDS, scale=2.0)
        social_score, social_terms = cls.keyword_signal(tokens, cls.SOCIAL_KEYWORDS)
        reward_score, reward_terms = cls.keyword_signal(tokens, cls.REWARD_KEYWORDS)
        positive_score, positive_terms = cls.keyword_signal(tokens, cls.OUTCOME_POSITIVE_KEYWORDS, scale=2.0)
        strength = _clamp(max(computed_strength, threat_score, negative_score, social_score, reward_score, positive_score))
        requested = max(1e-9, _clamp(attention))
        return EncodingDynamicsResult(
            encoding_source="heuristic",
            encoding_strength=strength,
            prediction_error=_clamp(_coerce_float(raw_input.get("prediction_error"), 0.0)),
            surprise=_clamp(_coerce_float(raw_input.get("surprise", raw_input.get("total_surprise")), strength)),
            arousal=_clamp(arousal),
            raw_drive=strength,
            raw_strength=strength,
            attention_budget_total=requested,
            attention_budget_requested=requested,
            attention_budget_granted=requested,
            attention_budget_denied=0.0,
            fallback_evidence={
                "threat_keywords": threat_terms,
                "negative_outcome_keywords": negative_terms,
                "social_keywords": social_terms,
                "reward_keywords": reward_terms,
                "positive_outcome_keywords": positive_terms,
                "guardrails": cls.guardrails(tokens),
            },
        )

    @classmethod
    def record(cls, source: str) -> None:
        cls.usage_histogram[source] = cls.usage_histogram.get(source, 0) + 1

    @classmethod
    def histogram(cls) -> dict[str, int]:
        return dict(cls.usage_histogram)


@dataclass(frozen=True)
class SalienceConfig:
    w_arousal: float = 0.30
    w_attention: float = 0.20
    w_novelty: float = 0.20
    w_relevance: float = 0.30
    relevance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "goal": 0.2,
            "threat": 0.2,
            "self": 0.2,
            "social": 0.2,
            "reward": 0.2,
        }
    )


@dataclass(frozen=True)
class RelevanceEvidence:
    score: float
    structured_signals: dict[str, float] = field(default_factory=dict)
    fallback_signals: dict[str, float] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "structured_signals": dict(self.structured_signals),
            "fallback_signals": dict(self.fallback_signals),
            "evidence": list(self.evidence),
            "guardrails": list(self.guardrails),
        }

    def evidence_strings(self) -> list[str]:
        values = list(self.evidence)
        values.extend(self.guardrails)
        return values or ["signal:none"]


@dataclass(frozen=True)
class EncodingSignals:
    goal: RelevanceEvidence
    threat: RelevanceEvidence
    self: RelevanceEvidence
    social: RelevanceEvidence
    reward: RelevanceEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal.to_dict(),
            "threat": self.threat.to_dict(),
            "self": self.self.to_dict(),
            "social": self.social.to_dict(),
            "reward": self.reward.to_dict(),
        }


@dataclass(frozen=True)
class EncodingAudit:
    signals: EncodingSignals
    relevance_audit: dict[str, object]
    memory_class_reason: str
    source_type_reason: str
    anchor_reasoning: dict[str, object]
    retention_priority: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_evidence": self.signals.goal.evidence_strings(),
            "threat_evidence": self.signals.threat.evidence_strings(),
            "self_evidence": self.signals.self.evidence_strings(),
            "social_evidence": self.signals.social.evidence_strings(),
            "reward_evidence": self.signals.reward.evidence_strings(),
            "signal_breakdown": self.signals.to_dict(),
            "relevance_audit": dict(self.relevance_audit),
            "memory_class_reason": self.memory_class_reason,
            "source_type_reason": self.source_type_reason,
            "anchor_reasoning": dict(self.anchor_reasoning),
            "retention_priority": dict(self.retention_priority),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    values: list[float] = []
    for item in value:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values


def _coerce_bool(value: Any) -> bool:
    return bool(value)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _tokenize(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                tokens.update(_tokenize(item))
            continue
        if isinstance(value, dict):
            for item in value.values():
                tokens.update(_tokenize(item))
            continue
        text = _normalize_text(value).lower()
        tokens.update(token for token in re.split(r"[^a-z0-9_]+", text) if token)
    return tokens


def _has_dynamic_encoding_signals(raw_input: dict[str, Any]) -> bool:
    has_prediction_error = raw_input.get("fep_prediction_error") is not None or raw_input.get("prediction_error") is not None
    has_surprise = any(
        raw_input.get(key) is not None
        for key in ("surprise", "total_surprise", "weighted_surprise")
    )
    has_budget = raw_input.get("available_attention_budget") is not None or raw_input.get("attention_budget") is not None
    return bool(has_prediction_error and has_surprise and raw_input.get("arousal") is not None and has_budget)


def _keyword_signal(tokens: set[str], keywords: set[str], *, scale: float = 3.0) -> tuple[float, list[str]]:
    evidence = sorted(tokens & keywords)
    return _clamp(len(evidence) / scale), evidence


def _overlap_signal(left: set[str], right: set[str], *, scale: float = 3.0) -> tuple[float, list[str]]:
    evidence = sorted(left & right)
    return _clamp(len(evidence) / scale), evidence


def _build_relevance_evidence(
    *,
    structured_signals: dict[str, float],
    fallback_signals: dict[str, float],
    evidence: list[str],
    guardrails: list[str] | None = None,
) -> RelevanceEvidence:
    score_candidates = [0.0]
    score_candidates.extend(value for value in structured_signals.values() if value > 0.0)
    score_candidates.extend(value for value in fallback_signals.values() if value > 0.0)
    score = _clamp(max(score_candidates))
    return RelevanceEvidence(
        score=score,
        structured_signals={key: _clamp(value) for key, value in structured_signals.items() if value > 0.0},
        fallback_signals={key: _clamp(value) for key, value in fallback_signals.items() if value > 0.0},
        evidence=tuple(evidence),
        guardrails=tuple(guardrails or ()),
    )


def score_goal_relevance(raw_input: dict[str, Any], current_state: dict[str, Any]) -> RelevanceEvidence:
    input_tokens = _tokenize(
        raw_input.get("content"),
        raw_input.get("semantic_tags"),
        raw_input.get("goal_cues"),
        raw_input.get("commitments"),
        raw_input.get("content_keywords"),
    )
    goal_tokens = _tokenize(
        current_state.get("active_goals"),
        current_state.get("goals"),
        current_state.get("goal_keywords"),
        current_state.get("active_commitments"),
    )
    overlap_score, overlap = _overlap_signal(input_tokens, goal_tokens)
    explicit = _clamp(_coerce_float(raw_input.get("goal_relevance_hint"), 0.0))
    active_commitment_overlap, commitment_overlap = _overlap_signal(
        _tokenize(raw_input.get("commitments")),
        _tokenize(current_state.get("active_commitments"), current_state.get("identity_commitments")),
        scale=2.0,
    )
    evidence = [f"goal_overlap:{item}" for item in overlap]
    evidence.extend(f"goal_commitment:{item}" for item in commitment_overlap)
    if explicit > 0.0:
        evidence.append(f"goal_hint:{explicit:.2f}")
    return _build_relevance_evidence(
        structured_signals={
            "goal_overlap": overlap_score,
            "goal_commitment_overlap": active_commitment_overlap,
            "goal_hint": explicit,
        },
        fallback_signals={},
        evidence=evidence or ["goal_overlap:none"],
    )


def score_threat_relevance(raw_input: dict[str, Any], current_state: dict[str, Any]) -> RelevanceEvidence:
    input_tokens = _tokenize(
        raw_input.get("content"),
        raw_input.get("semantic_tags"),
        raw_input.get("context_tags"),
        raw_input.get("threat_cues"),
        raw_input.get("outcome"),
        raw_input.get("predicted_outcome"),
    )
    raw_threat = _clamp(_coerce_float(raw_input.get("threat_level"), 0.0))
    risk = _clamp(_coerce_float(raw_input.get("risk"), 0.0))
    penalty_signal = _clamp(abs(_coerce_float(raw_input.get("penalty_signal"), 0.0)))
    loss_signal = _clamp(abs(_coerce_float(raw_input.get("loss_signal"), 0.0)))
    state_threat = _clamp(_coerce_float(current_state.get("threat_level"), 0.0) * 0.5)
    negative_reward_signal = _clamp(abs(min(0.0, _coerce_float(raw_input.get("reward_signal"), 0.0))))
    evidence: list[str] = []
    fallback_score = 0.0
    if not _has_dynamic_encoding_signals(raw_input):
        fallback_score, fallback_evidence = FallbackHeuristic.threat_evidence(input_tokens)
        evidence.extend(fallback_evidence)
    if raw_threat > 0.0:
        evidence.append(f"raw_threat:{raw_threat:.2f}")
    if risk > 0.0:
        evidence.append(f"risk:{risk:.2f}")
    if penalty_signal > 0.0:
        evidence.append(f"penalty_signal:{penalty_signal:.2f}")
    if loss_signal > 0.0:
        evidence.append(f"loss_signal:{loss_signal:.2f}")
    if negative_reward_signal > 0.0:
        evidence.append(f"negative_reward_signal:{negative_reward_signal:.2f}")
    if state_threat > 0.0:
        evidence.append(f"state_threat:{state_threat:.2f}")
    return _build_relevance_evidence(
        structured_signals={
            "raw_threat": raw_threat,
            "risk": risk,
            "penalty_signal": penalty_signal,
            "loss_signal": loss_signal,
            "negative_reward_signal": negative_reward_signal,
            "state_threat": state_threat,
        },
        fallback_signals={"heuristic_threat_signal": fallback_score},
        evidence=evidence or ["threat_signal:none"],
    )


def score_self_relevance(raw_input: dict[str, Any], current_state: dict[str, Any]) -> RelevanceEvidence:
    role_overlap_score, role_overlap = _overlap_signal(
        _tokenize(raw_input.get("roles")),
        _tokenize(current_state.get("identity_roles"), current_state.get("identity_active_themes")),
        scale=2.0,
    )
    relationship_overlap_score, relationship_overlap = _overlap_signal(
        _tokenize(raw_input.get("relationships")),
        _tokenize(current_state.get("important_relationships"), current_state.get("identity_active_themes")),
        scale=2.0,
    )
    commitment_overlap_score, commitment_overlap = _overlap_signal(
        _tokenize(raw_input.get("commitments")),
        _tokenize(current_state.get("active_commitments"), current_state.get("identity_commitments")),
        scale=2.0,
    )
    narrative_overlap_score, narrative_overlap = _overlap_signal(
        _tokenize(raw_input.get("narrative_nodes"), raw_input.get("identity_themes")),
        _tokenize(
            current_state.get("identity_themes"),
            current_state.get("self_narrative_keywords"),
            current_state.get("identity_active_themes"),
        ),
        scale=2.0,
    )
    explicit = _clamp(_coerce_float(raw_input.get("identity_relevance_hint"), 0.0))
    evidence: list[str] = []
    evidence.extend(f"role_continuity:{item}" for item in role_overlap)
    evidence.extend(f"relationship_continuity:{item}" for item in relationship_overlap)
    evidence.extend(f"commitment_continuity:{item}" for item in commitment_overlap)
    evidence.extend(f"narrative_continuity:{item}" for item in narrative_overlap)
    if explicit > 0.0:
        evidence.append(f"identity_hint:{explicit:.2f}")

    content_tokens = _tokenize(raw_input.get("content"), raw_input.get("semantic_tags"))
    guardrails: list[str] = []
    guardrails.extend(FallbackHeuristic.guardrails(content_tokens))
    if _clamp(_coerce_float(raw_input.get("arousal"), 0.0)) >= 0.7:
        guardrails.append("guard:high_arousal_not_identity")
    if not evidence:
        evidence.append("identity_continuity:none")

    structured = {
        "role_continuity": role_overlap_score,
        "relationship_continuity": relationship_overlap_score,
        "commitment_continuity": commitment_overlap_score,
        "narrative_continuity": narrative_overlap_score,
        "identity_hint": explicit,
    }
    score = max(structured.values(), default=0.0)
    if score <= 0.0 and guardrails:
        score = 0.0
    return RelevanceEvidence(
        score=_clamp(score),
        structured_signals={key: value for key, value in structured.items() if value > 0.0},
        fallback_signals={},
        evidence=tuple(evidence),
        guardrails=tuple(guardrails),
    )


def score_social_relevance(raw_input: dict[str, Any], current_state: dict[str, Any]) -> RelevanceEvidence:
    relationship_overlap_score, relationship_overlap = _overlap_signal(
        _tokenize(raw_input.get("relationships"), raw_input.get("social_agents")),
        _tokenize(current_state.get("important_relationships"), current_state.get("social_agents")),
        scale=2.0,
    )
    input_tokens = _tokenize(
        raw_input.get("content"),
        raw_input.get("semantic_tags"),
        raw_input.get("relationships"),
        raw_input.get("social_agents"),
    )
    social_flag = 0.25 if _coerce_bool(current_state.get("social_context_active")) else 0.0
    evidence = [f"social_overlap:{item}" for item in relationship_overlap]
    fallback_score = 0.0
    if not _has_dynamic_encoding_signals(raw_input):
        fallback_score, fallback_evidence = FallbackHeuristic.social_evidence(input_tokens)
        evidence.extend(fallback_evidence)
    if social_flag > 0.0:
        evidence.append("social_context_active")
    return _build_relevance_evidence(
        structured_signals={
            "relationship_overlap": relationship_overlap_score,
            "social_context_active": social_flag,
        },
        fallback_signals={"heuristic_social_signal": fallback_score},
        evidence=evidence or ["social_signal:none"],
    )


def score_reward_relevance(raw_input: dict[str, Any], current_state: dict[str, Any]) -> RelevanceEvidence:
    input_tokens = _tokenize(
        raw_input.get("content"),
        raw_input.get("semantic_tags"),
        raw_input.get("reward_cues"),
        raw_input.get("outcome"),
        raw_input.get("predicted_outcome"),
    )
    reward_signal = _clamp(max(0.0, _coerce_float(raw_input.get("reward_signal"), 0.0)))
    context_bonus = 0.2 if _coerce_bool(current_state.get("reward_context_active")) else 0.0
    evidence: list[str] = []
    fallback_score = 0.0
    if not _has_dynamic_encoding_signals(raw_input):
        fallback_score, fallback_evidence = FallbackHeuristic.reward_evidence(input_tokens)
        evidence.extend(fallback_evidence)
    if reward_signal > 0.0:
        evidence.append(f"reward_signal:{reward_signal:.2f}")
    if context_bonus > 0.0:
        evidence.append("reward_context_active")
    return _build_relevance_evidence(
        structured_signals={
            "reward_signal": reward_signal,
            "reward_context_active": context_bonus,
        },
        fallback_signals={
            "heuristic_reward_signal": fallback_score,
        },
        evidence=evidence or ["reward_signal:none"],
    )


def aggregate_relevance(
    *,
    goal: float,
    threat: float,
    self: float,
    social: float,
    reward: float,
    config: SalienceConfig | None = None,
) -> tuple[float, dict[str, object]]:
    effective = config or SalienceConfig()
    weights = dict(effective.relevance_weights)
    contributions = {
        "goal": weights.get("goal", 0.0) * goal,
        "threat": weights.get("threat", 0.0) * threat,
        "self": weights.get("self", 0.0) * self,
        "social": weights.get("social", 0.0) * social,
        "reward": weights.get("reward", 0.0) * reward,
    }
    total_weight = sum(weights.values()) or 1.0
    score = _clamp(sum(contributions.values()) / total_weight)
    audit = {
        "weights": weights,
        "inputs": {
            "goal": goal,
            "threat": threat,
            "self": self,
            "social": social,
            "reward": reward,
        },
        "contributions": contributions,
        "formula": (
            "relevance = "
            f"({weights.get('goal', 0.0):.2f}*goal + "
            f"{weights.get('threat', 0.0):.2f}*threat + "
            f"{weights.get('self', 0.0):.2f}*self + "
            f"{weights.get('social', 0.0):.2f}*social + "
            f"{weights.get('reward', 0.0):.2f}*reward) / {total_weight:.2f}"
        ),
    }
    return score, audit


def compute_salience(
    arousal: float,
    encoding_attention: float,
    novelty: float,
    relevance: float,
    config: SalienceConfig | None = None,
) -> float:
    effective = config or SalienceConfig()
    return _clamp(
        (effective.w_arousal * _clamp(arousal))
        + (effective.w_attention * _clamp(encoding_attention))
        + (effective.w_novelty * _clamp(novelty))
        + (effective.w_relevance * _clamp(relevance))
    )


def build_salience_audit(entry: MemoryEntry, config: SalienceConfig | None = None) -> dict[str, object]:
    effective = config or SalienceConfig()
    contributions = {
        "arousal": effective.w_arousal * entry.arousal,
        "attention": effective.w_attention * entry.encoding_attention,
        "novelty": effective.w_novelty * entry.novelty,
        "relevance": effective.w_relevance * entry.relevance,
    }
    return {
        "weights": {
            "arousal": effective.w_arousal,
            "attention": effective.w_attention,
            "novelty": effective.w_novelty,
            "relevance": effective.w_relevance,
        },
        "inputs": {
            "arousal": entry.arousal,
            "attention": entry.encoding_attention,
            "novelty": entry.novelty,
            "relevance": entry.relevance,
            "relevance_goal": entry.relevance_goal,
            "relevance_threat": entry.relevance_threat,
            "relevance_self": entry.relevance_self,
            "relevance_social": entry.relevance_social,
            "relevance_reward": entry.relevance_reward,
        },
        "contributions": contributions,
        "formula": (
            f"salience = {effective.w_arousal:.2f}*arousal + "
            f"{effective.w_attention:.2f}*attention + "
            f"{effective.w_novelty:.2f}*novelty + "
            f"{effective.w_relevance:.2f}*relevance"
        ),
        "salience": compute_salience(
            entry.arousal,
            entry.encoding_attention,
            entry.novelty,
            entry.relevance,
            effective,
        ),
    }


def format_salience_audit(entry: MemoryEntry, config: SalienceConfig | None = None) -> str:
    audit = build_salience_audit(entry, config)
    inputs = audit["inputs"]
    return (
        f"arousal={inputs['arousal']:.3f}, "
        f"attention={inputs['attention']:.3f}, "
        f"novelty={inputs['novelty']:.3f}, "
        f"relevance={inputs['relevance']:.3f} "
        f"(goal={inputs['relevance_goal']:.3f}, threat={inputs['relevance_threat']:.3f}, "
        f"self={inputs['relevance_self']:.3f}, social={inputs['relevance_social']:.3f}, "
        f"reward={inputs['relevance_reward']:.3f}) -> salience={audit['salience']:.3f}"
    )


def _build_encoding_signals(raw_input: dict[str, Any], current_state: dict[str, Any]) -> EncodingSignals:
    return EncodingSignals(
        goal=score_goal_relevance(raw_input, current_state),
        threat=score_threat_relevance(raw_input, current_state),
        self=score_self_relevance(raw_input, current_state),
        social=score_social_relevance(raw_input, current_state),
        reward=score_reward_relevance(raw_input, current_state),
    )


def _infer_memory_class(raw_input: dict[str, Any], signals: EncodingSignals) -> tuple[MemoryClass, str]:
    hint = _normalize_text(raw_input.get("memory_class")).lower()
    if hint in {item.value for item in MemoryClass}:
        return MemoryClass(hint), "explicit_memory_class_hint"
    if _string_list(raw_input.get("procedure_steps")):
        return MemoryClass.PROCEDURAL, "procedure_steps_present"
    if _coerce_bool(raw_input.get("inferred")) or _coerce_bool(raw_input.get("reconstructed")):
        return MemoryClass.INFERRED, "explicit_inference_flag"
    if (
        _coerce_bool(raw_input.get("semantic_pattern"))
        or len(_string_list(raw_input.get("supporting_episode_ids"))) >= 2
        or bool(_normalize_text(raw_input.get("abstraction_reason")))
        or bool(_string_list(raw_input.get("predictive_use_cases")))
    ):
        return MemoryClass.SEMANTIC, "semantic_supporting_evidence"
    if signals.self.score >= 0.75 and len(_string_list(raw_input.get("supporting_episode_ids"))) >= 2:
        return MemoryClass.SEMANTIC, "identity_cluster_consolidation"
    return MemoryClass.EPISODIC, "episodic_default"


def _infer_source_type(raw_input: dict[str, Any], memory_class: MemoryClass) -> tuple[SourceType, str]:
    hint = _normalize_text(raw_input.get("source_type")).lower()
    if hint in {item.value for item in SourceType}:
        return SourceType(hint), "explicit_source_type_hint"
    if memory_class is MemoryClass.INFERRED:
        return SourceType.INFERENCE, "inferred_memory_default"
    if _coerce_bool(raw_input.get("heard_from")):
        return SourceType.HEARSAY, "heard_from_flag"
    if _coerce_bool(raw_input.get("rehearsed")):
        return SourceType.REHEARSAL, "rehearsed_flag"
    if _coerce_bool(raw_input.get("reconstructed")):
        return SourceType.RECONSTRUCTION, "reconstructed_flag"
    return SourceType.EXPERIENCE, "experience_default"


def _derive_content(raw_input: dict[str, Any], memory_class: MemoryClass) -> str:
    explicit = _normalize_text(raw_input.get("content") or raw_input.get("summary"))
    if explicit:
        return explicit
    if memory_class is MemoryClass.PROCEDURAL:
        steps = _string_list(raw_input.get("procedure_steps"))
        if steps:
            return f"Procedure: {' -> '.join(steps[:3])}"
    action = _normalize_text(raw_input.get("action")) or "observe"
    outcome = _normalize_text(raw_input.get("outcome")) or _normalize_text(raw_input.get("predicted_outcome")) or "unknown outcome"
    event_time = _normalize_text(raw_input.get("event_time")) or "unspecified time"
    return f"At {event_time}, {action} led to {outcome}"


def _derive_tags(raw_input: dict[str, Any], content: str) -> tuple[list[str], list[str]]:
    semantic_tags = _string_list(raw_input.get("semantic_tags"))
    context_tags = _string_list(raw_input.get("context_tags"))
    if not semantic_tags:
        semantic_tags = sorted(token for token in _tokenize(content) if len(token) > 3)[:6]
    if not context_tags:
        context_tags = _string_list(raw_input.get("execution_contexts"))
    return semantic_tags, context_tags


def _derive_anchor_slots(
    raw_input: dict[str, Any],
    memory_class: MemoryClass,
    signals: EncodingSignals,
) -> tuple[dict[str, str | None], dict[str, str], dict[str, object]]:
    slots = dict(raw_input.get("anchor_slots", {})) if isinstance(raw_input.get("anchor_slots"), dict) else {}
    slots.setdefault("time", _normalize_text(raw_input.get("event_time")) or None)
    slots.setdefault("place", _normalize_text(raw_input.get("place")) or None)
    agents = _string_list(raw_input.get("agents")) or _string_list(raw_input.get("social_agents")) or _string_list(raw_input.get("relationships"))
    slots.setdefault("agents", ", ".join(agents) if agents else None)
    slots.setdefault("action", _normalize_text(raw_input.get("action")) or None)
    slots.setdefault(
        "outcome",
        _normalize_text(raw_input.get("outcome") or raw_input.get("predicted_outcome")) or None,
    )
    strengths = dict(raw_input.get("anchor_strengths", {})) if isinstance(raw_input.get("anchor_strengths"), dict) else {}
    reasoning: list[str] = []
    if memory_class is MemoryClass.EPISODIC and not strengths:
        strengths = {
            "time": AnchorStrength.WEAK.value,
            "place": AnchorStrength.WEAK.value,
            "agents": AnchorStrength.STRONG.value,
            "action": AnchorStrength.STRONG.value,
            "outcome": AnchorStrength.STRONG.value,
        }
        reasoning.append("episodic_default_protected_core")
        if signals.threat.score >= 0.6:
            strengths["outcome"] = AnchorStrength.LOCKED.value
            strengths["action"] = AnchorStrength.STRONG.value
            reasoning.append("threat_strengthened_outcome")
        if signals.self.score >= 0.6 and slots.get("agents"):
            strengths["agents"] = AnchorStrength.LOCKED.value
            reasoning.append("identity_strengthened_agents")
        if signals.social.score >= 0.5 and slots.get("agents"):
            strengths["agents"] = AnchorStrength.STRONG.value
            reasoning.append("social_context_strengthened_agents")
        if _coerce_bool(raw_input.get("time_is_core")) and slots.get("time"):
            strengths["time"] = AnchorStrength.STRONG.value
            reasoning.append("time_marked_core")
        if _coerce_bool(raw_input.get("place_is_core")) and slots.get("place"):
            strengths["place"] = AnchorStrength.STRONG.value
            reasoning.append("place_marked_core")
    return slots, strengths, {
        "slots_present": {key: value is not None for key, value in slots.items()},
        "reasons": reasoning,
    }


def _derive_retention_priority(signals: EncodingSignals, salience: float, novelty: float, arousal: float) -> dict[str, object]:
    reasons: list[str] = []
    if signals.self.score >= 0.75:
        reasons.append("identity_continuity_priority")
    if signals.threat.score >= 0.6:
        reasons.append("threat_priority")
    if novelty >= 0.75 and signals.self.score < 0.25:
        reasons.append("novelty_only_noise_risk")
    if arousal >= 0.9 or salience >= EMERGENCY_SALIENCE_THRESHOLD:
        reasons.append("emergency_consolidation")
    return {
        "priority_score": _clamp(max(salience, signals.self.score, signals.threat.score)),
        "reasons": reasons,
    }


def _default_lineage_metadata(
    raw_input: dict[str, Any],
    memory_class: MemoryClass,
    signals: EncodingSignals,
) -> tuple[list[str], dict[str, object]]:
    derived_from = _string_list(raw_input.get("derived_from"))
    support_ids = _string_list(raw_input.get("supporting_episode_ids"))
    if support_ids:
        derived_from = list(dict.fromkeys([*derived_from, *support_ids]))
    metadata = dict(raw_input.get("compression_metadata", {})) if isinstance(raw_input.get("compression_metadata"), dict) else {}
    if memory_class not in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}:
        return derived_from, metadata

    lineage_type = _normalize_text(metadata.get("lineage_type") or raw_input.get("lineage_type"))
    abstraction_reason = _normalize_text(metadata.get("abstraction_reason") or raw_input.get("abstraction_reason"))
    predictive_use_cases = _string_list(metadata.get("predictive_use_cases")) or _string_list(raw_input.get("predictive_use_cases"))

    if not lineage_type:
        if memory_class is MemoryClass.SEMANTIC and signals.self.score >= 0.6:
            lineage_type = "identity_consolidation"
        elif memory_class is MemoryClass.SEMANTIC:
            lineage_type = "episodic_compression"
        else:
            lineage_type = "pattern_extraction"
    if not abstraction_reason:
        if support_ids:
            abstraction_reason = "stabilized pattern across supporting episodes"
        elif memory_class is MemoryClass.INFERRED:
            abstraction_reason = "candidate pattern inferred from partial evidence"
        else:
            abstraction_reason = "semantic abstraction from repeated experience"
    if not predictive_use_cases:
        if memory_class is MemoryClass.SEMANTIC:
            predictive_use_cases = ["pattern-guided recall", "future expectation shaping"]
        else:
            predictive_use_cases = ["low-confidence planning hint", "candidate explanation"]

    metadata["lineage_type"] = lineage_type
    metadata["abstraction_reason"] = abstraction_reason
    metadata["predictive_use_cases"] = predictive_use_cases
    if support_ids:
        metadata.setdefault("support_entry_ids", support_ids)
    return derived_from, metadata


def _style_value(
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None,
    key: str,
    default: float = 0.0,
) -> float:
    if cognitive_style is None:
        return default
    if hasattr(cognitive_style, key):
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


def _effective_attention(raw_attention: float, signals: list[float], attention_selectivity: float) -> tuple[float, dict[str, object]]:
    active_signal_count = sum(1 for score in signals if score >= 0.30)
    k = max(1, int(round(5 - (attention_selectivity * 4))))
    is_top_signal = active_signal_count <= k
    multiplier = 1.0 if is_top_signal else (1.0 - (attention_selectivity * 0.5))
    return _clamp(raw_attention * multiplier), {
        "raw_attention": round(raw_attention, 6),
        "attention_selectivity": round(attention_selectivity, 6),
        "active_signal_count": active_signal_count,
        "top_k": k,
        "is_top_signal": is_top_signal,
        "multiplier": round(multiplier, 6),
    }


def _focus_tags_for_selectivity(
    semantic_tags: list[str],
    context_tags: list[str],
    *,
    attention_selectivity: float,
) -> tuple[list[str], list[str], dict[str, object]]:
    semantic_limit = max(2, int(round(6 - (attention_selectivity * 4))))
    context_limit = max(1, int(round(4 - (attention_selectivity * 3))))
    focused_semantic = list(semantic_tags[:semantic_limit])
    focused_context = list(context_tags[:context_limit])
    return focused_semantic, focused_context, {
        "semantic_limit": semantic_limit,
        "context_limit": context_limit,
        "semantic_before": list(semantic_tags),
        "semantic_after": list(focused_semantic),
        "context_before": list(context_tags),
        "context_after": list(focused_context),
    }


def encode_memory(
    raw_input: dict[str, Any],
    current_state: dict[str, Any],
    config: SalienceConfig,
    *,
    agent_state=None,
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
) -> MemoryEntry:
    state_context = merge_state_context(
        current_state,
        agent_state=agent_state,
        cognitive_style=cognitive_style,
    )
    signals = _build_encoding_signals(raw_input, state_context)
    dynamic_regulation = resolve_dynamic_salience(
        raw_input,
        state_context,
        config,
        agent_state=agent_state,
    )
    memory_class, memory_class_reason = _infer_memory_class(raw_input, signals)
    source_type, source_type_reason = _infer_source_type(raw_input, memory_class)
    content = _derive_content(raw_input, memory_class)
    valence = _coerce_float(raw_input.get("valence"), 0.0)
    arousal = _clamp(
        _coerce_float(
            raw_input.get("arousal"),
            max(abs(valence) * 0.5, _coerce_float(raw_input.get("threat_level"), 0.0)),
        )
    )
    uncertainty_sensitivity = _clamp(_style_value(cognitive_style, "uncertainty_sensitivity", 0.0))
    error_aversion = _clamp(_style_value(cognitive_style, "error_aversion", 0.0))
    attention_selectivity = _clamp(_style_value(cognitive_style, "attention_selectivity", 0.0))
    semantic_tags, context_tags = _derive_tags(raw_input, content)
    semantic_tags, context_tags, tag_focus_audit = _focus_tags_for_selectivity(
        semantic_tags,
        context_tags,
        attention_selectivity=attention_selectivity,
    )
    anchor_slots, anchor_strengths, anchor_reasoning = _derive_anchor_slots(raw_input, memory_class, signals)
    if dynamic_regulation.identity_match_ratio > 0.0 and signals.self.score >= 0.35:
        if anchor_slots.get("agents"):
            anchor_strengths["agents"] = AnchorStrength.LOCKED.value
            anchor_reasoning["reasons"].append("state_identity_locked_agents")
        if anchor_slots.get("time"):
            anchor_strengths["time"] = AnchorStrength.STRONG.value
            anchor_reasoning["reasons"].append("state_identity_strengthened_time")
    if float(state_context.get("threat_level", 0.0)) >= 0.60 and memory_class is MemoryClass.EPISODIC:
        for key in ("agents", "action", "outcome"):
            anchor_strengths[key] = AnchorStrength.LOCKED.value if key == "outcome" else AnchorStrength.STRONG.value
        anchor_reasoning["reasons"].append("state_threat_strengthened_core")
    if valence < 0.0:
        arousal = _clamp(arousal * (1.0 + (error_aversion * 0.25)))
    raw_attention = _clamp(_coerce_float(raw_input.get("encoding_attention", raw_input.get("attention", 0.5)), 0.5))
    novelty = _clamp(
        _coerce_float(
            raw_input.get("novelty"),
            max(
                _coerce_float(raw_input.get("prediction_error"), 0.0),
                _coerce_float(raw_input.get("total_surprise"), 0.0),
            ),
        )
    )
    novelty = _clamp(novelty * (1.0 + (uncertainty_sensitivity * 0.3)))
    effective_self_relevance = _clamp(
        signals.self.score * dynamic_regulation.effective_relevance_self_multiplier
    )
    attention, attention_audit = _effective_attention(
        raw_attention,
        [
            signals.goal.score,
            signals.threat.score,
            signals.self.score,
            signals.social.score,
            signals.reward.score,
            novelty,
            arousal,
        ],
        attention_selectivity,
    )
    effective_config = SalienceConfig(
        w_arousal=dynamic_regulation.effective_w_arousal,
        w_attention=dynamic_regulation.effective_w_attention,
        w_novelty=dynamic_regulation.effective_w_novelty,
        w_relevance=dynamic_regulation.effective_w_relevance,
        relevance_weights=dict(config.relevance_weights),
    )

    relevance, relevance_audit = aggregate_relevance(
        goal=signals.goal.score,
        threat=signals.threat.score,
        self=effective_self_relevance,
        social=signals.social.score,
        reward=signals.reward.score,
        config=effective_config,
    )
    salience = compute_salience(arousal, attention, novelty, relevance, effective_config)
    dynamics_result = EncodingDynamics.from_mapping(
        {
            **raw_input,
            "arousal": arousal,
            "encoding_attention": attention,
        }
    )
    if dynamics_result is None:
        dynamics_result = FallbackHeuristic.score(
            raw_input,
            salience,
            arousal=arousal,
            attention=attention,
        )
    else:
        salience = dynamics_result.encoding_strength
        novelty = _clamp(max(novelty, dynamics_result.prediction_error))
        attention = _clamp(dynamics_result.attention_budget_granted)
    FallbackHeuristic.record(dynamics_result.encoding_source)
    store_level = StoreLevel.LONG if (
        arousal > EMERGENCY_AROUSAL_THRESHOLD or salience > EMERGENCY_SALIENCE_THRESHOLD
    ) else StoreLevel.SHORT
    if (
        store_level is StoreLevel.SHORT
        and dynamic_regulation.identity_match_ratio > 0.0
        and effective_self_relevance >= 0.50
        and novelty <= 0.45
    ):
        memory_class_reason = f"{memory_class_reason}+identity_active_retention_bias_pending_store"
    if (
        memory_class is MemoryClass.EPISODIC
        and float(state_context.get("threat_level", 0.0)) >= 0.60
        and arousal >= 0.55
    ):
        memory_class_reason = f"{memory_class_reason}+threat_snapshot_bias"

    procedure_steps = _string_list(raw_input.get("procedure_steps"))
    step_confidence = [
        _clamp(_coerce_float(item, 0.8))
        for item in raw_input.get("step_confidence", [])
    ] if isinstance(raw_input.get("step_confidence"), list) else []
    execution_contexts = _string_list(raw_input.get("execution_contexts"))
    confidence_defaults = SOURCE_CONFIDENCE_DEFAULTS[source_type]
    abstractness = _coerce_float(raw_input.get("abstractness"), {
        MemoryClass.EPISODIC: 0.2,
        MemoryClass.SEMANTIC: 0.75,
        MemoryClass.PROCEDURAL: 0.45,
        MemoryClass.INFERRED: 0.80,
    }[memory_class])

    derived_from, metadata = _default_lineage_metadata(raw_input, memory_class, signals)
    audit = EncodingAudit(
        signals=signals,
        relevance_audit=relevance_audit,
        memory_class_reason=memory_class_reason,
        source_type_reason=source_type_reason,
        anchor_reasoning=anchor_reasoning,
        retention_priority=_derive_retention_priority(signals, salience, novelty, arousal),
    )
    metadata["encoding_audit"] = audit.to_dict()
    metadata["dynamic_salience_audit"] = dynamic_regulation.audit.to_dict()
    metadata["m410_encoding_dynamics"] = dynamics_result.to_dict()
    metadata["encoding_source"] = dynamics_result.encoding_source
    metadata["encoding_strength"] = dynamics_result.encoding_strength
    metadata["fep_prediction_error"] = dynamics_result.prediction_error
    metadata["surprise"] = dynamics_result.surprise
    metadata["arousal"] = dynamics_result.arousal
    metadata["attention_budget_total"] = dynamics_result.attention_budget_total
    metadata["attention_budget_granted"] = dynamics_result.attention_budget_granted
    metadata["attention_budget_denied"] = dynamics_result.attention_budget_denied
    metadata["heuristic_usage_histogram"] = FallbackHeuristic.histogram()
    metadata["cognitive_style_effects"] = {
        "uncertainty_sensitivity": round(uncertainty_sensitivity, 6),
        "error_aversion": round(error_aversion, 6),
        "attention_selectivity": round(attention_selectivity, 6),
        "effective_attention": dict(attention_audit),
        "tag_focus": dict(tag_focus_audit),
        "effective_novelty": round(novelty, 6),
        "effective_arousal": round(arousal, 6),
    }
    metadata["structural_decisions"] = {
        "memory_class_reason": memory_class_reason,
        "store_level": store_level.value,
        "anchor_reasoning": dict(anchor_reasoning),
    }

    entry = MemoryEntry(
        content=content,
        memory_class=memory_class,
        store_level=store_level,
        source_type=source_type,
        created_at=int(_coerce_float(raw_input.get("created_at", raw_input.get("cycle", 0)), 0.0)),
        last_accessed=int(_coerce_float(raw_input.get("created_at", raw_input.get("cycle", 0)), 0.0)),
        valence=valence,
        arousal=arousal,
        encoding_attention=attention,
        novelty=novelty,
        relevance_goal=signals.goal.score,
        relevance_threat=signals.threat.score,
        relevance_self=effective_self_relevance,
        relevance_social=signals.social.score,
        relevance_reward=signals.reward.score,
        relevance=relevance,
        salience=salience,
        trace_strength=max(0.05, salience),
        accessibility=max(0.05, _clamp((salience * 0.85) + (attention * 0.15))),
        abstractness=_clamp(abstractness),
        source_confidence=confidence_defaults[0],
        reality_confidence=confidence_defaults[1],
        semantic_tags=semantic_tags,
        context_tags=context_tags,
        anchor_slots=anchor_slots,
        anchor_strengths=anchor_strengths,
        procedure_steps=procedure_steps,
        step_confidence=step_confidence,
        execution_contexts=execution_contexts,
        mood_context=_normalize_text(raw_input.get("mood_context") or state_context.get("recent_mood_baseline")),
        retrieval_count=max(0, int(_coerce_float(raw_input.get("retrieval_count"), 0.0))),
        support_count=max(1, int(_coerce_float(raw_input.get("support_count"), 1.0))),
        counterevidence_count=max(0, int(_coerce_float(raw_input.get("counterevidence_count"), 0.0))),
        competing_interpretations=_string_list(raw_input.get("competing_interpretations")) or None,
        compression_metadata=metadata or None,
        derived_from=derived_from,
        is_dormant=bool(raw_input.get("is_dormant", False)),
        state_vector=_coerce_float_list(raw_input.get("state_vector", raw_input.get("embedding"))),
    )
    if memory_class is MemoryClass.PROCEDURAL and not execution_contexts:
        entry.execution_contexts = context_tags or ["general"]
    if "source_confidence" in raw_input:
        entry.source_confidence = _clamp(_coerce_float(raw_input.get("source_confidence")))
    if "reality_confidence" in raw_input:
        entry.reality_confidence = _clamp(_coerce_float(raw_input.get("reality_confidence")))
    return entry
