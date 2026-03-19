"""Extended personality parameter space for inverse inference.

Provides data structures for the full 10-dimension personality analysis
result, where each inferred parameter carries evidence, confidence, and
reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Base confidence-rated wrapper
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceRated:
    """A single inferred value with evidence and confidence."""
    value: Any  # float | str | dict | list
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ConfidenceRated:
        return cls(
            value=payload.get("value"),
            confidence=str(payload.get("confidence", "low")),  # type: ignore[arg-type]
            evidence=list(payload.get("evidence", [])),  # type: ignore[arg-type]
            reasoning=str(payload.get("reasoning", "")),
        )


# ---------------------------------------------------------------------------
# Evidence items
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """A piece of evidence extracted from source material."""
    excerpt: str
    source_index: int  # index into the materials list
    category: str  # "behavioral", "emotional", "cognitive", "relational", "value"
    appraisal_relevance: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvidenceItem:
        return cls(
            excerpt=str(payload.get("excerpt", "")),
            source_index=int(payload.get("source_index", 0)),
            category=str(payload.get("category", "behavioral")),
            appraisal_relevance=dict(payload.get("appraisal_relevance", {})),  # type: ignore[arg-type]
            tags=list(payload.get("tags", [])),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Dimension A: Core Priors
# ---------------------------------------------------------------------------

@dataclass
class CorePriors:
    """Fundamental beliefs about self, others, and the world."""
    self_worth: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    self_efficacy: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    other_reliability: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    other_predictability: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    world_safety: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    world_fairness: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k).to_dict() for k in (
            "self_worth", "self_efficacy", "other_reliability",
            "other_predictability", "world_safety", "world_fairness",
        )}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CorePriors:
        kw = {}
        for key in ("self_worth", "self_efficacy", "other_reliability",
                     "other_predictability", "world_safety", "world_fairness"):
            raw = payload.get(key)
            if isinstance(raw, dict):
                kw[key] = ConfidenceRated.from_dict(raw)
        return cls(**kw)


# ---------------------------------------------------------------------------
# Dimension B: Precision Allocation
# ---------------------------------------------------------------------------

@dataclass
class PrecisionAllocation:
    """How attention / precision is distributed across channels."""
    hypersensitive_channels: list[ConfidenceRated] = field(default_factory=list)
    blind_spots: list[ConfidenceRated] = field(default_factory=list)
    internal_vs_external: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.0))
    immediate_vs_narrative: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "hypersensitive_channels": [c.to_dict() for c in self.hypersensitive_channels],
            "blind_spots": [c.to_dict() for c in self.blind_spots],
            "internal_vs_external": self.internal_vs_external.to_dict(),
            "immediate_vs_narrative": self.immediate_vs_narrative.to_dict(),
        }


# ---------------------------------------------------------------------------
# Dimension C: Value Hierarchy
# ---------------------------------------------------------------------------

@dataclass
class AnalysisValueHierarchy:
    """Ranked values inferred from behavior and language."""
    ranked_values: list[tuple[str, ConfidenceRated]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ranked_values": [
                {"name": name, **cr.to_dict()} for name, cr in self.ranked_values
            ],
        }


# ---------------------------------------------------------------------------
# Dimension D: Cognitive Style
# ---------------------------------------------------------------------------

@dataclass
class CognitiveStyle:
    """Characteristic cognitive processing patterns."""
    abstract_vs_concrete: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.0))
    global_vs_detail: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.0))
    causal_attribution_tendency: ConfidenceRated = field(default_factory=lambda: ConfidenceRated("internal"))
    reflective_depth: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    coherence_need: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    ambiguity_tolerance: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k).to_dict() for k in (
            "abstract_vs_concrete", "global_vs_detail", "causal_attribution_tendency",
            "reflective_depth", "coherence_need", "ambiguity_tolerance",
        )}


# ---------------------------------------------------------------------------
# Dimension E: Affective Dynamics
# ---------------------------------------------------------------------------

@dataclass
class AffectiveDynamics:
    """Emotional processing patterns and baseline states."""
    baseline_arousal: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    recovery_speed: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.5))
    dominant_emotions: list[ConfidenceRated] = field(default_factory=list)
    emotion_channel_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_arousal": self.baseline_arousal.to_dict(),
            "recovery_speed": self.recovery_speed.to_dict(),
            "dominant_emotions": [e.to_dict() for e in self.dominant_emotions],
            "emotion_channel_weights": dict(self.emotion_channel_weights),
        }


# ---------------------------------------------------------------------------
# Dimension F: Social Orientation
# ---------------------------------------------------------------------------

@dataclass
class SocialOrientation:
    """Social interaction strategy weights."""
    orientation_weights: dict[str, ConfidenceRated] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_weights": {
                k: v.to_dict() for k, v in self.orientation_weights.items()
            },
        }


# ---------------------------------------------------------------------------
# Dimension G: Self Model Profile
# ---------------------------------------------------------------------------

@dataclass
class SelfModelProfile:
    """How the person models themselves."""
    self_narrative: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(""))
    identity_consistency_needs: list[ConfidenceRated] = field(default_factory=list)
    identity_threats: list[ConfidenceRated] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "self_narrative": self.self_narrative.to_dict(),
            "identity_consistency_needs": [c.to_dict() for c in self.identity_consistency_needs],
            "identity_threats": [c.to_dict() for c in self.identity_threats],
        }


# ---------------------------------------------------------------------------
# Dimension H: Other Model Profile
# ---------------------------------------------------------------------------

@dataclass
class OtherModelProfile:
    """Templates for modeling different categories of others."""
    intimate_model: ConfidenceRated = field(default_factory=lambda: ConfidenceRated({}))
    authority_model: ConfidenceRated = field(default_factory=lambda: ConfidenceRated({}))
    peer_model: ConfidenceRated = field(default_factory=lambda: ConfidenceRated({}))
    weaker_model: ConfidenceRated = field(default_factory=lambda: ConfidenceRated({}))
    stranger_model: ConfidenceRated = field(default_factory=lambda: ConfidenceRated({}))

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k).to_dict() for k in (
            "intimate_model", "authority_model", "peer_model",
            "weaker_model", "stranger_model",
        )}


# ---------------------------------------------------------------------------
# Dimension I: Relational Templates
# ---------------------------------------------------------------------------

@dataclass
class RelationalTemplates:
    """Recurring patterns in different relational contexts."""
    intimate_patterns: list[ConfidenceRated] = field(default_factory=list)
    cooperative_patterns: list[ConfidenceRated] = field(default_factory=list)
    power_patterns: list[ConfidenceRated] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "intimate_patterns": [c.to_dict() for c in self.intimate_patterns],
            "cooperative_patterns": [c.to_dict() for c in self.cooperative_patterns],
            "power_patterns": [c.to_dict() for c in self.power_patterns],
        }


# ---------------------------------------------------------------------------
# Dimension J: Temporal Structure
# ---------------------------------------------------------------------------

@dataclass
class TemporalStructure:
    """How attention is distributed across past, present, and future."""
    past_trauma_weight: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.33))
    present_pressure_weight: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.34))
    future_imagination_weight: ConfidenceRated = field(default_factory=lambda: ConfidenceRated(0.33))

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k).to_dict() for k in (
            "past_trauma_weight", "present_pressure_weight", "future_imagination_weight",
        )}


# ---------------------------------------------------------------------------
# Defense & Strategy profiles
# ---------------------------------------------------------------------------

@dataclass
class DefenseMechanismEntry:
    """A single defense mechanism with cost-benefit analysis."""
    name: str
    target_error: str
    short_term_benefit: str
    long_term_cost: str
    triggers: list[str] = field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DefenseMechanismProfile:
    """Full defense mechanism inventory."""
    mechanisms: list[DefenseMechanismEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"mechanisms": [m.to_dict() for m in self.mechanisms]}


@dataclass
class StrategyProfile:
    """Preferred and blocked action strategies."""
    preferred_strategies: list[ConfidenceRated] = field(default_factory=list)
    cost_analysis: dict[str, ConfidenceRated] = field(default_factory=dict)
    blocked_strategies: list[ConfidenceRated] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "preferred_strategies": [s.to_dict() for s in self.preferred_strategies],
            "cost_analysis": {k: v.to_dict() for k, v in self.cost_analysis.items()},
            "blocked_strategies": [s.to_dict() for s in self.blocked_strategies],
        }


# ---------------------------------------------------------------------------
# Feedback loops & predictions
# ---------------------------------------------------------------------------

@dataclass
class FeedbackLoop:
    """A closed-loop dynamic linking personality parameters."""
    name: str
    components: list[str]
    description: str
    valence: Literal["reinforcing", "balancing"] = "reinforcing"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ConditionalPrediction:
    """A behavioral prediction conditioned on a scenario."""
    scenario: str
    predicted_behavior: str
    confidence: Literal["high", "medium", "low"] = "low"
    reasoning: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Predictive hypothesis (internal intermediate)
# ---------------------------------------------------------------------------

@dataclass
class PredictiveHypothesis:
    """The system's best guess about the person's generative model."""
    core_prediction: str = ""
    dominant_drives: list[str] = field(default_factory=list)
    primary_threat_model: str = ""
    preferred_error_reduction: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stability analysis (internal intermediate)
# ---------------------------------------------------------------------------

@dataclass
class StabilityAnalysis:
    """Stable core, fragile points, and plastic points."""
    stable_core: list[ConfidenceRated] = field(default_factory=list)
    fragile_points: list[ConfidenceRated] = field(default_factory=list)
    plastic_points: list[ConfidenceRated] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "stable_core": [c.to_dict() for c in self.stable_core],
            "fragile_points": [c.to_dict() for c in self.fragile_points],
            "plastic_points": [c.to_dict() for c in self.plastic_points],
        }


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class PersonalityAnalysisResult:
    """Complete personality analysis output (12 sections)."""
    # Section 1: Summary
    summary: str = ""
    # Section 2: Raw evidence
    evidence_list: list[EvidenceItem] = field(default_factory=list)
    # Section 3: Core priors + value hierarchy
    core_priors: CorePriors = field(default_factory=CorePriors)
    value_hierarchy: AnalysisValueHierarchy = field(default_factory=AnalysisValueHierarchy)
    # Section 4: Precision + cognitive style
    precision_allocation: PrecisionAllocation = field(default_factory=PrecisionAllocation)
    cognitive_style: CognitiveStyle = field(default_factory=CognitiveStyle)
    # Section 5: Affect + defense
    affective_dynamics: AffectiveDynamics = field(default_factory=AffectiveDynamics)
    defense_mechanisms: DefenseMechanismProfile = field(default_factory=DefenseMechanismProfile)
    # Section 6: Relational templates + social strategy
    social_orientation: SocialOrientation = field(default_factory=SocialOrientation)
    relational_templates: RelationalTemplates = field(default_factory=RelationalTemplates)
    # Section 7: Self/Other model
    self_model_profile: SelfModelProfile = field(default_factory=SelfModelProfile)
    other_model_profile: OtherModelProfile = field(default_factory=OtherModelProfile)
    # Section 8: Closed-loop diagram
    feedback_loops: list[FeedbackLoop] = field(default_factory=list)
    # Section 9: Development history
    developmental_inferences: list[ConfidenceRated] = field(default_factory=list)
    # Section 10: Stability
    stable_core: list[ConfidenceRated] = field(default_factory=list)
    fragile_points: list[ConfidenceRated] = field(default_factory=list)
    plastic_points: list[ConfidenceRated] = field(default_factory=list)
    # Section 11: Predictions
    behavioral_predictions: list[ConditionalPrediction] = field(default_factory=list)
    # Section 12: Uncertainty
    missing_evidence: list[str] = field(default_factory=list)
    unresolvable_questions: list[str] = field(default_factory=list)
    # Section 13: One-line conclusion
    one_line_conclusion: str = ""
    # Metadata
    big_five: dict[str, float] = field(default_factory=dict)
    via_strengths: dict[str, float] = field(default_factory=dict)
    temporal_structure: TemporalStructure = field(default_factory=TemporalStructure)
    strategy_profile: StrategyProfile = field(default_factory=StrategyProfile)
    analysis_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "core_priors": self.core_priors.to_dict(),
            "value_hierarchy": self.value_hierarchy.to_dict(),
            "precision_allocation": self.precision_allocation.to_dict(),
            "cognitive_style": self.cognitive_style.to_dict(),
            "affective_dynamics": self.affective_dynamics.to_dict(),
            "defense_mechanisms": self.defense_mechanisms.to_dict(),
            "social_orientation": self.social_orientation.to_dict(),
            "relational_templates": self.relational_templates.to_dict(),
            "self_model_profile": self.self_model_profile.to_dict(),
            "other_model_profile": self.other_model_profile.to_dict(),
            "feedback_loops": [f.to_dict() for f in self.feedback_loops],
            "developmental_inferences": [d.to_dict() for d in self.developmental_inferences],
            "stable_core": [s.to_dict() for s in self.stable_core],
            "fragile_points": [f.to_dict() for f in self.fragile_points],
            "plastic_points": [p.to_dict() for p in self.plastic_points],
            "behavioral_predictions": [b.to_dict() for b in self.behavioral_predictions],
            "missing_evidence": list(self.missing_evidence),
            "unresolvable_questions": list(self.unresolvable_questions),
            "one_line_conclusion": self.one_line_conclusion,
            "big_five": dict(self.big_five),
            "via_strengths": dict(self.via_strengths),
            "temporal_structure": self.temporal_structure.to_dict(),
            "strategy_profile": self.strategy_profile.to_dict(),
            "analysis_confidence": self.analysis_confidence,
        }
