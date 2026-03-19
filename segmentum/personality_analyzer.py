"""Personality Analyzer: 10-step inverse inference engine.

Accepts text/behavioral materials and uses the existing FEP/Active Inference
infrastructure to build a full personality generative model via reverse
inference from evidence.

Two modes:
  A) Rule-based — uses NarrativeCompiler + mathematical projections only.
  B) LLM-enhanced — augments evidence extraction and narrative pattern
     recognition via an OpenAI-compatible API.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from statistics import mean
from typing import Any

from .analysis_types import (
    AffectiveDynamics,
    AnalysisValueHierarchy,
    CognitiveStyle,
    ConditionalPrediction,
    ConfidenceRated,
    CorePriors,
    DefenseMechanismEntry,
    DefenseMechanismProfile,
    EvidenceItem,
    FeedbackLoop,
    OtherModelProfile,
    PersonalityAnalysisResult,
    PrecisionAllocation,
    PredictiveHypothesis,
    RelationalTemplates,
    SelfModelProfile,
    SocialOrientation,
    StabilityAnalysis,
    StrategyProfile,
    TemporalStructure,
)
from .narrative_compiler import NarrativeCompiler
from .narrative_types import AppraisalVector, NarrativeEpisode
from .self_model import PersonalitySignal
from .via_projection import VIAProjection


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _confidence_from_count(n: int) -> str:
    if n >= 5:
        return "high"
    if n >= 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Classical defense mechanism mappings
# ---------------------------------------------------------------------------

_DEFENSE_MECHANISMS: list[dict[str, Any]] = [
    {
        "name": "repression",
        "target_error": "intolerable affect",
        "short_term_benefit": "immediate anxiety reduction",
        "long_term_cost": "loss of self-knowledge, symptom formation",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.4 + (1 - b5["openness"]) * 0.3,
        "strategy": "suppress",
    },
    {
        "name": "denial",
        "target_error": "threatening reality",
        "short_term_benefit": "maintains current world model",
        "long_term_cost": "reality-model divergence grows",
        "personality_bias": lambda b5: (1 - b5["openness"]) * 0.4 + b5["neuroticism"] * 0.2,
        "strategy": "suppress",
    },
    {
        "name": "projection",
        "target_error": "unacceptable self-attributes",
        "short_term_benefit": "preserves self-image",
        "long_term_cost": "distorted other-models, paranoid drift",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.3 + (1 - b5["agreeableness"]) * 0.3,
        "strategy": "redirect",
    },
    {
        "name": "rationalization",
        "target_error": "ego-dystonic behavior",
        "short_term_benefit": "coherent self-narrative maintained",
        "long_term_cost": "blocks genuine self-correction",
        "personality_bias": lambda b5: b5["conscientiousness"] * 0.3 + b5["openness"] * 0.2,
        "strategy": "assimilate",
    },
    {
        "name": "intellectualization",
        "target_error": "overwhelming emotion",
        "short_term_benefit": "affect neutralized via abstraction",
        "long_term_cost": "emotional disconnection",
        "personality_bias": lambda b5: b5["openness"] * 0.35 + (1 - b5["extraversion"]) * 0.2,
        "strategy": "assimilate",
    },
    {
        "name": "displacement",
        "target_error": "dangerous target for aggression",
        "short_term_benefit": "releases tension safely",
        "long_term_cost": "collateral relational damage",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.3 + (1 - b5["agreeableness"]) * 0.25,
        "strategy": "redirect",
    },
    {
        "name": "reaction_formation",
        "target_error": "unacceptable impulse",
        "short_term_benefit": "impulse fully masked",
        "long_term_cost": "rigidity, authenticity loss",
        "personality_bias": lambda b5: b5["conscientiousness"] * 0.35 + b5["neuroticism"] * 0.2,
        "strategy": "assimilate",
    },
    {
        "name": "sublimation",
        "target_error": "socially unacceptable drives",
        "short_term_benefit": "productive channel for energy",
        "long_term_cost": "minimal (most adaptive)",
        "personality_bias": lambda b5: b5["openness"] * 0.35 + b5["conscientiousness"] * 0.25,
        "strategy": "accommodate",
    },
    {
        "name": "regression",
        "target_error": "overwhelming complexity",
        "short_term_benefit": "retreats to simpler coping",
        "long_term_cost": "developmental stagnation",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.4 + (1 - b5["conscientiousness"]) * 0.2,
        "strategy": "suppress",
    },
    {
        "name": "isolation_of_affect",
        "target_error": "emotion-laden memory",
        "short_term_benefit": "processes event without overwhelm",
        "long_term_cost": "emotional numbing",
        "personality_bias": lambda b5: (1 - b5["extraversion"]) * 0.3 + b5["conscientiousness"] * 0.25,
        "strategy": "suppress",
    },
    {
        "name": "undoing",
        "target_error": "guilt over past action",
        "short_term_benefit": "symbolic reparation",
        "long_term_cost": "compulsive repetition",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.3 + b5["conscientiousness"] * 0.3,
        "strategy": "assimilate",
    },
    {
        "name": "splitting",
        "target_error": "ambivalent object representation",
        "short_term_benefit": "simplified relational model",
        "long_term_cost": "unstable relationships, black-white thinking",
        "personality_bias": lambda b5: b5["neuroticism"] * 0.35 + (1 - b5["agreeableness"]) * 0.2,
        "strategy": "suppress",
    },
    {
        "name": "idealization",
        "target_error": "inadequacy in attachment figure",
        "short_term_benefit": "preserved attachment security",
        "long_term_cost": "inevitable disillusionment",
        "personality_bias": lambda b5: b5["agreeableness"] * 0.3 + b5["neuroticism"] * 0.2,
        "strategy": "assimilate",
    },
    {
        "name": "passive_aggression",
        "target_error": "forbidden direct aggression",
        "short_term_benefit": "avoids direct conflict",
        "long_term_cost": "trust erosion, relational damage",
        "personality_bias": lambda b5: (1 - b5["extraversion"]) * 0.25 + (1 - b5["agreeableness"]) * 0.25 + b5["neuroticism"] * 0.2,
        "strategy": "redirect",
    },
    {
        "name": "humor",
        "target_error": "anxiety-provoking situation",
        "short_term_benefit": "social bonding + tension release",
        "long_term_cost": "can trivialize real issues",
        "personality_bias": lambda b5: b5["extraversion"] * 0.3 + b5["openness"] * 0.25,
        "strategy": "accommodate",
    },
]

# Value dimensions for ranking
_VALUE_DIMENSIONS = [
    "survival", "safety", "control", "dignity", "relation",
    "achievement", "freedom", "truth", "meaning", "contribution",
]


class PersonalityAnalyzer:
    """Orchestrates the 10-step personality analysis pipeline.

    Parameters
    ----------
    llm_config : dict | None
        If provided, enables LLM-enhanced analysis.  Keys:
        ``api_key``, ``model``, ``base_url``, ``timeout_seconds``.
    """

    def __init__(self, llm_config: dict[str, Any] | None = None) -> None:
        self.compiler = NarrativeCompiler()
        self.via = VIAProjection()
        self._llm_config = llm_config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        materials: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> PersonalityAnalysisResult:
        """Full 10-step personality analysis pipeline."""
        metadata = metadata or {}

        # Step 1: Evidence extraction
        evidence, episodes, appraisals, personality_signals = self._extract_evidence(materials)

        # Aggregate appraisals and Big Five
        agg_appraisal = self._aggregate_appraisals(appraisals)
        big_five = self._aggregate_big_five(personality_signals)
        confidence_level = _confidence_from_count(len(materials))

        # Step 2: Predictive hypothesis
        hypothesis = self._build_predictive_hypothesis(evidence, agg_appraisal, big_five)

        # Step 3: Parameter space modeling
        core_priors = self._infer_core_priors(agg_appraisal, big_five, confidence_level)
        cognitive_style = self._infer_cognitive_style(agg_appraisal, big_five, confidence_level)
        affective = self._infer_affective_dynamics(agg_appraisal, big_five, confidence_level)
        social = self._infer_social_orientation(big_five, confidence_level)
        precision = self._infer_precision_allocation(agg_appraisal, big_five, confidence_level)
        temporal = self._infer_temporal_structure(agg_appraisal, big_five, confidence_level)
        value_hierarchy = self._infer_value_hierarchy(agg_appraisal, big_five, confidence_level)

        # Step 4: Defense mechanism analysis
        defenses = self._analyze_defenses(big_five, agg_appraisal)

        # Step 5: Strategy selection model
        strategies = self._model_strategies(big_five, defenses, agg_appraisal)

        # Step 6: Closed-loop dynamics
        loops = self._build_feedback_loops(core_priors, defenses, strategies, big_five)

        # Step 7: Development history inference
        dev_inferences = self._infer_development(evidence, agg_appraisal, big_five, defenses)

        # Step 8: Stability analysis
        stability = self._analyze_stability(big_five, defenses, agg_appraisal)

        # Step 9: Behavioral predictions
        predictions = self._predict_behaviors(big_five, strategies, loops, agg_appraisal)

        # Self/Other/Relational models
        self_model_profile = self._infer_self_model(big_five, core_priors, confidence_level)
        other_model_profile = self._infer_other_model(big_five, agg_appraisal, confidence_level)
        relational = self._infer_relational_templates(big_five, social, confidence_level)

        # VIA strengths
        via_profile = self.via.project(
            openness=big_five["openness"],
            conscientiousness=big_five["conscientiousness"],
            extraversion=big_five["extraversion"],
            agreeableness=big_five["agreeableness"],
            neuroticism=big_five["neuroticism"],
        )

        # Uncertainty assessment
        missing = self._assess_missing_evidence(evidence, materials)

        # Step 10: Compile report
        summary = self._generate_summary(big_five, core_priors, defenses, hypothesis)
        conclusion = self._generate_conclusion(big_five, hypothesis)
        overall_confidence = self._compute_overall_confidence(
            len(materials), len(evidence), confidence_level
        )

        return PersonalityAnalysisResult(
            summary=summary,
            evidence_list=evidence,
            core_priors=core_priors,
            value_hierarchy=value_hierarchy,
            precision_allocation=precision,
            cognitive_style=cognitive_style,
            affective_dynamics=affective,
            defense_mechanisms=defenses,
            social_orientation=social,
            relational_templates=relational,
            self_model_profile=self_model_profile,
            other_model_profile=other_model_profile,
            feedback_loops=loops,
            developmental_inferences=dev_inferences,
            stable_core=stability.stable_core,
            fragile_points=stability.fragile_points,
            plastic_points=stability.plastic_points,
            behavioral_predictions=predictions,
            missing_evidence=missing,
            unresolvable_questions=self._unresolvable_questions(evidence),
            one_line_conclusion=conclusion,
            big_five=big_five,
            via_strengths=via_profile.to_dict(),
            temporal_structure=temporal,
            strategy_profile=strategies,
            analysis_confidence=overall_confidence,
        )

    # ------------------------------------------------------------------
    # Step 1: Evidence extraction
    # ------------------------------------------------------------------

    def _extract_evidence(
        self, materials: list[str],
    ) -> tuple[
        list[EvidenceItem],
        list[NarrativeEpisode],
        list[AppraisalVector],
        list[PersonalitySignal],
    ]:
        evidence: list[EvidenceItem] = []
        episodes: list[NarrativeEpisode] = []
        appraisals: list[AppraisalVector] = []
        signals: list[PersonalitySignal] = []

        for idx, text in enumerate(materials):
            episode = NarrativeEpisode(
                episode_id=f"mat-{idx:04d}-{uuid.uuid4().hex[:8]}",
                timestamp=idx,
                source="analysis_input",
                raw_text=text,
                tags=["analysis_material"],
            )
            episodes.append(episode)

            compiled_event = self.compiler.compile_event(episode)
            appraisal = self.compiler.appraise_event(compiled_event)
            appraisals.append(appraisal)
            signal = self.compiler.extract_personality_signal(appraisal)
            signals.append(signal)

            # Determine evidence category from event type
            category = self._categorize_event(compiled_event.event_type)

            appraisal_dict = appraisal.to_dict()
            # Find the top relevant appraisal dimensions
            top_dims = sorted(
                appraisal_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:4]
            relevance = {k: round(v, 3) for k, v in top_dims if abs(v) > 0.05}

            tags = [compiled_event.event_type, compiled_event.outcome_type]
            if compiled_event.annotations.get("low_signal"):
                tags.append("low_signal")

            evidence.append(EvidenceItem(
                excerpt=text[:500],  # truncate very long texts
                source_index=idx,
                category=category,
                appraisal_relevance=relevance,
                tags=tags,
            ))

        return evidence, episodes, appraisals, signals

    def _categorize_event(self, event_type: str) -> str:
        mapping = {
            "predator_attack": "behavioral",
            "witnessed_death": "emotional",
            "social_exclusion": "relational",
            "rescue": "relational",
            "exploration": "cognitive",
            "resource_gain": "behavioral",
            "unknown_event": "behavioral",
        }
        return mapping.get(event_type, "behavioral")

    # ------------------------------------------------------------------
    # Appraisal / Big Five aggregation
    # ------------------------------------------------------------------

    def _aggregate_appraisals(self, appraisals: list[AppraisalVector]) -> dict[str, float]:
        if not appraisals:
            return AppraisalVector().to_dict()
        fields = list(AppraisalVector().to_dict().keys())
        result: dict[str, float] = {}
        for f in fields:
            vals = [getattr(a, f) for a in appraisals]
            result[f] = round(_safe_mean(vals), 4)
        return result

    def _aggregate_big_five(self, signals: list[PersonalitySignal]) -> dict[str, float]:
        if not signals:
            return {t: 0.5 for t in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")}
        traits = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        result: dict[str, float] = {}
        for trait in traits:
            deltas = [getattr(s, f"{trait}_delta") for s in signals]
            result[trait] = _clamp(0.5 + _safe_mean(deltas))
        return result

    # ------------------------------------------------------------------
    # Step 2: Predictive hypothesis
    # ------------------------------------------------------------------

    def _build_predictive_hypothesis(
        self,
        evidence: list[EvidenceItem],
        agg: dict[str, float],
        big_five: dict[str, float],
    ) -> PredictiveHypothesis:
        # Determine dominant drives from appraisal pattern
        drives: list[str] = []
        if agg.get("physical_threat", 0) > 0.3:
            drives.append("safety-seeking")
        if agg.get("social_threat", 0) > 0.3:
            drives.append("social-validation")
        if agg.get("attachment_signal", 0) > 0.2:
            drives.append("attachment")
        if agg.get("self_efficacy_impact", 0) > 0.2:
            drives.append("mastery")
        if agg.get("novelty", 0) > 0.3:
            drives.append("exploration")
        if not drives:
            drives.append("homeostasis")

        # Primary threat model
        threat_dims = [
            ("physical_threat", "physical harm"),
            ("social_threat", "social rejection"),
            ("uncertainty", "unpredictability"),
            ("meaning_violation", "meaning collapse"),
        ]
        primary_threat = max(threat_dims, key=lambda td: agg.get(td[0], 0))[1]

        # Preferred error reduction
        if big_five["openness"] > 0.6:
            pref_reduction = "accommodate (update model)"
        elif big_five["conscientiousness"] > 0.6:
            pref_reduction = "assimilate (reframe experience)"
        elif big_five["neuroticism"] > 0.6:
            pref_reduction = "suppress (attenuate signal)"
        else:
            pref_reduction = "redirect (substitute pathway)"

        core = (
            f"Person primarily seeks {', '.join(drives[:3])}; "
            f"main threat axis is {primary_threat}."
        )

        return PredictiveHypothesis(
            core_prediction=core,
            dominant_drives=drives,
            primary_threat_model=primary_threat,
            preferred_error_reduction=pref_reduction,
        )

    # ------------------------------------------------------------------
    # Step 3: Parameter space modeling
    # ------------------------------------------------------------------

    def _infer_core_priors(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> CorePriors:
        self_worth = _clamp(
            0.5
            + agg.get("self_efficacy_impact", 0) * 0.35
            + agg.get("attachment_signal", 0) * 0.25
            - agg.get("social_threat", 0) * 0.20
        )
        self_efficacy = _clamp(
            0.5
            + agg.get("self_efficacy_impact", 0) * 0.40
            + agg.get("controllability", 0) * 0.25
            - agg.get("uncertainty", 0) * 0.15
        )
        other_reliability = _clamp(
            0.5
            + agg.get("trust_impact", 0) * 0.40
            + agg.get("attachment_signal", 0) * 0.20
            - agg.get("social_threat", 0) * 0.15
        )
        other_pred = _clamp(
            0.5
            + agg.get("trust_impact", 0) * 0.25
            - agg.get("uncertainty", 0) * 0.20
            + agg.get("controllability", 0) * 0.15
        )
        world_safety = _clamp(
            0.5
            - agg.get("physical_threat", 0) * 0.35
            - agg.get("uncertainty", 0) * 0.25
            + agg.get("controllability", 0) * 0.20
        )
        world_fairness = _clamp(
            0.5
            - agg.get("meaning_violation", 0) * 0.30
            + agg.get("controllability", 0) * 0.20
            - agg.get("social_threat", 0) * 0.15
        )

        def _cr(val: float, reasoning: str) -> ConfidenceRated:
            return ConfidenceRated(round(val, 4), conf, [], reasoning)

        return CorePriors(
            self_worth=_cr(self_worth, "derived from self_efficacy_impact + attachment - social_threat"),
            self_efficacy=_cr(self_efficacy, "derived from self_efficacy_impact + controllability - uncertainty"),
            other_reliability=_cr(other_reliability, "derived from trust_impact + attachment - social_threat"),
            other_predictability=_cr(other_pred, "derived from trust_impact - uncertainty + controllability"),
            world_safety=_cr(world_safety, "derived from -physical_threat - uncertainty + controllability"),
            world_fairness=_cr(world_fairness, "derived from -meaning_violation + controllability - social_threat"),
        )

    def _infer_cognitive_style(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> CognitiveStyle:
        o, c, n = b5["openness"], b5["conscientiousness"], b5["neuroticism"]
        mv = agg.get("meaning_violation", 0)
        unc = agg.get("uncertainty", 0)

        abstract_vs_concrete = _clamp((o - 0.5) * 0.6 + mv * 0.15, -1, 1)
        ambiguity_tol = _clamp(o * 0.4 - n * 0.3 + (1 - unc) * 0.2)
        coherence_need = _clamp(c * 0.4 + (1 - o) * 0.3)
        reflective_depth = _clamp(o * 0.35 + c * 0.25)

        # Global vs detail: high openness → global, high conscientiousness → detail
        global_vs_detail = _clamp((o - c) * 0.5, -1, 1)

        # Causal attribution: internal vs external
        if agg.get("self_efficacy_impact", 0) > 0.1:
            attribution = "internal"
        elif agg.get("controllability", 0) < -0.1:
            attribution = "external"
        else:
            attribution = "balanced"

        def _cr(val: Any, reasoning: str) -> ConfidenceRated:
            return ConfidenceRated(round(val, 4) if isinstance(val, float) else val, conf, [], reasoning)

        return CognitiveStyle(
            abstract_vs_concrete=_cr(abstract_vs_concrete, "openness*0.6 + meaning_violation*0.15"),
            global_vs_detail=_cr(global_vs_detail, "(openness - conscientiousness)*0.5"),
            causal_attribution_tendency=_cr(attribution, "from self_efficacy_impact vs controllability"),
            reflective_depth=_cr(reflective_depth, "openness*0.35 + conscientiousness*0.25"),
            coherence_need=_cr(coherence_need, "conscientiousness*0.4 + (1-openness)*0.3"),
            ambiguity_tolerance=_cr(ambiguity_tol, "openness*0.4 - neuroticism*0.3 + (1-uncertainty)*0.2"),
        )

    def _infer_affective_dynamics(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> AffectiveDynamics:
        n = b5["neuroticism"]
        pt = agg.get("physical_threat", 0)
        st = agg.get("social_threat", 0)
        unc = agg.get("uncertainty", 0)
        se = agg.get("self_efficacy_impact", 0)
        ctrl = agg.get("controllability", 0)

        arousal = _clamp(n * 0.4 + _safe_mean([pt, st]) * 0.3)
        recovery = _clamp((1 - n) * 0.4 + max(0, se) * 0.3 + max(0, ctrl) * 0.2)

        # Emotion channel weights
        shame = _clamp(st * 0.4 + (1 - max(0, se)) * 0.3)
        anger = _clamp((1 - max(0, ctrl)) * 0.3 + pt * 0.2)
        anxiety = _clamp(unc * 0.4 + pt * 0.3)
        sadness = _clamp(agg.get("loss", 0) * 0.4 + st * 0.15)
        void = _clamp(agg.get("meaning_violation", 0) * 0.4 + agg.get("loss", 0) * 0.2)
        disgust = _clamp(agg.get("contamination", 0) * 0.5 + agg.get("moral_salience", 0) * 0.2)

        weights = {
            "shame": round(shame, 4),
            "anger": round(anger, 4),
            "anxiety": round(anxiety, 4),
            "sadness": round(sadness, 4),
            "void": round(void, 4),
            "disgust": round(disgust, 4),
        }
        # Top dominant emotions
        sorted_emotions = sorted(weights.items(), key=lambda kv: -kv[1])
        dominant = [
            ConfidenceRated(name, conf, [], f"weight={w:.3f}")
            for name, w in sorted_emotions[:3]
            if w > 0.1
        ]

        return AffectiveDynamics(
            baseline_arousal=ConfidenceRated(round(arousal, 4), conf, [], "neuroticism*0.4 + mean_threat*0.3"),
            recovery_speed=ConfidenceRated(round(recovery, 4), conf, [], "(1-N)*0.4 + SE*0.3 + ctrl*0.2"),
            dominant_emotions=dominant,
            emotion_channel_weights=weights,
        )

    def _infer_social_orientation(
        self, b5: dict[str, float], conf: str,
    ) -> SocialOrientation:
        o = b5["openness"]
        c = b5["conscientiousness"]
        e = b5["extraversion"]
        a = b5["agreeableness"]
        n = b5["neuroticism"]

        orientations = {
            "compete": _clamp((1 - a) * 0.4 + e * 0.2),
            "cooperate": _clamp(a * 0.4 + e * 0.2),
            "attach": _clamp(e * 0.3 + n * 0.2 + a * 0.15),
            "avoid": _clamp((1 - e) * 0.3 + n * 0.25),
            "please": _clamp(a * 0.35 + n * 0.25 - (1 - a) * 0.15),
            "dominate": _clamp(e * 0.3 + (1 - a) * 0.3),
            "observe": _clamp(o * 0.3 + (1 - e) * 0.2 + c * 0.15),
        }

        return SocialOrientation(
            orientation_weights={
                k: ConfidenceRated(round(v, 4), conf, [], f"Big Five projection")
                for k, v in orientations.items()
            }
        )

    def _infer_precision_allocation(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> PrecisionAllocation:
        n = b5["neuroticism"]
        e = b5["extraversion"]

        # Hypersensitive channels: high appraisal values
        hyper: list[ConfidenceRated] = []
        if agg.get("physical_threat", 0) > 0.3:
            hyper.append(ConfidenceRated("danger", conf, [], "high physical_threat appraisal"))
        if agg.get("social_threat", 0) > 0.3:
            hyper.append(ConfidenceRated("social_rejection", conf, [], "high social_threat appraisal"))
        if agg.get("attachment_signal", 0) > 0.3:
            hyper.append(ConfidenceRated("attachment", conf, [], "high attachment_signal"))
        if n > 0.6:
            hyper.append(ConfidenceRated("threat_general", conf, [], "high neuroticism"))

        # Blind spots: suppressed channels
        blind: list[ConfidenceRated] = []
        if agg.get("novelty", 0) < 0.1 and b5["openness"] < 0.4:
            blind.append(ConfidenceRated("novelty", conf, [], "low novelty + low openness"))
        if agg.get("attachment_signal", 0) < -0.1:
            blind.append(ConfidenceRated("attachment", conf, [], "negative attachment signal"))

        # Internal vs external balance
        int_ext = _clamp((1 - e) * 0.4 + n * 0.2 - e * 0.2, -1, 1)
        # Immediate vs narrative
        imm_nar = _clamp(
            agg.get("uncertainty", 0) * 0.3
            - agg.get("meaning_violation", 0) * 0.2
            - b5["openness"] * 0.2,
            -1, 1,
        )

        return PrecisionAllocation(
            hypersensitive_channels=hyper,
            blind_spots=blind,
            internal_vs_external=ConfidenceRated(round(int_ext, 4), conf, [], "introversion bias"),
            immediate_vs_narrative=ConfidenceRated(round(imm_nar, 4), conf, [], "present vs story focus"),
        )

    def _infer_temporal_structure(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> TemporalStructure:
        loss = agg.get("loss", 0)
        ctrl = agg.get("controllability", 0)
        pt = agg.get("physical_threat", 0)
        unc = agg.get("uncertainty", 0)
        se = agg.get("self_efficacy_impact", 0)

        past = loss * 0.4 + max(0, (1 - max(0, ctrl))) * 0.2
        present = pt * 0.3 + unc * 0.3
        future = max(0, ctrl) * 0.3 + max(0, se) * 0.3

        total = past + present + future
        if total > 0:
            past /= total
            present /= total
            future /= total
        else:
            past = present = future = 1 / 3

        return TemporalStructure(
            past_trauma_weight=ConfidenceRated(round(past, 4), conf, [], "loss*0.4 + (1-ctrl)*0.2"),
            present_pressure_weight=ConfidenceRated(round(present, 4), conf, [], "threat*0.3 + uncertainty*0.3"),
            future_imagination_weight=ConfidenceRated(round(future, 4), conf, [], "ctrl*0.3 + SE*0.3"),
        )

    def _infer_value_hierarchy(
        self, agg: dict[str, float], b5: dict[str, float], conf: str,
    ) -> AnalysisValueHierarchy:
        o, c, e, a, n = b5["openness"], b5["conscientiousness"], b5["extraversion"], b5["agreeableness"], b5["neuroticism"]
        pt = agg.get("physical_threat", 0)
        st = agg.get("social_threat", 0)
        se = agg.get("self_efficacy_impact", 0)
        att = agg.get("attachment_signal", 0)

        scores = {
            "survival": _clamp(0.5 + pt * 0.3 + n * 0.15),
            "safety": _clamp(0.5 + n * 0.25 + pt * 0.15 - o * 0.1),
            "control": _clamp(0.5 + c * 0.2 + max(0, se) * 0.15 - agg.get("uncertainty", 0) * 0.1),
            "dignity": _clamp(0.5 + st * 0.2 + n * 0.1),
            "relation": _clamp(0.5 + a * 0.2 + e * 0.15 + max(0, att) * 0.1),
            "achievement": _clamp(0.5 + c * 0.25 + max(0, se) * 0.15),
            "freedom": _clamp(0.5 + o * 0.25 + (1 - c) * 0.1),
            "truth": _clamp(0.5 + o * 0.2 + c * 0.1),
            "meaning": _clamp(0.5 + o * 0.2 + agg.get("meaning_violation", 0) * 0.15),
            "contribution": _clamp(0.5 + a * 0.2 + e * 0.1),
        }

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return AnalysisValueHierarchy(
            ranked_values=[
                (name, ConfidenceRated(round(score, 4), conf, [], "Big Five + appraisal projection"))
                for name, score in ranked
            ]
        )

    # ------------------------------------------------------------------
    # Step 4: Defense mechanism analysis
    # ------------------------------------------------------------------

    def _analyze_defenses(
        self, b5: dict[str, float], agg: dict[str, float],
    ) -> DefenseMechanismProfile:
        entries: list[DefenseMechanismEntry] = []
        for mech in _DEFENSE_MECHANISMS:
            bias = mech["personality_bias"](b5)
            # Only include mechanisms with meaningful activation
            if bias > 0.25:
                triggers: list[str] = []
                if mech["strategy"] == "suppress" and b5["neuroticism"] > 0.5:
                    triggers.append("high anxiety / threat")
                if mech["strategy"] == "assimilate" and b5["conscientiousness"] > 0.5:
                    triggers.append("ego-inconsistent feedback")
                if mech["strategy"] == "redirect" and agg.get("social_threat", 0) > 0.3:
                    triggers.append("social conflict")
                if mech["strategy"] == "accommodate" and b5["openness"] > 0.5:
                    triggers.append("novel challenge")
                if not triggers:
                    triggers.append("general stress")

                conf = "high" if bias > 0.5 else "medium" if bias > 0.35 else "low"
                entries.append(DefenseMechanismEntry(
                    name=mech["name"],
                    target_error=mech["target_error"],
                    short_term_benefit=mech["short_term_benefit"],
                    long_term_cost=mech["long_term_cost"],
                    triggers=triggers,
                    confidence=conf,
                ))

        # Sort by confidence descending
        conf_order = {"high": 0, "medium": 1, "low": 2}
        entries.sort(key=lambda e: conf_order.get(e.confidence, 3))
        return DefenseMechanismProfile(mechanisms=entries)

    # ------------------------------------------------------------------
    # Step 5: Strategy model
    # ------------------------------------------------------------------

    def _model_strategies(
        self, b5: dict[str, float], defenses: DefenseMechanismProfile,
        agg: dict[str, float],
    ) -> StrategyProfile:
        o, c, e, a, n = b5["openness"], b5["conscientiousness"], b5["extraversion"], b5["agreeableness"], b5["neuroticism"]

        strategy_scores = {
            "accommodate": _clamp(o * 0.35 + a * 0.2 + (1 - n) * 0.15),
            "assimilate": _clamp(c * 0.35 + o * 0.15 + (1 - n) * 0.1),
            "suppress": _clamp(n * 0.35 + (1 - o) * 0.2 + (1 - e) * 0.1),
            "redirect": _clamp(e * 0.25 + (1 - a) * 0.2 + n * 0.15),
            "withdraw": _clamp((1 - e) * 0.3 + n * 0.25 + (1 - a) * 0.1),
            "control": _clamp(c * 0.3 + (1 - a) * 0.2 + e * 0.15),
        }

        ranked = sorted(strategy_scores.items(), key=lambda kv: -kv[1])
        preferred = [
            ConfidenceRated(name, "medium", [], f"score={score:.3f}")
            for name, score in ranked[:3]
        ]

        # Cost analysis
        cost_analysis: dict[str, ConfidenceRated] = {}
        for name, score in strategy_scores.items():
            if name == "suppress":
                cost = ConfidenceRated("high long-term cost", "medium", [], "precision debt accumulation")
            elif name == "accommodate":
                cost = ConfidenceRated("low long-term cost", "medium", [], "model update is adaptive")
            elif name == "withdraw":
                cost = ConfidenceRated("medium cost", "medium", [], "social isolation risk")
            else:
                cost = ConfidenceRated("moderate cost", "low", [], "context-dependent")
            cost_analysis[name] = cost

        # Blocked strategies: lowest scoring
        blocked = [
            ConfidenceRated(name, "low", [], f"score={score:.3f}, personality disfavors")
            for name, score in ranked[-2:]
        ]

        return StrategyProfile(
            preferred_strategies=preferred,
            cost_analysis=cost_analysis,
            blocked_strategies=blocked,
        )

    # ------------------------------------------------------------------
    # Step 6: Feedback loops
    # ------------------------------------------------------------------

    def _build_feedback_loops(
        self,
        priors: CorePriors,
        defenses: DefenseMechanismProfile,
        strategies: StrategyProfile,
        b5: dict[str, float],
    ) -> list[FeedbackLoop]:
        loops: list[FeedbackLoop] = []

        # Loop 1: Anxiety → Suppression → Blind spots → Surprise → More anxiety
        if b5["neuroticism"] > 0.5:
            loops.append(FeedbackLoop(
                name="Anxiety-Suppression Spiral",
                components=["high neuroticism", "suppress strategy", "precision debt", "surprise increase", "anxiety"],
                description=(
                    "High baseline anxiety triggers suppression of threatening signals. "
                    "This creates blind spots, increasing unexpected prediction errors, "
                    "which further raises anxiety."
                ),
                valence="reinforcing",
            ))

        # Loop 2: Low self-worth → Please/Accommodate → Self-neglect → Confirmation
        if priors.self_worth.value < 0.4 and b5["agreeableness"] > 0.55:  # type: ignore[operator]
            loops.append(FeedbackLoop(
                name="Low Self-Worth Accommodation Loop",
                components=["low self_worth", "please orientation", "self-neglect", "reduced agency", "self_worth confirms"],
                description=(
                    "Low self-worth drives people-pleasing behavior, which sacrifices "
                    "personal needs, reducing self-efficacy evidence, and confirming "
                    "the low self-worth prior."
                ),
                valence="reinforcing",
            ))

        # Loop 3: Exploration → Mastery → Confidence → More exploration
        if b5["openness"] > 0.55 and priors.self_efficacy.value > 0.5:  # type: ignore[operator]
            loops.append(FeedbackLoop(
                name="Exploration-Mastery Virtuous Cycle",
                components=["high openness", "exploration", "novel learning", "self_efficacy increase", "curiosity"],
                description=(
                    "High openness drives exploration, generating mastery experiences "
                    "that reinforce self-efficacy, enabling further exploration."
                ),
                valence="reinforcing",
            ))

        # Loop 4: Distrust → Avoidance → No corrective experience → Distrust persists
        if priors.other_reliability.value < 0.4:  # type: ignore[operator]
            loops.append(FeedbackLoop(
                name="Distrust-Avoidance Loop",
                components=["low other_reliability", "social avoidance", "no corrective contact", "distrust maintained"],
                description=(
                    "Low trust in others leads to social avoidance, preventing "
                    "positive relational experiences that could update the trust prior."
                ),
                valence="reinforcing",
            ))

        # Loop 5: Control need → Rigidity → Surprise when world doesn't comply → More control
        if b5["conscientiousness"] > 0.6 and b5["neuroticism"] > 0.45:
            loops.append(FeedbackLoop(
                name="Control-Rigidity Spiral",
                components=["high control need", "rigid planning", "unexpected deviation", "anxiety", "increased control"],
                description=(
                    "Strong need for control produces rigid behavior. When reality "
                    "deviates from plan, prediction error is amplified, driving even "
                    "stronger control attempts."
                ),
                valence="reinforcing",
            ))

        if not loops:
            loops.append(FeedbackLoop(
                name="Baseline Homeostatic Loop",
                components=["perception", "prediction", "error", "update"],
                description="Standard predictive processing cycle with balanced error correction.",
                valence="balancing",
            ))

        return loops

    # ------------------------------------------------------------------
    # Step 7: Development history inference
    # ------------------------------------------------------------------

    def _infer_development(
        self,
        evidence: list[EvidenceItem],
        agg: dict[str, float],
        b5: dict[str, float],
        defenses: DefenseMechanismProfile,
    ) -> list[ConfidenceRated]:
        inferences: list[ConfidenceRated] = []

        # High social threat → early relational disruption
        if agg.get("social_threat", 0) > 0.3:
            inferences.append(ConfidenceRated(
                "Possible early experiences of social rejection or exclusion shaped "
                "heightened sensitivity to social threat signals.",
                "low", [], "inferred from elevated social_threat appraisal",
            ))

        # High attachment + trust → secure base experience
        if agg.get("attachment_signal", 0) > 0.3 and agg.get("trust_impact", 0) > 0.2:
            inferences.append(ConfidenceRated(
                "Evidence of positive attachment experiences; likely had at least one "
                "reliable caregiver or close relationship.",
                "medium", [], "inferred from positive attachment + trust signals",
            ))

        # High contamination/moral salience → possible exposure to moral injury
        if agg.get("contamination", 0) > 0.3 or agg.get("moral_salience", 0) > 0.3:
            inferences.append(ConfidenceRated(
                "Exposure to morally distressing events may have shaped heightened "
                "contamination sensitivity and moral vigilance.",
                "low", [], "inferred from contamination/moral_salience signals",
            ))

        # Many suppress-type defenses → possible environment that punished expression
        suppress_count = sum(
            1 for m in defenses.mechanisms
            if m.name in ("repression", "denial", "isolation_of_affect", "splitting", "regression")
        )
        if suppress_count >= 2:
            inferences.append(ConfidenceRated(
                "Reliance on suppression-class defenses suggests an environment where "
                "emotional expression was discouraged or punished.",
                "low", [], f"{suppress_count} suppression-type defenses active",
            ))

        # High exploration + low threat → enriched environment
        if agg.get("novelty", 0) > 0.3 and agg.get("physical_threat", 0) < 0.2:
            inferences.append(ConfidenceRated(
                "Pattern suggests access to a relatively safe, stimulating environment "
                "that encouraged exploration.",
                "low", [], "high novelty + low physical threat",
            ))

        if not inferences:
            inferences.append(ConfidenceRated(
                "Insufficient evidence for developmental inferences.",
                "low", [], "limited material",
            ))

        return inferences

    # ------------------------------------------------------------------
    # Step 8: Stability analysis
    # ------------------------------------------------------------------

    def _analyze_stability(
        self,
        b5: dict[str, float],
        defenses: DefenseMechanismProfile,
        agg: dict[str, float],
    ) -> StabilityAnalysis:
        stable: list[ConfidenceRated] = []
        fragile: list[ConfidenceRated] = []
        plastic: list[ConfidenceRated] = []

        # Traits far from neutral are more stable (crystallized)
        for trait, val in b5.items():
            deviation = abs(val - 0.5)
            if deviation > 0.15:
                stable.append(ConfidenceRated(
                    f"{trait} ({val:.2f})", "medium", [],
                    f"deviation {deviation:.2f} from neutral suggests crystallized trait",
                ))
            elif deviation < 0.05:
                plastic.append(ConfidenceRated(
                    f"{trait} ({val:.2f})", "low", [],
                    "near-neutral, highly malleable",
                ))

        # Defense-heavy areas are fragile
        for mech in defenses.mechanisms[:3]:
            if mech.confidence in ("high", "medium"):
                fragile.append(ConfidenceRated(
                    f"Area protected by {mech.name}", "medium", [],
                    f"active defense suggests underlying vulnerability in {mech.target_error}",
                ))

        # High uncertainty areas are plastic
        if agg.get("uncertainty", 0) > 0.3:
            plastic.append(ConfidenceRated(
                "World model under uncertainty", "medium", [],
                "high uncertainty indicates model still updating",
            ))

        if not stable:
            stable.append(ConfidenceRated("no clearly crystallized traits detected", "low", [], ""))
        if not fragile:
            fragile.append(ConfidenceRated("no clear fragility points detected", "low", [], ""))
        if not plastic:
            plastic.append(ConfidenceRated("no clearly plastic dimensions detected", "low", [], ""))

        return StabilityAnalysis(
            stable_core=stable,
            fragile_points=fragile,
            plastic_points=plastic,
        )

    # ------------------------------------------------------------------
    # Step 9: Behavioral predictions
    # ------------------------------------------------------------------

    def _predict_behaviors(
        self,
        b5: dict[str, float],
        strategies: StrategyProfile,
        loops: list[FeedbackLoop],
        agg: dict[str, float],
    ) -> list[ConditionalPrediction]:
        predictions: list[ConditionalPrediction] = []

        # Prediction from dominant strategy
        if strategies.preferred_strategies:
            top = strategies.preferred_strategies[0].value
            predictions.append(ConditionalPrediction(
                scenario="Under moderate stress",
                predicted_behavior=f"Most likely to use '{top}' strategy as primary coping response.",
                confidence="medium",
                reasoning=f"Highest-ranked strategy from personality projection.",
            ))

        # Social prediction
        if b5["extraversion"] > 0.6:
            predictions.append(ConditionalPrediction(
                scenario="In novel social situations",
                predicted_behavior="Will seek engagement and social connection; likely to initiate contact.",
                confidence="medium",
                reasoning="High extraversion drives social approach behavior.",
            ))
        elif b5["extraversion"] < 0.4:
            predictions.append(ConditionalPrediction(
                scenario="In novel social situations",
                predicted_behavior="Will observe before engaging; may avoid large groups.",
                confidence="medium",
                reasoning="Low extraversion predicts cautious social approach.",
            ))

        # Threat response prediction
        if b5["neuroticism"] > 0.6:
            predictions.append(ConditionalPrediction(
                scenario="Facing unexpected threat or criticism",
                predicted_behavior="Heightened physiological arousal with slow recovery; may ruminate.",
                confidence="medium",
                reasoning="High neuroticism predicts amplified threat response.",
            ))

        # Achievement prediction
        if b5["conscientiousness"] > 0.6:
            predictions.append(ConditionalPrediction(
                scenario="Given an ambiguous task with no clear instructions",
                predicted_behavior="Will create structure and clear goals; may feel uncomfortable with ambiguity.",
                confidence="medium",
                reasoning="High conscientiousness drives structure-seeking.",
            ))

        # Feedback loop predictions
        for loop in loops:
            if loop.valence == "reinforcing" and "spiral" in loop.name.lower():
                predictions.append(ConditionalPrediction(
                    scenario=f"Under prolonged stress (activating '{loop.name}')",
                    predicted_behavior="Risk of escalating cycle; external intervention may be needed to break loop.",
                    confidence="low",
                    reasoning=f"Reinforcing loop: {' → '.join(loop.components[:3])}...",
                ))

        if not predictions:
            predictions.append(ConditionalPrediction(
                scenario="General",
                predicted_behavior="Insufficient data for specific behavioral predictions.",
                confidence="low",
                reasoning="Limited evidence base.",
            ))

        return predictions

    # ------------------------------------------------------------------
    # Self / Other / Relational model inference
    # ------------------------------------------------------------------

    def _infer_self_model(
        self, b5: dict[str, float], priors: CorePriors, conf: str,
    ) -> SelfModelProfile:
        # Self-narrative from dominant traits
        dominant = max(b5.items(), key=lambda kv: abs(kv[1] - 0.5))
        narrative = f"Core identity organized around {dominant[0]} (value={dominant[1]:.2f})."

        threats: list[ConfidenceRated] = []
        if priors.self_worth.value < 0.4:  # type: ignore[operator]
            threats.append(ConfidenceRated("worthlessness", conf, [], "low self_worth prior"))
        if priors.self_efficacy.value < 0.4:  # type: ignore[operator]
            threats.append(ConfidenceRated("incompetence", conf, [], "low self_efficacy prior"))

        consistency: list[ConfidenceRated] = []
        if b5["conscientiousness"] > 0.6:
            consistency.append(ConfidenceRated("moral consistency", conf, [], "high conscientiousness"))
        if b5["openness"] < 0.4:
            consistency.append(ConfidenceRated("tradition / stability", conf, [], "low openness"))

        return SelfModelProfile(
            self_narrative=ConfidenceRated(narrative, conf, [], "from dominant Big Five trait"),
            identity_consistency_needs=consistency,
            identity_threats=threats,
        )

    def _infer_other_model(
        self, b5: dict[str, float], agg: dict[str, float], conf: str,
    ) -> OtherModelProfile:
        trust_base = agg.get("trust_impact", 0)
        attachment = agg.get("attachment_signal", 0)

        def _model(label: str, trust_mod: float, pred_mod: float) -> ConfidenceRated:
            t = _clamp(0.5 + trust_base * 0.3 + trust_mod)
            p = _clamp(0.5 + pred_mod)
            return ConfidenceRated(
                {"trust": round(t, 3), "predictability": round(p, 3)},
                conf, [],
                f"{label} template from appraisal + Big Five",
            )

        return OtherModelProfile(
            intimate_model=_model("intimate", attachment * 0.2, b5["agreeableness"] * 0.15),
            authority_model=_model("authority", -b5["neuroticism"] * 0.1, b5["conscientiousness"] * 0.1),
            peer_model=_model("peer", b5["agreeableness"] * 0.1, 0.0),
            weaker_model=_model("weaker", b5["agreeableness"] * 0.15, 0.05),
            stranger_model=_model("stranger", -0.1, -b5["neuroticism"] * 0.1),
        )

    def _infer_relational_templates(
        self, b5: dict[str, float], social: SocialOrientation, conf: str,
    ) -> RelationalTemplates:
        intimate: list[ConfidenceRated] = []
        coop: list[ConfidenceRated] = []
        power: list[ConfidenceRated] = []

        if b5["agreeableness"] > 0.6:
            intimate.append(ConfidenceRated("warm, accommodating in close relationships", conf, [], "high A"))
            coop.append(ConfidenceRated("strong team player, consensus-seeking", conf, [], "high A"))
        elif b5["agreeableness"] < 0.4:
            intimate.append(ConfidenceRated("maintains emotional distance, values autonomy", conf, [], "low A"))
            coop.append(ConfidenceRated("independent operator, may resist group norms", conf, [], "low A"))

        if b5["extraversion"] > 0.6:
            intimate.append(ConfidenceRated("seeks frequent contact, expressive", conf, [], "high E"))
            power.append(ConfidenceRated("naturally assumes leadership roles", conf, [], "high E"))
        elif b5["extraversion"] < 0.4:
            intimate.append(ConfidenceRated("needs space, selective sharing", conf, [], "low E"))
            power.append(ConfidenceRated("prefers advisory over directive roles", conf, [], "low E"))

        if b5["neuroticism"] > 0.6:
            intimate.append(ConfidenceRated("anxious attachment patterns possible", conf, [], "high N"))

        return RelationalTemplates(
            intimate_patterns=intimate or [ConfidenceRated("insufficient data", "low", [], "")],
            cooperative_patterns=coop or [ConfidenceRated("insufficient data", "low", [], "")],
            power_patterns=power or [ConfidenceRated("insufficient data", "low", [], "")],
        )

    # ------------------------------------------------------------------
    # Step 10: Report compilation helpers
    # ------------------------------------------------------------------

    def _assess_missing_evidence(
        self, evidence: list[EvidenceItem], materials: list[str],
    ) -> list[str]:
        missing: list[str] = []
        categories = {e.category for e in evidence}
        if "relational" not in categories:
            missing.append("No relational/social interaction evidence found.")
        if "emotional" not in categories:
            missing.append("No emotional processing evidence found.")
        if "cognitive" not in categories:
            missing.append("No cognitive/exploration evidence found.")
        if len(materials) < 3:
            missing.append("Limited material volume; confidence is reduced.")
        low_signal_count = sum(1 for e in evidence if "low_signal" in e.tags)
        if low_signal_count > len(evidence) * 0.5:
            missing.append("Many materials produced weak semantic signal; richer narratives would improve analysis.")
        return missing

    def _unresolvable_questions(self, evidence: list[EvidenceItem]) -> list[str]:
        questions: list[str] = []
        categories = {e.category for e in evidence}
        if "relational" not in categories:
            questions.append("Attachment style cannot be reliably determined without relational evidence.")
        if len(evidence) < 5:
            questions.append("Trait stability vs. state effects cannot be distinguished with limited samples.")
        questions.append("Early developmental history requires biographical data not available from behavioral samples alone.")
        return questions

    def _generate_summary(
        self,
        b5: dict[str, float],
        priors: CorePriors,
        defenses: DefenseMechanismProfile,
        hypothesis: PredictiveHypothesis,
    ) -> str:
        # Dominant trait
        traits_sorted = sorted(b5.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
        top_traits = [(t, v) for t, v in traits_sorted[:2]]
        trait_desc = ", ".join(
            f"{'high' if v > 0.5 else 'low'} {t}" for t, v in top_traits
        )

        # Top defenses
        top_def = ", ".join(m.name for m in defenses.mechanisms[:2]) if defenses.mechanisms else "none identified"

        return (
            f"Personality organized around {trait_desc}. "
            f"Primary drives: {', '.join(hypothesis.dominant_drives[:3])}. "
            f"Main threat axis: {hypothesis.primary_threat_model}. "
            f"Preferred error reduction: {hypothesis.preferred_error_reduction}. "
            f"Key defenses: {top_def}."
        )

    def _generate_conclusion(
        self, b5: dict[str, float], hypothesis: PredictiveHypothesis,
    ) -> str:
        drive = hypothesis.dominant_drives[0] if hypothesis.dominant_drives else "homeostasis"
        return (
            f"A {drive}-oriented system with "
            f"{hypothesis.primary_threat_model} as primary vulnerability axis."
        )

    def _compute_overall_confidence(
        self, n_materials: int, n_evidence: int, base_conf: str,
    ) -> float:
        base = {"high": 0.7, "medium": 0.5, "low": 0.3}.get(base_conf, 0.3)
        # Bonus for more evidence
        volume_bonus = min(0.2, n_evidence * 0.02)
        return round(_clamp(base + volume_bonus), 3)
