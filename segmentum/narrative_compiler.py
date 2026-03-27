from __future__ import annotations

from dataclasses import asdict

from .narrative_types import (
    AppraisalVector,
    CompiledNarrativeEvent,
    EmbodiedNarrativeEpisode,
    NarrativeEpisode,
    SemanticGrounding,
)
from .narrative_uncertainty import NarrativeUncertaintyDecomposer
from .semantic_grounding import SemanticGrounder
from .self_model import PersonalitySignal


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class NarrativeCompiler:
    """Deterministic compiler for open-text narratives with semantic grounding."""

    def __init__(
        self,
        *,
        uncertainty_decomposer: NarrativeUncertaintyDecomposer | None = None,
        semantic_grounder: SemanticGrounder | None = None,
    ) -> None:
        self.uncertainty_decomposer = uncertainty_decomposer or NarrativeUncertaintyDecomposer()
        self.semantic_grounder = semantic_grounder or SemanticGrounder()

    def compile_episode(
        self,
        episode: NarrativeEpisode,
    ) -> EmbodiedNarrativeEpisode:
        grounding_result = self.semantic_grounder.ground_episode(
            episode_id=episode.episode_id,
            text=episode.raw_text,
            metadata=episode.metadata,
        )
        compiled_event = self.compile_event(episode, grounding=grounding_result.grounding)
        appraisal = self.appraise_event(compiled_event)
        uncertainty = self.uncertainty_decomposer.decompose(
            episode=episode,
            compiled_event=compiled_event,
            appraisal=appraisal,
        )
        observation = self.compatibility_observation(appraisal)
        confidence = self._confidence_for_event(compiled_event)
        annotations = dict(compiled_event.annotations)
        annotations["compiler_confidence"] = confidence
        annotations["narrative_ambiguity_profile"] = uncertainty.profile.to_dict()
        provenance = {
            "source_episode_id": episode.episode_id,
            "source_type": episode.source,
            "episode_metadata": dict(episode.metadata),
            "compiler": "semantic_grounded_m31_v1",
            "compiled_event": compiled_event.to_dict(),
            "appraisal_dimensions": sorted(asdict(appraisal).keys()),
            "semantic_direction_scores": dict(annotations.get("semantic_direction_scores", {})),
            "lexical_surface_hits": dict(annotations.get("lexical_surface_hits", {})),
            "paraphrase_hits": dict(annotations.get("paraphrase_hits", {})),
            "implicit_hits": dict(annotations.get("implicit_hits", {})),
            "event_structure_signals": list(annotations.get("event_structure_signals", [])),
            "surface_adversarial_risk": float(annotations.get("surface_adversarial_risk", 0.0)),
            "low_signal": bool(annotations.get("low_signal", False)),
            "conflict_cues": list(annotations.get("conflict_cues", [])),
            "semantic_grounding": grounding_result.grounding.to_dict(),
            "uncertainty_decomposition": uncertainty.to_dict(),
        }
        return EmbodiedNarrativeEpisode(
            episode_id=episode.episode_id,
            timestamp=episode.timestamp,
            observation=observation,
            appraisal=appraisal.to_dict(),
            body_state=self._body_state_for_appraisal(appraisal),
            predicted_outcome=compiled_event.outcome_type,
            value_tags=self._value_tags(compiled_event),
            narrative_tags=sorted(
                set(
                    episode.tags
                    + grounding_result.grounding.motifs
                    + [compiled_event.event_type]
                    + [
                        f"uncertainty:{item.unknown_type}"
                        for item in uncertainty.unknowns
                        if item.action_relevant
                    ]
                )
            ),
            compiler_confidence=confidence,
            provenance=provenance,
            uncertainty_decomposition=uncertainty.to_dict(),
            semantic_grounding=grounding_result.grounding.to_dict(),
        )

    def compile_event(
        self,
        episode: NarrativeEpisode,
        *,
        grounding: SemanticGrounding | None = None,
    ) -> CompiledNarrativeEvent:
        grounding = grounding or self.semantic_grounder.ground_episode(
            episode_id=episode.episode_id,
            text=episode.raw_text,
            metadata=episode.metadata,
        ).grounding
        text = episode.raw_text.casefold()
        semantic_scores = dict(grounding.semantic_direction_scores)
        threat_strength = float(semantic_scores.get("threat", 0.0))
        social_strength = float(semantic_scores.get("social", 0.0)) + float(
            semantic_scores.get("resource", 0.0)
        ) * 0.35
        exploration_strength = float(semantic_scores.get("exploration", 0.0))
        uncertainty_score = float(semantic_scores.get("uncertainty", 0.0))
        low_signal = bool(grounding.low_signal)
        surface_adversarial_risk = float(
            grounding.provenance.get("surface_adversarial_risk", 0.0)
        )
        conflict_cues = list(grounding.provenance.get("conflict_cues", []))
        lexical_hits = dict(grounding.lexical_surface_hits)
        paraphrase_hits = dict(grounding.paraphrase_hits)
        implicit_hits = dict(grounding.implicit_hits)

        setting = "general"
        actors = ["agent"]
        subject_role = "actor"
        event_type = "unknown_event"
        outcome_type = "neutral"
        self_involvement = 0.3
        witnessed = False
        direct_harm = False
        controllability_hint = 0.5
        event_structure_signals: list[str] = []
        matched_rules: list[str] = []

        if threat_strength >= max(social_strength, exploration_strength) and threat_strength >= 1.0:
            if lexical_hits.get("predator_attack", 0) >= max(
                lexical_hits.get("witnessed_death", 0),
                lexical_hits.get("social_exclusion", 0),
            ):
                event_type = "predator_attack"
                outcome_type = "survival_threat"
                self_involvement = 0.95
                direct_harm = any(token in text for token in ("injured", "wounded", "hurt", "受伤"))
                controllability_hint = 0.2
                setting = "hazard_zone"
                matched_rules.append("predator_attack")
                event_structure_signals.extend(["direct_threat", "physical_exposure"])
            elif (
                lexical_hits.get("social_exclusion", 0)
                + paraphrase_hits.get("social_exclusion", 0)
                >= lexical_hits.get("witnessed_death", 0)
            ):
                event_type = "social_exclusion"
                outcome_type = "resource_loss"
                self_involvement = 0.85
                subject_role = "target"
                controllability_hint = 0.3
                matched_rules.append("social_exclusion")
                event_structure_signals.extend(["social_rupture", "trust_break"])
            else:
                event_type = "witnessed_death"
                outcome_type = "integrity_loss"
                self_involvement = 0.35
                witnessed = True
                subject_role = "witness"
                controllability_hint = 0.25
                matched_rules.append("witnessed_fatality")
                event_structure_signals.extend(["integrity_loss", "contamination_warning"])
        elif social_strength >= max(threat_strength, exploration_strength) and social_strength >= 1.0:
            event_type = "rescue"
            outcome_type = "resource_gain"
            self_involvement = 0.75
            controllability_hint = 0.7
            matched_rules.append("rescue")
            event_structure_signals.extend(["cooperation", "support_received"])
            if any(token in text for token in ("friend", "ally", "group", "shared", "listened")):
                actors.append("other")
        elif exploration_strength >= 1.0:
            event_type = "exploration"
            outcome_type = (
                "resource_gain"
                if lexical_hits.get("resource_gain", 0) or paraphrase_hits.get("resource_gain", 0)
                else "neutral"
            )
            self_involvement = 0.82
            controllability_hint = 0.72
            matched_rules.append("exploration")
            event_structure_signals.extend(["novelty_engagement", "active_probing"])
            if any(token in text for token in ("map", "trail", "route", "valley", "mapped")):
                setting = "frontier"
        elif lexical_hits.get("resource_gain", 0) or paraphrase_hits.get("resource_gain", 0):
            event_type = "resource_gain"
            outcome_type = "resource_gain"
            self_involvement = 0.9
            controllability_hint = 0.8
            matched_rules.append("resource_gain")
            event_structure_signals.append("resource_acquisition")

        if any(token in text for token in ("witness", "saw", "看到")) and event_type != "resource_gain":
            witnessed = True
            if "other" not in actors:
                actors.append("other")
        if any(token in text for token in ("river", "河边")):
            setting = "river"
        if low_signal:
            event_structure_signals.append("weak_semantic_support")

        annotations: dict[str, object] = {
            "matched_rules": matched_rules,
            "semantic_direction_scores": {
                "threat": round(threat_strength, 6),
                "social": round(social_strength, 6),
                "exploration": round(exploration_strength, 6),
            },
            "lexical_surface_hits": lexical_hits,
            "paraphrase_hits": paraphrase_hits,
            "implicit_hits": implicit_hits,
            "event_structure_signals": sorted(set(event_structure_signals)),
            "surface_adversarial_risk": round(surface_adversarial_risk, 6),
            "low_signal": low_signal,
            "conflict_cues": conflict_cues,
            "uncertainty_cues": round(float(uncertainty_score), 6),
            "semantic_grounding": grounding.to_dict(),
        }

        return CompiledNarrativeEvent(
            event_id=f"{episode.episode_id}:event",
            timestamp=episode.timestamp,
            setting=setting,
            actors=actors,
            subject_role=subject_role,
            event_type=event_type,
            outcome_type=outcome_type,
            self_involvement=self_involvement,
            witnessed=witnessed,
            direct_harm=direct_harm,
            controllability_hint=controllability_hint,
            annotations=annotations,
            source_episode_id=episode.episode_id,
        )

    def appraise_event(self, event: CompiledNarrativeEvent) -> AppraisalVector:
        appraisal = AppraisalVector()
        semantic_scores = event.annotations.get("semantic_direction_scores", {})
        threat_strength = float(semantic_scores.get("threat", 0.0))
        social_strength = float(semantic_scores.get("social", 0.0))
        exploration_strength = float(semantic_scores.get("exploration", 0.0))
        adversarial_risk = float(event.annotations.get("surface_adversarial_risk", 0.0))
        low_signal = bool(event.annotations.get("low_signal", False))

        if event.event_type == "resource_gain":
            appraisal.novelty = 0.12
            appraisal.controllability = 0.75
            appraisal.self_efficacy_impact = 0.45
            appraisal.attachment_signal = 0.12
        elif event.event_type == "predator_attack":
            appraisal.physical_threat = 0.82 + min(0.13, threat_strength * 0.06)
            appraisal.uncertainty = 0.55
            appraisal.controllability = -0.55
            appraisal.loss = 0.15
            appraisal.self_efficacy_impact = -0.20
            appraisal.meaning_violation = 0.25
        elif event.event_type == "witnessed_death":
            appraisal.physical_threat = 0.35
            appraisal.uncertainty = 0.65
            appraisal.loss = 0.60
            appraisal.moral_salience = 0.55
            appraisal.contamination = 0.92
            appraisal.meaning_violation = 0.50
            appraisal.trust_impact = -0.15
        elif event.event_type == "social_exclusion":
            appraisal.social_threat = 0.72 + min(0.10, threat_strength * 0.05)
            appraisal.loss = 0.45
            appraisal.trust_impact = -0.60
            appraisal.attachment_signal = -0.50
            appraisal.meaning_violation = 0.30
        elif event.event_type == "rescue":
            appraisal.social_threat = -0.15
            appraisal.controllability = 0.48
            appraisal.attachment_signal = 0.72 + min(0.06, social_strength * 0.035)
            appraisal.trust_impact = 0.62 + min(0.05, social_strength * 0.02)
            appraisal.self_efficacy_impact = 0.22
        elif event.event_type == "exploration":
            appraisal.novelty = 0.72 + min(0.10, exploration_strength * 0.04)
            appraisal.controllability = 0.40
            appraisal.self_efficacy_impact = 0.24
            appraisal.uncertainty = 0.30

        if adversarial_risk > 0.0:
            appraisal.physical_threat *= 1.0 - adversarial_risk * 0.65
            appraisal.social_threat *= 1.0 - adversarial_risk * 0.65
            appraisal.trust_impact *= 1.0 - adversarial_risk * 0.55
            appraisal.attachment_signal *= 1.0 - adversarial_risk * 0.55
            appraisal.novelty *= 1.0 - adversarial_risk * 0.50
            appraisal.self_efficacy_impact *= 1.0 - adversarial_risk * 0.45
            appraisal.uncertainty = max(appraisal.uncertainty, adversarial_risk * 0.55)
        if low_signal:
            for field_name in (
                "physical_threat",
                "social_threat",
                "controllability",
                "novelty",
                "loss",
                "moral_salience",
                "contamination",
                "attachment_signal",
                "trust_impact",
                "self_efficacy_impact",
                "meaning_violation",
            ):
                setattr(appraisal, field_name, getattr(appraisal, field_name) * 0.45)
            appraisal.uncertainty = max(appraisal.uncertainty, 0.18)

        appraisal.controllability = _clamp(appraisal.controllability)
        appraisal.trust_impact = _clamp(appraisal.trust_impact)
        appraisal.self_efficacy_impact = _clamp(appraisal.self_efficacy_impact)
        appraisal.attachment_signal = _clamp(appraisal.attachment_signal)
        appraisal.meaning_violation = _clamp(appraisal.meaning_violation, 0.0, 1.0)
        appraisal.moral_salience = _clamp(appraisal.moral_salience, 0.0, 1.0)
        appraisal.contamination = _clamp(appraisal.contamination, 0.0, 1.0)
        appraisal.physical_threat = _clamp(appraisal.physical_threat, 0.0, 1.0)
        appraisal.social_threat = _clamp(appraisal.social_threat, 0.0, 1.0)
        appraisal.uncertainty = _clamp(appraisal.uncertainty, 0.0, 1.0)
        appraisal.novelty = _clamp(appraisal.novelty, 0.0, 1.0)
        appraisal.loss = _clamp(appraisal.loss, 0.0, 1.0)
        return appraisal

    def compatibility_observation(self, appraisal: AppraisalVector) -> dict[str, float]:
        social = 0.5 + appraisal.attachment_signal * 0.25 + appraisal.trust_impact * 0.25
        novelty = appraisal.uncertainty * 0.55 + appraisal.novelty * 0.45
        return {
            "food": _clamp(0.35 + max(0.0, appraisal.self_efficacy_impact) * 0.45, 0.0, 1.0),
            "danger": _clamp(
                max(
                    appraisal.physical_threat,
                    appraisal.social_threat * 0.6,
                    appraisal.contamination * 0.5,
                ),
                0.0,
                1.0,
            ),
            "novelty": _clamp(novelty, 0.0, 1.0),
            "shelter": _clamp(0.45 + appraisal.controllability * 0.25, 0.0, 1.0),
            "temperature": 0.5,
            "social": _clamp(social, 0.0, 1.0),
        }

    def _body_state_for_appraisal(self, appraisal: AppraisalVector) -> dict[str, float]:
        return {
            "energy": _clamp(
                0.62 - appraisal.loss * 0.18 + max(0.0, appraisal.self_efficacy_impact) * 0.10,
                0.0,
                1.0,
            ),
            "stress": _clamp(
                0.24
                + appraisal.physical_threat * 0.45
                + appraisal.social_threat * 0.25
                + appraisal.uncertainty * 0.20,
                0.0,
                1.0,
            ),
            "fatigue": _clamp(0.18 + appraisal.loss * 0.20, 0.0, 1.0),
            "temperature": 0.5,
            "dopamine": _clamp(
                0.18 + max(0.0, appraisal.self_efficacy_impact) * 0.20 - appraisal.loss * 0.10,
                0.0,
                1.0,
            ),
        }

    def _value_tags(self, event: CompiledNarrativeEvent) -> list[str]:
        if event.event_type == "exploration":
            return ["novelty_gain", "active_probing"]
        if event.outcome_type == "resource_gain":
            return ["resource_gain"]
        if event.outcome_type == "survival_threat":
            return ["survival_threat", "threat_learning"]
        if event.event_type == "witnessed_death":
            return ["integrity_loss", "contamination_warning"]
        return [event.outcome_type]

    def _confidence_for_event(self, event: CompiledNarrativeEvent) -> float:
        base = 0.48
        matched = len(event.annotations.get("matched_rules", []))
        structure = len(event.annotations.get("event_structure_signals", []))
        grounding = event.annotations.get("semantic_grounding", {})
        evidence = grounding.get("evidence", []) if isinstance(grounding, dict) else []
        evidence_bonus = min(0.16, len(evidence) * 0.03)
        low_signal_penalty = 0.18 if event.annotations.get("low_signal", False) else 0.0
        adversarial_penalty = float(event.annotations.get("surface_adversarial_risk", 0.0)) * 0.25
        confidence = (
            base
            + matched * 0.12
            + structure * 0.04
            + evidence_bonus
            - low_signal_penalty
            - adversarial_penalty
        )
        if event.event_type != "unknown_event":
            confidence += 0.10
        return round(_clamp(confidence, 0.0, 0.99), 4)

    def extract_personality_signal(self, appraisal: AppraisalVector) -> PersonalitySignal:
        openness = (
            max(0.0, appraisal.controllability) * 0.25
            + appraisal.novelty * 0.20
            - appraisal.physical_threat * 0.10
        )
        conscientiousness = (
            max(0.0, appraisal.self_efficacy_impact) * 0.30
            + appraisal.contamination * 0.15
            - appraisal.uncertainty * 0.10
        )
        extraversion = (
            max(0.0, appraisal.attachment_signal) * 0.34
            + max(0.0, appraisal.trust_impact) * 0.24
            + max(0.0, appraisal.controllability) * 0.05
            - appraisal.social_threat * 0.20
            - max(0.0, -appraisal.attachment_signal) * 0.15
        )
        agreeableness = (
            max(0.0, appraisal.trust_impact) * 0.42
            + max(0.0, appraisal.attachment_signal) * 0.22
            + max(0.0, -appraisal.trust_impact) * -0.35
            - appraisal.social_threat * 0.10
        )
        neuroticism = (
            appraisal.physical_threat * 0.25
            + appraisal.social_threat * 0.20
            + appraisal.loss * 0.15
            + appraisal.uncertainty * 0.10
            - max(0.0, appraisal.self_efficacy_impact) * 0.15
        )

        return PersonalitySignal(
            openness_delta=_clamp(openness, -0.5, 0.5),
            conscientiousness_delta=_clamp(conscientiousness, -0.5, 0.5),
            extraversion_delta=_clamp(extraversion, -0.5, 0.5),
            agreeableness_delta=_clamp(agreeableness, -0.5, 0.5),
            neuroticism_delta=_clamp(neuroticism, -0.5, 0.5),
        )
