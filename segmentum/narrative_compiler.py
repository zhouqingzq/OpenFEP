from __future__ import annotations

from dataclasses import asdict

from .narrative_types import (
    AppraisalVector,
    CompiledNarrativeEvent,
    EmbodiedNarrativeEpisode,
    NarrativeEpisode,
)
from .self_model import PersonalitySignal


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class NarrativeCompiler:
    """Deterministic rule-based compiler for seeded narrative scenarios."""

    def compile_episode(
        self,
        episode: NarrativeEpisode,
    ) -> EmbodiedNarrativeEpisode:
        compiled_event = self.compile_event(episode)
        appraisal = self.appraise_event(compiled_event)
        observation = self.compatibility_observation(appraisal)
        confidence = self._confidence_for_event(compiled_event)
        return EmbodiedNarrativeEpisode(
            episode_id=episode.episode_id,
            timestamp=episode.timestamp,
            observation=observation,
            appraisal=appraisal.to_dict(),
            body_state=self._body_state_for_appraisal(appraisal),
            predicted_outcome=compiled_event.outcome_type,
            value_tags=self._value_tags(compiled_event),
            narrative_tags=sorted(set(episode.tags + [compiled_event.event_type])),
            compiler_confidence=confidence,
            provenance={
                "source_episode_id": episode.episode_id,
                "source_type": episode.source,
                "compiler": "rule_based_m25_v1",
                "compiled_event": compiled_event.to_dict(),
                "appraisal_dimensions": sorted(asdict(appraisal).keys()),
            },
        )

    def compile_event(self, episode: NarrativeEpisode) -> CompiledNarrativeEvent:
        text = episode.raw_text.casefold()
        setting = "general"
        actors = ["agent"]
        subject_role = "actor"
        event_type = "unknown_event"
        outcome_type = "neutral"
        self_involvement = 0.3
        witnessed = False
        direct_harm = False
        controllability_hint = 0.5
        annotations: dict[str, object] = {"matched_rules": []}

        if any(token in text for token in ("找到", "吃的", "食物", "food", "resource", "berries")):
            event_type = "resource_gain"
            outcome_type = "resource_gain"
            self_involvement = 0.9
            controllability_hint = 0.8
            annotations["matched_rules"].append("resource_gain")
        elif any(token in text for token in ("攻击", "鳄鱼", "predator", "near miss", "袭击")):
            event_type = "predator_attack"
            outcome_type = "survival_threat"
            self_involvement = 0.95
            direct_harm = any(token in text for token in ("受伤", "injured", "wounded"))
            controllability_hint = 0.2
            setting = "hazard_zone"
            annotations["matched_rules"].append("predator_attack")
            if "河边" in text or "river" in text:
                annotations["setting_hint"] = "river"
        elif any(token in text for token in ("毒蘑菇", "poison", "contamin", "死去", "death", "fatal")):
            event_type = "witnessed_death"
            outcome_type = "integrity_loss"
            self_involvement = 0.35
            witnessed = True
            subject_role = "witness"
            controllability_hint = 0.25
            annotations["matched_rules"].append("witnessed_fatality")
        elif any(token in text for token in ("排斥", "羞辱", "betray", "exclude", "humiliat")):
            event_type = "social_exclusion"
            outcome_type = "resource_loss"
            self_involvement = 0.85
            witnessed = False
            subject_role = "target"
            controllability_hint = 0.3
            annotations["matched_rules"].append("social_exclusion")
        elif any(token in text for token in ("帮助", "救", "rescue", "protect")):
            event_type = "rescue"
            outcome_type = "resource_gain"
            self_involvement = 0.75
            controllability_hint = 0.7
            annotations["matched_rules"].append("rescue")

        if any(token in text for token in ("看到", "witness", "saw")) and event_type != "resource_gain":
            witnessed = True
            actors.append("other")

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
        if event.event_type == "resource_gain":
            appraisal.novelty = 0.20
            appraisal.controllability = 0.75
            appraisal.self_efficacy_impact = 0.45
        elif event.event_type == "predator_attack":
            appraisal.physical_threat = 0.95
            appraisal.uncertainty = 0.70
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
            appraisal.social_threat = 0.82
            appraisal.loss = 0.45
            appraisal.trust_impact = -0.60
            appraisal.attachment_signal = -0.50
            appraisal.meaning_violation = 0.30
        elif event.event_type == "rescue":
            appraisal.social_threat = -0.15
            appraisal.controllability = 0.45
            appraisal.attachment_signal = 0.75
            appraisal.trust_impact = 0.60
            appraisal.self_efficacy_impact = 0.20

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
            "energy": _clamp(0.62 - appraisal.loss * 0.18 + max(0.0, appraisal.self_efficacy_impact) * 0.10, 0.0, 1.0),
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
            "dopamine": _clamp(0.18 + max(0.0, appraisal.self_efficacy_impact) * 0.20 - appraisal.loss * 0.10, 0.0, 1.0),
        }

    def _value_tags(self, event: CompiledNarrativeEvent) -> list[str]:
        if event.outcome_type == "resource_gain":
            return ["resource_gain"]
        if event.outcome_type == "survival_threat":
            return ["survival_threat", "threat_learning"]
        if event.event_type == "witnessed_death":
            return ["integrity_loss", "contamination_warning"]
        return [event.outcome_type]

    def _confidence_for_event(self, event: CompiledNarrativeEvent) -> float:
        base = 0.55
        matched = len(event.annotations.get("matched_rules", []))
        confidence = base + matched * 0.12
        if event.event_type != "unknown_event":
            confidence += 0.10
        return round(_clamp(confidence, 0.0, 0.99), 4)

    def extract_personality_signal(self, appraisal: AppraisalVector) -> PersonalitySignal:
        """Extract Big Five personality trait deltas from an appraisal vector.

        Maps appraisal dimensions to personality trait signals following
        the M2.6 mapping table. Returns deltas in [-0.5, 0.5] range.
        """
        # Openness: increased by controllability (agency) and novelty
        openness = (
            max(0.0, appraisal.controllability) * 0.25
            + appraisal.novelty * 0.20
            - appraisal.physical_threat * 0.10
        )

        # Conscientiousness: increased by self-efficacy and contamination caution
        conscientiousness = (
            max(0.0, appraisal.self_efficacy_impact) * 0.30
            + appraisal.contamination * 0.15
            - appraisal.uncertainty * 0.10
        )

        # Extraversion: increased by positive social signals
        extraversion = (
            max(0.0, appraisal.attachment_signal) * 0.30
            + max(0.0, appraisal.trust_impact) * 0.20
            - appraisal.social_threat * 0.20
            - max(0.0, -appraisal.attachment_signal) * 0.15
        )

        # Agreeableness: increased by trust, decreased by betrayal
        agreeableness = (
            max(0.0, appraisal.trust_impact) * 0.38
            + max(0.0, appraisal.attachment_signal) * 0.20
            + max(0.0, -appraisal.trust_impact) * -0.35
            - appraisal.social_threat * 0.10
        )

        # Neuroticism: increased by threats, loss, social threat
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
