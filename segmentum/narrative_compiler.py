from __future__ import annotations

from dataclasses import asdict

from .narrative_types import (
    AppraisalVector,
    CompiledNarrativeEvent,
    EmbodiedNarrativeEpisode,
    NarrativeEpisode,
)
from .self_model import PersonalitySignal


_NEGATION_MARKERS = (
    "not",
    "never",
    "fictional",
    "poster",
    "quote",
    "quoted",
    "pretend",
    "pretended",
    "imagined",
    "just a word",
    "training slogan",
    "wasn't real",
    "didn't happen",
    "no one actually",
    "bu shi",
    "mei you",
    "不是",
)

_SEMANTIC_CUES: dict[str, tuple[str, ...]] = {
    "resource_gain": (
        "food",
        "resource",
        "berries",
        "supplies",
        "found provisions",
        "shared meal",
        "safe resources",
        "找到",
        "吃的",
        "食物",
        "鎵惧埌",
        "鍚冪殑",
        "椋熺墿",
    ),
    "predator_attack": (
        "predator",
        "attack",
        "ambush",
        "pounced",
        "near miss",
        "unsafe crossing",
        "wounded",
        "injured",
        "trap snapped",
        "攻击",
        "鳄鱼",
        "河边",
        "受伤",
        "鏀诲嚮",
        "槌勯奔",
        "琚嚮",
    ),
    "witnessed_death": (
        "poison",
        "toxic",
        "contamin",
        "fatal",
        "death",
        "corpse",
        "body collapsed",
        "died",
        "毒蘑菇",
        "死去",
        "看到一个人",
        "姣掕槕",
        "姝诲幓",
        "death",
    ),
    "social_exclusion": (
        "betray",
        "excluded",
        "rejected",
        "abandoned",
        "humiliat",
        "deceived",
        "lied to",
        "trust broken",
        "left me outside",
        "鎺掓枼",
        "缇炶颈",
    ),
    "rescue": (
        "rescue",
        "protect",
        "helped",
        "help",
        "shared",
        "cooperate",
        "friend",
        "ally",
        "stayed nearby",
        "safe contact",
        "repair",
        "帮助",
        "保护",
        "甯姪",
        "鏁?",
    ),
    "exploration": (
        "explore",
        "explored",
        "mapped",
        "map",
        "search",
        "searched",
        "question",
        "experiment",
        "adapted",
        "adapt",
        "curious",
        "trail",
        "unfamiliar",
        "new signals",
        "probe",
        "learned the route",
        "探索",
        "地图",
        "适应",
    ),
    "uncertainty": (
        "unclear",
        "uncertain",
        "didn't know",
        "unpredictable",
        "ambiguous",
        "mixed signals",
        "contradictory",
        "uncertain",
    ),
}

_EVENT_TO_DIRECTION = {
    "predator_attack": "threat",
    "witnessed_death": "threat",
    "social_exclusion": "threat",
    "rescue": "social",
    "resource_gain": "social",
    "exploration": "exploration",
}

_SEMANTIC_CUES["rescue"] = _SEMANTIC_CUES["rescue"] + (
    "rescued",
    "saved",
    "救",
    "救了",
    "救助",
    "好人",
)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _match_count(text: str, cues: tuple[str, ...]) -> int:
    return sum(1 for cue in cues if cue and cue in text)


class NarrativeCompiler:
    """Deterministic rule-based compiler for seeded and open-text narratives."""

    def compile_episode(
        self,
        episode: NarrativeEpisode,
    ) -> EmbodiedNarrativeEpisode:
        compiled_event = self.compile_event(episode)
        appraisal = self.appraise_event(compiled_event)
        observation = self.compatibility_observation(appraisal)
        confidence = self._confidence_for_event(compiled_event)
        annotations = dict(compiled_event.annotations)
        annotations["compiler_confidence"] = confidence
        provenance = {
            "source_episode_id": episode.episode_id,
            "source_type": episode.source,
            "episode_metadata": dict(episode.metadata),
            "compiler": "rule_based_m221_v1",
            "compiled_event": compiled_event.to_dict(),
            "appraisal_dimensions": sorted(asdict(appraisal).keys()),
            "semantic_direction_scores": dict(
                annotations.get("semantic_direction_scores", {})
            ),
            "lexical_surface_hits": dict(annotations.get("lexical_surface_hits", {})),
            "event_structure_signals": list(
                annotations.get("event_structure_signals", [])
            ),
            "surface_adversarial_risk": float(
                annotations.get("surface_adversarial_risk", 0.0)
            ),
            "low_signal": bool(annotations.get("low_signal", False)),
            "conflict_cues": list(annotations.get("conflict_cues", [])),
        }
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
            provenance=provenance,
        )

    def compile_event(self, episode: NarrativeEpisode) -> CompiledNarrativeEvent:
        text = episode.raw_text.casefold()
        lexical_hits = {
            name: _match_count(text, cues)
            for name, cues in _SEMANTIC_CUES.items()
        }
        negation_hits = _match_count(text, _NEGATION_MARKERS)
        threat_score = (
            lexical_hits["predator_attack"]
            + lexical_hits["witnessed_death"] * 0.9
            + lexical_hits["social_exclusion"] * 0.9
        )
        social_score = lexical_hits["rescue"] + lexical_hits["resource_gain"] * 0.35
        exploration_score = lexical_hits["exploration"] + lexical_hits["resource_gain"] * 0.15
        uncertainty_score = lexical_hits["uncertainty"] + float(
            "but" in text or "however" in text or "但是" in text
        )
        contradiction_cues: list[str] = []
        if any(token in text for token in ("but", "however", "yet", "although", "但是", "但", "却")):
            contradiction_cues.append("contrastive_connector")
        if any(token in text for token in ("at first", "later", "once", "now", "起初", "后来")):
            contradiction_cues.append("temporal_shift")
        surface_adversarial_risk = min(
            1.0,
            (negation_hits * 0.22)
            + max(0.0, (threat_score + social_score + exploration_score) - 5.0) * 0.05,
        )

        semantic_threat = max(0.0, threat_score - negation_hits * 0.7)
        semantic_social = max(0.0, social_score - negation_hits * 0.45)
        semantic_exploration = max(0.0, exploration_score - negation_hits * 0.40)
        semantic_direction_scores = {
            "threat": round(semantic_threat, 6),
            "social": round(semantic_social, 6),
            "exploration": round(semantic_exploration, 6),
        }
        semantic_total = sum(semantic_direction_scores.values())
        low_signal = semantic_total <= 1.0 and uncertainty_score >= 0.0

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

        if semantic_threat >= max(semantic_social, semantic_exploration) and semantic_threat >= 1.0:
            if lexical_hits["predator_attack"] >= max(
                lexical_hits["witnessed_death"],
                lexical_hits["social_exclusion"],
            ):
                event_type = "predator_attack"
                outcome_type = "survival_threat"
                self_involvement = 0.95
                direct_harm = any(
                    token in text for token in ("injured", "wounded", "hurt", "scar", "鍙椾激")
                )
                controllability_hint = 0.2
                setting = "hazard_zone"
                matched_rules.append("predator_attack")
                event_structure_signals.extend(["direct_threat", "physical_exposure"])
            elif lexical_hits["social_exclusion"] >= lexical_hits["witnessed_death"]:
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
        elif semantic_social >= max(semantic_threat, semantic_exploration) and semantic_social >= 1.0:
            event_type = "rescue"
            outcome_type = "resource_gain"
            self_involvement = 0.75
            controllability_hint = 0.7
            matched_rules.append("rescue")
            event_structure_signals.extend(["cooperation", "support_received"])
            if any(token in text for token in ("friend", "ally", "group", "shared")):
                actors.append("other")
        elif semantic_exploration >= 1.0:
            event_type = "exploration"
            outcome_type = "resource_gain" if lexical_hits["resource_gain"] else "neutral"
            self_involvement = 0.82
            controllability_hint = 0.72
            matched_rules.append("exploration")
            event_structure_signals.extend(["novelty_engagement", "active_probing"])
            if any(token in text for token in ("map", "trail", "route", "valley")):
                setting = "frontier"
        elif lexical_hits["resource_gain"] >= 1:
            event_type = "resource_gain"
            outcome_type = "resource_gain"
            self_involvement = 0.9
            controllability_hint = 0.8
            matched_rules.append("resource_gain")
            event_structure_signals.append("resource_acquisition")

        if any(token in text for token in ("witness", "saw", "鐪嬪埌")) and event_type != "resource_gain":
            witnessed = True
            if "other" not in actors:
                actors.append("other")
        if any(token in text for token in ("river", "娌宠竟")):
            setting = "river"
        if low_signal:
            event_structure_signals.append("weak_semantic_support")

        annotations: dict[str, object] = {
            "matched_rules": matched_rules,
            "semantic_direction_scores": semantic_direction_scores,
            "lexical_surface_hits": lexical_hits,
            "event_structure_signals": sorted(set(event_structure_signals)),
            "surface_adversarial_risk": round(surface_adversarial_risk, 6),
            "low_signal": low_signal,
            "conflict_cues": contradiction_cues,
            "uncertainty_cues": round(float(uncertainty_score), 6),
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
            appraisal.controllability = 0.45
            appraisal.attachment_signal = 0.70 + min(0.05, social_strength * 0.03)
            appraisal.trust_impact = 0.60
            appraisal.self_efficacy_impact = 0.20
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
        low_signal_penalty = 0.18 if event.annotations.get("low_signal", False) else 0.0
        adversarial_penalty = float(event.annotations.get("surface_adversarial_risk", 0.0)) * 0.25
        confidence = base + matched * 0.12 + structure * 0.04 - low_signal_penalty - adversarial_penalty
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
            max(0.0, appraisal.attachment_signal) * 0.30
            + max(0.0, appraisal.trust_impact) * 0.20
            - appraisal.social_threat * 0.20
            - max(0.0, -appraisal.attachment_signal) * 0.15
        )
        agreeableness = (
            max(0.0, appraisal.trust_impact) * 0.38
            + max(0.0, appraisal.attachment_signal) * 0.20
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
