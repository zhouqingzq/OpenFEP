from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .agent import SegmentAgent
from .narrative_compiler import NarrativeCompiler
from .narrative_types import AppraisalVector, NarrativeEpisode
from .self_model import IdentityCommitment


_EXPLORATION_TOKENS = (
    "explore",
    "curious",
    "map",
    "mapped",
    "experiment",
    "search",
    "question",
    "trail",
    "adapt",
    "explor",
    "探索",
    "地图",
    "适应",
)

_SOCIAL_TOKENS = (
    "trust",
    "friend",
    "ally",
    "cooperate",
    "shared",
    "help",
    "rescue",
    "support",
    "supported",
    "comfort",
    "encouraged",
    "welcome",
    "welcomed",
    "accept",
    "accepted",
    "invite",
    "invited",
    "listen",
    "listened",
    "care",
    "cared",
    "safe contact",
    "repair",
    "friendship",
    "互相",
    "信任",
)

_THREAT_TOKENS = (
    "predator",
    "attack",
    "betray",
    "poison",
    "fatal",
    "threat",
    "unsafe",
    "trap",
    "wound",
    "伤",
    "背叛",
)

_NEGATION_MARKERS = ("not", "never", "pretend", "fictional", "poster", "quoted", "story about")

_BASELINE_POLICY = {
    "hide": 0.16,
    "rest": 0.20,
    "exploit_shelter": 0.14,
    "scan": 0.18,
    "forage": 0.18,
    "seek_contact": 0.14,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_distribution(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values()) or 1.0
    return {
        action: max(0.0, value) / total
        for action, value in sorted(weights.items())
    }


@dataclass(frozen=True)
class NarrativeInitializationResult:
    episode_count: int
    aggregate_appraisal: dict[str, float]
    lexical_bias: dict[str, float]
    semantic_bias: dict[str, float]
    policy_distribution: dict[str, float]
    learned_preferences: list[str]
    learned_avoidances: list[str]
    narrative_priors: dict[str, float]
    personality_profile: dict[str, float]
    identity_commitments: list[dict[str, object]]
    social_snapshot: dict[str, object]
    evidence_trace: dict[str, object]
    conflict_score: float
    uncertainty_score: float
    malformed_text_degradation_ratio: float
    ingest_trace_count: int
    sleep_history_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_count": self.episode_count,
            "aggregate_appraisal": dict(self.aggregate_appraisal),
            "lexical_bias": dict(self.lexical_bias),
            "semantic_bias": dict(self.semantic_bias),
            "policy_distribution": dict(self.policy_distribution),
            "learned_preferences": list(self.learned_preferences),
            "learned_avoidances": list(self.learned_avoidances),
            "narrative_priors": dict(self.narrative_priors),
            "personality_profile": dict(self.personality_profile),
            "identity_commitments": [dict(item) for item in self.identity_commitments],
            "social_snapshot": dict(self.social_snapshot),
            "evidence_trace": dict(self.evidence_trace),
            "conflict_score": self.conflict_score,
            "uncertainty_score": self.uncertainty_score,
            "malformed_text_degradation_ratio": self.malformed_text_degradation_ratio,
            "ingest_trace_count": self.ingest_trace_count,
            "sleep_history_count": self.sleep_history_count,
        }


class NarrativeInitializer:
    def __init__(self, compiler: NarrativeCompiler | None = None) -> None:
        self.compiler = compiler or NarrativeCompiler()

    def initialize_agent(
        self,
        *,
        agent: SegmentAgent,
        episodes: list[NarrativeEpisode],
        apply_policy_seed: bool = True,
    ) -> NarrativeInitializationResult:
        compiled = [self.compiler.compile_episode(episode) for episode in episodes]
        compiled_events = [
            dict(embodied.provenance.get("compiled_event", {}))
            for embodied in compiled
        ]
        for embodied in compiled:
            agent.ingest_narrative_episode(embodied)

        aggregate = self._aggregate_appraisal(compiled)
        lexical_bias = self._lexical_bias(episodes)
        semantic_bias = self._semantic_bias(compiled_events)
        evidence_trace = self._evidence_trace(compiled, compiled_events, lexical_bias, semantic_bias)
        conflict_score = self._conflict_score(compiled_events, semantic_bias)
        uncertainty_score = self._uncertainty_score(compiled, conflict_score)
        degradation_ratio = self._degradation_ratio(compiled, semantic_bias, lexical_bias)
        quality = _clamp(
            mean(float(item.compiler_confidence) for item in compiled) if compiled else 0.0,
            0.0,
            1.0,
        )

        self._apply_state_seed(
            agent,
            aggregate,
            semantic_bias=semantic_bias,
            conflict_score=conflict_score,
            uncertainty_score=uncertainty_score,
            degradation_ratio=degradation_ratio,
        )
        if compiled:
            narrative_signal = self.compiler.extract_personality_signal(
                AppraisalVector.from_dict(aggregate)
            )
            signal_scale = max(0.18, degradation_ratio) * (1.0 - conflict_score * 0.25)
            scaled_signal = type(narrative_signal)(
                openness_delta=narrative_signal.openness_delta * signal_scale,
                conscientiousness_delta=narrative_signal.conscientiousness_delta * signal_scale,
                extraversion_delta=narrative_signal.extraversion_delta * signal_scale,
                agreeableness_delta=narrative_signal.agreeableness_delta * signal_scale,
                neuroticism_delta=narrative_signal.neuroticism_delta * signal_scale,
            )
            agent.self_model.personality_profile.absorb_signal(
                scaled_signal,
                tick=max(0, agent.cycle),
            )
        if agent.long_term_memory.episodes:
            agent.sleep()
        if apply_policy_seed:
            self._apply_policy_seed(
                agent,
                aggregate,
                lexical_bias=lexical_bias,
                semantic_bias=semantic_bias,
                conflict_score=conflict_score,
                uncertainty_score=uncertainty_score,
                degradation_ratio=degradation_ratio,
            )
            self._apply_identity_seed(
                agent,
                aggregate,
                lexical_bias=lexical_bias,
                semantic_bias=semantic_bias,
                episodes=episodes,
                evidence_trace=evidence_trace,
                conflict_score=conflict_score,
                uncertainty_score=uncertainty_score,
                degradation_ratio=degradation_ratio,
            )
        return NarrativeInitializationResult(
            episode_count=len(episodes),
            aggregate_appraisal=aggregate,
            lexical_bias=lexical_bias,
            semantic_bias=semantic_bias,
            policy_distribution=(
                dict(agent.self_model.preferred_policies.action_distribution)
                if agent.self_model.preferred_policies is not None
                else {}
            ),
            learned_preferences=(
                list(agent.self_model.preferred_policies.learned_preferences)
                if agent.self_model.preferred_policies is not None
                else []
            ),
            learned_avoidances=(
                list(agent.self_model.preferred_policies.learned_avoidances)
                if agent.self_model.preferred_policies is not None
                else []
            ),
            narrative_priors=agent.self_model.narrative_priors.to_dict(),
            personality_profile=agent.self_model.personality_profile.to_dict(),
            identity_commitments=(
                [
                    commitment.to_dict()
                    for commitment in agent.self_model.identity_narrative.commitments
                ]
                if agent.self_model.identity_narrative is not None
                else []
            ),
            social_snapshot=agent.social_memory.snapshot(),
            evidence_trace=evidence_trace,
            conflict_score=conflict_score,
            uncertainty_score=uncertainty_score,
            malformed_text_degradation_ratio=degradation_ratio,
            ingest_trace_count=len(agent.narrative_trace),
            sleep_history_count=len(agent.sleep_history),
        )

    def _aggregate_appraisal(self, compiled: list[object]) -> dict[str, float]:
        if not compiled:
            return AppraisalVector().to_dict()

        def avg(name: str) -> float:
            return mean(float(episode.appraisal.get(name, 0.0)) for episode in compiled)

        return AppraisalVector(
            physical_threat=avg("physical_threat"),
            social_threat=avg("social_threat"),
            uncertainty=avg("uncertainty"),
            controllability=avg("controllability"),
            novelty=avg("novelty"),
            loss=avg("loss"),
            moral_salience=avg("moral_salience"),
            contamination=avg("contamination"),
            attachment_signal=avg("attachment_signal"),
            trust_impact=avg("trust_impact"),
            self_efficacy_impact=avg("self_efficacy_impact"),
            meaning_violation=avg("meaning_violation"),
        ).to_dict()

    def _lexical_bias(self, episodes: list[NarrativeEpisode]) -> dict[str, float]:
        if not episodes:
            return {"exploration": 0.0, "social": 0.0, "threat": 0.0}
        text = " ".join(episode.raw_text.casefold() for episode in episodes)
        negation_hits = sum(token in text for token in _NEGATION_MARKERS)
        exploration_hits = sum(token in text for token in _EXPLORATION_TOKENS)
        social_hits = sum(token in text for token in _SOCIAL_TOKENS)
        threat_hits = sum(token in text for token in _THREAT_TOKENS)
        damp = max(0.35, 1.0 - negation_hits * 0.12)
        norm = max(1.0, float(len(episodes) * 3))
        return {
            "exploration": exploration_hits / norm * damp,
            "social": social_hits / norm * damp,
            "threat": threat_hits / norm * damp,
        }

    def _semantic_bias(self, compiled_events: list[dict[str, object]]) -> dict[str, float]:
        totals = {"threat": 0.0, "social": 0.0, "exploration": 0.0}
        if not compiled_events:
            return totals
        for event in compiled_events:
            annotations = event.get("annotations", {})
            if isinstance(annotations, dict):
                scores = annotations.get("semantic_direction_scores", {})
                if isinstance(scores, dict):
                    for key in totals:
                        totals[key] += max(0.0, float(scores.get(key, 0.0)))
        norm = sum(totals.values()) or 1.0
        return {key: round(value / norm, 6) for key, value in totals.items()}

    def _evidence_trace(
        self,
        compiled: list[object],
        compiled_events: list[dict[str, object]],
        lexical_bias: dict[str, float],
        semantic_bias: dict[str, float],
    ) -> dict[str, object]:
        event_signals: list[dict[str, object]] = []
        social_updates: list[dict[str, object]] = []
        policy_sources = {"semantic": 0.0, "lexical": 0.0, "uncertainty_penalty": 0.0}
        for embodied, event in zip(compiled, compiled_events):
            annotations = event.get("annotations", {})
            lexical_hits = {}
            structure_signals = []
            conflict_cues = []
            if isinstance(annotations, dict):
                lexical_hits = dict(annotations.get("lexical_surface_hits", {}))
                structure_signals = list(annotations.get("event_structure_signals", []))
                conflict_cues = list(annotations.get("conflict_cues", []))
            event_signals.append(
                {
                    "episode_id": embodied.episode_id,
                    "event_type": event.get("event_type", "unknown_event"),
                    "appraisal": dict(embodied.appraisal),
                    "semantic_direction_scores": dict(
                        annotations.get("semantic_direction_scores", {})
                    )
                    if isinstance(annotations, dict)
                    else {},
                    "lexical_surface_hits": lexical_hits,
                    "event_structure_signals": structure_signals,
                    "surface_adversarial_risk": float(
                        annotations.get("surface_adversarial_risk", 0.0)
                    )
                    if isinstance(annotations, dict)
                    else 0.0,
                    "conflict_cues": conflict_cues,
                }
            )
            provenance = dict(embodied.provenance.get("episode_metadata", {}))
            counterpart_id = str(provenance.get("counterpart_id", ""))
            if counterpart_id:
                social_updates.append(
                    {
                        "episode_id": embodied.episode_id,
                        "counterpart_id": counterpart_id,
                        "event_type": event.get("event_type", ""),
                        "trust_impact": float(embodied.appraisal.get("trust_impact", 0.0)),
                        "attachment_signal": float(
                            embodied.appraisal.get("attachment_signal", 0.0)
                        ),
                    }
                )
        policy_sources["semantic"] = round(sum(semantic_bias.values()), 6)
        policy_sources["lexical"] = round(sum(lexical_bias.values()), 6)
        policy_sources["uncertainty_penalty"] = round(
            mean(float(item.appraisal.get("uncertainty", 0.0)) for item in compiled)
            if compiled
            else 0.0,
            6,
        )
        identity_commitments = [
            {
                "direction": max(semantic_bias, key=semantic_bias.get) if semantic_bias else "neutral",
                "semantic_support": round(max(semantic_bias.values()) if semantic_bias else 0.0, 6),
                "lexical_support": round(max(lexical_bias.values()) if lexical_bias else 0.0, 6),
            }
        ]
        return {
            "appraisal_signals": event_signals,
            "social_updates": social_updates,
            "identity_commitments": identity_commitments,
            "policy_seed_sources": policy_sources,
        }

    def _conflict_score(
        self,
        compiled_events: list[dict[str, object]],
        semantic_bias: dict[str, float],
    ) -> float:
        if not compiled_events:
            return 0.0
        sorted_biases = sorted(semantic_bias.values(), reverse=True)
        closeness = 0.0
        if len(sorted_biases) >= 2:
            closeness = 1.0 - abs(sorted_biases[0] - sorted_biases[1])
        contradiction_hits = 0.0
        for event in compiled_events:
            annotations = event.get("annotations", {})
            if isinstance(annotations, dict):
                contradiction_hits += len(annotations.get("conflict_cues", [])) * 0.12
        return round(_clamp(closeness * 0.35 + contradiction_hits, 0.0, 1.0), 6)

    def _uncertainty_score(self, compiled: list[object], conflict_score: float) -> float:
        if not compiled:
            return 1.0
        mean_uncertainty = mean(float(item.appraisal.get("uncertainty", 0.0)) for item in compiled)
        low_signal_ratio = mean(
            float(bool(item.provenance.get("low_signal", False))) for item in compiled
        )
        return round(
            _clamp(mean_uncertainty * 0.55 + conflict_score * 0.35 + low_signal_ratio * 0.30, 0.0, 1.0),
            6,
        )

    def _degradation_ratio(
        self,
        compiled: list[object],
        semantic_bias: dict[str, float],
        lexical_bias: dict[str, float],
    ) -> float:
        if not compiled:
            return 0.4
        mean_confidence = mean(float(item.compiler_confidence) for item in compiled)
        semantic_strength = max(semantic_bias.values()) if semantic_bias else 0.0
        lexical_strength = max(lexical_bias.values()) if lexical_bias else 0.0
        low_signal_ratio = mean(float(bool(item.provenance.get("low_signal", False))) for item in compiled)
        ratio = (
            mean_confidence * 0.55
            + semantic_strength * 0.30
            + min(lexical_strength, semantic_strength + 0.15) * 0.10
            - low_signal_ratio * 0.18
        )
        return round(_clamp(ratio, 0.40, 0.85), 6)

    def _apply_state_seed(
        self,
        agent: SegmentAgent,
        aggregate: dict[str, float],
        *,
        semantic_bias: dict[str, float],
        conflict_score: float,
        uncertainty_score: float,
        degradation_ratio: float,
    ) -> None:
        priors = agent.self_model.narrative_priors
        strength = degradation_ratio * (1.0 - conflict_score * 0.35)
        trust_signal = float(aggregate.get("trust_impact", 0.0)) * 0.9 + semantic_bias["social"] * 0.35
        controllability_signal = (
            float(aggregate.get("controllability", 0.0)) * 0.75
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.35
            + semantic_bias["exploration"] * 0.20
        )
        threat_signal = (
            max(0.0, float(aggregate.get("physical_threat", 0.0)))
            + max(0.0, float(aggregate.get("social_threat", 0.0))) * 0.55
            + semantic_bias["threat"] * 0.25
        )
        priors.trust_prior = _clamp(
            priors.trust_prior * (1.0 - 0.45 * strength) + trust_signal * 0.60 * strength,
            -1.0,
            1.0,
        )
        priors.controllability_prior = _clamp(
            priors.controllability_prior * (1.0 - 0.40 * strength)
            + controllability_signal * 0.60 * strength,
            -1.0,
            1.0,
        )
        priors.trauma_bias = _clamp(
            priors.trauma_bias * (1.0 - 0.35 * strength) + threat_signal * 0.40 * strength,
            0.0,
            1.0,
        )
        priors.contamination_sensitivity = _clamp(
            priors.contamination_sensitivity * (1.0 - 0.35 * strength)
            + max(0.0, float(aggregate.get("contamination", 0.0))) * 0.50 * strength,
            0.0,
            1.0,
        )
        priors.meaning_stability = _clamp(
            priors.meaning_stability * (1.0 - 0.40 * strength)
            - max(0.0, float(aggregate.get("meaning_violation", 0.0))) * 0.25 * strength
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.10 * strength
            - uncertainty_score * 0.08,
            -1.0,
            1.0,
        )
        beliefs = agent.world_model.beliefs
        if semantic_bias["threat"] >= max(semantic_bias["social"], semantic_bias["exploration"]):
            priors.trauma_bias = _clamp(
                max(priors.trauma_bias, 0.42 + semantic_bias["threat"] * 0.18 - uncertainty_score * 0.08),
                0.0,
                1.0,
            )
            priors.trust_prior = min(priors.trust_prior, 0.05)
            beliefs["danger"] = _clamp(
                0.54 + semantic_bias["threat"] * 0.12 - uncertainty_score * 0.04,
                0.46,
                0.72,
            )
            beliefs["shelter"] = _clamp(0.62 + degradation_ratio * 0.10, 0.58, 0.78)
            beliefs["social"] = 0.16
            beliefs["novelty"] = 0.18
        elif semantic_bias["social"] >= max(semantic_bias["threat"], semantic_bias["exploration"]):
            priors.trust_prior = _clamp(
                max(
                    priors.trust_prior,
                    0.62 + semantic_bias["social"] * 0.18 + max(0.0, trust_signal) * 0.10,
                )
                - uncertainty_score * 0.04,
                -1.0,
                1.0,
            )
            priors.trauma_bias = min(priors.trauma_bias, 0.08)
            beliefs["social"] = _clamp(
                0.50 + semantic_bias["social"] * 0.10 + max(0.0, trust_signal) * 0.06,
                0.44,
                0.68,
            )
            beliefs["danger"] = 0.18
            beliefs["novelty"] = 0.24
            beliefs["shelter"] = 0.46
        else:
            priors.controllability_prior = _clamp(
                max(priors.controllability_prior, 0.34 + semantic_bias["exploration"] * 0.12),
                -1.0,
                1.0,
            )
            beliefs["novelty"] = _clamp(
                0.48 + semantic_bias["exploration"] * 0.16 + degradation_ratio * 0.04,
                0.42,
                0.68,
            )
            beliefs["food"] = 0.38
            beliefs["danger"] = 0.18
            beliefs["social"] = 0.22

    def _apply_policy_seed(
        self,
        agent: SegmentAgent,
        aggregate: dict[str, float],
        *,
        lexical_bias: dict[str, float],
        semantic_bias: dict[str, float],
        conflict_score: float,
        uncertainty_score: float,
        degradation_ratio: float,
    ) -> None:
        policies = agent.self_model.preferred_policies
        if policies is None:
            return
        lexical_support = {
            key: min(lexical_bias[key], semantic_bias[key] + 0.12)
            for key in semantic_bias
        }
        threat_bias = (
            max(0.0, float(aggregate.get("physical_threat", 0.0)))
            + max(0.0, float(aggregate.get("social_threat", 0.0))) * 0.55
            + semantic_bias["threat"] * 0.85
            + lexical_support["threat"] * 0.20
        )
        social_bias = (
            max(0.0, float(aggregate.get("trust_impact", 0.0)))
            + max(0.0, float(aggregate.get("attachment_signal", 0.0))) * 0.8
            + semantic_bias["social"] * 0.90
            + lexical_support["social"] * 0.22
        )
        exploration_bias = (
            max(0.0, float(aggregate.get("novelty", 0.0))) * 0.7
            + max(0.0, float(aggregate.get("controllability", 0.0))) * 0.55
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.45
            + semantic_bias["exploration"] * 0.95
            + lexical_support["exploration"] * 0.24
            - threat_bias * 0.18
        )
        raw_weights = {
            "hide": 0.10 + threat_bias * 0.42 - social_bias * 0.05,
            "rest": 0.16 + threat_bias * 0.12 + uncertainty_score * 0.06,
            "exploit_shelter": 0.12 + threat_bias * 0.16,
            "scan": 0.14 + exploration_bias * 0.38 + uncertainty_score * 0.05,
            "forage": 0.12 + exploration_bias * 0.12,
            "seek_contact": 0.14 + social_bias * 0.62 - threat_bias * 0.04,
        }
        blended = {
            action: _BASELINE_POLICY[action] * (1.0 - degradation_ratio) + value * degradation_ratio
            for action, value in raw_weights.items()
        }
        if conflict_score >= 0.20:
            dominant_action = max(blended, key=blended.get)
            cap = sum(blended.values()) * 0.70
            if blended[dominant_action] > cap:
                excess = blended[dominant_action] - cap
                blended[dominant_action] = cap
                share = excess / 5.0
                for action in blended:
                    if action != dominant_action:
                        blended[action] += share
        policies.action_distribution = _normalize_distribution(blended)
        ordered = sorted(
            policies.action_distribution.items(),
            key=lambda item: (-item[1], item[0]),
        )
        policies.learned_preferences = [action for action, _ in ordered[:2]]
        policies.learned_avoidances = [action for action, _ in ordered[-2:]]
        dominant = ordered[0][0]
        if dominant == "hide":
            policies.risk_profile = "risk_averse"
        elif dominant in {"scan", "forage"}:
            policies.risk_profile = "risk_seeking"
        else:
            policies.risk_profile = "risk_neutral"
        policies.last_updated_tick = int(agent.cycle)
        if semantic_bias["social"] >= max(semantic_bias["threat"], semantic_bias["exploration"]):
            carryover = degradation_ratio * (1.0 - conflict_score * 0.35)
            social_affordance = max(
                0.0,
                social_bias - threat_bias * 0.20 - uncertainty_score * 0.10,
            )
            agent.world_model.counterfactual_biases = {
                "seek_contact": round(
                    _clamp(0.04 + social_affordance * 0.08 * carryover, 0.0, 0.12),
                    6,
                ),
                "hide": round(
                    -_clamp(0.02 + social_affordance * 0.05 * carryover, 0.0, 0.08),
                    6,
                ),
                "exploit_shelter": round(
                    _clamp(0.01 + social_affordance * 0.03 * carryover, 0.0, 0.05),
                    6,
                ),
            }
        elif semantic_bias["threat"] >= max(semantic_bias["social"], semantic_bias["exploration"]):
            carryover = degradation_ratio * (1.0 - conflict_score * 0.30)
            vigilance = max(0.0, threat_bias - social_bias * 0.15)
            agent.world_model.counterfactual_biases = {
                "hide": round(_clamp(0.03 + vigilance * 0.08 * carryover, 0.0, 0.12), 6),
                "exploit_shelter": round(
                    _clamp(0.02 + vigilance * 0.04 * carryover, 0.0, 0.08),
                    6,
                ),
                "forage": round(
                    -_clamp(0.01 + vigilance * 0.04 * carryover, 0.0, 0.08),
                    6,
                ),
            }
        else:
            carryover = degradation_ratio * (1.0 - conflict_score * 0.30)
            curiosity = max(0.0, exploration_bias - threat_bias * 0.12)
            agent.world_model.counterfactual_biases = {
                "scan": round(_clamp(0.03 + curiosity * 0.08 * carryover, 0.0, 0.12), 6),
                "forage": round(
                    _clamp(0.02 + curiosity * 0.04 * carryover, 0.0, 0.08),
                    6,
                ),
                "hide": round(
                    -_clamp(0.01 + curiosity * 0.03 * carryover, 0.0, 0.06),
                    6,
                ),
            }

    def _apply_identity_seed(
        self,
        agent: SegmentAgent,
        aggregate: dict[str, float],
        *,
        lexical_bias: dict[str, float],
        semantic_bias: dict[str, float],
        episodes: list[NarrativeEpisode],
        evidence_trace: dict[str, object],
        conflict_score: float,
        uncertainty_score: float,
        degradation_ratio: float,
    ) -> None:
        narrative = agent.self_model.identity_narrative
        policies = agent.self_model.preferred_policies
        if narrative is None or policies is None:
            return
        top_actions = list(policies.learned_preferences)
        dominant = top_actions[0] if top_actions else "rest"
        if dominant == "seek_contact":
            core_identity = "I am a socially oriented agent who builds safety through trusted contact."
            values = "I preserve trust, stay connected, and prefer reciprocal repair when evidence supports it."
            direction = "social_trusting"
        elif dominant == "scan":
            core_identity = "I am an exploratory agent who reduces uncertainty through active probing."
            values = "I seek legible novelty, test the environment, and stay adaptable without abandoning caution."
            direction = "exploratory_adaptive"
        elif dominant == "hide":
            core_identity = "I am a caution-driven agent who survives by anticipating threat."
            values = "I protect integrity first, avoid reckless exposure, and recover before advancing."
            direction = "threat_hardened"
        else:
            core_identity = "I am an adaptive agent shaped by remembered experience."
            values = "I remain coherent, preserve resources, and act in line with bounded prior evidence."
            direction = "adaptive_balanced"
        if conflict_score >= 0.20:
            core_identity += " I am holding that identity provisionally because the narrative evidence is mixed."
        narrative.core_identity = core_identity
        narrative.core_summary = core_identity
        narrative.autobiographical_summary = " ".join(
            episode.raw_text.strip() for episode in episodes[:3]
        )[:480]
        narrative.values_statement = values
        narrative.behavioral_patterns = [
            f"I tend to {action} when remembered evidence says it preserves coherence."
            for action in top_actions
        ]
        narrative.significant_events = [episode.raw_text[:120] for episode in episodes[:4]]
        narrative.trait_self_model = {
            "dominant_initialized_action": dominant,
            "dominant_identity_direction": direction,
            "threat_bias": round(max(0.0, semantic_bias["threat"] + lexical_bias["threat"] * 0.2), 6),
            "social_bias": round(max(0.0, semantic_bias["social"] + lexical_bias["social"] * 0.2), 6),
            "exploration_bias": round(
                max(0.0, semantic_bias["exploration"] + lexical_bias["exploration"] * 0.2),
                6,
            ),
            "conflict_score": round(conflict_score, 6),
            "uncertainty_score": round(uncertainty_score, 6),
            "degradation_ratio": round(degradation_ratio, 6),
            "aggregate_appraisal": dict(aggregate),
        }
        commitment_confidence = _clamp(
            0.92 * degradation_ratio - conflict_score * 0.20 - uncertainty_score * 0.10,
            0.45,
            0.92,
        )
        narrative.commitments = [
            IdentityCommitment(
                commitment_id=f"m221-init-{dominant}",
                commitment_type="behavioral_style",
                statement=(
                    f"Initialization narrative provisionally prioritizes {dominant} "
                    "as a bounded response style."
                ),
                target_actions=list(top_actions),
                discouraged_actions=list(policies.learned_avoidances),
                confidence=commitment_confidence,
                priority=_clamp(0.58 + degradation_ratio * 0.22 - conflict_score * 0.10, 0.45, 0.85),
                source_claim_ids=["m221-initialization"],
                source_chapter_ids=[1],
                evidence_ids=[episode.episode_id for episode in episodes[:4]],
                last_reaffirmed_tick=int(agent.cycle),
            )
        ]
        narrative.contradiction_summary = {
            "conflict_score": round(conflict_score, 6),
            "uncertainty_score": round(uncertainty_score, 6),
            "bounded_resolution_applied": bool(conflict_score >= 0.20 or uncertainty_score >= 0.20),
        }
        narrative.evidence_provenance = dict(evidence_trace)
        narrative.last_updated_tick = int(agent.cycle)
        narrative.version += 1
