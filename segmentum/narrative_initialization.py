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
    "探索",
    "好奇",
    "绘制",
    "地图",
    "实验",
    "寻找",
    "搜寻",
)

_SOCIAL_TOKENS = (
    "trust",
    "friend",
    "ally",
    "cooperate",
    "shared",
    "help",
    "rescue",
    "safe contact",
    "信任",
    "朋友",
    "同伴",
    "合作",
    "帮助",
    "救",
    "团体",
)

_THREAT_TOKENS = (
    "predator",
    "attack",
    "betray",
    "poison",
    "fatal",
    "threat",
    "袭击",
    "背叛",
    "毒",
    "死亡",
    "威胁",
    "排斥",
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class NarrativeInitializationResult:
    episode_count: int
    aggregate_appraisal: dict[str, float]
    lexical_bias: dict[str, float]
    policy_distribution: dict[str, float]
    learned_preferences: list[str]
    learned_avoidances: list[str]
    narrative_priors: dict[str, float]
    personality_profile: dict[str, float]
    identity_commitments: list[dict[str, object]]
    social_snapshot: dict[str, object]
    ingest_trace_count: int
    sleep_history_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_count": self.episode_count,
            "aggregate_appraisal": dict(self.aggregate_appraisal),
            "lexical_bias": dict(self.lexical_bias),
            "policy_distribution": dict(self.policy_distribution),
            "learned_preferences": list(self.learned_preferences),
            "learned_avoidances": list(self.learned_avoidances),
            "narrative_priors": dict(self.narrative_priors),
            "personality_profile": dict(self.personality_profile),
            "identity_commitments": [dict(item) for item in self.identity_commitments],
            "social_snapshot": dict(self.social_snapshot),
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
        for embodied in compiled:
            agent.ingest_narrative_episode(embodied)

        aggregate = self._aggregate_appraisal(compiled)
        lexical_bias = self._lexical_bias(episodes)
        self._apply_state_seed(agent, aggregate)
        if compiled:
            narrative_signal = self.compiler.extract_personality_signal(
                AppraisalVector.from_dict(aggregate)
            )
            agent.self_model.personality_profile.absorb_signal(
                narrative_signal,
                tick=max(0, agent.cycle),
            )
        if agent.long_term_memory.episodes:
            agent.sleep()
        if apply_policy_seed:
            self._apply_policy_seed(agent, aggregate, lexical_bias)
            self._apply_identity_seed(agent, aggregate, lexical_bias, episodes)
        return NarrativeInitializationResult(
            episode_count=len(episodes),
            aggregate_appraisal=aggregate,
            lexical_bias=lexical_bias,
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
            ingest_trace_count=len(agent.narrative_trace),
            sleep_history_count=len(agent.sleep_history),
        )

    def _aggregate_appraisal(self, compiled) -> dict[str, float]:
        if not compiled:
            return AppraisalVector().to_dict()

        def avg(name: str) -> float:
            return mean(
                float(episode.appraisal.get(name, 0.0))
                for episode in compiled
            )

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
        exploration_hits = sum(token in text for token in _EXPLORATION_TOKENS)
        social_hits = sum(token in text for token in _SOCIAL_TOKENS)
        threat_hits = sum(token in text for token in _THREAT_TOKENS)
        norm = max(1.0, float(len(episodes) * 2))
        return {
            "exploration": exploration_hits / norm,
            "social": social_hits / norm,
            "threat": threat_hits / norm,
        }

    def _apply_state_seed(self, agent: SegmentAgent, aggregate: dict[str, float]) -> None:
        priors = agent.self_model.narrative_priors
        threat_signal = (
            max(0.0, float(aggregate.get("physical_threat", 0.0)))
            + max(0.0, float(aggregate.get("social_threat", 0.0))) * 0.55
        )
        social_signal = (
            max(0.0, float(aggregate.get("trust_impact", 0.0)))
            + max(0.0, float(aggregate.get("attachment_signal", 0.0))) * 0.8
        )
        exploration_signal = (
            max(0.0, float(aggregate.get("novelty", 0.0)))
            + max(0.0, float(aggregate.get("controllability", 0.0))) * 0.6
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.45
        )
        priors.trust_prior = _clamp(
            priors.trust_prior * 0.6 + float(aggregate.get("trust_impact", 0.0)) * 0.9,
            -1.0,
            1.0,
        )
        priors.controllability_prior = _clamp(
            priors.controllability_prior * 0.55
            + float(aggregate.get("controllability", 0.0)) * 0.75
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.35,
            -1.0,
            1.0,
        )
        beliefs = agent.world_model.beliefs
        if threat_signal >= max(social_signal, exploration_signal):
            beliefs["danger"] = 0.18
            beliefs["shelter"] = 0.70
            beliefs["social"] = 0.20
        elif social_signal >= max(threat_signal, exploration_signal):
            beliefs["social"] = 0.18
            beliefs["danger"] = 0.28
            beliefs["novelty"] = 0.38
        else:
            beliefs["novelty"] = 0.12
            beliefs["food"] = 0.30
            beliefs["danger"] = 0.26
        priors.trauma_bias = _clamp(
            priors.trauma_bias * 0.6
            + (
                max(0.0, float(aggregate.get("physical_threat", 0.0)))
                + max(0.0, float(aggregate.get("social_threat", 0.0))) * 0.55
                + max(0.0, float(aggregate.get("loss", 0.0))) * 0.45
            )
            * 0.55,
            0.0,
            1.0,
        )
        priors.contamination_sensitivity = _clamp(
            priors.contamination_sensitivity * 0.6
            + max(0.0, float(aggregate.get("contamination", 0.0))) * 0.7,
            0.0,
            1.0,
        )
        priors.meaning_stability = _clamp(
            priors.meaning_stability * 0.6
            - max(0.0, float(aggregate.get("meaning_violation", 0.0))) * 0.35
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.15,
            -1.0,
            1.0,
        )

    def _apply_policy_seed(
        self,
        agent: SegmentAgent,
        aggregate: dict[str, float],
        lexical_bias: dict[str, float],
    ) -> None:
        policies = agent.self_model.preferred_policies
        if policies is None:
            return
        threat_bias = (
            max(0.0, float(aggregate.get("physical_threat", 0.0)))
            + max(0.0, float(aggregate.get("social_threat", 0.0))) * 0.55
            + lexical_bias["threat"] * 0.55
        )
        social_bias = (
            max(0.0, float(aggregate.get("trust_impact", 0.0)))
            + max(0.0, float(aggregate.get("attachment_signal", 0.0))) * 0.8
            + lexical_bias["social"] * 0.6
        )
        exploration_bias = (
            max(0.0, float(aggregate.get("novelty", 0.0))) * 0.7
            + max(0.0, float(aggregate.get("controllability", 0.0))) * 0.55
            + max(0.0, float(aggregate.get("self_efficacy_impact", 0.0))) * 0.45
            + lexical_bias["exploration"] * 0.75
            - threat_bias * 0.30
        )
        if social_bias >= max(threat_bias, exploration_bias):
            weights = {
                "hide": 0.06 + threat_bias * 0.12,
                "rest": 0.18,
                "exploit_shelter": 0.10,
                "scan": 0.06 + max(0.0, exploration_bias) * 0.08,
                "forage": 0.06,
                "seek_contact": 0.34 + social_bias * 0.70,
            }
        elif threat_bias >= max(social_bias, exploration_bias):
            weights = {
                "hide": 0.22 + threat_bias * 0.60,
                "rest": 0.22 + threat_bias * 0.10,
                "exploit_shelter": 0.16 + threat_bias * 0.20,
                "scan": 0.06 + max(0.0, exploration_bias) * 0.06,
                "forage": 0.04,
                "seek_contact": 0.04,
            }
        else:
            weights = {
                "hide": 0.08 + threat_bias * 0.18,
                "rest": 0.14,
                "exploit_shelter": 0.10,
                "scan": 0.24 + max(0.0, exploration_bias) * 0.42,
                "forage": 0.14 + max(0.0, exploration_bias) * 0.18,
                "seek_contact": 0.06 + max(0.0, social_bias) * 0.14,
            }
        total = sum(weights.values()) or 1.0
        policies.action_distribution = {
            action: value / total for action, value in weights.items()
        }
        ordered = sorted(
            policies.action_distribution.items(),
            key=lambda item: (-item[1], item[0]),
        )
        policies.learned_preferences = [action for action, _ in ordered[:2]]
        policies.learned_avoidances = [action for action, _ in ordered[-2:]]
        if threat_bias >= max(social_bias, exploration_bias):
            policies.risk_profile = "risk_averse"
        elif exploration_bias >= max(threat_bias, social_bias):
            policies.risk_profile = "risk_seeking"
        else:
            policies.risk_profile = "risk_neutral"
        policies.last_updated_tick = int(agent.cycle)

    def _apply_identity_seed(
        self,
        agent: SegmentAgent,
        aggregate: dict[str, float],
        lexical_bias: dict[str, float],
        episodes: list[NarrativeEpisode],
    ) -> None:
        narrative = agent.self_model.identity_narrative
        policies = agent.self_model.preferred_policies
        if narrative is None or policies is None:
            return
        top_actions = list(policies.learned_preferences)
        dominant = top_actions[0] if top_actions else "rest"
        if dominant == "seek_contact":
            core_identity = "I am a socially oriented agent who builds safety through trusted contact."
            values = "I should preserve trust, stay connected, and repair cooperative bonds when possible."
        elif dominant == "scan":
            core_identity = "I am an exploratory agent who reduces uncertainty through active probing."
            values = "I should seek legible novelty, test the environment, and stay adaptable."
        elif dominant == "hide":
            core_identity = "I am a caution-driven agent who survives by anticipating threat."
            values = "I should protect integrity first, avoid reckless exposure, and recover before advancing."
        else:
            core_identity = "I am an adaptive agent shaped by remembered experience."
            values = "I should remain coherent, preserve resources, and act in line with learned priorities."
        narrative.core_identity = core_identity
        narrative.core_summary = core_identity
        narrative.autobiographical_summary = " ".join(
            episode.raw_text.strip() for episode in episodes[:3]
        )[:480]
        narrative.values_statement = values
        narrative.behavioral_patterns = [
            f"I tend to {action} when experience says it preserves coherence."
            for action in top_actions
        ]
        narrative.significant_events = [episode.raw_text[:120] for episode in episodes[:4]]
        narrative.trait_self_model = {
            "dominant_initialized_action": dominant,
            "threat_bias": round(
                max(
                    0.0,
                    float(aggregate.get("physical_threat", 0.0))
                    + float(aggregate.get("social_threat", 0.0)) * 0.5
                    + lexical_bias["threat"] * 0.5,
                ),
                6,
            ),
            "social_bias": round(
                max(
                    0.0,
                    float(aggregate.get("trust_impact", 0.0))
                    + float(aggregate.get("attachment_signal", 0.0)) * 0.7
                    + lexical_bias["social"] * 0.5,
                ),
                6,
            ),
            "exploration_bias": round(
                max(
                    0.0,
                    float(aggregate.get("novelty", 0.0))
                    + float(aggregate.get("controllability", 0.0)) * 0.6
                    + lexical_bias["exploration"] * 0.6,
                ),
                6,
            ),
        }
        narrative.commitments = [
            IdentityCommitment(
                commitment_id=f"m220-init-{dominant}",
                commitment_type="behavioral_style",
                statement=f"Initialization narrative prioritizes {dominant} as a stable response style.",
                target_actions=list(top_actions),
                discouraged_actions=list(policies.learned_avoidances),
                confidence=0.88,
                priority=0.84,
                source_claim_ids=["m220-initialization"],
                source_chapter_ids=[1],
                evidence_ids=[episode.episode_id for episode in episodes[:4]],
                last_reaffirmed_tick=int(agent.cycle),
            )
        ]
        narrative.last_updated_tick = int(agent.cycle)
        narrative.version += 1
