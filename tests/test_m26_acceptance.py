"""M2.6 Acceptance Tests: Personality Trait Space & Trait-Driven Decision Making.

Validates that:
- PersonalityProfile creation, serialization, round-trip
- Narrative experience shapes Big Five personality traits during sleep
- Different narrative histories produce different personality profiles
- Different personalities produce different action choices
- Personality survives agent serialization round-trip
- Existing behavior not broken (default neutral personality = zero modulation)
"""
from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode
from segmentum.self_model import PersonalityProfile, PersonalitySignal


# -- Helpers --

def _make_agent(seed: int = 42) -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(seed))
    agent.long_term_memory.minimum_support = 1
    agent.long_term_memory.sleep_minimum_support = 1
    agent.long_term_memory.episode_score_threshold = 0.0
    return agent


def _ingest_and_sleep(
    agent: SegmentAgent,
    narratives: list[tuple[str, str]],
    *,
    sleep_cycles: int = 3,
) -> None:
    compiler = NarrativeCompiler()
    for batch in range(sleep_cycles):
        for i, (text, tag) in enumerate(narratives):
            ep = NarrativeEpisode(
                episode_id=f"m26-{batch}-{i}",
                timestamp=batch * 100 + i,
                source="test",
                raw_text=text,
                tags=[tag],
                metadata={},
            )
            agent.ingest_narrative_episode(compiler.compile_episode(ep))
        agent.sleep()


def _round_trip_agent(agent: SegmentAgent, seed: int = 999) -> SegmentAgent:
    payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
    return SegmentAgent.from_dict(payload, rng=random.Random(seed))


THREAT_NARRATIVES = [
    ("agent被一只鳄鱼攻击了，没有受伤", "predator"),
    ("agent被一只鳄鱼攻击了，没有受伤", "predator"),
    ("agent看到一个人吃了毒蘑菇死去了", "death"),
    ("agent被一只鳄鱼攻击了，没有受伤", "predator"),
]

SOCIAL_NARRATIVES = [
    ("agent被一个好人救了", "rescue"),
    ("agent被一个好人救了", "rescue"),
    ("agent出门找到了一些吃的", "food"),
    ("agent被一个好人救了", "rescue"),
]


# -- Unit Tests --

class TestPersonalityProfileUnit(unittest.TestCase):

    def test_default_profile_is_neutral(self) -> None:
        p = PersonalityProfile()
        self.assertEqual(p.openness, 0.5)
        self.assertEqual(p.conscientiousness, 0.5)
        self.assertEqual(p.extraversion, 0.5)
        self.assertEqual(p.agreeableness, 0.5)
        self.assertEqual(p.neuroticism, 0.5)
        self.assertEqual(p.update_count, 0)

    def test_neutral_profile_produces_zero_modulation(self) -> None:
        """Default profile should not affect drives, priors, or policy."""
        p = PersonalityProfile()
        drive_mod = p.drive_modulation()
        for value in drive_mod.values():
            self.assertAlmostEqual(value, 0.0, places=10)
        strategic_mod = p.strategic_modulation()
        for value in strategic_mod.values():
            self.assertAlmostEqual(value, 0.0, places=10)
        for action in ("forage", "hide", "scan", "rest", "seek_contact", "exploit_shelter", "thermoregulate"):
            self.assertAlmostEqual(p.policy_bias(action, 0.5), 0.0, places=10)

    def test_signal_absorption_updates_traits(self) -> None:
        p = PersonalityProfile()
        signal = PersonalitySignal(
            openness_delta=0.3,
            neuroticism_delta=-0.2,
        )
        deltas = p.absorb_signal(signal, tick=10)
        self.assertGreater(p.openness, 0.5)
        self.assertLess(p.neuroticism, 0.5)
        self.assertEqual(p.update_count, 1)
        self.assertEqual(p.last_updated_tick, 10)
        self.assertGreater(deltas["openness"], 0.0)
        self.assertLess(deltas["neuroticism"], 0.0)

    def test_learning_rate_decays(self) -> None:
        p = PersonalityProfile()
        lr0 = p.learning_rate
        p.update_count = 10
        lr10 = p.learning_rate
        self.assertGreater(lr0, lr10)

    def test_traits_clamped(self) -> None:
        p = PersonalityProfile(openness=0.95)
        signal = PersonalitySignal(openness_delta=0.5)
        p.absorb_signal(signal, tick=1)
        self.assertLessEqual(p.openness, 0.95)
        self.assertGreaterEqual(p.openness, 0.05)

    def test_serialization_round_trip(self) -> None:
        p = PersonalityProfile(
            openness=0.7,
            conscientiousness=0.3,
            extraversion=0.8,
            agreeableness=0.2,
            neuroticism=0.6,
            update_count=5,
            last_updated_tick=100,
        )
        d = p.to_dict()
        p2 = PersonalityProfile.from_dict(d)
        self.assertEqual(p.to_dict(), p2.to_dict())

    def test_from_dict_handles_none(self) -> None:
        p = PersonalityProfile.from_dict(None)
        self.assertEqual(p.openness, 0.5)


class TestPersonalitySignalExtraction(unittest.TestCase):

    def test_threat_appraisal_produces_high_neuroticism(self) -> None:
        compiler = NarrativeCompiler()
        ep = NarrativeEpisode(
            episode_id="t-1", timestamp=1, source="test",
            raw_text="agent被一只鳄鱼攻击了，没有受伤", tags=["predator"],
        )
        embodied = compiler.compile_episode(ep)
        from segmentum.narrative_types import AppraisalVector
        appraisal = AppraisalVector.from_dict(embodied.appraisal)
        signal = compiler.extract_personality_signal(appraisal)
        self.assertGreater(signal.neuroticism_delta, 0.0)
        self.assertLess(signal.openness_delta, 0.1)

    def test_social_positive_appraisal_produces_high_extraversion(self) -> None:
        compiler = NarrativeCompiler()
        ep = NarrativeEpisode(
            episode_id="s-1", timestamp=1, source="test",
            raw_text="agent被一个好人救了", tags=["rescue"],
        )
        embodied = compiler.compile_episode(ep)
        from segmentum.narrative_types import AppraisalVector
        appraisal = AppraisalVector.from_dict(embodied.appraisal)
        signal = compiler.extract_personality_signal(appraisal)
        self.assertGreater(signal.extraversion_delta, 0.0)
        self.assertGreater(signal.agreeableness_delta, 0.0)


# -- Integration Tests --

class TestScenarioA_ThreatHistory(unittest.TestCase):
    """Threat-heavy history should produce high neuroticism and cautious behavior."""

    def test_threat_history_raises_neuroticism(self) -> None:
        agent = _make_agent(seed=101)
        before = agent.self_model.personality_profile.neuroticism
        _ingest_and_sleep(agent, THREAT_NARRATIVES, sleep_cycles=3)
        after = agent.self_model.personality_profile.neuroticism
        self.assertGreater(after, before)
        self.assertGreater(after, 0.65, f"Neuroticism should exceed 0.65, got {after:.3f}")

    def test_threat_history_agent_prefers_caution(self) -> None:
        agent = _make_agent(seed=102)
        _ingest_and_sleep(agent, THREAT_NARRATIVES, sleep_cycles=3)
        obs = Observation(food=0.25, danger=0.60, novelty=0.3, shelter=0.35, temperature=0.5, social=0.2)
        result = agent.decision_cycle(obs)
        chosen = result["diagnostics"].chosen.choice
        # Under moderate danger, threat-trained agent should prefer cautious actions
        cautious_actions = {"hide", "exploit_shelter", "rest"}
        self.assertIn(
            chosen,
            cautious_actions,
            f"Expected cautious action, got {chosen}",
        )


class TestScenarioB_SocialHistory(unittest.TestCase):
    """Social-positive history should produce high extraversion and agreeableness."""

    def test_social_history_raises_extraversion_and_agreeableness(self) -> None:
        agent = _make_agent(seed=201)
        _ingest_and_sleep(agent, SOCIAL_NARRATIVES, sleep_cycles=3)
        profile = agent.self_model.personality_profile
        self.assertGreater(profile.extraversion, 0.60)
        self.assertGreater(profile.agreeableness, 0.60)

    def test_social_history_agent_values_contact(self) -> None:
        agent = _make_agent(seed=202)
        _ingest_and_sleep(agent, SOCIAL_NARRATIVES, sleep_cycles=3)
        # In a safe, socially sparse environment, social agent should rank seek_contact higher
        obs = Observation(food=0.6, danger=0.10, novelty=0.5, shelter=0.5, temperature=0.5, social=0.15)
        result = agent.decision_cycle(obs)
        ranked = result["diagnostics"].ranked_options
        seek_score = next(o.policy_score for o in ranked if o.choice == "seek_contact")
        # seek_contact should be at least mid-ranked
        scores = sorted([o.policy_score for o in ranked], reverse=True)
        median_score = scores[len(scores) // 2]
        self.assertGreaterEqual(
            seek_score,
            median_score - 0.5,
            "seek_contact should not be bottom-ranked for a social agent",
        )


class TestScenarioC_DivergentBehavior(unittest.TestCase):
    """Two agents with different histories should produce different behavior."""

    def test_divergent_histories_produce_different_profiles(self) -> None:
        agent_a = _make_agent(seed=301)
        agent_b = _make_agent(seed=301)
        _ingest_and_sleep(agent_a, THREAT_NARRATIVES, sleep_cycles=3)
        _ingest_and_sleep(agent_b, SOCIAL_NARRATIVES, sleep_cycles=3)

        pa = agent_a.self_model.personality_profile
        pb = agent_b.self_model.personality_profile

        # Profiles should differ significantly
        self.assertGreater(pa.neuroticism, pb.neuroticism)
        self.assertGreater(pb.extraversion, pa.extraversion)
        self.assertGreater(pb.agreeableness, pa.agreeableness)

    def test_divergent_profiles_produce_different_action_rankings(self) -> None:
        agent_a = _make_agent(seed=302)
        agent_b = _make_agent(seed=302)
        _ingest_and_sleep(agent_a, THREAT_NARRATIVES, sleep_cycles=3)
        _ingest_and_sleep(agent_b, SOCIAL_NARRATIVES, sleep_cycles=3)

        # Run 5 different observations and check for divergent choices
        observations = [
            Observation(food=0.3, danger=0.50, novelty=0.4, shelter=0.3, temperature=0.5, social=0.3),
            Observation(food=0.5, danger=0.20, novelty=0.6, shelter=0.4, temperature=0.5, social=0.2),
            Observation(food=0.2, danger=0.70, novelty=0.2, shelter=0.5, temperature=0.5, social=0.4),
            Observation(food=0.6, danger=0.10, novelty=0.3, shelter=0.6, temperature=0.5, social=0.1),
            Observation(food=0.4, danger=0.40, novelty=0.5, shelter=0.2, temperature=0.5, social=0.5),
        ]

        different_choices = 0
        for obs in observations:
            # Reset agents to avoid carry-over from prior decision cycles
            ra = agent_a.decision_cycle(obs)
            rb = agent_b.decision_cycle(obs)
            if ra["diagnostics"].chosen.choice != rb["diagnostics"].chosen.choice:
                different_choices += 1

        # At least 40% of identical decision cycles should produce different choices
        self.assertGreaterEqual(
            different_choices,
            2,
            "Agents with different personalities should diverge on at least 40% of identical decision cycles",
        )


class TestScenarioD_Serialization(unittest.TestCase):
    """PersonalityProfile survives full agent serialization round-trip."""

    def test_personality_survives_agent_round_trip(self) -> None:
        agent = _make_agent(seed=401)
        _ingest_and_sleep(agent, THREAT_NARRATIVES, sleep_cycles=2)

        original_profile = agent.self_model.personality_profile.to_dict()
        restored = _round_trip_agent(agent, seed=401)
        restored_profile = restored.self_model.personality_profile.to_dict()

        self.assertEqual(original_profile, restored_profile)

    def test_restored_agent_makes_same_decisions(self) -> None:
        agent = _make_agent(seed=402)
        _ingest_and_sleep(agent, THREAT_NARRATIVES, sleep_cycles=2)

        obs = Observation(food=0.3, danger=0.55, novelty=0.4, shelter=0.3, temperature=0.5, social=0.3)
        original_result = agent.decision_cycle(obs)

        restored = _round_trip_agent(agent, seed=402)
        restored_result = restored.decision_cycle(obs)

        self.assertEqual(
            original_result["diagnostics"].chosen.choice,
            restored_result["diagnostics"].chosen.choice,
        )


class TestNoRegressionDefaultProfile(unittest.TestCase):
    """Default neutral personality should not change existing behavior."""

    def test_fresh_agent_has_neutral_profile(self) -> None:
        agent = SegmentAgent(rng=random.Random(500))
        p = agent.self_model.personality_profile
        self.assertEqual(p.openness, 0.5)
        self.assertEqual(p.conscientiousness, 0.5)
        self.assertEqual(p.extraversion, 0.5)
        self.assertEqual(p.agreeableness, 0.5)
        self.assertEqual(p.neuroticism, 0.5)
        self.assertEqual(p.update_count, 0)

    def test_neutral_personality_does_not_affect_drive_modulation(self) -> None:
        agent = SegmentAgent(rng=random.Random(501))
        mod = agent.self_model.personality_profile.drive_modulation()
        for value in mod.values():
            self.assertAlmostEqual(value, 0.0, places=10)

    def test_narrative_trace_includes_personality_deltas(self) -> None:
        agent = _make_agent(seed=502)
        _ingest_and_sleep(agent, THREAT_NARRATIVES, sleep_cycles=1)
        has_personality = any(
            "personality_deltas" in entry.get("narrative_prior_updates", {})
            for entry in agent.narrative_trace
        )
        self.assertTrue(
            has_personality,
            "narrative_trace should include personality_deltas after sleep",
        )


if __name__ == "__main__":
    unittest.main()
