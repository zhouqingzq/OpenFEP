from __future__ import annotations

import json
import random
import threading
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.sleep_consolidator import LLMSleepRuleRefiner, SleepLLMExtractor
from segmentum.types import SleepRule


class SleepActionFreezeTests(unittest.TestCase):
    """Verify that the action space is frozen during sleep."""

    def test_decision_cycle_raises_while_sleeping(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent._sleeping = True
        with self.assertRaises(RuntimeError) as ctx:
            agent.decision_cycle(
                Observation(food=0.5, danger=0.2, novelty=0.3, shelter=0.4, temperature=0.5, social=0.3)
            )
        self.assertIn("frozen", str(ctx.exception))

    def test_sleeping_flag_resets_after_sleep_completes(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.sleep()
        self.assertFalse(agent._sleeping)

    def test_sleeping_flag_resets_even_on_error(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        # Force an error inside _sleep_inner by corrupting episodes
        agent.episodes = [None]  # type: ignore[list-item]
        try:
            agent.sleep()
        except Exception:
            pass
        self.assertFalse(agent._sleeping)


class LLMSleepRuleRefinerTests(unittest.TestCase):
    """Unit tests for LLMSleepRuleRefiner parsing logic."""

    def _make_rule(self, **overrides: object) -> SleepRule:
        defaults = dict(
            rule_id="sleep-1-0-forage-survival_threat",
            type="risk_pattern",
            cluster=0,
            action="forage",
            observed_outcome="survival_threat",
            confidence=0.80,
            support=5,
            average_surprise=120.0,
            average_prediction_error=0.22,
            timestamp=6,
        )
        defaults.update(overrides)
        return SleepRule(**defaults)  # type: ignore[arg-type]

    def test_parse_valid_json_array(self) -> None:
        rule = self._make_rule()
        text = json.dumps([{
            "rule_id": rule.rule_id,
            "type": rule.type,
            "cluster": rule.cluster,
            "action": rule.action,
            "observed_outcome": rule.observed_outcome,
            "confidence": 0.92,
            "support": rule.support,
            "average_surprise": rule.average_surprise,
            "average_prediction_error": rule.average_prediction_error,
            "timestamp": rule.timestamp,
        }])
        parsed = LLMSleepRuleRefiner._parse_rules(text, [rule])
        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0].confidence, 0.92)

    def test_parse_markdown_fenced_json(self) -> None:
        rule = self._make_rule()
        inner = json.dumps([{
            "rule_id": rule.rule_id,
            "type": rule.type,
            "cluster": rule.cluster,
            "action": rule.action,
            "observed_outcome": rule.observed_outcome,
            "confidence": 0.85,
            "support": rule.support,
            "average_surprise": rule.average_surprise,
            "average_prediction_error": rule.average_prediction_error,
            "timestamp": rule.timestamp,
        }])
        text = f"```json\n{inner}\n```"
        parsed = LLMSleepRuleRefiner._parse_rules(text, [rule])
        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0].confidence, 0.85)

    def test_parse_garbage_returns_fallback(self) -> None:
        rule = self._make_rule()
        parsed = LLMSleepRuleRefiner._parse_rules("not json at all", [rule])
        self.assertEqual(parsed, [rule])

    def test_parse_empty_array_returns_fallback(self) -> None:
        rule = self._make_rule()
        parsed = LLMSleepRuleRefiner._parse_rules("[]", [rule])
        self.assertEqual(parsed, [rule])

    def test_parse_clamps_confidence(self) -> None:
        rule = self._make_rule()
        text = json.dumps([{
            "rule_id": rule.rule_id,
            "type": rule.type,
            "cluster": rule.cluster,
            "action": rule.action,
            "observed_outcome": rule.observed_outcome,
            "confidence": 1.5,
            "support": rule.support,
            "average_surprise": rule.average_surprise,
            "average_prediction_error": rule.average_prediction_error,
            "timestamp": rule.timestamp,
        }])
        parsed = LLMSleepRuleRefiner._parse_rules(text, [rule])
        self.assertLessEqual(parsed[0].confidence, 0.99)

    def test_refiner_without_api_key_returns_original(self) -> None:
        refiner = LLMSleepRuleRefiner(api_key="")
        rule = self._make_rule()
        result = refiner.refine_sleep_rules(summary={}, rules=[rule], episodes=[])
        self.assertEqual(result, [rule])

    def test_extractor_delegates_to_refiner(self) -> None:
        """SleepLLMExtractor correctly calls LLMSleepRuleRefiner.refine_sleep_rules."""
        rule = self._make_rule()
        adjusted = self._make_rule(confidence=0.95)
        class FakeRefiner:
            def refine_sleep_rules(self, *, summary, rules, episodes):
                return [adjusted]
        extractor = SleepLLMExtractor(llm=FakeRefiner())
        result = extractor([rule], [{"cluster_id": 0, "action_taken": "forage",
                                     "predicted_outcome": "survival_threat",
                                     "prediction_error": 0.22, "total_surprise": 1005.0}])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.95)


class SleepConsolidationLoopTests(unittest.TestCase):
    def test_sleep_consolidation_reduces_surprise_on_repeat_anomaly(self) -> None:
        agent = SegmentAgent(rng=random.Random(29))
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1

        observation = {
            "food": 0.38,
            "danger": 0.58,
            "novelty": 0.22,
            "shelter": 0.18,
            "temperature": 0.46,
            "social": 0.18,
        }
        prediction = {
            "food": 0.72,
            "danger": 0.18,
            "novelty": 0.42,
            "shelter": 0.42,
            "temperature": 0.50,
            "social": 0.30,
        }
        errors = {
            key: observation[key] - prediction[key]
            for key in observation
        }
        harmful_outcome = {
            "energy_delta": -0.08,
            "stress_delta": 0.24,
            "fatigue_delta": 0.16,
            "temperature_delta": 0.02,
            "free_energy_drop": -0.42,
        }

        before_sleep = agent.long_term_memory.maybe_store_episode(
            cycle=1,
            observation=observation,
            prediction=prediction,
            errors=errors,
            action="forage",
            outcome=harmful_outcome,
            body_state={
                "energy": 0.18,
                "stress": 0.82,
                "fatigue": 0.32,
                "temperature": 0.46,
            },
        )
        self.assertTrue(before_sleep.episode_created)
        for cycle in range(2, 6):
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=observation,
                prediction=prediction,
                errors=errors,
                action="forage",
                outcome=harmful_outcome,
                body_state={
                    "energy": 0.18,
                    "stress": 0.82,
                    "fatigue": 0.32,
                    "temperature": 0.46,
                },
            )

        episodes_before_sleep = len(agent.long_term_memory.episodes)
        summary = agent.sleep()

        self.assertGreater(summary.rules_extracted, 0)
        self.assertGreater(summary.threat_updates, 0)
        self.assertGreater(summary.preference_updates, 0)
        self.assertTrue(agent.semantic_memory)

        diagnostics = agent.decision_cycle(Observation(**observation))["diagnostics"]
        chosen_action = diagnostics.chosen.choice
        repeated_outcome = (
            harmful_outcome
            if chosen_action == "forage"
            else {
                "energy_delta": -0.02,
                "stress_delta": -0.06,
                "fatigue_delta": -0.04,
                "temperature_delta": 0.0,
                "free_energy_drop": 0.06,
            }
        )
        after_sleep = agent.long_term_memory.maybe_store_episode(
            cycle=100,
            observation=observation,
            prediction=prediction,
            errors=errors,
            action=chosen_action,
            outcome=repeated_outcome,
            body_state={
                "energy": 0.24,
                "stress": 0.38,
                "fatigue": 0.24,
                "temperature": 0.47,
            },
        )

        self.assertLess(after_sleep.total_surprise, before_sleep.total_surprise)
        self.assertTrue((not after_sleep.episode_created) or (chosen_action != "forage"))
        self.assertLessEqual(len(agent.long_term_memory.episodes), episodes_before_sleep)
        self.assertGreater(agent.world_model.get_threat_prior(0), 0.0)


if __name__ == "__main__":
    unittest.main()
