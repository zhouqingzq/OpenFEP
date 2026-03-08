from __future__ import annotations

import unittest

from segmentum.agent import SegmentAgent
from segmentum.drives import DriveSystem, StrategicLayer
from segmentum.environment import Observation
from segmentum.evaluation import RunMetrics
from segmentum.fep import infer_policy
from segmentum.memory import LongTermMemory
from segmentum.predictive_coding import (
    InteroceptiveBeliefState,
    compose_upstream_observation,
    predictive_coding_profile,
)
from segmentum.runtime import SegmentRuntime
from segmentum.state import AgentState, Strategy, TickInput
from segmentum.world_model import GenerativeWorldModel


class FepPolicyTests(unittest.TestCase):
    def test_infer_policy_prefers_escape_under_high_pressure_and_surprise(self) -> None:
        state = AgentState(
            internal_energy=0.15,
            prediction_error=0.82,
            boredom=0.08,
            surprise_load=0.75,
        )
        tick_input = TickInput(
            resource_pressure=0.88,
            surprise_signal=0.92,
            boredom_signal=0.02,
            energy_drain=0.04,
        )

        policy = infer_policy(state, tick_input)

        self.assertEqual(policy.chosen_strategy, Strategy.ESCAPE)
        self.assertGreater(policy.pragmatic_value, policy.epistemic_value)
        self.assertLess(policy.escape_efe, policy.explore_efe)
        self.assertLess(policy.escape_efe, policy.exploit_efe)


class MemoryRetrievalTests(unittest.TestCase):
    def test_retrieve_similar_prefers_recent_matching_episodes(self) -> None:
        memory = LongTermMemory()
        reference = {
            "food": 0.80,
            "danger": 0.20,
            "novelty": 0.60,
            "shelter": 0.50,
            "temperature": 0.45,
            "social": 0.40,
        }

        memory.store_episode(
            cycle=1,
            observation=reference,
            prediction=reference,
            errors={key: 0.0 for key in reference},
            action="hide",
            outcome={"free_energy_drop": 0.30},
        )
        memory.store_episode(
            cycle=2,
            observation={
                "food": 0.76,
                "danger": 0.24,
                "novelty": 0.56,
                "shelter": 0.48,
                "temperature": 0.47,
                "social": 0.44,
            },
            prediction=reference,
            errors={key: 0.0 for key in reference},
            action="forage",
            outcome={"free_energy_drop": 0.20},
        )
        memory.store_episode(
            cycle=9,
            observation=reference,
            prediction=reference,
            errors={key: 0.0 for key in reference},
            action="scan",
            outcome={"free_energy_drop": 0.10},
        )

        hits = memory.retrieve_similar(
            current_observation=reference,
            current_body_state={"cycle": 10, "energy": 0.50, "stress": 0.20},
            k=2,
        )

        self.assertEqual(len(hits), 2)
        self.assertEqual([episode["action"] for episode in hits], ["scan", "forage"])
        self.assertEqual(hits[0]["cycle"], 9)


class WorldModelUpdateTests(unittest.TestCase):
    def test_update_from_error_moves_beliefs_toward_observation(self) -> None:
        model = GenerativeWorldModel()
        target_food = 0.82
        target_danger = 0.10
        start_food = model.beliefs["food"]
        start_danger = model.beliefs["danger"]

        model.update_from_error(
            {
                "food": target_food - start_food,
                "danger": target_danger - start_danger,
            }
        )

        self.assertGreater(model.beliefs["food"], start_food)
        self.assertLess(model.beliefs["danger"], start_danger)
        self.assertLess(abs(target_food - model.beliefs["food"]), abs(target_food - start_food))
        self.assertLess(
            abs(target_danger - model.beliefs["danger"]),
            abs(target_danger - start_danger),
        )


class HierarchicalBeliefStateTests(unittest.TestCase):
    def test_large_prediction_error_propagates_weighted_residual(self) -> None:
        state = InteroceptiveBeliefState()
        update = state.posterior_update(
            incoming_observation={
                "food": 0.95,
                "danger": 0.05,
                "novelty": 0.50,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.30,
            },
            top_down_prediction={
                "food": 0.20,
                "danger": 0.80,
                "novelty": 0.50,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.30,
            },
        )

        self.assertTrue(update.digestion_exceeded)
        self.assertGreater(abs(update.propagated_error["food"]), 0.0)
        self.assertGreater(abs(update.propagated_error["danger"]), 0.0)
        self.assertGreater(update.error_precision["food"], 0.80)
        self.assertGreater(update.kalman_gain["food"], 0.0)

    def test_small_prediction_error_is_locally_absorbed(self) -> None:
        state = InteroceptiveBeliefState()
        update = state.posterior_update(
            incoming_observation={
                "food": 0.51,
                "danger": 0.31,
                "novelty": 0.50,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.30,
            },
            top_down_prediction={
                "food": 0.50,
                "danger": 0.30,
                "novelty": 0.50,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.30,
            },
        )

        self.assertFalse(update.digestion_exceeded)
        self.assertEqual(update.propagated_error["food"], 0.0)
        self.assertEqual(update.propagated_error["danger"], 0.0)

    def test_runtime_accepts_predictive_profile_override(self) -> None:
        runtime = SegmentRuntime.load_or_create(
            seed=17,
            reset=True,
            predictive_hyperparameters=predictive_coding_profile("hair_trigger"),
            reset_predictive_precisions=True,
        )

        self.assertAlmostEqual(
            runtime.agent.interoceptive_layer.belief_state.digestion_threshold,
            0.03,
        )
        self.assertAlmostEqual(
            runtime.agent.world_model.sensorimotor_layer.belief_state.base_error_precision,
            0.90,
        )
        self.assertAlmostEqual(
            runtime.agent.strategic_layer.belief_state.initial_precision,
            0.98,
        )

    def test_strategic_dispatch_uses_belief_state_fast_weights(self) -> None:
        layer = StrategicLayer()
        drives = DriveSystem()
        layer.belief_state.top_down_mix = 0.0

        priors, prediction = layer.dispatch_prediction(
            energy=0.80,
            stress=0.20,
            fatigue=0.10,
            temperature=0.50,
            dopamine=0.10,
            drive_system=drives,
        )

        self.assertEqual(prediction, layer.belief_state.beliefs)

        layer.belief_state.top_down_mix = 1.0
        priors, prediction = layer.dispatch_prediction(
            energy=0.80,
            stress=0.20,
            fatigue=0.10,
            temperature=0.50,
            dopamine=0.10,
            drive_system=drives,
        )

        self.assertEqual(prediction, priors)

    def test_perceive_exposes_explicit_hierarchical_signals(self) -> None:
        agent = SegmentAgent()
        observation = Observation(
            food=0.92,
            danger=0.08,
            novelty=0.64,
            shelter=0.51,
            temperature=0.47,
            social=0.35,
        )

        _, prediction, _, _, hierarchy = agent.perceive(observation)

        self.assertEqual(prediction, hierarchy.interoceptive_prediction)
        self.assertEqual(
            hierarchy.interoceptive_prediction,
            hierarchy.interoceptive_update.prediction,
        )
        self.assertEqual(
            hierarchy.sensorimotor_observation,
            compose_upstream_observation(
                hierarchy.sensorimotor_prediction,
                hierarchy.interoceptive_update.propagated_error,
            ),
        )
        self.assertEqual(
            hierarchy.strategic_observation,
            compose_upstream_observation(
                hierarchy.strategic_prediction,
                hierarchy.sensorimotor_update.propagated_error,
            ),
        )


class SummaryMetricsTests(unittest.TestCase):
    def test_summary_exposes_baseline_report_fields(self) -> None:
        metrics = RunMetrics()
        metrics.record_cycle(
            choice="rest",
            free_energy_before=1.0,
            free_energy_after=0.7,
            energy=0.4,
            stress=0.2,
            memory_hits=1,
            slept=False,
            alive=True,
        )
        metrics.record_cycle(
            choice="rest",
            free_energy_before=0.9,
            free_energy_after=0.8,
            energy=0.35,
            stress=0.25,
            memory_hits=0,
            slept=False,
            alive=True,
        )
        metrics.telemetry_error_count = 1
        metrics.persistence_error_count = 2
        metrics.termination_reason = "cycles_exhausted"

        summary = metrics.summary()

        self.assertEqual(summary["memory_hit_rate"], 0.5)
        self.assertEqual(summary["termination_reason"], "cycles_exhausted")
        self.assertEqual(summary["unique_actions"], 1)
        self.assertAlmostEqual(summary["action_entropy"], 0.0)
        self.assertEqual(summary["dominant_action"], "rest")
        self.assertAlmostEqual(summary["dominant_action_share"], 1.0)
        self.assertEqual(summary["max_action_streak"], 2)
        self.assertEqual(summary["telemetry_error_count"], 1)
        self.assertEqual(summary["persistence_error_count"], 2)
        self.assertEqual(summary["action_distribution"], {"rest": 1.0})


class DeterminismSmokeTests(unittest.TestCase):
    def test_same_seed_produces_consistent_summary_for_longer_run(self) -> None:
        first_summary = SegmentRuntime.load_or_create(seed=41, reset=True).run(
            cycles=64,
            verbose=False,
        )
        second_summary = SegmentRuntime.load_or_create(seed=41, reset=True).run(
            cycles=64,
            verbose=False,
        )

        self.assertEqual(first_summary.keys(), second_summary.keys())
        for key in first_summary:
            first_value = first_summary[key]
            second_value = second_summary[key]
            if isinstance(first_value, float):
                self.assertAlmostEqual(first_value, second_value, places=12, msg=key)
            else:
                self.assertEqual(first_value, second_value, key)


class BehaviorRegressionTests(unittest.TestCase):
    def test_long_run_does_not_collapse_to_single_action(self) -> None:
        summary = SegmentRuntime.load_or_create(seed=17, reset=True).run(
            cycles=128,
            verbose=False,
        )

        self.assertGreaterEqual(summary["unique_actions"], 3)
        self.assertGreaterEqual(summary["action_entropy"], 0.20)
        self.assertLessEqual(summary["dominant_action_share"], 0.92)


if __name__ == "__main__":
    unittest.main()
