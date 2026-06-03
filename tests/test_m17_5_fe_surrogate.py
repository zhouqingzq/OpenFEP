from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.fep_surrogate import build_free_energy_surrogate
from segmentum.preferences import Goal


class TestM175FreeEnergySurrogate(unittest.TestCase):
    def test_compute_free_energy_matches_canonical_owner(self) -> None:
        agent = SegmentAgent(rng=random.Random(7))
        agent.energy = 0.32
        agent.stress = 0.18
        agent.fatigue = 0.27
        agent.temperature = 0.61
        errors = {
            "food": -0.20,
            "danger": 0.50,
            "novelty": -0.10,
            "shelter": 0.25,
            "temperature": -0.08,
            "social": 0.12,
        }

        computed = agent.compute_free_energy(errors)
        expected = build_free_energy_surrogate(
            errors=errors,
            body_state={
                "energy": agent.energy,
                "stress": agent.stress,
                "fatigue": agent.fatigue,
                "temperature": agent.temperature,
            },
        ).free_energy_surrogate

        self.assertAlmostEqual(computed, expected)

    def test_project_action_emits_canonical_breakdowns(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        observed = {
            "food": 0.20,
            "danger": 0.90,
            "novelty": 0.25,
            "shelter": 0.30,
            "temperature": 0.48,
            "social": 0.15,
        }
        priors = agent.strategic_layer.priors(
            agent.energy,
            agent.stress,
            agent.fatigue,
            agent.temperature,
            agent.dopamine,
            agent.drive_system,
        )
        prediction = agent.world_model.predict(priors)
        errors = {
            key: observed.get(key, 0.0) - prediction.get(key, 0.0)
            for key in sorted(set(observed) | set(prediction))
        }
        free_energy_before = agent.compute_free_energy(errors)

        projected = agent._project_action(
            action="hide",
            observed=observed,
            prediction=prediction,
            priors=priors,
            free_energy_before=free_energy_before,
            current_cluster_id=None,
            active_goal=Goal.SURVIVAL,
        )

        fe_breakdown = projected["free_energy_surrogate"]
        efe_breakdown = projected["expected_free_energy_surrogate"]
        predicted_effects = projected["predicted_effects"]

        self.assertIn("free_energy_surrogate", fe_breakdown)
        self.assertIn("precision_weighted_prediction_error", fe_breakdown)
        self.assertIn("expected_free_energy_surrogate", efe_breakdown)
        self.assertIn("risk_cost", efe_breakdown)
        self.assertAlmostEqual(
            float(predicted_effects["free_energy_drop"]),
            free_energy_before - float(fe_breakdown["free_energy_surrogate"]),
        )
        self.assertAlmostEqual(
            float(projected["expected_free_energy"]),
            float(efe_breakdown["expected_free_energy_surrogate"]),
            places=6,
        )

    def test_stored_episode_carries_canonical_fe_aliases(self) -> None:
        agent = SegmentAgent(rng=random.Random(17))
        agent.cycle = 1

        decision = agent.integrate_outcome(
            choice="hide",
            observed={
                "food": 0.20,
                "danger": 0.95,
                "novelty": 0.20,
                "shelter": 0.30,
                "temperature": 0.45,
                "social": 0.20,
            },
            prediction={
                "food": 0.70,
                "danger": 0.10,
                "novelty": 0.45,
                "shelter": 0.55,
                "temperature": 0.50,
                "social": 0.35,
            },
            errors={
                "food": -0.50,
                "danger": 0.85,
                "novelty": -0.25,
                "shelter": -0.25,
                "temperature": -0.05,
                "social": -0.15,
            },
            free_energy_before=0.20,
            free_energy_after=0.60,
        )

        self.assertTrue(decision.episode_created)
        stored = agent.long_term_memory.episodes[-1]
        self.assertIn("free_energy_surrogate_breakdown", stored)
        self.assertIn("free_energy_surrogate", stored)
        self.assertIn("precision_weighted_prediction_error", stored)
        self.assertAlmostEqual(float(stored["free_energy_delta"]), -0.40)
        self.assertAlmostEqual(float(stored["outcome"]["free_energy_drop"]), -0.40)


if __name__ == "__main__":
    unittest.main()
