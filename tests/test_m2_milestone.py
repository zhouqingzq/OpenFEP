from __future__ import annotations

import json
import random
import unittest
from dataclasses import asdict

from segmentum.agent import SegmentAgent
from segmentum.preferences import Goal
from tests.test_counterfactual import OBS_DANGEROUS, _populate_dangerous_episodes


def _round_trip_agent(agent: SegmentAgent, seed: int = 999) -> SegmentAgent:
    payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
    return SegmentAgent.from_dict(payload, rng=random.Random(seed))


def _score_for_action(agent: SegmentAgent, action: str) -> float:
    diagnostics = agent.decision_cycle(OBS_DANGEROUS)["diagnostics"]
    return next(option.policy_score for option in diagnostics.ranked_options if option.choice == action)


class TestM2Milestone(unittest.TestCase):
    def test_past_influence_changes_future_action_ranking(self) -> None:
        baseline_agent = SegmentAgent(rng=random.Random(101))
        baseline_forage = _score_for_action(baseline_agent, "forage")

        influenced_agent = SegmentAgent(rng=random.Random(101))
        influenced_agent.long_term_memory.minimum_support = 1
        influenced_agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(influenced_agent, count=6)
        influenced_agent.long_term_memory.assign_clusters()
        influenced_agent.cycle = 20
        influenced_agent.sleep()

        diagnostics = influenced_agent.decision_cycle(OBS_DANGEROUS)["diagnostics"]
        influenced_forage = next(
            option.policy_score for option in diagnostics.ranked_options if option.choice == "forage"
        )

        self.assertLess(influenced_forage, baseline_forage)
        self.assertNotEqual(diagnostics.chosen.choice, "forage")

    def test_identity_continuity_survives_json_reload(self) -> None:
        agent = SegmentAgent(rng=random.Random(17))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()

        continuous = _round_trip_agent(agent, seed=77)
        restored = _round_trip_agent(agent, seed=77)

        continuous_actions: list[str] = []
        restored_actions: list[str] = []
        continuous_components: list[str] = []
        restored_components: list[str] = []
        for _ in range(8):
            continuous_diag = continuous.decision_cycle(OBS_DANGEROUS)["diagnostics"]
            restored_diag = restored.decision_cycle(OBS_DANGEROUS)["diagnostics"]
            continuous_actions.append(continuous_diag.chosen.choice)
            restored_actions.append(restored_diag.chosen.choice)
            continuous_components.append(continuous_diag.chosen.dominant_component)
            restored_components.append(restored_diag.chosen.dominant_component)

        self.assertEqual(
            continuous.self_model.identity_narrative.to_dict(),
            restored.self_model.identity_narrative.to_dict(),
        )
        self.assertEqual(
            continuous.self_model.preferred_policies.to_dict(),
            restored.self_model.preferred_policies.to_dict(),
        )
        self.assertEqual(continuous_actions, restored_actions)
        self.assertEqual(continuous_components, restored_components)

    def test_project_action_emits_goal_weighted_expected_free_energy(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        observed = asdict(OBS_DANGEROUS)
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

        expected = agent.long_term_memory.preference_model.expected_free_energy(
            outcome=str(projected["predicted_outcome"]),
            predicted_error=float(projected["predicted_error"]),
            action_ambiguity=float(projected["action_ambiguity"]),
            goal=Goal.SURVIVAL,
            baseline_risk=float(projected["risk"]),
        )

        self.assertIn("expected_free_energy", projected)
        self.assertAlmostEqual(float(projected["expected_free_energy"]), expected)
        self.assertGreaterEqual(float(projected["expected_free_energy"]), float(projected["risk"]))
    def test_explain_structured_states_typical_or_deviating_pattern(self) -> None:
        agent = SegmentAgent(rng=random.Random(55))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()
        agent.decision_cycle(OBS_DANGEROUS)

        details = agent.explain_decision_details()

        self.assertIn("active_goal", details)
        self.assertIn("goal_alignment", details)
        self.assertIn("dominant_component", details)
        self.assertIn("historical_action_frequency", details)
        self.assertIn("identity_consistency", details)
        self.assertTrue(
            "This choice is consistent with my established pattern" in details["text"]
            or "This choice deviates from my usual pattern" in details["text"]
        )


if __name__ == "__main__":
    unittest.main()



