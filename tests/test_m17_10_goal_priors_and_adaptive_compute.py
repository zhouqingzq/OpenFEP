from __future__ import annotations

import unittest
from unittest.mock import patch

from segmentum.action_schema import ActionSchema
from segmentum.adaptive_compute import decide_adaptive_compute
from segmentum.agent import SegmentAgent
from segmentum.goal_priors import build_goal_prior_adjustment
from segmentum.preferences import Goal


def _state(
    *,
    danger: float = 0.25,
    novelty: float = 0.35,
    social: float = 0.45,
    energy: float = 0.60,
    stress: float = 0.25,
    fatigue: float = 0.20,
) -> dict[str, object]:
    return {
        "observation": {
            "danger": danger,
            "novelty": novelty,
            "social": social,
            "food": 0.20,
            "shelter": 0.25,
        },
        "body_state": {
            "energy": energy,
            "stress": stress,
            "fatigue": fatigue,
            "temperature": 0.50,
        },
    }


class GoalPriorsAndAdaptiveComputeTests(unittest.TestCase):
    def test_same_observation_differs_by_active_goal_before_final_reranking(self) -> None:
        agent = SegmentAgent()
        observation = {"danger": 0.22, "novelty": 0.42, "social": 0.30, "food": 0.24, "shelter": 0.28}
        prediction = {"danger": 0.26, "novelty": 0.30, "social": 0.22, "food": 0.20, "shelter": 0.25}
        priors = dict(prediction)
        current_state = {
            "observation": dict(observation),
            "prediction": dict(prediction),
            "errors": {key: observation[key] - prediction[key] for key in observation},
            "body_state": agent._current_body_state(),
        }
        survival_prior = build_goal_prior_adjustment(
            active_goal=Goal.SURVIVAL,
            current_state=current_state,
            goal_context={"active_goal": "SURVIVAL", "urgency_scores": {"SURVIVAL": 0.80}},
        )
        resource_prior = build_goal_prior_adjustment(
            active_goal=Goal.RESOURCES,
            current_state=current_state,
            goal_context={"active_goal": "RESOURCES", "urgency_scores": {"RESOURCES": 0.80}},
        )
        self.assertIsNotNone(survival_prior)
        self.assertIsNotNone(resource_prior)
        survival_option = agent._project_action(
            action="forage",
            observed=observation,
            prediction=prediction,
            priors=priors,
            free_energy_before=0.40,
            current_cluster_id=None,
            active_goal=Goal.SURVIVAL,
            memory_context={"goal_prior": survival_prior.to_dict(), "actions": {}},
        )
        resource_option = agent._project_action(
            action="forage",
            observed=observation,
            prediction=prediction,
            priors=priors,
            free_energy_before=0.40,
            current_cluster_id=None,
            active_goal=Goal.RESOURCES,
            memory_context={"goal_prior": resource_prior.to_dict(), "actions": {}},
        )
        self.assertTrue(survival_option["goal_prior_applied"])
        self.assertTrue(resource_option["goal_prior_applied"])
        self.assertNotEqual(
            survival_option["projected_observation"]["danger"],
            resource_option["projected_observation"]["danger"],
        )
        self.assertNotEqual(
            survival_option["projected_observation"]["food"],
            resource_option["projected_observation"]["food"],
        )
        self.assertNotEqual(
            survival_option["expected_free_energy"],
            resource_option["expected_free_energy"],
        )

    def test_goal_prior_is_channel_bounded_and_auditable(self) -> None:
        prior = build_goal_prior_adjustment(
            active_goal=Goal.SOCIAL,
            current_state=_state(),
            goal_context={"active_goal": "SOCIAL", "urgency_scores": {"SOCIAL": 0.70}},
        )
        self.assertIsNotNone(prior)
        payload = prior.to_dict()
        self.assertEqual(payload["active_goal"], "SOCIAL")
        self.assertIn("prior_channel_shifts", payload)
        self.assertIn("modality_shifts", payload)
        self.assertIn("reason_codes", payload)
        for value in payload["prior_channel_shifts"].values():
            self.assertLessEqual(abs(float(value)), 0.30)
        for value in payload["modality_shifts"].values():
            self.assertLessEqual(abs(float(value)), 0.14)

    def test_goal_prior_does_not_override_strong_contradictory_observation(self) -> None:
        prior = build_goal_prior_adjustment(
            active_goal=Goal.RESOURCES,
            current_state=_state(danger=0.90, energy=0.32, stress=0.76, fatigue=0.74),
            goal_context={"active_goal": "RESOURCES", "urgency_scores": {"RESOURCES": 0.88}},
        )
        self.assertIsNotNone(prior)
        payload = prior.to_dict()
        self.assertLess(payload["contradiction_guard"], 0.6)
        self.assertIn("strong_observed_danger_limits_goal_prior", payload["reason_codes"])
        self.assertLessEqual(payload["modality_shifts"]["danger"], 0.05)

    def test_adaptive_compute_reduces_budget_in_stable_low_conflict_regime(self) -> None:
        decision = decide_adaptive_compute(
            field={
                "field_flatness": 0.12,
                "conflict_density": 0.08,
                "counterfactual_audit": {"chosen_decision_subsequent_fe": 0.16},
            },
            goal_context={"active_goal": "CONTROL", "urgency_scores": {"CONTROL": 0.28}},
            prediction_error_surrogate=0.18,
            base_retrieval_k=3,
            base_path_k=2,
            candidate_action_count=7,
        )
        payload = decision.to_dict()
        self.assertEqual(payload["confidence_regime"], "low")
        self.assertEqual(payload["retrieval_k"], 2)
        self.assertEqual(payload["path_neighborhood_k"], 1)
        self.assertEqual(payload["verification_target_limit"], 1)
        self.assertEqual(payload["candidate_action_limit"], 4)
        self.assertFalse(payload["field_refinement_enabled"])

    def test_adaptive_compute_increases_budget_in_flat_or_conflicted_regime(self) -> None:
        decision = decide_adaptive_compute(
            field={
                "field_flatness": 0.74,
                "conflict_density": 0.58,
                "counterfactual_audit": {
                    "chosen_decision_subsequent_fe": 0.54,
                    "status": "field_required",
                },
            },
            goal_context={"active_goal": "SURVIVAL", "urgency_scores": {"SURVIVAL": 0.85}},
            prediction_error_surrogate=0.49,
            base_retrieval_k=3,
            base_path_k=2,
            candidate_action_count=7,
        )
        payload = decision.to_dict()
        self.assertEqual(payload["confidence_regime"], "high")
        self.assertGreaterEqual(payload["retrieval_k"], 5)
        self.assertGreaterEqual(payload["path_neighborhood_k"], 4)
        self.assertGreaterEqual(payload["verification_target_limit"], 4)
        self.assertGreaterEqual(payload["candidate_action_limit"], 6)
        self.assertTrue(payload["field_refinement_enabled"])

    def test_at_least_three_live_budgets_are_controlled(self) -> None:
        agent = SegmentAgent()
        agent.last_retrieval_result = {
            "adaptive_compute": {
                "candidate_action_limit": 3,
                "verification_target_limit": 4,
                "retrieval_k": 5,
                "path_neighborhood_k": 4,
            },
            "goal_prior": {},
            "local_field": {},
        }
        memory_context = agent._zero_memory_context(
            observed={"danger": 0.20, "novelty": 0.35, "social": 0.30},
            baseline_prediction={"danger": 0.24, "novelty": 0.22, "social": 0.20},
            errors={"danger": -0.04, "novelty": 0.13, "social": 0.10},
            summary="test",
            active_paths=[],
            local_field={},
        )
        many_actions = [
            ActionSchema(name=name, cost_estimate=0.05 + (index * 0.01))
            for index, name in enumerate(
                ["rest", "hide", "exploit_shelter", "thermoregulate", "forage", "scan", "seek_contact"]
            )
        ]
        with patch.object(agent, "_available_action_schemas", return_value=many_actions):
            options = agent.evaluate_action_options(
                observed={"danger": 0.20, "novelty": 0.35, "social": 0.30},
                prediction={"danger": 0.24, "novelty": 0.22, "social": 0.20},
                priors={"danger": 0.24, "novelty": 0.22, "social": 0.20},
                free_energy_before=0.30,
                current_cluster_id=None,
                active_goal=Goal.CONTROL,
                memory_context=memory_context,
            )
        self.assertEqual(len(options), 3)
        self.assertEqual(memory_context["adaptive_compute"]["verification_target_limit"], 4)
        self.assertEqual(memory_context["adaptive_compute"]["retrieval_k"], 5)
        self.assertEqual(memory_context["adaptive_compute"]["path_neighborhood_k"], 4)

    def test_goal_urgency_can_raise_compute_when_other_signals_are_borderline(self) -> None:
        decision = decide_adaptive_compute(
            field={
                "field_flatness": 0.40,
                "conflict_density": 0.30,
                "counterfactual_audit": {"chosen_decision_subsequent_fe": 0.34},
            },
            goal_context={"active_goal": "SURVIVAL", "urgency_scores": {"SURVIVAL": 0.90}},
            prediction_error_surrogate=0.34,
            base_retrieval_k=3,
            base_path_k=2,
            candidate_action_count=6,
        )
        self.assertEqual(decision.confidence_regime, "high")
        self.assertIn("goal_urgency_escalation", decision.escalation_reason_codes)

    def test_goal_alignment_score_is_not_the_only_goal_effect(self) -> None:
        agent = SegmentAgent()
        observed = {"danger": 0.18, "novelty": 0.50, "social": 0.28, "food": 0.24, "shelter": 0.30}
        prediction = {"danger": 0.24, "novelty": 0.24, "social": 0.20, "food": 0.20, "shelter": 0.26}
        current_state = {
            "observation": dict(observed),
            "prediction": dict(prediction),
            "errors": {key: observed[key] - prediction[key] for key in observed},
            "body_state": agent._current_body_state(),
        }
        control_prior = build_goal_prior_adjustment(
            active_goal=Goal.CONTROL,
            current_state=current_state,
            goal_context={"active_goal": "CONTROL", "urgency_scores": {"CONTROL": 0.75}},
        )
        social_prior = build_goal_prior_adjustment(
            active_goal=Goal.SOCIAL,
            current_state=current_state,
            goal_context={"active_goal": "SOCIAL", "urgency_scores": {"SOCIAL": 0.75}},
        )
        control_option = agent._project_action(
            action="seek_contact",
            observed=observed,
            prediction=prediction,
            priors=dict(prediction),
            free_energy_before=0.36,
            current_cluster_id=None,
            active_goal=Goal.CONTROL,
            memory_context={"goal_prior": control_prior.to_dict(), "actions": {}},
        )
        social_option = agent._project_action(
            action="seek_contact",
            observed=observed,
            prediction=prediction,
            priors=dict(prediction),
            free_energy_before=0.36,
            current_cluster_id=None,
            active_goal=Goal.SOCIAL,
            memory_context={"goal_prior": social_prior.to_dict(), "actions": {}},
        )
        self.assertNotEqual(
            control_option["goal_prior"]["modality_shifts"],
            social_option["goal_prior"]["modality_shifts"],
        )
        self.assertNotEqual(
            control_option["projected_observation"]["social"],
            social_option["projected_observation"]["social"],
        )

    def test_adaptive_compute_decision_is_deterministic(self) -> None:
        payload = {
            "field": {
                "field_flatness": 0.51,
                "conflict_density": 0.27,
                "counterfactual_audit": {"chosen_decision_subsequent_fe": 0.31},
            },
            "goal_context": {"active_goal": "CONTROL", "urgency_scores": {"CONTROL": 0.41}},
            "prediction_error_surrogate": 0.29,
            "base_retrieval_k": 3,
            "base_path_k": 2,
            "candidate_action_count": 6,
        }
        first = decide_adaptive_compute(**payload).to_dict()
        second = decide_adaptive_compute(**payload).to_dict()
        self.assertEqual(first, second)

    def test_escalation_reduces_subsequent_fe_or_is_audited_as_no_gain(self) -> None:
        no_gain = decide_adaptive_compute(
            field={
                "field_flatness": 0.80,
                "conflict_density": 0.62,
                "counterfactual_audit": {
                    "chosen_decision_subsequent_fe": 0.52,
                    "status": "field_divergent_no_gain",
                },
            },
            goal_context={"active_goal": "SURVIVAL", "urgency_scores": {"SURVIVAL": 0.84}},
            prediction_error_surrogate=0.48,
            base_retrieval_k=3,
            base_path_k=2,
            candidate_action_count=7,
        ).to_dict()
        self.assertTrue(no_gain["escalation_no_gain"])
        self.assertIn("escalation_no_gain", no_gain["escalation_reason_codes"])


if __name__ == "__main__":
    unittest.main()
