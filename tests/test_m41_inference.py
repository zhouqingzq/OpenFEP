from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import CognitiveStyleParameters, run_cognitive_style_trial
from segmentum.m41_inference import infer_cognitive_style, rank_alternative_explanations, summarize_parameter_recovery


class TestM41Inference(unittest.TestCase):
    def test_inference_recovers_profile_directionally_from_behavior_logs(self) -> None:
        target = CognitiveStyleParameters(
            uncertainty_sensitivity=0.84,
            error_aversion=0.28,
            exploration_bias=0.88,
            attention_selectivity=0.52,
            confidence_gain=0.48,
            update_rigidity=0.34,
            resource_pressure_sensitivity=0.36,
            virtual_prediction_error_gain=0.26,
        )
        payload = run_cognitive_style_trial(target, seed=71)
        inference = infer_cognitive_style(payload["logs"])
        recovery = summarize_parameter_recovery(inference["inferred_parameters"], target.to_dict())

        self.assertEqual(inference["analysis_type"], "behavior_to_style_inference")
        self.assertGreaterEqual(inference["fit_confidence"], 0.55)
        self.assertLessEqual(recovery["mae"], 0.25)
        self.assertEqual(inference["classification"]["predicted_profile"], "high_exploration_low_caution")

    def test_sparse_logs_mark_parameters_unidentifiable(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters(), seed=72)
        inference = infer_cognitive_style(payload["logs"][:2])
        self.assertTrue(inference["unidentifiable_parameters"])
        self.assertTrue(inference["alternative_explanations"])

    def test_alternative_explanations_are_ranked_for_weak_evidence(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters(), seed=73)
        ranked = rank_alternative_explanations(payload["logs"][:2])
        self.assertGreaterEqual(len(ranked), 1)
        self.assertIn("parameter", ranked[0])
        self.assertIn("explanation", ranked[0])


if __name__ == "__main__":
    unittest.main()
