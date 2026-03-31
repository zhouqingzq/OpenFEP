from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import CognitiveStyleParameters, run_cognitive_style_trial
from segmentum.m41_explanations import build_behavior_explanation_report, explain_decision_record
from segmentum.m41_inference import infer_cognitive_style


class TestM41Explanations(unittest.TestCase):
    def test_explanation_layer_outputs_decision_drivers_and_mechanism_graph(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters(), seed=81)
        inference = infer_cognitive_style(payload["logs"])
        report = build_behavior_explanation_report(payload["logs"], parameters=inference["inferred_parameters"])
        decision = explain_decision_record(payload["logs"][0], parameters=inference["inferred_parameters"])

        self.assertEqual(report["analysis_type"], "behavior_explanation_report")
        self.assertTrue(report["dominant_mechanisms"])
        self.assertTrue(report["mechanism_graph"]["edges"])
        self.assertIn("top_drivers", decision)
        self.assertGreaterEqual(len(decision["top_drivers"]), 1)


if __name__ == "__main__":
    unittest.main()
