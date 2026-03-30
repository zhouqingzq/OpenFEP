from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import CognitiveStyleParameters, reconstruct_behavior_patterns, run_cognitive_style_trial


class TestM41DecisionLogging(unittest.TestCase):
    def test_canonical_trace_reconstructs_three_behavior_patterns(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        patterns = reconstruct_behavior_patterns(payload["logs"])
        labels = {item["label"] for item in patterns}
        self.assertIn("directed_exploration", labels)
        self.assertIn("resource_conservation", labels)
        self.assertIn("confidence_sharpening", labels)

    def test_resource_ablation_changes_action_sequence(self) -> None:
        full = run_cognitive_style_trial(CognitiveStyleParameters())
        ablated = run_cognitive_style_trial(CognitiveStyleParameters(), ablate_resource_pressure=True)
        self.assertNotEqual(full["summary"]["selected_actions"], ablated["summary"]["selected_actions"])


if __name__ == "__main__":
    unittest.main()
