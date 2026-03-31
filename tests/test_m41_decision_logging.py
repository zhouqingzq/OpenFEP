from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import (
    CognitiveStyleParameters,
    PARAMETER_REFERENCE,
    audit_decision_log,
    parameter_causality_matrix,
    reconstruct_behavior_patterns,
    run_cognitive_style_trial,
)


class TestM41DecisionLogging(unittest.TestCase):
    def test_canonical_trace_reconstructs_four_behavior_patterns(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        patterns = reconstruct_behavior_patterns(payload["logs"])
        labels = {item["label"] for item in patterns}
        self.assertIn("directed_exploration", labels)
        self.assertIn("resource_conservation", labels)
        self.assertIn("confidence_sharpening", labels)
        self.assertIn("counterfactual_avoidance", labels)

    def test_resource_ablation_changes_action_sequence(self) -> None:
        full = run_cognitive_style_trial(CognitiveStyleParameters())
        ablated = run_cognitive_style_trial(CognitiveStyleParameters(), ablate_resource_pressure=True)
        self.assertNotEqual(full["summary"]["selected_actions"], ablated["summary"]["selected_actions"])

    def test_parameter_snapshot_is_present_for_each_record(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        for record in payload["logs"]:
            self.assertEqual(set(record["parameter_snapshot"].keys()), set(CognitiveStyleParameters.schema()["required"]))

    def test_parameter_snapshot_is_required_for_log_audit(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        missing_snapshot = dict(payload["logs"][0])
        missing_snapshot.pop("parameter_snapshot", None)
        audit = audit_decision_log([missing_snapshot, *payload["logs"][1:]])
        self.assertGreater(audit["missing_field_counts"]["parameter_snapshot"], 0)
        self.assertLess(audit["parameter_snapshot_complete_rate"], 1.0)
        self.assertGreater(audit["invalid_rate"], 0.0)

    def test_all_eight_parameters_have_independent_causal_probes(self) -> None:
        matrix = parameter_causality_matrix()
        self.assertEqual(set(matrix.keys()), set(PARAMETER_REFERENCE.keys()))
        for parameter_name, probe in matrix.items():
            with self.subTest(parameter=parameter_name):
                self.assertTrue(probe["identifiable"], msg=probe)

    def test_decision_log_completeness_is_within_threshold(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        audit = audit_decision_log(payload["logs"])
        self.assertLessEqual(audit["invalid_rate"], 0.05)
        self.assertEqual(audit["parameter_snapshot_complete_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
