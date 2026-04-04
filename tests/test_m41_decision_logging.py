from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import (
    CognitiveStyleParameters,
    PARAMETER_REFERENCE,
    audit_decision_log,
    compute_trial_variation,
    parameter_intervention_sensitivity_matrix,
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

    def test_resource_ablation_changes_behavior_distribution(self) -> None:
        full = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)
        ablated = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41, ablate_resource_pressure=True)
        variation = compute_trial_variation(full, ablated)
        self.assertTrue(variation["varies"], msg=variation)

    def test_seed_changes_trial_behavior(self) -> None:
        first = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)
        second = run_cognitive_style_trial(CognitiveStyleParameters(), seed=42)
        variation = compute_trial_variation(first, second)
        self.assertTrue(variation["varies"], msg=variation)

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

    def test_log_audit_rejects_out_of_range_and_semantically_invalid_values(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        corrupted = dict(payload["logs"][0])
        corrupted["internal_confidence"] = 1.5
        corrupted["attention_allocation"] = {"evidence": 0.9, "uncertainty": 0.9}
        corrupted["selected_action"] = "teleport"
        audit = audit_decision_log([corrupted, *payload["logs"][1:]])
        self.assertGreater(audit["invalid_value_counts"]["internal_confidence"], 0)
        self.assertGreater(audit["semantic_invalid_counts"]["attention_allocation_not_normalized"], 0)
        self.assertGreater(audit["semantic_invalid_counts"]["selected_action_unknown"], 0)
        self.assertGreater(audit["invalid_rate"], 0.0)

    def test_all_eight_parameters_have_independent_intervention_probes(self) -> None:
        matrix = parameter_intervention_sensitivity_matrix()
        self.assertEqual(set(matrix.keys()), set(PARAMETER_REFERENCE.keys()))
        for parameter_name, probe in matrix.items():
            with self.subTest(parameter=parameter_name):
                self.assertEqual(probe["analysis_type"], "intervention_sensitivity")
                self.assertTrue(probe["identifiable"], msg=probe)

    def test_decision_log_completeness_is_within_threshold(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        audit = audit_decision_log(payload["logs"])
        self.assertLessEqual(audit["invalid_rate"], 0.05)
        self.assertEqual(audit["parameter_snapshot_complete_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
