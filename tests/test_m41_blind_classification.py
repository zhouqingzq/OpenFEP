from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import (
    BLIND_CLASSIFICATION_FEATURES,
    PROFILE_REGISTRY,
    blind_classification_experiment,
)


class TestM41BlindClassification(unittest.TestCase):
    def test_blind_classification_uses_split_and_beats_baseline(self) -> None:
        experiment = blind_classification_experiment()
        self.assertEqual(experiment["analysis_type"], "toy_internal_distinguishability")
        self.assertEqual(experiment["generator_family"], "same_generator_family")
        self.assertFalse(experiment["external_validation"])
        self.assertGreaterEqual(len(experiment["profiles"]), 3)
        self.assertEqual(set(experiment["profiles"]), set(PROFILE_REGISTRY))
        self.assertIn("train_seeds", experiment["train_eval_split"])
        self.assertIn("eval_seeds", experiment["train_eval_split"])
        self.assertGreaterEqual(experiment["accuracy"], 0.80)
        self.assertGreater(experiment["accuracy"], experiment["baseline_accuracy"])

    def test_blind_samples_hide_true_profile_labels(self) -> None:
        experiment = blind_classification_experiment()
        for sample in experiment["blind_samples"]:
            self.assertNotIn("true_profile", sample)
            self.assertNotIn("parameter_snapshot", sample)
            self.assertEqual(set(sample["metrics"]).issubset(set(BLIND_CLASSIFICATION_FEATURES)), True)

    def test_per_class_recall_and_feature_set_are_reported(self) -> None:
        experiment = blind_classification_experiment()
        self.assertEqual(experiment["feature_set"], BLIND_CLASSIFICATION_FEATURES)
        self.assertTrue(any("toy benchmark" in item for item in experiment["validation_limits"]))
        self.assertTrue(any("same generator family" in item for item in experiment["validation_limits"]))
        self.assertTrue(any("external blind validation" in item for item in experiment["validation_limits"]))
        self.assertTrue(all(payload["recall"] >= 0.75 for payload in experiment["per_class"].values()))


if __name__ == "__main__":
    unittest.main()
