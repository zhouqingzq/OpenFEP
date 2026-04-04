from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m4_cognitive_style import (
    BLIND_CLASSIFICATION_FEATURES,
    PROFILE_REGISTRY,
    blind_classification_experiment,
    synthetic_profile_distinguishability_benchmark,
)


class TestM41BlindClassification(unittest.TestCase):
    def test_cross_generator_blind_classification_beats_null_baseline(self) -> None:
        experiment = synthetic_profile_distinguishability_benchmark()
        self.assertEqual(experiment["analysis_type"], "synthetic_holdout_profile_distinguishability")
        self.assertEqual(experiment["legacy_analysis_type"], "cross_generator_blind_classification")
        self.assertEqual(experiment["benchmark_scope"], "sidecar same-framework profile distinguishability on synthetic holdout")
        self.assertEqual(experiment["claim_envelope"], "sidecar_synthetic_diagnostic")
        self.assertEqual(experiment["legacy_status"], "m42_plus_preresearch_sidecar")
        self.assertEqual(experiment["generator_family"], "same_framework_synthetic_holdout")
        self.assertFalse(experiment["external_validation"])
        self.assertEqual(experiment["validation_type"], "synthetic_holdout_same_framework")
        self.assertTrue(experiment["not_acceptance_evidence"])
        self.assertEqual(set(experiment["profiles"]), set(PROFILE_REGISTRY))
        self.assertGreater(experiment["accuracy"], experiment["threshold"])
        self.assertTrue(all(payload["recall"] >= 0.60 for payload in experiment["per_class"].values()))
        artifact = experiment["classifier_artifact"]
        self.assertEqual(artifact["model_type"], "nearest_centroid")
        self.assertEqual(set(artifact["class_centroids"]), set(PROFILE_REGISTRY))
        self.assertEqual(artifact["feature_set"], experiment["feature_set"])
        self.assertEqual(artifact["train_seeds"], experiment["train_eval_split"]["train_seeds"])
        self.assertIn("does not prove M4.2 benchmark recovery", experiment["validation_limits"][-1])

    def test_blind_samples_hide_labels_and_snapshots(self) -> None:
        experiment = synthetic_profile_distinguishability_benchmark()
        for sample in experiment["blind_samples"]:
            self.assertNotIn("true_profile", sample)
            self.assertNotIn("parameter_snapshot", sample)
            self.assertTrue(set(sample["metrics"]).issubset(set(BLIND_CLASSIFICATION_FEATURES)))
        self.assertEqual(experiment["training_summary"], experiment["classifier_artifact"]["training_summary"])

    def test_no_handwritten_profile_routing_remains(self) -> None:
        source = Path("segmentum/m4_cognitive_style.py").read_text(encoding="utf-8")
        self.assertNotIn("_cross_generator_predict", source)
        self.assertNotIn("low_strategy_threshold", source)

    def test_legacy_blind_classification_entrypoint_remains_alias(self) -> None:
        self.assertEqual(blind_classification_experiment(), synthetic_profile_distinguishability_benchmark())


if __name__ == "__main__":
    unittest.main()
