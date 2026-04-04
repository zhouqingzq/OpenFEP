from __future__ import annotations

import unittest

from segmentum.m41_baselines import (
    baseline_scope_note,
    infer_confidence_only_baseline,
    infer_risk_resource_baseline,
    infer_simple_rl_heuristic_baseline,
    random_baseline_inference,
)
from segmentum.m4_cognitive_style import PARAMETER_REFERENCE, PROFILE_REGISTRY, run_cognitive_style_trial
from segmentum.m41_inference import infer_cognitive_style


class TestM41Baselines(unittest.TestCase):
    def test_baseline_scope_is_explicitly_legacy_same_framework(self) -> None:
        note = baseline_scope_note()
        self.assertEqual(note["claim_envelope"], "sidecar_synthetic_diagnostic")
        self.assertEqual(note["legacy_status"], "m42_plus_preresearch_sidecar")
        self.assertEqual(note["validation_type"], "synthetic_holdout_same_framework")
        self.assertEqual(note["interpretation"], "synthetic-family baseline comparison only")

    def test_baseline_fit_confidence_is_data_driven(self) -> None:
        full_records = run_cognitive_style_trial(PROFILE_REGISTRY["balanced_midline"], seed=41)["logs"]
        sparse_records = full_records[:2]
        alternate_records = run_cognitive_style_trial(PROFILE_REGISTRY["low_exploration_high_caution"], seed=41, stress=True)["logs"]

        for inference_fn in (
            infer_risk_resource_baseline,
            infer_confidence_only_baseline,
            infer_simple_rl_heuristic_baseline,
        ):
            with self.subTest(model=inference_fn.__name__):
                full = inference_fn(full_records)
                sparse = inference_fn(sparse_records)
                alternate = inference_fn(alternate_records)
                self.assertGreaterEqual(full["fit_confidence"], 0.0)
                self.assertLessEqual(full["fit_confidence"], 1.0)
                self.assertGreaterEqual(sparse["fit_confidence"], 0.0)
                self.assertLessEqual(sparse["fit_confidence"], 1.0)
                self.assertNotEqual(full["fit_confidence"], sparse["fit_confidence"])
                self.assertNotEqual(full["fit_confidence"], alternate["fit_confidence"])

    def test_random_baseline_is_controlled_uniform_sampler(self) -> None:
        records_a = run_cognitive_style_trial(PROFILE_REGISTRY["balanced_midline"], seed=41)["logs"]
        records_b = run_cognitive_style_trial(PROFILE_REGISTRY["low_exploration_high_caution"], seed=42, stress=True)["logs"]

        first = random_baseline_inference(records_a)
        second = random_baseline_inference(records_a)
        third = random_baseline_inference(records_b)

        self.assertEqual(first["fit_confidence"], 0.0)
        self.assertEqual(first["sampling_strategy"], "controlled_uniform_0_1")
        self.assertEqual(first["sampling_seed"], second["sampling_seed"])
        self.assertEqual(first["inferred_parameters"], second["inferred_parameters"])
        self.assertNotEqual(first["sampling_seed"], third["sampling_seed"])

        changed = 0
        for parameter_name in PARAMETER_REFERENCE:
            value = first["inferred_parameters"][parameter_name]
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            changed += int(first["inferred_parameters"][parameter_name] != third["inferred_parameters"][parameter_name])
        self.assertGreaterEqual(changed, 1)

    def test_style_inference_and_random_baseline_remain_distinct_on_synthetic_records(self) -> None:
        records = run_cognitive_style_trial(PROFILE_REGISTRY["balanced_midline"], seed=41)["logs"]
        main = infer_cognitive_style(records)
        random = random_baseline_inference(records)
        self.assertGreater(main["fit_confidence"], 0.0)
        self.assertEqual(random["fit_confidence"], 0.0)
        self.assertNotEqual(main["inferred_parameters"], random["inferred_parameters"])
        self.assertEqual(random["model_label"], "random_baseline")


if __name__ == "__main__":
    unittest.main()
