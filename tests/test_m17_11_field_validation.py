from __future__ import annotations

import unittest
from pathlib import Path

from segmentum.m17_11_field_ablation import (
    load_field_validation_corpus,
    render_field_validation_report,
    run_field_validation,
)


FIXTURE_DIR = Path("fixtures/m17_11")
TRAIN_PATH = FIXTURE_DIR / "train_corpus.json"
HELD_OUT_PATH = FIXTURE_DIR / "held_out_corpus.json"


class FieldValidationHarnessTests(unittest.TestCase):
    def test_corpus_train_and_held_out_do_not_overlap(self) -> None:
        corpus = load_field_validation_corpus(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
        )
        train_ids = {item["fixture_id"] for item in corpus["train"]}
        held_out_ids = {item["fixture_id"] for item in corpus["held_out"]}
        self.assertFalse(train_ids & held_out_ids)

    def test_replay_is_deterministic_for_same_corpus_and_seed(self) -> None:
        first = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=7,
        )
        second = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=7,
        )
        self.assertEqual(first, second)

    def test_field_advantage_metrics_use_m17_5_surrogate_as_outcome(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        self.assertEqual(
            result["held_out_metrics"]["outcome_quantity"],
            "m17_5_expected_free_energy_surrogate",
        )

    def test_baselines_include_best_single_naive_topk_and_field_off(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        row = result["held_out_metrics"]["paired_rows"][0]
        self.assertIn("best_single_fe", row)
        self.assertIn("naive_topk_fe", row)
        self.assertIn("field_off_fe", row)

    def test_regression_rate_is_reported_and_nonhidden(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        metrics = result["held_out_metrics"]
        self.assertIn("regression_rate", metrics)
        self.assertGreater(metrics["regression_rate"], 0.0)
        self.assertIn("p90_regression_magnitude", metrics)

    def test_multi_cycle_trajectory_declines_more_than_frozen_memory_control(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        self.assertLess(
            result["trajectory"]["full_loop_mean_slope"],
            result["trajectory"]["frozen_memory_mean_slope"],
        )

    def test_ablation_matrix_toggles_one_component_at_a_time(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        names = [item["ablation"] for item in result["ablation_matrix"]]
        self.assertEqual(
            names,
            [
                "m17_6_credit",
                "m17_7_reconsolidation",
                "m17_8_paths",
                "m17_9_field",
                "m17_10_goal_priors",
                "m17_10_adaptive_compute",
            ],
        )

    def test_ablating_field_falls_back_without_crash(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        field_ablation = next(
            item for item in result["ablation_matrix"] if item["ablation"] == "m17_9_field"
        )
        self.assertIn("held_out_metrics", field_ablation)
        self.assertIn("paired_rows", field_ablation["held_out_metrics"])

    def test_component_with_no_measurable_contribution_is_flagged(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        goal_prior_ablation = next(
            item for item in result["ablation_matrix"] if item["ablation"] == "m17_10_goal_priors"
        )
        self.assertTrue(goal_prior_ablation["component_no_measurable_contribution"])

    def test_metrics_fit_on_train_are_reported_on_held_out_only(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        self.assertEqual(result["params_fit_on"], "train")
        self.assertEqual(result["metrics_reported_on"], "held_out")
        self.assertFalse(result["fixtures_overlap"])

    def test_leakage_between_fit_and_report_fixtures_is_detected(self) -> None:
        result = run_field_validation(
            train_path=HELD_OUT_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        self.assertTrue(result["leakage_detected"])
        self.assertTrue(result["fixtures_overlap"])

    def test_report_preserves_honesty_about_negative_result(self) -> None:
        result = run_field_validation(
            train_path=TRAIN_PATH,
            held_out_path=HELD_OUT_PATH,
            seed=0,
        )
        self.assertLessEqual(result["held_out_metrics"]["mean_fe_advantage_vs_naive_topk"], 0.0)
        report = render_field_validation_report(result)
        self.assertIn("regression_rate", report)
        self.assertIn("M17.5 expected-free-energy surrogate", report)


if __name__ == "__main__":
    unittest.main()
