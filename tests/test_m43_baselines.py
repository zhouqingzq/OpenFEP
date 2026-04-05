from __future__ import annotations

import unittest

from segmentum.m43_baselines import (
    fit_confidence_logistic_baseline,
    run_confidence_human_match_ceiling,
    run_confidence_random_baseline,
    run_confidence_stimulus_only_baseline,
    run_igt_frequency_matching_baseline,
    run_igt_human_behavior_baseline,
)
from segmentum.m43_modeling import run_fitted_confidence_agent
from segmentum.m4_benchmarks import BenchmarkTrial, IowaTrial, default_acceptance_benchmark_root


class TestM43Baselines(unittest.TestCase):
    def _make_igt_trials(
        self,
        *,
        subject_prefix: str,
        split: str,
        trial_index: int,
        deck_counts: dict[str, int],
    ) -> list[IowaTrial]:
        trials: list[IowaTrial] = []
        running = 0
        for deck in "ABCD":
            for count in range(int(deck_counts.get(deck, 0))):
                running += 1
                reward = 100 if deck in {"A", "B"} else 50
                trials.append(
                    IowaTrial(
                        f"{subject_prefix}-{trial_index}-{deck}-{count}",
                        f"{subject_prefix}-{running}",
                        deck,
                        reward,
                        0,
                        reward,
                        deck in {"C", "D"},
                        trial_index,
                        split,
                        f"{subject_prefix}.csv",
                    )
                )
        return trials

    def test_confidence_stimulus_only_baseline_beats_random_on_aligned_known_inputs(self) -> None:
        training = [
            BenchmarkTrial("t1", "s1", "sess1", 0.8, "right", "right", 0.8, 500, "train", "demo", "a.csv"),
            BenchmarkTrial("t2", "s2", "sess2", -0.7, "left", "left", 0.7, 500, "train", "demo", "a.csv"),
        ]
        evaluation = [
            BenchmarkTrial("e1", "s3", "sess3", 0.9, "right", "right", 0.9, 500, "heldout", "demo", "a.csv"),
            BenchmarkTrial("e2", "s4", "sess4", -0.9, "left", "left", 0.9, 500, "heldout", "demo", "a.csv"),
        ]
        stimulus_only = run_confidence_stimulus_only_baseline(training, evaluation)
        random_payload = run_confidence_random_baseline(evaluation, seed=7)
        self.assertGreater(stimulus_only["metrics"]["heldout_likelihood"], random_payload["metrics"]["heldout_likelihood"])
        self.assertGreaterEqual(stimulus_only["fixed_confidence"], 0.5)

    def test_confidence_logistic_baseline_fits_nontrivial_coefficients(self) -> None:
        training = [
            BenchmarkTrial("t1", "s1", "sess1", 0.8, "right", "right", 0.8, 500, "train", "demo", "a.csv"),
            BenchmarkTrial("t2", "s2", "sess2", 0.6, "right", "right", 0.7, 520, "train", "demo", "a.csv"),
            BenchmarkTrial("t3", "s3", "sess3", -0.7, "left", "left", 0.7, 540, "train", "demo", "a.csv"),
            BenchmarkTrial("t4", "s4", "sess4", -0.9, "left", "left", 0.9, 560, "train", "demo", "a.csv"),
        ]
        coefficients = fit_confidence_logistic_baseline(training, seed=11)
        self.assertGreater(coefficients["choice_slope"], 0.0)
        self.assertNotEqual(coefficients["confidence_slope"], 0.0)

    def test_human_match_ceiling_uses_condition_majority_vote(self) -> None:
        training = [
            BenchmarkTrial("t1", "s1", "sess1", 0.5, "right", "right", 0.7, 500, "train", "demo", "a.csv"),
            BenchmarkTrial("t2", "s2", "sess2", 0.5, "right", "right", 0.6, 510, "train", "demo", "a.csv"),
            BenchmarkTrial("t3", "s3", "sess3", 0.5, "right", "left", 0.3, 520, "train", "demo", "a.csv"),
        ]
        evaluation = [BenchmarkTrial("e1", "s4", "sess4", 0.5, "right", "right", 0.9, 530, "heldout", "demo", "a.csv")]
        payload = run_confidence_human_match_ceiling(training, evaluation)
        self.assertEqual(payload["predictions"][0]["predicted_choice"], "right")
        self.assertGreater(payload["predictions"][0]["predicted_probability_right"], 0.5)

    def test_igt_human_behavior_baseline_uses_training_distribution(self) -> None:
        training = [
            IowaTrial("t1", "s1", "A", 100, 0, 100, False, 1, "train", "s1.csv"),
            IowaTrial("t2", "s2", "A", 100, 0, 100, False, 1, "train", "s2.csv"),
            IowaTrial("t3", "s3", "C", 50, 0, 50, True, 1, "train", "s3.csv"),
        ]
        evaluation = [IowaTrial("e1", "h1", "A", 100, 0, 100, False, 1, "heldout", "h1.csv")]
        payload = run_igt_human_behavior_baseline(training, evaluation)
        self.assertEqual(payload["predictions"][0]["chosen_deck"], "A")
        self.assertGreater(payload["predictions"][0]["predicted_confidence"], 0.5)

    def test_igt_frequency_matching_samples_trial_index_human_frequencies(self) -> None:
        training = self._make_igt_trials(
            subject_prefix="train",
            split="train",
            trial_index=7,
            deck_counts={"A": 8, "B": 4, "C": 3, "D": 1},
        )
        evaluation = [
            IowaTrial(f"e{i}", f"h{i}", "A", 100, 0, 100, False, 7, "heldout", "heldout.csv")
            for i in range(2000)
        ]

        payload = run_igt_frequency_matching_baseline(training, evaluation, seed=19)

        chosen_counts = {deck: 0 for deck in "ABCD"}
        for row in payload["predictions"]:
            chosen_counts[row["chosen_deck"]] += 1
        chosen_rates = {deck: chosen_counts[deck] / len(payload["predictions"]) for deck in "ABCD"}
        target_rates = {"A": 0.5, "B": 0.25, "C": 0.1875, "D": 0.0625}

        self.assertEqual(payload["model_label"], "frequency_matching")
        for deck, target in target_rates.items():
            self.assertAlmostEqual(chosen_rates[deck], target, delta=0.03)
        self.assertAlmostEqual(
            payload["fitted_model"]["deck_probabilities_by_trial_index"][7]["A"],
            0.5,
        )

    def test_igt_frequency_matching_uses_trial_index_distribution_not_simulated_rewards(self) -> None:
        training = []
        training.extend(self._make_igt_trials(subject_prefix="train", split="train", trial_index=1, deck_counts={"C": 5}))
        training.extend(self._make_igt_trials(subject_prefix="train", split="train", trial_index=2, deck_counts={"A": 5}))
        evaluation = [
            IowaTrial("e1", "h1", "A", 100, 0, 100, False, 1, "heldout", "heldout.csv"),
            IowaTrial("e2", "h1", "A", 100, 0, 100, False, 2, "heldout", "heldout.csv"),
        ]

        payload = run_igt_frequency_matching_baseline(training, evaluation, seed=5)

        self.assertEqual([row["chosen_deck"] for row in payload["predictions"]], ["C", "A"])
        self.assertEqual([row["predicted_confidence"] for row in payload["predictions"]], [1.0, 1.0])
        self.assertEqual(payload["predictions"][0]["reward"], 50)
        self.assertEqual(payload["predictions"][1]["reward"], 100)

    def test_igt_frequency_matching_falls_back_to_global_human_frequency(self) -> None:
        training = []
        training.extend(self._make_igt_trials(subject_prefix="train", split="train", trial_index=1, deck_counts={"A": 3, "D": 1}))
        training.extend(self._make_igt_trials(subject_prefix="train", split="train", trial_index=2, deck_counts={"A": 3, "D": 1}))
        evaluation = [
            IowaTrial(f"e{i}", f"h{i}", "A", 100, 0, 100, False, 99, "heldout", "heldout.csv")
            for i in range(800)
        ]

        payload = run_igt_frequency_matching_baseline(training, evaluation, seed=23)

        chosen_counts = {deck: 0 for deck in "ABCD"}
        for row in payload["predictions"]:
            chosen_counts[row["chosen_deck"]] += 1
        self.assertAlmostEqual(chosen_counts["A"] / len(payload["predictions"]), 0.75, delta=0.04)
        self.assertAlmostEqual(chosen_counts["D"] / len(payload["predictions"]), 0.25, delta=0.04)
        self.assertEqual(payload["fitted_model"]["global_deck_probabilities"]["A"], 0.75)
        self.assertEqual(payload["fitted_model"]["global_deck_probabilities"]["D"], 0.25)

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_baseline_tracks_exist_in_real_fit_payload(self) -> None:
        payload = run_fitted_confidence_agent(
            benchmark_root=default_acceptance_benchmark_root(),
            sample_limits={
                "confidence_train_max_trials": 1000,
                "confidence_validation_max_trials": 1000,
                "confidence_heldout_max_trials": 1200,
            },
        )
        self.assertEqual(payload["source_type"], "external_bundle")
        self.assertEqual(payload["baselines"]["random"]["trial_count"], payload["trial_count"])
        self.assertEqual(payload["baselines"]["stimulus_only"]["trial_count"], payload["trial_count"])


if __name__ == "__main__":
    unittest.main()
