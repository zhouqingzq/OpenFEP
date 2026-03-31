from __future__ import annotations

import unittest

from segmentum.m4_benchmarks import compute_behavioral_seed_summaries, evaluate_seed_tolerance_gate, same_seed_triple_replay


class TestM42Reproducibility(unittest.TestCase):
    def test_same_seed_triple_replay_is_exact(self) -> None:
        replay = same_seed_triple_replay(
            "two_armed_bandit",
            seed=91,
            run_kwargs={"trial_count": 50, "include_predictions": False, "include_subject_summary": False},
        )
        self.assertTrue(replay["exact_match"])

    def test_different_seeds_produce_different_behavioral_summaries(self) -> None:
        summary = compute_behavioral_seed_summaries(
            "two_armed_bandit",
            seeds=[1, 2, 3, 4, 5, 6, 7, 8],
            run_kwargs={"trial_count": 50, "include_predictions": False, "include_subject_summary": False},
        )
        self.assertTrue(summary["different_seeds_differ"])
        self.assertGreater(summary["behavioral_summary"]["mean_reward"]["variance"], 0.0)

    def test_bootstrap_tolerance_gate_passes_for_seed_level_mean_reward(self) -> None:
        summary = compute_behavioral_seed_summaries(
            "two_armed_bandit",
            seeds=[1, 2, 3, 4, 5, 6, 7, 8],
            run_kwargs={"trial_count": 50, "include_predictions": False, "include_subject_summary": False},
        )
        gate = evaluate_seed_tolerance_gate(
            summary["seed_summaries"],
            metric_name="mean_reward",
            bootstrap_seed=99,
            lower_bound=0.55,
            upper_bound=0.8,
            min_variance=0.0001,
        )
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
