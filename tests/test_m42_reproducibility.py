from __future__ import annotations

import json
from pathlib import Path
import unittest

from segmentum.m42_audit import M42_REPRO_PATH, write_m42_acceptance_artifacts
from segmentum.m4_benchmarks import compute_behavioral_seed_summaries, default_acceptance_benchmark_root, same_seed_triple_replay


class TestM42Reproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if default_acceptance_benchmark_root() is None:
            raise unittest.SkipTest("external acceptance bundle required")
        write_m42_acceptance_artifacts()
        cls.report = json.loads(Path(M42_REPRO_PATH).read_text(encoding="utf-8"))
        cls.selected_subject_id = cls.report["replay_basis"]["selected_subject_id"]

    def test_same_seed_triple_replay_is_exact_on_external_igt(self) -> None:
        replay = same_seed_triple_replay(
            "iowa_gambling_task",
            seed=44,
            run_kwargs={
                "benchmark_root": default_acceptance_benchmark_root(),
                "selected_subject_id": self.selected_subject_id,
                "protocol_mode": "standard_100",
                "include_predictions": False,
                "include_subject_summary": False,
            },
        )
        self.assertTrue(replay["exact_match"])

    def test_different_seeds_produce_different_external_igt_sequences(self) -> None:
        summary = compute_behavioral_seed_summaries(
            "iowa_gambling_task",
            seeds=[41, 42, 43, 45],
            run_kwargs={
                "benchmark_root": default_acceptance_benchmark_root(),
                "selected_subject_id": self.selected_subject_id,
                "protocol_mode": "standard_100",
                "include_predictions": False,
                "include_subject_summary": False,
            },
        )
        self.assertTrue(summary["different_seeds_differ"])
        self.assertGreater(summary["unique_behavior_sequences"], 1)
        self.assertGreater(summary["sequence_diff_summary"]["max_diff_count"], 0)
        self.assertGreaterEqual(summary["behavioral_evidence"]["varying_behavior_metric_count"], 2)
        self.assertIn("mean_confidence", summary["behavioral_summary"])
        self.assertIn("reward_fit", summary["behavioral_summary"])
        self.assertTrue(summary["reproducibility_gate"]["passed"])
        row = summary["seed_summaries"][0]
        self.assertIn("deck_distribution", row)
        self.assertIn("mean_confidence", row)
        self.assertIn("reward_fit", row)

    def test_generated_repro_report_is_external_bundle_anchored(self) -> None:
        self.assertEqual(self.report["replay_basis"]["benchmark_id"], "iowa_gambling_task")
        self.assertEqual(self.report["replay_basis"]["source_type"], "external_bundle")
        self.assertFalse(self.report["replay_basis"]["smoke_test_only"])
        self.assertFalse(self.report["replay_basis"]["is_synthetic"])
        self.assertTrue(self.report["replay_basis"]["acceptance_requires_external_bundle"])
        self.assertGreater(self.report["different_seed_behavior"]["sequence_diff_summary"]["max_diff_count"], 0)
        self.assertGreaterEqual(self.report["different_seed_behavior"]["behavioral_evidence"]["varying_behavior_metric_count"], 2)
        self.assertTrue(self.report["different_seed_behavior"]["reproducibility_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
