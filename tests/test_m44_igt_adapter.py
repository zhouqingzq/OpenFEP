from __future__ import annotations

import unittest

from segmentum.m4_benchmarks import preprocess_iowa_gambling_task, run_iowa_gambling_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM44IgtAdapter(unittest.TestCase):
    def test_preprocess_produces_all_required_splits(self) -> None:
        payload = preprocess_iowa_gambling_task(allow_smoke_test=True)
        splits = {row["split"] for row in payload["trials"]}
        self.assertEqual(payload["manifest"]["benchmark_id"], "iowa_gambling_task")
        self.assertEqual(splits, {"train", "validation", "heldout"})
        self.assertEqual(payload["benchmark_status"]["benchmark_state"], "blocked_missing_external_bundle")

    def test_igt_benchmark_run_is_deterministic(self) -> None:
        first = run_iowa_gambling_benchmark(CognitiveStyleParameters(), seed=44, allow_smoke_test=True)
        second = run_iowa_gambling_benchmark(CognitiveStyleParameters(), seed=44, allow_smoke_test=True)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertIn("deck_match_rate", first["metrics"])


if __name__ == "__main__":
    unittest.main()
