from __future__ import annotations

import unittest

from segmentum.m4_benchmarks import run_confidence_database_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM42ConfidenceBenchmark(unittest.TestCase):
    def test_benchmark_run_is_deterministic_on_fixed_seed(self) -> None:
        first = run_confidence_database_benchmark(CognitiveStyleParameters(), seed=42)
        second = run_confidence_database_benchmark(CognitiveStyleParameters(), seed=42)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["predictions"], second["predictions"])

    def test_neutral_ablation_underperforms_full_profile(self) -> None:
        full = run_confidence_database_benchmark(CognitiveStyleParameters(), seed=42)
        neutral = run_confidence_database_benchmark(
            CognitiveStyleParameters(resource_pressure_sensitivity=0.0, confidence_gain=0.5, error_aversion=0.5),
            seed=42,
        )
        self.assertGreater(full["metrics"]["heldout_likelihood"], neutral["metrics"]["heldout_likelihood"])


if __name__ == "__main__":
    unittest.main()
