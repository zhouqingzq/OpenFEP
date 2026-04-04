from __future__ import annotations

import json
from pathlib import Path
import unittest

from segmentum.m42_audit import M42_CONFIDENCE_TRACE_PATH, write_m42_acceptance_artifacts
from segmentum.m4_benchmarks import default_acceptance_benchmark_root


class TestM42ConfidenceBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if default_acceptance_benchmark_root() is None:
            raise unittest.SkipTest("external acceptance bundle required")
        write_m42_acceptance_artifacts()
        cls.payload = json.loads(Path(M42_CONFIDENCE_TRACE_PATH).read_text(encoding="utf-8"))

    def test_full_trial_export_matches_trial_count(self) -> None:
        self.assertEqual(self.payload["trial_count"], len(self.payload["trial_trace"]))
        self.assertTrue(self.payload["trial_export_validation"]["ok"])
        self.assertTrue(self.payload["leakage_check"]["ok"])
        self.assertTrue(self.payload["leakage_check"]["subject"]["ok"])
        self.assertTrue(self.payload["leakage_check"]["session"]["ok"])

    def test_human_aligned_fields_are_present(self) -> None:
        row = self.payload["trial_trace"][0]
        for field_name in (
            "trial_id",
            "subject_id",
            "session_id",
            "split",
            "stimulus_strength",
            "evidence_strength",
            "agent_choice",
            "agent_confidence_rating",
            "correct_choice",
            "human_choice",
            "human_confidence",
        ):
            self.assertIn(field_name, row)
        self.assertEqual(row["source_dataset"], self.payload["trial_trace"][0]["source_dataset"])


if __name__ == "__main__":
    unittest.main()
