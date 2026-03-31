from __future__ import annotations

import json
from pathlib import Path
import unittest

from segmentum.m42_audit import M42_IGT_TRACE_PATH, write_m42_acceptance_artifacts
from segmentum.m4_benchmarks import STANDARD_IGT_TRIAL_COUNT, default_acceptance_benchmark_root


class TestM44IgtAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if default_acceptance_benchmark_root() is None:
            raise unittest.SkipTest("external acceptance bundle required")
        write_m42_acceptance_artifacts()
        cls.payload = json.loads(Path(M42_IGT_TRACE_PATH).read_text(encoding="utf-8"))

    def test_standard_protocol_is_exactly_100_trials(self) -> None:
        self.assertEqual(self.payload["protocol_validation"]["standard_trial_count"], STANDARD_IGT_TRIAL_COUNT)
        self.assertEqual(self.payload["trial_count"], STANDARD_IGT_TRIAL_COUNT)
        self.assertEqual(self.payload["trial_trace"][0]["trial_index"], 1)
        self.assertEqual(self.payload["trial_trace"][-1]["trial_index"], STANDARD_IGT_TRIAL_COUNT)

    def test_trial_records_include_cumulative_gain_and_internal_state(self) -> None:
        row = self.payload["trial_trace"][0]
        self.assertIn("cumulative_gain", row)
        self.assertIn("internal_state_snapshot", row)
        self.assertTrue(row["internal_state_snapshot"])
        for key in ("deck_history", "value_estimates", "last_outcome", "loss_streak", "confidence"):
            self.assertIn(key, row["internal_state_snapshot"])


if __name__ == "__main__":
    unittest.main()
