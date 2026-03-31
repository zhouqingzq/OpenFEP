from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import PARAMETER_REFERENCE
from segmentum.m41_identifiability import build_identifiability_report


class TestM41Identifiability(unittest.TestCase):
    def test_identifiability_report_covers_recovery_coupling_and_intervals(self) -> None:
        payload = build_identifiability_report()
        self.assertEqual(payload["analysis_type"], "parameter_identifiability_report")
        self.assertEqual(set(payload["parameter_recovery"].keys()), set(PARAMETER_REFERENCE.keys()))
        self.assertEqual(set(payload["parameter_coupling"].keys()), set(PARAMETER_REFERENCE.keys()))
        for parameter_name, report in payload["parameter_recovery"].items():
            with self.subTest(parameter=parameter_name):
                self.assertIn("mean_abs_error", report)
                self.assertIn("identifiable_rate", report)
                self.assertIn("coupled_with", report)
                self.assertIn("non_identifiable_interval", report)
        self.assertGreaterEqual(payload["sample_counts"]["total"], 10)


if __name__ == "__main__":
    unittest.main()
