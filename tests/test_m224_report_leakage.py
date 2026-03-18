from __future__ import annotations

import unittest

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark


class TestM224ReportLeakage(unittest.TestCase):
    def test_non_broadcast_content_does_not_appear_in_full_workspace_report(self) -> None:
        payload = run_m224_workspace_benchmark(seed_set=[SEED_SET[0]])
        full_trial = next(
            trial for trial in payload["trials"] if trial["variant"] == "full_workspace"
        )
        report = full_trial["report"]
        self.assertLessEqual(report["report_leakage_rate"], 0.05)
        self.assertLessEqual(report["suppressed_content_intrusion_rate"], 0.05)
        self.assertTrue(set(report["channels"]).issubset(set(report["broadcast_channels"])))


if __name__ == "__main__":
    unittest.main()
