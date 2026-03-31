from __future__ import annotations

import json
from pathlib import Path
import unittest

from segmentum.m42_audit import M42_CONFIDENCE_TRACE_PATH, M42_IGT_TRACE_PATH, M42_REPRO_PATH, M42_REPORT_PATH, write_m42_acceptance_artifacts
from segmentum.m4_benchmarks import default_acceptance_benchmark_root


class TestM42Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        write_m42_acceptance_artifacts()
        cls.report = json.loads(Path(M42_REPORT_PATH).read_text(encoding="utf-8"))

    def test_acceptance_report_tracks_current_status(self) -> None:
        root = default_acceptance_benchmark_root()
        if root is None:
            self.assertEqual(self.report["status"], "FAIL")
            self.assertEqual(self.report["acceptance_state"], "blocked_missing_external_bundle")
            return
        self.assertEqual(self.report["status"], "PASS")
        self.assertEqual(self.report["acceptance_state"], "acceptance_pass")
        self.assertEqual(self.report["tracks"]["confidence_database"]["status"], "pass")
        self.assertEqual(self.report["tracks"]["iowa_gambling_task"]["status"], "pass")
        self.assertEqual(self.report["tracks"]["two_armed_bandit"]["status"], "pass")
        self.assertEqual(self.report["tracks"]["reproducibility"]["status"], "pass")

    def test_acceptance_artifacts_exist(self) -> None:
        for path in (M42_REPORT_PATH, M42_CONFIDENCE_TRACE_PATH, M42_IGT_TRACE_PATH, M42_REPRO_PATH):
            self.assertTrue(Path(path).exists(), str(path))


if __name__ == "__main__":
    unittest.main()
