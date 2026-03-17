from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m220_benchmarks import run_m220_acceptance_suite, write_m220_acceptance_artifacts


class TestM220Acceptance(unittest.TestCase):
    def test_acceptance_suite_passes_gating_bundle(self) -> None:
        payload = run_m220_acceptance_suite(seed=220, cycles=24, repeats=2)

        self.assertEqual(payload["milestone_id"], "M2.20")
        self.assertTrue(payload["acceptance"]["passed"])
        self.assertGreaterEqual(len(payload["acceptance"]["significant_metrics"]), 3)
        self.assertGreaterEqual(len(payload["acceptance"]["effect_metrics"]), 3)
        self.assertTrue(payload["acceptance"]["causality_passed"])
        self.assertTrue(payload["acceptance"]["ablation_passed"])
        self.assertTrue(payload["acceptance"]["stress_passed"])
        self.assertTrue(payload["acceptance"]["determinism_passed"])

    def test_artifact_writer_emits_required_report_fields(self) -> None:
        written = write_m220_acceptance_artifacts(seed=220, cycles=24, repeats=2)
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))

        for field in (
            "milestone_id",
            "status",
            "generated_at",
            "seed_set",
            "artifacts",
            "tests",
            "gates",
            "findings",
            "residual_risks",
            "freshness",
            "recommendation",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M2.20")
        self.assertEqual(report["recommendation"], "ACCEPT")


if __name__ == "__main__":
    unittest.main()
