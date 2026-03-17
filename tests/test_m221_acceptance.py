from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m221_benchmarks import run_m221_open_narrative_benchmark, write_m221_acceptance_artifacts


class TestM221Acceptance(unittest.TestCase):
    def test_acceptance_bundle_passes(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        self.assertEqual(payload["milestone_id"], "M2.21")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["recommendation"], "ACCEPT")
        self.assertTrue(all(payload["gates"].values()))

    def test_acceptance_artifact_schema_is_complete(self) -> None:
        written = write_m221_acceptance_artifacts(cycles=24)
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
        for field in (
            "milestone_id",
            "status",
            "recommendation",
            "generated_at",
            "seed_set",
            "artifacts",
            "tests",
            "gates",
            "significant_metric_count",
            "effect_metric_count",
            "per_scenario_breakdown",
            "per_variant_breakdown",
            "residual_risks",
            "freshness",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M2.21")
        self.assertTrue(report["freshness"]["generated_this_round"])


if __name__ == "__main__":
    unittest.main()
