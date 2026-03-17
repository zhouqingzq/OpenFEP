from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m222_benchmarks import write_m222_acceptance_artifacts


class TestM222Acceptance(unittest.TestCase):
    def test_artifact_schema_complete(self) -> None:
        written = write_m222_acceptance_artifacts(seed_set=[222], long_run_cycles=24, restart_pre_cycles=12, restart_post_cycles=12)
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
        summary = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))
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
            "protocol_breakdown",
            "ablation_breakdown",
            "restart_breakdown",
            "stress_recovery_breakdown",
            "residual_risks",
            "freshness",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M2.22")
        self.assertTrue(report["freshness"]["generated_this_round"])
        self.assertEqual(report["protocol_breakdown"], summary["protocol_breakdown"])
        self.assertIn("repair_summary", report)
        self.assertIn("homeostasis_advantage", report["repair_summary"])
        self.assertIn("restart_anchor_repairs", report["repair_summary"])


if __name__ == "__main__":
    unittest.main()
