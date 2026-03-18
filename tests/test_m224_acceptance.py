from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark, write_m224_acceptance_artifacts


class TestM224Acceptance(unittest.TestCase):
    def test_same_seed_same_workspace_protocol_is_deterministic(self) -> None:
        left = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
        right = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
        self.assertEqual(left["acceptance_report"]["goal_details"], right["acceptance_report"]["goal_details"])

    def test_acceptance_artifact_schema_complete(self) -> None:
        written = write_m224_acceptance_artifacts(seed_set=list(SEED_SET))
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
        artifact = json.loads(Path(written["workspace_causality"]).read_text(encoding="utf-8"))
        for field in (
            "milestone_id",
            "status",
            "recommendation",
            "generated_at",
            "codebase_version",
            "seed_set",
            "protocols",
            "variants",
            "variant_metrics",
            "paired_comparisons",
            "significant_metric_count",
            "effect_metric_count",
            "integration_breakdown",
            "protocol_breakdown",
            "report_breakdown",
            "capacity_breakdown",
            "persistence_breakdown",
            "downstream_causality_breakdown",
            "gates",
            "goal_details",
            "artifacts",
            "residual_risks",
            "freshness",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M2.24")
        self.assertTrue(report["freshness"]["generated_this_round"])
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["determinism"])
        self.assertTrue(report["gates"]["artifact_schema_complete"])
        self.assertTrue(report["gates"]["runtime_integration"])
        self.assertTrue(report["gates"]["semantic_report_leakage"])
        self.assertIn("variant_metrics", artifact)
        self.assertIn("paired_comparisons", artifact)
        self.assertIn("runtime_integration", artifact)
        self.assertIn("open_world_runtime", artifact)


if __name__ == "__main__":
    unittest.main()
