from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m223_benchmarks import SEED_SET, write_m223_acceptance_artifacts


class TestM223Acceptance(unittest.TestCase):
    def test_acceptance_artifact_schema_complete(self) -> None:
        written = write_m223_acceptance_artifacts(seed_set=list(SEED_SET))
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
        trace = json.loads(Path(written["trace"]).read_text(encoding="utf-8"))
        for field in (
            "milestone_id",
            "status",
            "recommendation",
            "generated_at",
            "seed_set",
            "artifacts",
            "tests",
            "scenario_definitions",
            "variant_metrics",
            "scenario_breakdown",
            "repair_breakdown",
            "stress_breakdown",
            "bounded_update_breakdown",
            "chapter_transition_breakdown",
            "protocol_integrity",
            "sample_independence_checks",
            "metric_counting_rules",
            "gates",
            "goal_details",
            "residual_risks",
            "freshness",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M2.23")
        self.assertTrue(report["freshness"]["generated_this_round"])
        self.assertEqual(report["seed_set"], SEED_SET)
        self.assertTrue(report["protocol_integrity"]["seed_set_complete"])
        self.assertTrue(report["protocol_integrity"]["scenario_set_complete"])
        self.assertTrue(report["protocol_integrity"]["condition_set_complete"])
        self.assertIn("self_inconsistency_detection_rate", report["metric_counting_rules"])
        self.assertIn("repair_success_rate", report["metric_counting_rules"])
        sample = trace["trace"][0]
        for field in (
            "active_commitments",
            "relevant_commitments",
            "commitment_compatibility_score",
            "self_inconsistency_error",
            "conflict_type",
            "severity_level",
            "behavioral_classification",
            "repair_triggered",
            "repair_policy",
            "repair_result",
            "actual_conflict_event",
            "detection_basis",
            "false_positive_basis",
            "repair_success_basis",
        ):
            self.assertIn(field, sample)


if __name__ == "__main__":
    unittest.main()
