from __future__ import annotations

import unittest

from segmentum.m224_benchmarks import run_m224_open_world_runtime_probe, run_m224_runtime_integration_probe


class TestM224RuntimeIntegration(unittest.TestCase):
    def test_runtime_probe_proves_real_workspace_wiring(self) -> None:
        payload = run_m224_runtime_integration_probe()
        self.assertTrue(payload["passed"])
        self.assertNotEqual(
            payload["full_workspace"]["choice"],
            payload["no_workspace"]["choice"],
        )
        full_report = payload["full_workspace"]["conscious_report"]
        self.assertTrue(full_report["leakage_free"])
        self.assertTrue(
            set(full_report["channels"]).isdisjoint(set(full_report["suppressed_channels"]))
        )
        maintenance = payload["full_workspace"]["explanation_details"]["workspace_maintenance_priority"]
        self.assertGreater(maintenance["priority_gain"], 0.0)
        self.assertTrue(payload["full_workspace"]["agenda"]["active_tasks"])
        review = payload["full_workspace"]["metacognitive_review"]
        self.assertTrue(review["review_required"])
        self.assertTrue(review["workspace_conflict_channels"])
        self.assertTrue(payload["full_workspace"]["semantic_report"]["semantic_leakage_free"])

    def test_open_world_runtime_probe_proves_bounded_real_world_evidence(self) -> None:
        payload = run_m224_open_world_runtime_probe()
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["checks"]["workspace_trace_present"])
        self.assertTrue(payload["checks"]["workspace_bias_observed"])
        self.assertTrue(payload["checks"]["workspace_absent_when_disabled"])
        self.assertTrue(payload["checks"]["maintenance_priority_delta_present"])
        self.assertTrue(payload["checks"]["semantic_report_leakage_free"])


if __name__ == "__main__":
    unittest.main()
