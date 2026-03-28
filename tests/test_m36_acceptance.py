from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m3_audit import M36_REPORT_PATH, write_m36_acceptance_artifacts


class TestM36Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m36_acceptance_artifacts()
        report = json.loads(Path(M36_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M3.6")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["semantic_growth_controlled"]["passed"])
        self.assertTrue(report["gates"]["process_motivation_observable"]["passed"])
        self.assertTrue(report["gates"]["style_diversity_and_stability"]["passed"])
        self.assertTrue(report["gates"]["restart_continuity"]["passed"])
        self.assertTrue(report["gates"]["open_world_non_scripted"]["passed"])
        self.assertTrue(report["gates"]["bounded_growth_under_stress"]["passed"])


if __name__ == "__main__":
    unittest.main()
