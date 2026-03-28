from __future__ import annotations

import unittest

from segmentum.m3_open_world_trial import run_open_world_growth_trial


class TestM36OpenWorldGrowth(unittest.TestCase):
    def test_trial_exhibits_growth_process_and_style_diversity(self) -> None:
        payload = run_open_world_growth_trial()
        summary = payload["summary"]
        self.assertGreaterEqual(summary["schema_count_mean"], 3.0)
        self.assertTrue(summary["process_observability"])
        self.assertGreaterEqual(summary["style_label_diversity"], 3)
        self.assertTrue(summary["restart_continuity"])

    def test_flattened_ablation_reduces_growth_and_style_diversity(self) -> None:
        full = run_open_world_growth_trial()
        ablated = run_open_world_growth_trial(ablation_mode="flattened")
        self.assertLess(ablated["summary"]["schema_count_mean"], full["summary"]["schema_count_mean"])
        self.assertLess(ablated["summary"]["style_label_diversity"], full["summary"]["style_label_diversity"])
        self.assertFalse(ablated["summary"]["process_observability"])


if __name__ == "__main__":
    unittest.main()
