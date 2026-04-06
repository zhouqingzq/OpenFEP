from __future__ import annotations

import json
import unittest

from segmentum.m48_audit import M48_REPORT_PATH, write_m48_acceptance_artifacts
from segmentum.m48_open_world import benchmark_open_world_projection, simulate_open_world_projection
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM48ParameterProjection(unittest.TestCase):
    def test_projection_run_is_deterministic(self) -> None:
        first = simulate_open_world_projection(CognitiveStyleParameters(), seed=45)
        second = simulate_open_world_projection(CognitiveStyleParameters(), seed=45)
        self.assertEqual(first, second)

    def test_profile_mapping_is_observable(self) -> None:
        payload = benchmark_open_world_projection(seed=45)
        self.assertTrue(all(payload["correspondence"].values()))
        self.assertEqual(payload["probe_type"], "synthetic_probe")
        self.assertTrue(payload["live_cli_loop"]["summary"]["live_integration"])

    def test_selected_action_matches_top_scored_candidate(self) -> None:
        payload = simulate_open_world_projection(CognitiveStyleParameters(), seed=45)
        for row in payload["logs"]:
            top_candidate = max(row["decision"]["candidate_actions"], key=lambda item: item["total_score"])
            self.assertEqual(row["decision"]["selected_action"], top_candidate["action"]["name"])

    def test_report_marks_synthetic_probe_explicitly(self) -> None:
        write_m48_acceptance_artifacts()
        payload = json.loads(M48_REPORT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(payload["headline_metrics"]["synthetic_probe"])
        self.assertTrue(payload["headline_metrics"]["live_integration"])


if __name__ == "__main__":
    unittest.main()
