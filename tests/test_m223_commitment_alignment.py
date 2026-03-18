from __future__ import annotations

import unittest

from segmentum.m223_benchmarks import build_m223_scenarios, run_m223_self_consistency_benchmark
from segmentum.self_model import build_default_self_model


class TestM223CommitmentAlignment(unittest.TestCase):
    def test_same_seed_same_protocol_is_deterministic(self) -> None:
        first = run_m223_self_consistency_benchmark(seed_set=[223])
        second = run_m223_self_consistency_benchmark(seed_set=[223])
        self.assertEqual(first["variant_metrics"], second["variant_metrics"])
        self.assertEqual(first["scenario_breakdown"], second["scenario_breakdown"])

    def test_commitments_change_relevant_decision_scoring(self) -> None:
        scenario = build_m223_scenarios()["temptation_conflict"]
        self_model = build_default_self_model()
        self_model.identity_narrative = scenario.narrative
        with_commitments = self_model.assess_action_commitments(
            action="forage",
            projected_state=scenario.conflict_condition.context,
            current_tick=1,
        )
        self_model.commitments_enabled = False
        without_commitments = self_model.assess_action_commitments(
            action="forage",
            projected_state=scenario.conflict_condition.context,
            current_tick=1,
        )
        self.assertLess(float(with_commitments["compatibility_score"]), float(without_commitments["compatibility_score"]))
        self.assertGreater(len(with_commitments["relevant_commitments"]), 0)


if __name__ == "__main__":
    unittest.main()
