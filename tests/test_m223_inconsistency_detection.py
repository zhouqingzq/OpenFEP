from __future__ import annotations

import unittest

from segmentum.m223_benchmarks import _evaluate_condition, build_m223_scenarios
from segmentum.self_model import build_default_self_model


class TestM223InconsistencyDetection(unittest.TestCase):
    def test_conflict_is_detected_and_classified(self) -> None:
        scenario = build_m223_scenarios()["stress_drift"]
        self_model = build_default_self_model()
        self_model.identity_narrative = scenario.narrative
        assessment = self_model.assess_action_commitments(
            action="forage",
            projected_state=scenario.conflict_condition.context,
            current_tick=3,
        )
        self.assertGreater(float(assessment["self_inconsistency_error"]), 0.15)
        self.assertEqual(assessment["conflict_type"], "stress_drift")
        self.assertIn(assessment["severity_level"], {"medium", "high"})

    def test_irrelevant_context_does_not_spuriously_trigger(self) -> None:
        scenario = build_m223_scenarios()["social_contradiction"]
        self_model = build_default_self_model()
        self_model.identity_narrative = scenario.narrative
        assessment = self_model.assess_action_commitments(
            action="seek_contact",
            projected_state=scenario.aligned_condition.context,
            current_tick=2,
        )
        self.assertFalse(bool(assessment["repair_triggered"]))
        self.assertNotEqual(assessment["consistency_classification"], "self_conflict")
        self.assertIn(assessment["behavioral_classification"], {"aligned", "healthy_adaptation", "over_rigidity"})

    def test_aligned_nonzero_error_is_not_auto_counted_as_detected(self) -> None:
        scenario = build_m223_scenarios()["adaptation_vs_betrayal"]
        record = _evaluate_condition(
            seed=223,
            scenario=scenario,
            condition=scenario.aligned_condition,
            commitments_enabled=True,
            repair_enabled=True,
            tick=1,
        )
        self.assertFalse(record["actual_conflict_event"])
        self.assertFalse(record["detected"])
        self.assertFalse(record["detection_basis"]["counted"])

    def test_adaptation_vs_betrayal_preserves_internal_behavior_classification(self) -> None:
        scenario = build_m223_scenarios()["adaptation_vs_betrayal"]
        self_model = build_default_self_model()
        self_model.identity_narrative = scenario.narrative
        aligned = self_model.assess_action_commitments(
            action="scan",
            projected_state=scenario.aligned_condition.context,
            current_tick=1,
        )
        conflict = self_model.assess_action_commitments(
            action="rest",
            projected_state=scenario.conflict_condition.context,
            current_tick=2,
        )
        self.assertEqual(aligned["behavioral_classification"], "healthy_adaptation")
        self.assertIn(conflict["behavioral_classification"], {"over_rigidity", "narrative_rationalization", "temporary_deviation", "self_conflict"})
        self.assertNotEqual(conflict["consistency_classification"], "aligned")


if __name__ == "__main__":
    unittest.main()
