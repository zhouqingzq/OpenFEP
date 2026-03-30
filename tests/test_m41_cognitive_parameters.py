from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import CognitiveStyleParameters, DecisionLogRecord, run_cognitive_style_trial


class TestM41CognitiveParameters(unittest.TestCase):
    def test_parameters_roundtrip_and_schema_version_are_stable(self) -> None:
        params = CognitiveStyleParameters(exploration_bias=0.81, resource_pressure_sensitivity=0.93)
        restored = CognitiveStyleParameters.from_dict(params.to_dict())
        self.assertEqual(params, restored)
        self.assertEqual(restored.schema_version, "m4.cognitive_style.v1")

    def test_decision_log_roundtrip_preserves_required_fields(self) -> None:
        payload = run_cognitive_style_trial(CognitiveStyleParameters())
        record = DecisionLogRecord.from_dict(payload["logs"][0])
        restored = record.to_dict()
        self.assertEqual(restored["schema_version"], "m4.decision_log.v1")
        self.assertIn("candidate_actions", restored)
        self.assertIn("resource_state", restored)


if __name__ == "__main__":
    unittest.main()
