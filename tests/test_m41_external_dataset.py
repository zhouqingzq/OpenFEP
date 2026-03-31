from __future__ import annotations

import unittest

from segmentum.m41_external_dataset import load_external_behavior_dataset, normalize_external_record


class TestM41ExternalDataset(unittest.TestCase):
    def test_external_dataset_loads_and_groups_sessions(self) -> None:
        dataset = load_external_behavior_dataset()
        self.assertEqual(dataset["schema_version"], "m41.external.dataset.v1")
        self.assertGreaterEqual(dataset["session_count"], 6)
        self.assertGreaterEqual(dataset["record_count"], 36)
        self.assertIn("ethnography_lab", dataset["sources"])

    def test_normalization_maps_external_event_to_decision_log_schema(self) -> None:
        dataset = load_external_behavior_dataset()
        session = dataset["sessions"][0]
        normalized = normalize_external_record(
            {
                "schema_version": "m41.external.event.v1",
                "source_name": session["source_name"],
                "subject_id": session["subject_id"],
                "session_id": session["session_id"],
                "task_name": session["task_name"],
                "tick": 99,
                "timestamp": "2026-03-01T00:00:00+00:00",
                "observation_evidence": {"evidence_strength": 0.5, "uncertainty": 0.6, "expected_error": 0.3, "imagined_risk": 0.2},
                "resource_state": {"energy": 0.5, "budget": 0.5, "stress": 0.4, "time_remaining": 0.5},
                "selected_action": "scan",
                "internal_confidence": 0.45,
            }
        )
        self.assertEqual(normalized["schema_version"], "m4.decision_log.v3")
        self.assertIn("candidate_actions", normalized)
        self.assertIn("parameter_snapshot", normalized)
        self.assertEqual(normalized["selected_action"], "scan")


if __name__ == "__main__":
    unittest.main()
