from __future__ import annotations

import unittest

from segmentum.inquiry_scheduler import semantic_uncertainty_priority_bonus
from segmentum.prediction_ledger import LedgerDiscrepancy, PredictionHypothesis
from segmentum.verification import semantic_priority_adjustment


class TestM33SemanticPredictionAblation(unittest.TestCase):
    def test_priority_helpers_and_ledger_semantic_fields_roundtrip(self) -> None:
        grounding = {"motifs": ["predator_attack"], "semantic_direction_scores": {"uncertainty": 0.4}}
        schemas = [
            {
                "schema_id": "schema:predator_attack",
                "motif_signature": ["predator_attack"],
                "dominant_direction": "threat",
                "confidence": 0.75,
            }
        ]
        priority = semantic_priority_adjustment(
            prediction_id="pred:m33",
            semantic_grounding=grounding,
            semantic_schemas=schemas,
        )
        inquiry_bonus = semantic_uncertainty_priority_bonus(
            semantic_grounding=grounding,
            semantic_schemas=schemas,
        )
        prediction = PredictionHypothesis(
            prediction_id="pred:m33",
            created_tick=1,
            last_updated_tick=1,
            source_module="test",
            prediction_type="environment_state",
            target_channels=("danger",),
            expected_state={"danger": 0.4},
            confidence=0.7,
            expected_horizon=1,
            linked_schema_ids=("schema:predator_attack",),
            semantic_provenance={"motifs": ["predator_attack"]},
        )
        discrepancy = LedgerDiscrepancy(
            discrepancy_id="disc:m33",
            label="semantic mismatch",
            source="prediction_error",
            discrepancy_type="semantic_uncertainty",
            created_tick=1,
            last_seen_tick=2,
            severity=0.5,
            linked_schema_ids=("schema:predator_attack",),
            semantic_provenance={"motifs": ["predator_attack"]},
        )

        self.assertGreater(priority["priority_delta"], 0.0)
        self.assertGreater(inquiry_bonus, 0.0)
        self.assertEqual(
            PredictionHypothesis.from_dict(prediction.to_dict()).linked_schema_ids,
            ("schema:predator_attack",),
        )
        self.assertEqual(
            LedgerDiscrepancy.from_dict(discrepancy.to_dict()).linked_schema_ids,
            ("schema:predator_attack",),
        )


if __name__ == "__main__":
    unittest.main()
