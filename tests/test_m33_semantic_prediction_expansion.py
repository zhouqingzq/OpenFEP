from __future__ import annotations

import unittest

from segmentum.predictive_coding import apply_schema_conditioned_prediction


class TestM33SemanticPredictionExpansion(unittest.TestCase):
    def test_schema_conditioning_changes_prediction_surface(self) -> None:
        conditioned, payload = apply_schema_conditioned_prediction(
            {"food": 0.5, "danger": 0.3, "novelty": 0.5, "shelter": 0.5, "temperature": 0.5, "social": 0.3},
            semantic_schemas=[
                {
                    "schema_id": "schema:predator_attack-exploration",
                    "motif_signature": ["predator_attack", "exploration"],
                    "dominant_direction": "threat",
                    "confidence": 0.82,
                }
            ],
            semantic_grounding={
                "motifs": ["predator_attack", "exploration"],
                "semantic_direction_scores": {"uncertainty": 0.6},
            },
        )
        self.assertGreater(conditioned["danger"], 0.3)
        self.assertTrue(payload["applied_schema_ids"])


if __name__ == "__main__":
    unittest.main()
