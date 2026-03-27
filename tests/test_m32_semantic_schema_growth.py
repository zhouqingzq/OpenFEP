from __future__ import annotations

import unittest

from segmentum.memory import LongTermMemory


def _episode(episode_id: str, motifs: list[str], outcome: str, confidence: float = 0.75) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "predicted_outcome": outcome,
        "compiler_confidence": confidence,
        "semantic_grounding": {
            "episode_id": episode_id,
            "motifs": motifs,
            "semantic_direction_scores": {"threat": 1.0 if "predator_attack" in motifs else 0.0, "social": 1.0 if "rescue" in motifs else 0.0},
            "lexical_surface_hits": {},
            "paraphrase_hits": {},
            "implicit_hits": {},
            "evidence": [],
            "supporting_segments": [],
            "provenance": {},
            "low_signal": False,
        },
        "narrative_tags": list(motifs),
        "source_type": "narrative",
        "continuity_tags": [],
    }


class TestM32SemanticSchemaGrowth(unittest.TestCase):
    def test_repeated_grounded_episodes_form_shared_schema(self) -> None:
        memory = LongTermMemory()
        memory.episodes = [
            _episode("ep-1", ["predator_attack", "exploration"], "survival_threat"),
            _episode("ep-2", ["predator_attack", "exploration"], "survival_threat", 0.82),
        ]
        memory._refresh_semantic_patterns()

        self.assertTrue(memory.semantic_schemas)
        schema = memory.semantic_schemas[0]
        self.assertEqual(schema["support_count"], 2)
        self.assertIn("predator_attack", schema["motif_signature"])

    def test_schema_state_roundtrips_through_memory_snapshot(self) -> None:
        memory = LongTermMemory()
        memory.episodes = [_episode("ep-1", ["rescue"], "resource_gain")]
        memory._refresh_semantic_patterns()

        restored = LongTermMemory.from_dict(memory.to_dict())
        self.assertEqual(restored.semantic_schemas, memory.semantic_schemas)


if __name__ == "__main__":
    unittest.main()
