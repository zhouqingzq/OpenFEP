from __future__ import annotations

import unittest

from segmentum.memory import LongTermMemory


def _episode(episode_id: str, motifs: list[str], outcome: str) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "predicted_outcome": outcome,
        "compiler_confidence": 0.72,
        "semantic_grounding": {
            "episode_id": episode_id,
            "motifs": motifs,
            "semantic_direction_scores": {"threat": 1.0 if "predator_attack" in motifs else 0.0},
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


class TestM32SchemaConflictResolution(unittest.TestCase):
    def test_conflicting_evidence_weaken_or_split_schema(self) -> None:
        memory = LongTermMemory()
        memory.episodes = [
            _episode("ep-1", ["predator_attack", "exploration"], "survival_threat"),
            _episode("ep-2", ["predator_attack", "exploration"], "resource_gain"),
        ]
        memory._refresh_semantic_patterns()

        self.assertTrue(
            memory.latest_schema_update["split_schema_ids"] or memory.latest_schema_update["weakened_schema_ids"]
        )


if __name__ == "__main__":
    unittest.main()
