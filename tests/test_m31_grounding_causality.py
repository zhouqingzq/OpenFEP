from __future__ import annotations

import unittest

from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode
from segmentum.semantic_grounding import SemanticGrounder


class NullGrounder(SemanticGrounder):
    def ground_episode(self, *, episode_id: str, text: str, metadata=None):  # type: ignore[override]
        return super().ground_episode(episode_id=episode_id, text="", metadata=metadata)


class TestM31GroundingCausality(unittest.TestCase):
    def test_disabling_grounding_degrades_paraphrase_interpretation(self) -> None:
        episode = NarrativeEpisode(
            episode_id="m31-causal",
            timestamp=1,
            source="chat",
            raw_text="They listened carefully, stayed nearby, and I felt safe enough to reconnect.",
            tags=[],
            metadata={},
        )
        grounded = NarrativeCompiler().compile_episode(episode)
        baseline = NarrativeCompiler(semantic_grounder=NullGrounder()).compile_episode(episode)

        self.assertGreater(grounded.appraisal["trust_impact"], baseline.appraisal["trust_impact"])
        self.assertNotEqual(grounded.semantic_grounding, baseline.semantic_grounding)


if __name__ == "__main__":
    unittest.main()
