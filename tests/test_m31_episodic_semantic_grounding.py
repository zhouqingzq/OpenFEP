from __future__ import annotations

import unittest

from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode, SemanticGrounding


class TestM31EpisodicSemanticGrounding(unittest.TestCase):
    def test_paraphrase_and_implicit_cues_map_to_unified_grounding(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m31-ground",
            timestamp=1,
            source="chat",
            raw_text=(
                "The group stood by me, made room for me, and I felt safe enough to reconnect."
            ),
            tags=["social"],
            metadata={},
        )

        compiled = compiler.compile_episode(episode)
        grounding = SemanticGrounding.from_dict(compiled.semantic_grounding)

        evidence_types = {item.source_type for item in grounding.evidence}
        self.assertIn("paraphrase", evidence_types)
        self.assertIn("implicit", evidence_types)
        self.assertIn("rescue", grounding.motifs)
        self.assertGreater(compiled.appraisal["trust_impact"], 0.4)

    def test_semantic_grounding_roundtrips(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m31-roundtrip",
            timestamp=2,
            source="chat",
            raw_text="The trap snapped again and something hunted us from the ridge.",
            tags=["threat"],
            metadata={},
        )
        compiled = compiler.compile_episode(episode)
        grounding = SemanticGrounding.from_dict(compiled.semantic_grounding)
        self.assertEqual(grounding.to_dict(), compiled.semantic_grounding)


if __name__ == "__main__":
    unittest.main()
