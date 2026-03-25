from __future__ import annotations

import unittest

from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode


class TestM233NarrativeRobustness(unittest.TestCase):
    def test_mixed_language_and_conflicting_evidence_remain_bounded(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m233-mixed",
            timestamp=1,
            source="chat",
            raw_text=(
                "He said danger was gone, 但是 later the trap snapped again, and I 不知道 "
                "whether it was bad luck, a persistent threat, or just the environment misleading us."
            ),
            tags=["mixed"],
            metadata={"environment_id": "ridge"},
        )

        compiled = compiler.compile_episode(episode)
        payload = compiled.uncertainty_decomposition

        self.assertLessEqual(len(payload["unknowns"]), 3)
        self.assertLessEqual(len(payload["competing_hypotheses"]), 9)
        self.assertGreaterEqual(payload["profile"]["decision_relevant_unknown_count"], 1)
        self.assertGreaterEqual(payload["profile"]["interpretive_competition"], 0.2)

    def test_low_signal_paraphrase_degrades_gracefully(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m233-low",
            timestamp=2,
            source="chat",
            raw_text="Maybe something weird happened, maybe not, and the details are fuzzy.",
            tags=["low_signal"],
            metadata={},
        )

        compiled = compiler.compile_episode(episode)
        payload = compiled.uncertainty_decomposition

        self.assertLessEqual(len(payload["unknowns"]), 3)
        self.assertGreaterEqual(payload["profile"]["surface_cue_count"], 0)
        self.assertIsInstance(payload["summary"], str)


if __name__ == "__main__":
    unittest.main()
