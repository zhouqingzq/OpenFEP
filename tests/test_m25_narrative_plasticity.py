from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode


class TestM25NarrativePlasticity(unittest.TestCase):
    def test_repeated_predator_logs_shift_action_ranking(self) -> None:
        observation = Observation(
            food=0.16,
            danger=0.84,
            novelty=0.20,
            shelter=0.12,
            temperature=0.48,
            social=0.18,
        )
        baseline = SegmentAgent(rng=random.Random(21))
        baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
        baseline_forage = next(
            option.policy_score
            for option in baseline_diag.ranked_options
            if option.choice == "forage"
        )

        experienced = SegmentAgent(rng=random.Random(21))
        experienced.long_term_memory.minimum_support = 1
        experienced.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()
        for index in range(4):
            experienced.ingest_narrative_episode(
                compiler.compile_episode(
                    NarrativeEpisode(
                        episode_id=f"n-m25-{index}",
                        timestamp=index + 1,
                        source="user_diary",
                        raw_text="第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。",
                        tags=["predator"],
                        metadata={},
                    )
                )
            )
        experienced.sleep()

        diag = experienced.decision_cycle(observation)["diagnostics"]
        forage_score = next(
            option.policy_score
            for option in diag.ranked_options
            if option.choice == "forage"
        )

        self.assertGreater(experienced.self_model.narrative_priors.trauma_bias, 0.0)
        self.assertLess(forage_score, baseline_forage)

    def test_witnessed_fatality_increases_contamination_caution(self) -> None:
        observation = Observation(
            food=0.28,
            danger=0.18,
            novelty=0.34,
            shelter=0.22,
            temperature=0.49,
            social=0.30,
        )
        baseline = SegmentAgent(rng=random.Random(31))
        baseline_diag = baseline.decision_cycle(observation)["diagnostics"]

        experienced = SegmentAgent(rng=random.Random(31))
        experienced.long_term_memory.minimum_support = 1
        experienced.long_term_memory.sleep_minimum_support = 1
        compiler = NarrativeCompiler()
        for index in range(4):
            experienced.ingest_narrative_episode(
                compiler.compile_episode(
                    NarrativeEpisode(
                        episode_id=f"n-m25-fatal-{index}",
                        timestamp=index + 1,
                        source="user_diary",
                        raw_text="第三天，agent看到一个人吃了毒蘑菇死去了。",
                        tags=["fatality"],
                        metadata={},
                    )
                )
            )
        experienced.sleep()

        diag = experienced.decision_cycle(observation)["diagnostics"]
        forage_rank = next(
            index
            for index, option in enumerate(diag.ranked_options, start=1)
            if option.choice == "forage"
        )
        baseline_forage_rank = next(
            index
            for index, option in enumerate(baseline_diag.ranked_options, start=1)
            if option.choice == "forage"
        )

        self.assertGreater(
            experienced.self_model.narrative_priors.contamination_sensitivity,
            0.0,
        )
        self.assertEqual(diag.chosen.choice, "hide")
        self.assertGreaterEqual(forage_rank, baseline_forage_rank)


if __name__ == "__main__":
    unittest.main()
