from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode


class TestNarrativeSleepConsolidation(unittest.TestCase):
    def test_sleep_updates_narrative_priors(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        compiler = NarrativeCompiler()
        for index in range(3):
            embodied = compiler.compile_episode(
                NarrativeEpisode(
                    episode_id=f"n-sleep-{index}",
                    timestamp=index + 1,
                    source="user_diary",
                    raw_text="第三天，agent看到一个人吃了毒蘑菇死去了。",
                    tags=["fatality"],
                    metadata={},
                )
            )
            agent.ingest_narrative_episode(embodied)

        before = agent.self_model.narrative_priors.to_dict()
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        agent.sleep()
        after = agent.self_model.narrative_priors.to_dict()

        self.assertGreater(after["contamination_sensitivity"], before["contamination_sensitivity"])
        self.assertLess(after["meaning_stability"], before["meaning_stability"])
        self.assertTrue(
            any("narrative_prior_updates" in entry for entry in agent.narrative_trace)
        )

    def test_narrative_priors_and_trace_survive_round_trip(self) -> None:
        agent = SegmentAgent(rng=random.Random(12))
        compiler = NarrativeCompiler()
        embodied = compiler.compile_episode(
            NarrativeEpisode(
                episode_id="n-roundtrip-1",
                timestamp=1,
                source="user_diary",
                raw_text="第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。",
                tags=["predator"],
                metadata={},
            )
        )
        agent.ingest_narrative_episode(embodied)
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        agent.sleep()

        restored = SegmentAgent.from_dict(agent.to_dict(), rng=random.Random(13))

        self.assertEqual(
            restored.self_model.narrative_priors.to_dict(),
            agent.self_model.narrative_priors.to_dict(),
        )
        self.assertEqual(restored.narrative_trace, agent.narrative_trace)


if __name__ == "__main__":
    unittest.main()
