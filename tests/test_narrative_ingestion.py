from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.narrative_ingestion import NarrativeIngestionService
from segmentum.narrative_types import NarrativeEpisode


class TestNarrativeIngestion(unittest.TestCase):
    def test_ingestion_stores_appraisal_payload(self) -> None:
        agent = SegmentAgent(rng=random.Random(7))
        service = NarrativeIngestionService()
        episode = NarrativeEpisode(
            episode_id="n-ingest-1",
            timestamp=2,
            source="user_diary",
            raw_text="第二天，agent昨天路过河边，被一只鳄鱼攻击了，没有受伤。",
            tags=["predator"],
            metadata={},
        )

        results = service.ingest(agent=agent, episodes=[episode])

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(len(agent.long_term_memory.episodes), 1)
        stored = agent.long_term_memory.episodes[-1]
        self.assertIn("appraisal", stored)
        self.assertIn("narrative_tags", stored)
        self.assertEqual(stored["source_episode_id"], "n-ingest-1")
        self.assertEqual(stored["source_type"], "user_diary")
        self.assertGreater(float(stored["compiler_confidence"]), 0.0)
        self.assertTrue(agent.narrative_trace)
        self.assertTrue(results[0]["ingestion"]["episode_created"])

    def test_service_attaches_sleep_summary_when_agent_should_sleep(self) -> None:
        agent = SegmentAgent(rng=random.Random(8))
        agent.long_term_memory.memory_threshold = 0
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        service = NarrativeIngestionService()
        episodes = [
            NarrativeEpisode(
                episode_id=f"n-ingest-sleep-{index}",
                timestamp=index + 1,
                source="user_diary",
                raw_text="第三天，agent看到一个人吃了毒蘑菇死去了。",
                tags=["fatality"],
                metadata={},
            )
            for index in range(2)
        ]

        results = service.ingest(agent=agent, episodes=episodes)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("sleep" in result for result in results))
        self.assertGreaterEqual(results[0]["sleep"]["sleep_cycle_id"], 1)
        self.assertIn("contamination_sensitivity", results[0]["sleep"]["narrative_prior_updates"])
        self.assertGreaterEqual(len(agent.sleep_history), 1)


if __name__ == "__main__":
    unittest.main()
