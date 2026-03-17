from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_ingestion import NarrativeIngestionService
from segmentum.narrative_types import NarrativeEpisode


class TestM216PersistentOthers(unittest.TestCase):
    def test_narrative_ingestion_builds_distinct_other_models(self) -> None:
        agent = SegmentAgent(rng=random.Random(4))
        service = NarrativeIngestionService()
        episodes = [
            NarrativeEpisode(
                episode_id="social:1",
                timestamp=1,
                source="world-alpha",
                raw_text="A trusted ally shared food and stayed nearby.",
                tags=["social_event", "cooperation", "repair"],
                metadata={
                    "counterpart_id": "ally_alice",
                    "counterpart_name": "Alice",
                    "trust_impact": 0.8,
                    "attachment_signal": 0.5,
                    "repair": True,
                    "event_type": "social_contact",
                },
            ),
            NarrativeEpisode(
                episode_id="social:2",
                timestamp=2,
                source="world-alpha",
                raw_text="A rival threatened the shelter entrance.",
                tags=["social_event", "betrayal", "rupture"],
                metadata={
                    "counterpart_id": "rival_bob",
                    "counterpart_name": "Bob",
                    "trust_impact": -0.8,
                    "social_threat": 0.9,
                    "rupture": True,
                    "event_type": "threat",
                },
            ),
        ]

        results = service.ingest(agent=agent, episodes=episodes)

        self.assertEqual(len(results), 2)
        self.assertEqual(set(agent.social_memory.others), {"ally_alice", "rival_bob"})
        alice = agent.social_memory.others["ally_alice"]
        bob = agent.social_memory.others["rival_bob"]
        self.assertGreater(alice.trust, bob.trust)
        self.assertLess(alice.threat, bob.threat)

    def test_social_memory_changes_action_policy(self) -> None:
        baseline = SegmentAgent(rng=random.Random(5))
        socialized = SegmentAgent(rng=random.Random(5))
        socialized.social_memory.observe_counterpart(
            other_id="rival_bob",
            tick=1,
            appraisal={
                "trust_impact": -0.9,
                "social_threat": 0.9,
                "attachment_signal": -0.3,
                "uncertainty": 0.2,
            },
            metadata={"counterpart_name": "Bob", "rupture": True},
            tags=["betrayal", "rupture"],
            event_type="threat",
        )

        observation = Observation(
            food=0.3,
            danger=0.1,
            novelty=0.3,
            shelter=0.3,
            temperature=0.5,
            social=0.05,
        )
        baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
        social_diag = socialized.decision_cycle(observation)["diagnostics"]

        baseline_scores = {
            option.choice: option.policy_score for option in baseline_diag.ranked_options
        }
        social_scores = {
            option.choice: option.policy_score for option in social_diag.ranked_options
        }
        self.assertGreater(social_scores["hide"], baseline_scores["hide"])
        self.assertLess(social_scores["seek_contact"], baseline_scores["seek_contact"])
        self.assertIn("rival_bob", social_diag.social_focus)
        self.assertIn("rival_bob:rupture-risk", social_diag.social_alerts)


if __name__ == "__main__":
    unittest.main()
