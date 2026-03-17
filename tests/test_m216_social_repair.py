from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation


class TestM216SocialRepair(unittest.TestCase):
    def test_repair_partially_restores_trust_under_bounds(self) -> None:
        agent = SegmentAgent(rng=random.Random(6))
        memory = agent.social_memory
        memory.observe_counterpart(
            other_id="rival_bob",
            tick=1,
            appraisal={
                "trust_impact": -0.9,
                "social_threat": 0.9,
                "attachment_signal": -0.4,
                "uncertainty": 0.3,
            },
            metadata={"counterpart_name": "Bob", "rupture": True},
            tags=["betrayal", "rupture"],
            event_type="threat",
        )
        trust_after_rupture = memory.others["rival_bob"].trust

        memory.observe_counterpart(
            other_id="rival_bob",
            tick=2,
            appraisal={
                "trust_impact": 0.35,
                "social_threat": 0.1,
                "attachment_signal": 0.2,
                "uncertainty": 0.2,
            },
            metadata={"counterpart_name": "Bob", "repair": True},
            tags=["repair", "apology"],
            event_type="repair_attempt",
        )
        repaired = memory.others["rival_bob"]

        self.assertGreater(repaired.trust, trust_after_rupture)
        self.assertLess(repaired.trust, 0.8)
        self.assertGreaterEqual(repaired.repair_count, 1)

    def test_social_memory_survives_restart_and_still_affects_behavior(self) -> None:
        agent = SegmentAgent(rng=random.Random(8))
        agent.social_memory.observe_counterpart(
            other_id="ally_alice",
            tick=1,
            appraisal={"trust_impact": 0.9, "attachment_signal": 0.6, "uncertainty": 0.1},
            metadata={"counterpart_name": "Alice", "repair": True},
            tags=["cooperation", "repair"],
            event_type="social_contact",
        )

        restored = SegmentAgent.from_dict(
            json.loads(json.dumps(agent.to_dict())),
            rng=random.Random(9),
        )
        self.assertIn("ally_alice", restored.social_memory.others)
        before = agent.social_memory.others["ally_alice"].to_dict()
        after = restored.social_memory.others["ally_alice"].to_dict()
        self.assertEqual(before, after)

        diagnostics = restored.decision_cycle(
            Observation(
                food=0.3,
                danger=0.1,
                novelty=0.3,
                shelter=0.3,
                temperature=0.5,
                social=0.05,
            )
        )["diagnostics"]
        self.assertIn("ally_alice", diagnostics.social_focus)
        self.assertGreater(diagnostics.social_snapshot["history_size"], 0)


if __name__ == "__main__":
    unittest.main()
