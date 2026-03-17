from __future__ import annotations

import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.workspace import GlobalWorkspace, GlobalWorkspaceState


class M213GlobalWorkspaceTests(unittest.TestCase):
    def test_workspace_broadcast_is_capacity_limited_and_roundtrippable(self) -> None:
        agent = SegmentAgent()
        agent.configure_attention_bottleneck(enabled=True, capacity=3)
        agent.configure_global_workspace(enabled=True, capacity=2)
        agent.decision_cycle(
            Observation(
                food=0.2,
                danger=0.7,
                novelty=0.95,
                shelter=0.2,
                temperature=0.5,
                social=0.1,
            )
        )

        state = agent.last_workspace_state
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(len(state.broadcast_contents), 2)

        restored = GlobalWorkspaceState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_workspace_state_survives_agent_roundtrip(self) -> None:
        agent = SegmentAgent()
        agent.configure_global_workspace(enabled=True, capacity=2)
        agent.decision_cycle(
            Observation(
                food=0.2,
                danger=0.1,
                novelty=1.0,
                shelter=0.1,
                temperature=0.5,
                social=0.1,
            )
        )
        payload = agent.to_dict()
        restored = SegmentAgent.from_dict(payload, rng=agent.rng)

        self.assertEqual(
            restored.global_workspace.to_dict(),
            agent.global_workspace.to_dict(),
        )


if __name__ == "__main__":
    unittest.main()
