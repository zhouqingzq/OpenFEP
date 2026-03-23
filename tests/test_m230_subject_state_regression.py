from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.subject_state import derive_subject_state


class TestM230SubjectStateRegression(unittest.TestCase):
    def test_subject_state_does_not_double_count_slow_continuity_modifier(self) -> None:
        agent = SegmentAgent(rng=random.Random(30))
        agent.slow_variable_learner.state.identity.commitment_stability = 0.88
        agent.slow_variable_learner.state.identity.continuity_resilience = 0.82
        agent.slow_variable_learner.state.values.hierarchy_stability = 0.80

        continuity_report = {
            "continuity_score": 0.73,
            "protected_anchor_ids": ["anchor:commitment"],
        }

        subject_state = derive_subject_state(
            agent,
            continuity_report=continuity_report,
            previous_state=agent.subject_state,
        )

        self.assertAlmostEqual(subject_state.continuity_score, 0.73, places=6)

    def test_subject_fragility_uses_reported_continuity_score_without_inflation(self) -> None:
        agent = SegmentAgent(rng=random.Random(31))
        agent.slow_variable_learner.state.identity.commitment_stability = 0.9
        agent.slow_variable_learner.state.identity.continuity_resilience = 0.9
        agent.slow_variable_learner.state.values.hierarchy_stability = 0.9

        continuity_report = {
            "continuity_score": 0.77,
            "protected_anchor_ids": ["anchor:continuity"],
        }

        subject_state = derive_subject_state(
            agent,
            continuity_report=continuity_report,
            previous_state=agent.subject_state,
        )

        self.assertTrue(subject_state.status_flags["continuity_fragile"])
        self.assertLess(subject_state.continuity_score, 0.78)


if __name__ == "__main__":
    unittest.main()
