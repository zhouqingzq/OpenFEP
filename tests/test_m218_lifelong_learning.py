from __future__ import annotations

import unittest

from segmentum.memory import (
    LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
    LongTermMemory,
)
from segmentum.self_model import PreferredPolicies, build_default_self_model


class TestM218LifelongLearning(unittest.TestCase):
    def test_action_collapse_guard_triggers_under_dominant_policy(self) -> None:
        model = build_default_self_model()
        model.preferred_policies = PreferredPolicies(
            action_distribution={"rest": 1.0},
            learned_avoidances=[],
        )
        model.update_continuity_audit(
            episodic_memory=[],
            archived_memory=[],
            action_history=["scan", "forage", "rest", "hide"],
            current_tick=1,
        )
        audit = model.update_continuity_audit(
            episodic_memory=[],
            archived_memory=[],
            action_history=["rest"] * 48,
            current_tick=2,
        )

        self.assertIn("action_collapse_guard", audit.interventions)
        self.assertGreaterEqual(
            audit.dominant_action_ratio,
            model.drift_budget.action_dominance_limit,
        )
        self.assertIn("rest", model.preferred_policies.learned_avoidances)

    def test_identity_anchors_remain_rehearsable_after_retirement(self) -> None:
        memory = LongTermMemory(
            episodes=[
                {
                    "episode_id": "ep-protected",
                    "cycle": 1,
                    "last_seen_cycle": 1,
                    "identity_critical": True,
                    "lifecycle_stage": LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
                },
                {
                    "episode_id": "ep-stale",
                    "cycle": 2,
                    "last_seen_cycle": 2,
                    "identity_critical": False,
                    "lifecycle_stage": "validated_episode",
                },
            ]
            + [
                {
                    "episode_id": f"ep-{index:03d}",
                    "cycle": index + 10,
                    "last_seen_cycle": index + 10,
                    "identity_critical": False,
                    "lifecycle_stage": "candidate_episode",
                }
                for index in range(140)
            ],
            max_active_age=128,
        )

        retired = memory.retire_stale_episodes(current_cycle=400, retain_recent=64)
        anchors = memory.protected_identity_anchors(limit=4)
        rehearsal = memory.rehearsal_batch(current_cycle=401, limit=2)

        self.assertGreater(retired, 0)
        self.assertEqual(anchors[0]["episode_id"], "ep-protected")
        self.assertEqual(rehearsal[0]["episode_id"], "ep-protected")
        self.assertTrue(memory.rehearsal_log)


if __name__ == "__main__":
    unittest.main()
