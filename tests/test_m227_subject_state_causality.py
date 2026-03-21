from __future__ import annotations

import unittest

from segmentum.homeostasis import HomeostasisState, MaintenanceAgenda
from segmentum.subject_state import (
    SubjectPriority,
    SubjectState,
    apply_subject_state_to_maintenance_agenda,
    subject_action_bias,
    subject_memory_threshold_delta,
)


class TestM227SubjectStateCausality(unittest.TestCase):
    def test_subject_state_changes_policy_memory_and_maintenance_paths(self) -> None:
        calm = SubjectState(
            tick=10,
            dominant_goal="CONTROL",
            current_phase="consolidation",
            continuity_score=0.96,
            subject_priority_stack=(
                SubjectPriority(
                    label="goal:control",
                    weight=0.35,
                    priority_type="goal",
                    preferred_actions=("scan",),
                ),
            ),
            status_flags={
                "threatened": False,
                "repairing": False,
                "overloaded": False,
                "socially_destabilized": False,
                "continuity_fragile": False,
            },
        )
        fragile = SubjectState(
            tick=10,
            dominant_goal="SURVIVAL",
            current_phase="survival_crisis",
            continuity_score=0.52,
            maintenance_pressure=0.88,
            identity_tension_level=0.63,
            self_inconsistency_level=0.54,
            active_commitments=("protect continuity",),
            subject_priority_stack=(
                SubjectPriority(
                    label="need:danger",
                    weight=0.92,
                    priority_type="need",
                    preferred_actions=("hide", "exploit_shelter"),
                    avoid_actions=("forage",),
                ),
                SubjectPriority(
                    label="continuity fragility",
                    weight=0.81,
                    priority_type="continuity",
                    preferred_actions=("scan", "rest"),
                    avoid_actions=("forage",),
                ),
            ),
            status_flags={
                "threatened": True,
                "repairing": True,
                "overloaded": True,
                "socially_destabilized": False,
                "continuity_fragile": True,
            },
        )
        agenda = MaintenanceAgenda(
            cycle=10,
            active_tasks=("energy_recovery",),
            recommended_action="forage",
            interrupt_action=None,
            policy_shift_strength=0.10,
            state=HomeostasisState(),
        )

        self.assertGreater(subject_action_bias(fragile, "hide"), subject_action_bias(calm, "hide"))
        self.assertLess(subject_action_bias(fragile, "forage"), subject_action_bias(calm, "forage"))
        self.assertLess(subject_memory_threshold_delta(fragile), subject_memory_threshold_delta(calm))

        updated_agenda, details = apply_subject_state_to_maintenance_agenda(fragile, agenda)
        self.assertGreater(updated_agenda.policy_shift_strength, agenda.policy_shift_strength)
        self.assertEqual(updated_agenda.recommended_action, "hide")
        self.assertIn("continuity_guard", updated_agenda.active_tasks)
        self.assertIn("repair_stabilization", updated_agenda.active_tasks)
        self.assertTrue(details["status_flags"]["continuity_fragile"])


if __name__ == "__main__":
    unittest.main()
