from __future__ import annotations

import unittest

from segmentum.homeostasis import HomeostasisState, MaintenanceAgenda
from segmentum.m227_audit import _baseline_subject_state, _fragile_subject_state
from segmentum.subject_state import apply_subject_state_to_maintenance_agenda, subject_memory_threshold_delta


class TestM227SubjectStateAblation(unittest.TestCase):
    def test_removing_subject_state_modulation_degrades_safety_reroute(self) -> None:
        baseline = _baseline_subject_state()
        fragile = _fragile_subject_state()
        agenda = MaintenanceAgenda(
            cycle=10,
            active_tasks=("energy_recovery",),
            recommended_action="forage",
            interrupt_action=None,
            policy_shift_strength=0.10,
            state=HomeostasisState(),
        )

        baseline_agenda, _ = apply_subject_state_to_maintenance_agenda(baseline, agenda)
        fragile_agenda, _ = apply_subject_state_to_maintenance_agenda(fragile, agenda)

        self.assertEqual(baseline_agenda.recommended_action, "forage")
        self.assertEqual(fragile_agenda.recommended_action, "hide")
        self.assertGreater(fragile_agenda.policy_shift_strength, baseline_agenda.policy_shift_strength)
        self.assertLess(subject_memory_threshold_delta(fragile), subject_memory_threshold_delta(baseline))


if __name__ == "__main__":
    unittest.main()
