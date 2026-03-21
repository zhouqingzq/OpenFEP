from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.homeostasis import HomeostasisState, MaintenanceAgenda
from segmentum.runtime import SegmentRuntime
from segmentum.subject_state import apply_subject_state_to_maintenance_agenda, derive_subject_state


class TestM227SubjectStateStress(unittest.TestCase):
    def test_subject_state_survives_acute_stress_and_restart_without_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=True,
            )
            runtime.agent.energy = 0.12
            runtime.agent.stress = 0.84
            runtime.agent.fatigue = 0.78
            runtime.agent.temperature = 0.67

            agenda = MaintenanceAgenda(
                cycle=runtime.agent.cycle,
                active_tasks=("energy_recovery",),
                recommended_action="forage",
                interrupt_action=None,
                policy_shift_strength=0.34,
                chronic_debt_pressure=0.41,
                protected_mode=True,
                state=HomeostasisState(),
            )
            subject_state = derive_subject_state(
                runtime.agent,
                maintenance_agenda=agenda,
                continuity_report={
                    "continuity_score": 0.58,
                    "protected_anchor_ids": ["stress-anchor-001"],
                },
                previous_state=runtime.subject_state,
            )
            updated_agenda, _ = apply_subject_state_to_maintenance_agenda(subject_state, agenda)
            runtime.subject_state = subject_state
            runtime.agent.subject_state = subject_state
            runtime.save_snapshot()

            self.assertTrue(subject_state.status_flags["threatened"])
            self.assertTrue(subject_state.status_flags["continuity_fragile"])
            self.assertEqual(updated_agenda.recommended_action, "hide")

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=False,
            )
            self.assertEqual(restored.subject_state.to_dict(), subject_state.to_dict())
            restored.step(verbose=False)
            self.assertGreaterEqual(restored.agent.cycle, 1)
            self.assertIn("continuity_fragile", restored.subject_state.status_flags)


if __name__ == "__main__":
    unittest.main()
