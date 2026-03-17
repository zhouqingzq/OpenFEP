from __future__ import annotations

import unittest

from segmentum.homeostasis import HomeostasisScheduler, HomeostasisState, MaintenanceAgenda


class M214HomeostasisSchedulerTests(unittest.TestCase):
    def test_scheduler_interrupts_for_critical_energy_and_fatigue(self) -> None:
        scheduler = HomeostasisScheduler()

        agenda = scheduler.assess(
            cycle=3,
            energy=0.08,
            stress=0.20,
            fatigue=0.92,
            temperature=0.50,
            telemetry_error_count=0,
            persistence_error_count=0,
            should_sleep=False,
        )

        self.assertEqual(agenda.interrupt_action, "rest")
        self.assertIn("fatigue_recovery", agenda.active_tasks)

    def test_scheduler_accumulates_and_restores_chronic_debt(self) -> None:
        scheduler = HomeostasisScheduler()

        for cycle in range(1, 5):
            agenda = scheduler.assess(
                cycle=cycle,
                energy=0.18,
                stress=0.78,
                fatigue=0.80,
                temperature=0.50,
                telemetry_error_count=1,
                persistence_error_count=0,
                should_sleep=True,
            )
        self.assertGreater(agenda.state.chronic_energy_debt, 0.0)
        self.assertGreater(agenda.state.chronic_fatigue_debt, 0.0)
        self.assertGreater(agenda.state.short_term_sleep_pressure, 0.45)

        recovered = scheduler.assess(
            cycle=5,
            energy=0.80,
            stress=0.20,
            fatigue=0.10,
            temperature=0.50,
            telemetry_error_count=0,
            persistence_error_count=0,
            should_sleep=False,
        )
        self.assertLess(
            recovered.state.chronic_energy_debt,
            agenda.state.chronic_energy_debt,
        )

    def test_agenda_roundtrip(self) -> None:
        agenda = MaintenanceAgenda(
            cycle=4,
            active_tasks=("energy_recovery", "sleep_pressure"),
            recommended_action="rest",
            interrupt_action="rest",
            interrupt_reason="critical energy/fatigue debt",
            sleep_recommended=True,
            memory_compaction_recommended=False,
            telemetry_backoff_recommended=False,
            state=HomeostasisState(acute_energy_debt=0.2, chronic_energy_debt=0.1),
        )
        restored = MaintenanceAgenda.from_dict(agenda.to_dict())
        self.assertEqual(restored.to_dict(), agenda.to_dict())


if __name__ == "__main__":
    unittest.main()
