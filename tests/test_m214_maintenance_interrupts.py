from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segmentum.runtime import SegmentRuntime


class M214MaintenanceInterruptTests(unittest.TestCase):
    def test_runtime_homeostasis_interrupt_overrides_default_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=17,
                reset=True,
            )
            runtime.agent.energy = 0.05
            runtime.agent.fatigue = 0.95
            runtime.agent.stress = 0.20

            result = runtime.step(verbose=False)

            self.assertEqual(result["choice"], "rest")
            record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["choice"], "rest")
            self.assertEqual(record["homeostasis"]["agenda"]["interrupt_action"], "rest")
            self.assertIn("Homeostasis interrupt", record["decision_loop"]["explanation"])

    def test_background_maintenance_reduces_stress_and_fatigue(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
        runtime.agent.stress = 0.90
        runtime.agent.fatigue = 0.82

        stress_before = runtime.agent.stress
        fatigue_before = runtime.agent.fatigue
        runtime.step(verbose=False)

        self.assertLess(runtime.agent.stress, stress_before)
        self.assertLess(runtime.agent.fatigue, fatigue_before)


if __name__ == "__main__":
    unittest.main()
