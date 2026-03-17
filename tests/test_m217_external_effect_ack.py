from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segmentum.action_schema import ActionSchema
from segmentum.runtime import SegmentRuntime


class TestM217ExternalEffectAck(unittest.TestCase):
    def test_failed_external_action_triggers_bounded_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = SegmentRuntime.load_or_create(
                state_path=root / "state.json",
                trace_path=root / "trace.jsonl",
                reset=True,
            )

            failed = runtime.execute_governed_action(
                ActionSchema(
                    name="unstable_workspace_note",
                    params={"path": "unstable.txt", "text": "should fail\n"},
                ),
                predicted_effects={"file_write": 1.0},
            )
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["dispatch"])
            self.assertIsNotNone(failed["repair"])
            repair_dispatch = failed["repair"]["dispatch"]
            self.assertIsNotNone(repair_dispatch)
            repair_path = root / "data" / "governed_actions" / "repair.log"
            self.assertTrue(repair_path.exists())
            self.assertIn("repair after unstable_workspace_note", repair_path.read_text(encoding="utf-8"))

    def test_governance_state_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            trace_path = root / "trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                reset=True,
            )
            runtime.execute_governed_action(
                ActionSchema(
                    name="write_workspace_note",
                    params={"path": "persist.txt", "text": "persist\n"},
                ),
                predicted_effects={"file_write": 1.0},
            )
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                reset=False,
            )
            self.assertEqual(
                runtime.governance.state.snapshot(),
                restored.governance.state.snapshot(),
            )
            traces = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(any(row.get("event") == "external_action" for row in traces))


if __name__ == "__main__":
    unittest.main()
