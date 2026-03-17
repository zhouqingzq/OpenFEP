from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.action_schema import ActionSchema
from segmentum.runtime import SegmentRuntime


class TestM217GovernedAction(unittest.TestCase):
    def test_governed_allowed_review_and_denied_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = SegmentRuntime.load_or_create(
                state_path=root / "state.json",
                trace_path=root / "trace.jsonl",
                reset=True,
            )

            allowed = runtime.execute_governed_action(
                ActionSchema(
                    name="write_workspace_note",
                    params={"path": "test_note.txt", "text": "m217 governed write\n"},
                ),
                predicted_effects={"file_write": 1.0},
            )
            self.assertEqual(allowed["status"], "allowed")
            self.assertIsNotNone(allowed["dispatch"])
            note_path = root / "data" / "governed_actions" / "test_note.txt"
            self.assertTrue(note_path.exists())
            self.assertEqual(note_path.read_text(encoding="utf-8"), "m217 governed write\n")

            review = runtime.execute_governed_action(
                ActionSchema(
                    name="fetch_remote_status",
                    params={"url": "https://example.com/status"},
                ),
                predicted_effects={"network_probe": 1.0},
            )
            self.assertEqual(review["status"], "review-required")
            self.assertIsNone(review["dispatch"])

            denied = runtime.execute_governed_action(
                ActionSchema(
                    name="delete_workspace_note",
                    params={"path": "test_note.txt"},
                ),
                predicted_effects={"file_delete": 1.0},
            )
            self.assertEqual(denied["status"], "denied")
            self.assertIsNone(denied["dispatch"])


if __name__ == "__main__":
    unittest.main()
