from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segmentum.runtime import SegmentRuntime


class TestM227SubjectState(unittest.TestCase):
    def test_subject_state_is_persistent_visible_and_explainable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=227,
                reset=True,
            )

            runtime.step(verbose=False)

            subject_state = runtime.subject_state
            self.assertEqual(subject_state.tick, runtime.agent.cycle)
            self.assertTrue(subject_state.core_identity_summary)
            self.assertTrue(subject_state.current_phase)
            self.assertTrue(subject_state.dominant_goal)
            self.assertIn("continuity_fragile", subject_state.status_flags)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("subject_state", payload)
            self.assertEqual(payload["subject_state"]["tick"], runtime.agent.cycle)

            trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(trace_lines)
            trace_record = json.loads(trace_lines[-1])
            self.assertIn("subject_state", trace_record)
            self.assertIn("subject_state", trace_record["decision_loop"]["explanation_details"])
            self.assertEqual(
                trace_record["subject_state"]["continuity_anchors"],
                list(subject_state.continuity_anchors),
            )

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=227,
                reset=False,
            )
            self.assertEqual(subject_state.to_dict(), restored.subject_state.to_dict())

            report = restored.agent.conscious_report()
            self.assertIn("Subject state:", report["text"])
            self.assertIn("subject_state", report)


if __name__ == "__main__":
    unittest.main()
