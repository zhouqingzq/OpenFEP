from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.runtime import SegmentRuntime


class TestM227SnapshotRoundTrip(unittest.TestCase):
    def test_restart_preserves_subject_state_continuity_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=True,
            )

            for _ in range(260):
                runtime.step(verbose=False)
                if runtime.subject_state.continuity_anchors:
                    break
            else:
                self.fail("expected subject-state continuity anchors before restart")

            before_anchors = tuple(runtime.subject_state.continuity_anchors)
            before_phase = runtime.subject_state.current_phase
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=False,
            )

            self.assertTupleEqual(before_anchors, restored.subject_state.continuity_anchors)
            self.assertEqual(before_phase, restored.subject_state.current_phase)
            self.assertGreater(restored.subject_state.continuity_score, 0.0)


if __name__ == "__main__":
    unittest.main()
