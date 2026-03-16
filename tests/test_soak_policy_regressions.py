from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from segmentum.runtime import SegmentRuntime
from tests._pytest_compat import pytest


@pytest.mark.stress
class SoakPolicyRegressionTests(unittest.TestCase):
    def test_m0_seed_17_maintains_switching_behavior_over_256_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = SegmentRuntime.load_or_create(
                state_path=Path(tmp_dir) / "segment_state.json",
                trace_path=Path(tmp_dir) / "segment_trace.jsonl",
                seed=17,
                reset=True,
            )
            summary = runtime.run(cycles=256, verbose=False)

        self.assertGreaterEqual(summary["unique_actions"], 3)
        self.assertGreaterEqual(summary["action_entropy"], 0.20)
        self.assertLessEqual(summary["dominant_action_share"], 0.92)
        self.assertLessEqual(summary["max_action_streak"], 48)
        self.assertGreaterEqual(summary["action_switch_count"], 24)


if __name__ == "__main__":
    unittest.main()
