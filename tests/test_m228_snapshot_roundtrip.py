from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segmentum.runtime import SegmentRuntime


class TestM228SnapshotRoundTrip(unittest.TestCase):
    def test_snapshot_and_trace_preserve_prediction_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=71,
                reset=True,
            )

            for _ in range(6):
                runtime.step(verbose=False)
                if runtime.agent.prediction_ledger.active_predictions() and runtime.agent.prediction_ledger.active_discrepancies():
                    break

            runtime.save_snapshot()
            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=71,
                reset=False,
            )

            self.assertTrue(restored.agent.prediction_ledger.active_predictions())
            self.assertTrue(restored.agent.prediction_ledger.active_discrepancies())
            self.assertEqual(
                restored.agent.prediction_ledger.top_discrepancies(1)[0].discrepancy_id,
                runtime.agent.prediction_ledger.top_discrepancies(1)[0].discrepancy_id,
            )

            lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(lines)
            trace_record = json.loads(lines[-1])
            self.assertIn("prediction_ledger", trace_record)
            self.assertIn("ledger_bias", trace_record["decision_loop"])
            self.assertIn("ledger_payload", trace_record["decision_loop"])


if __name__ == "__main__":
    unittest.main()
