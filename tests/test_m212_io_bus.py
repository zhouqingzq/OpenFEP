from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segmentum.logging_utils import ConsciousnessLogger
from segmentum.runtime import SegmentRuntime
from tests.test_runtime import FixedInnerSpeech, FixedInteroceptor


class M212RuntimeIOBusTests(unittest.TestCase):
    def test_runtime_records_perception_and_action_bus_in_trace_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            log_path = Path(tmp_dir) / "consciousness.log"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=17,
                reset=True,
            )
            runtime.interoceptor = FixedInteroceptor()
            runtime.inner_speech_engine = FixedInnerSpeech()
            runtime.consciousness_logger = ConsciousnessLogger(log_path=log_path)

            runtime.run(cycles=1, verbose=False, host_telemetry=True)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("io_bus", payload)
            self.assertGreaterEqual(payload["io_bus"]["perception_bus"]["packets_seen"], 3)
            self.assertEqual(payload["io_bus"]["action_bus"]["dispatch_count"], 1)

            trace_records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            record = trace_records[-1]
            primary = record["io"]["perception"]["primary_observation"]
            self.assertEqual(primary["source_type"], "simulated_world")
            self.assertEqual(len(primary["signals"]), 6)
            self.assertEqual(primary["adapter_name"], "SimulatedWorldAdapter")

            dispatch = record["io"]["action"]["dispatch"]
            ack = record["io"]["action"]["acknowledgment"]
            self.assertEqual(dispatch["adapter_name"], "SimulatedWorldActionAdapter")
            self.assertEqual(dispatch["action_name"], record["choice"])
            self.assertEqual(ack["action_name"], record["choice"])
            self.assertTrue(ack["success"])
            self.assertIn("energy_delta", ack["feedback"])

            host_tick = record["host_tick"]
            self.assertEqual(host_tick["input_packet"]["source_type"], "host_telemetry")
            self.assertEqual(
                host_tick["input_packet"]["adapter_name"],
                "ProcessInteroceptorAdapter",
            )


if __name__ == "__main__":
    unittest.main()
