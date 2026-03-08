from __future__ import annotations

import asyncio
import json
import random
import tempfile
from pathlib import Path
import unittest

from segmentum.daemon import HeartbeatDaemon
from segmentum.interoception import InteroceptionReading
from segmentum.logging_utils import ConsciousnessLogger
from segmentum.runtime import SegmentRuntime
from segmentum.state import Strategy


class FixedInteroceptor:
    def sample(self) -> InteroceptionReading:
        return InteroceptionReading(
            cpu_percent=8.0,
            memory_mb=96.0,
            cpu_prediction_error=0.0,
            memory_prediction_error=0.0,
            resource_pressure=0.0,
            energy_drain=0.02,
            boredom_signal=0.10,
            surprise_signal=0.0,
        )


class FixedInnerSpeech:
    async def generate(self, state, tick_input, policy) -> str:  # noqa: ANN001
        _ = (state, tick_input, policy)
        return "hold a steady baseline"


class FailingInteroceptor:
    def sample(self) -> InteroceptionReading:
        raise RuntimeError("telemetry failure")


class FailingWorld:
    def __init__(self) -> None:
        self.rng = random.Random(17)

    def observe(self):  # noqa: ANN001
        raise RuntimeError("sensor failure")


class SnapshotFailingRuntime(SegmentRuntime):
    def save_snapshot(self) -> None:
        raise OSError("disk full")


class SegmentRuntimePersistenceTests(unittest.TestCase):
    def test_same_seed_produces_identical_runs(self) -> None:
        first_runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
        first_summary = first_runtime.run(cycles=12, verbose=False)
        first_snapshot = first_runtime.export_snapshot()

        second_runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
        second_summary = second_runtime.run(cycles=12, verbose=False)
        second_snapshot = second_runtime.export_snapshot()

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_snapshot, second_snapshot)

    def test_resume_matches_continuous_run_with_same_seed(self) -> None:
        continuous_runtime = SegmentRuntime.load_or_create(seed=31, reset=True)
        continuous_runtime.run(cycles=12, verbose=False)
        continuous_snapshot = continuous_runtime.export_snapshot()

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"

            split_runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=31,
                reset=True,
            )
            split_runtime.run(cycles=5, verbose=False)

            reloaded_runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=99)
            reloaded_runtime.run(cycles=7, verbose=False)
            resumed_snapshot = reloaded_runtime.export_snapshot()

        self.assertEqual(continuous_snapshot, resumed_snapshot)

    def test_runtime_persists_and_recovers_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=11,
                reset=True,
            )
            first_summary = runtime.run(cycles=3, verbose=False)

            self.assertTrue(state_path.exists())
            self.assertTrue(trace_path.exists())
            self.assertEqual(first_summary["cycles_completed"], 3)
            trace_records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(trace_records), 3)
            self.assertEqual(trace_records[-1]["running_metrics"]["cycles_completed"], 3)
            hierarchy = trace_records[-1]["hierarchy"]
            self.assertIn("strategic_prediction", hierarchy)
            self.assertIn("sensorimotor_prediction", hierarchy)
            self.assertIn("interoceptive_update", hierarchy)
            self.assertIn(
                "propagated_error",
                hierarchy["interoceptive_update"],
            )

            reloaded = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=99,
            )
            second_summary = reloaded.run(cycles=2, verbose=False)

            self.assertGreaterEqual(second_summary["cycles_completed"], 5)
            self.assertEqual(reloaded.agent.cycle, 5)
            self.assertGreaterEqual(len(reloaded.agent.long_term_memory.episodes), 3)

    def test_runtime_persists_host_telemetry_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            log_path = Path(tmp_dir) / "consciousness.log"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=5,
                reset=True,
            )
            runtime.interoceptor = FixedInteroceptor()
            runtime.inner_speech_engine = FixedInnerSpeech()
            runtime.consciousness_logger = ConsciousnessLogger(log_path=log_path)

            runtime.run(cycles=2, verbose=False, host_telemetry=True)

            reloaded = SegmentRuntime.load_or_create(state_path=state_path, seed=99)

            self.assertEqual(reloaded.host_state.tick_count, 2)
            self.assertEqual(reloaded.host_state.last_strategy, Strategy.EXPLOIT)
            self.assertTrue(log_path.exists())

    def test_daemon_uses_unified_runtime_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            log_path = Path(tmp_dir) / "consciousness.log"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=5,
                reset=True,
            )
            runtime.interoceptor = FixedInteroceptor()
            runtime.inner_speech_engine = FixedInnerSpeech()
            runtime.consciousness_logger = ConsciousnessLogger(log_path=log_path)

            daemon = HeartbeatDaemon(tick_interval_seconds=0.0, runtime=runtime)
            summary = asyncio.run(daemon.run(max_ticks=2))

            self.assertEqual(summary["cycles_completed"], 2)
            self.assertEqual(runtime.host_state.tick_count, 2)
            self.assertEqual(runtime.agent.cycle, 2)
            self.assertTrue(log_path.exists())

    def test_runtime_converts_step_exceptions_into_termination_reason(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=7, reset=True)
        runtime.world = FailingWorld()

        summary = runtime.run(cycles=3, verbose=False)

        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(summary["termination_reason"], "exception:RuntimeError")

    def test_runtime_counts_telemetry_failures_without_stopping_loop(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=7, reset=True)
        runtime.interoceptor = FailingInteroceptor()

        summary = runtime.run(cycles=3, verbose=False, host_telemetry=True)

        self.assertEqual(summary["cycles_completed"], 3)
        self.assertEqual(summary["telemetry_error_count"], 3)
        self.assertEqual(summary["termination_reason"], "cycles_exhausted")

    def test_runtime_counts_snapshot_failures_without_stopping_loop(self) -> None:
        runtime = SnapshotFailingRuntime.load_or_create(seed=7, reset=True)

        summary = runtime.run(cycles=3, verbose=False)

        self.assertEqual(summary["cycles_completed"], 3)
        self.assertEqual(summary["persistence_error_count"], 4)
        self.assertEqual(summary["termination_reason"], "cycles_exhausted")

    def test_runtime_recovers_from_corrupt_snapshot_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            state_path.write_text("{not-json", encoding="utf-8")

            runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=13)

            self.assertEqual(runtime.state_load_status, "recovered_from_corrupt_snapshot")
            quarantined = list(Path(tmp_dir).glob("segment_state.corrupt_snapshot.*.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertFalse(state_path.exists())

    def test_runtime_recovers_from_unsupported_snapshot_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            state_path.write_text(
                json.dumps({"state_version": "9.9", "agent": {}, "world": {}}),
                encoding="utf-8",
            )

            runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=13)

            self.assertEqual(
                runtime.state_load_status,
                "recovered_from_unsupported_state_version",
            )
            quarantined = list(
                Path(tmp_dir).glob("segment_state.unsupported_state_version.*.json")
            )
            self.assertEqual(len(quarantined), 1)
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()
