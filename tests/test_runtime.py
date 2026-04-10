from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import json
import random
import tempfile
from pathlib import Path
import unittest
import warnings

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


class HTTPTimeout(RuntimeError):
    pass


class TokenLimitExceeded(RuntimeError):
    pass


class ExternalFailureWorld:
    def __init__(self) -> None:
        self.rng = random.Random(17)

    def observe(self):  # noqa: ANN001
        raise HTTPTimeout("upstream website timed out")


class InternalFailureWorld:
    def __init__(self) -> None:
        self.rng = random.Random(17)

    def observe(self):  # noqa: ANN001
        raise TokenLimitExceeded("local token budget exhausted")


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

    def test_sync_memory_awareness_preserves_authoritative_memory_store(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=31, reset=True)
        runtime.run(cycles=12, verbose=False)

        agent = runtime.agent
        self.assertGreater(len(agent.long_term_memory.episodes), 0)
        self.assertIs(agent.memory_store, agent.long_term_memory.memory_store)

        entries_before = len(agent.long_term_memory.memory_store.entries)
        self.assertGreater(entries_before, 0)
        self.assertEqual(entries_before, len(agent.long_term_memory.episodes))

        agent.sync_memory_awareness_to_long_term_memory()

        self.assertIs(agent.memory_store, agent.long_term_memory.memory_store)
        self.assertEqual(len(agent.memory_store.entries), entries_before)
        self.assertEqual(len(agent.memory_store.entries), len(agent.long_term_memory.episodes))

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
            snapshot_payload = json.loads(state_path.read_text(encoding="utf-8"))
            stored_episodes = snapshot_payload["agent"]["long_term_memory"]["episodes"]
            self.assertGreaterEqual(len(stored_episodes), 1)
            self.assertIn("embedding", stored_episodes[0])
            self.assertIn("value_score", stored_episodes[0])
            self.assertIn("predicted_outcome", stored_episodes[0])
            self.assertIn("preferred_probability", stored_episodes[0])
            self.assertIn("risk", stored_episodes[0])
            self.assertIn("total_surprise", stored_episodes[0])
            episodic_memory = trace_records[-1]["episodic_memory"]
            self.assertIn("value_score", episodic_memory)
            self.assertIn("predicted_outcome", episodic_memory)
            self.assertIn("preferred_probability", episodic_memory)
            self.assertIn("risk", episodic_memory)
            self.assertIn("prediction_error", episodic_memory)
            self.assertIn("total_surprise", episodic_memory)
            self.assertIn("episode_created", episodic_memory)
            decision_loop = trace_records[-1]["decision_loop"]
            self.assertIn("prediction_error", decision_loop)
            self.assertIn("predicted_outcome", decision_loop)
            self.assertIn("preferred_probability", decision_loop)
            self.assertIn("risk", decision_loop)
            self.assertIn("ambiguity", decision_loop)
            self.assertIn("expected_free_energy", decision_loop)
            self.assertIn("memory_bias", decision_loop)
            self.assertIn("pattern_bias", decision_loop)
            self.assertIn("policy_bias", decision_loop)
            self.assertIn("identity_bias", decision_loop)
            self.assertIn("policy_score", decision_loop)
            self.assertIn("policy_scores", decision_loop)
            self.assertIn("chosen_action", decision_loop)
            self.assertIn("total_surprise", decision_loop)
            self.assertIn("explanation", decision_loop)
            self.assertIn("policy_bias", trace_records[-1]["decision_ranking"][0])
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

    def test_default_runtime_uses_memory_store_without_deprecation_warnings(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=19, reset=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime.step(verbose=False)

        deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
        self.assertEqual(deprecations, [])
        self.assertEqual(runtime.agent.long_term_memory.memory_backend, "memory_store")
        self.assertEqual(runtime.agent.last_memory_context.get("memory_backend"), "memory_store")
        self.assertIn("recall_hypothesis", runtime.agent.last_memory_context)
        self.assertIn("reconstruction_trace", runtime.agent.last_memory_context)

    def test_sleep_does_not_compress_active_memory_below_recovery_floor(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=11, reset=True)

        runtime.run(cycles=5, verbose=False)

        self.assertGreaterEqual(
            len(runtime.agent.long_term_memory.episodes),
            runtime.agent.long_term_memory.minimum_active_episodes,
        )

    def test_integrate_outcome_keeps_negative_free_energy_drop_for_memory(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=17, reset=True)
        runtime.agent.cycle = 1

        decision = runtime.agent.integrate_outcome(
            choice="hide",
            observed={
                "food": 0.20,
                "danger": 0.95,
                "novelty": 0.20,
                "shelter": 0.30,
                "temperature": 0.45,
                "social": 0.20,
            },
            prediction={
                "food": 0.70,
                "danger": 0.10,
                "novelty": 0.45,
                "shelter": 0.55,
                "temperature": 0.50,
                "social": 0.35,
            },
            errors={
                "food": -0.50,
                "danger": 0.85,
                "novelty": -0.25,
                "shelter": -0.25,
                "temperature": -0.05,
                "social": -0.15,
            },
            free_energy_before=0.20,
            free_energy_after=0.60,
        )

        self.assertTrue(decision.episode_created)
        stored = runtime.agent.long_term_memory.episodes[-1]
        self.assertAlmostEqual(stored["outcome"]["free_energy_drop"], -0.40)
        self.assertEqual(stored["predicted_outcome"], "survival_threat")
        self.assertIn("risk", stored)
        self.assertAlmostEqual(stored["value_score"], -1.0)

    def test_explain_decision_is_deterministic(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=17, reset=True)

        runtime.step(verbose=False)

        first = runtime.agent.explain_decision()
        second = runtime.agent.explain_decision()
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("I chose "))
        self.assertIn("According to my preference model", first)

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

    def test_runtime_logs_world_error_classification_for_external_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                seed=7,
                reset=True,
                trace_path=trace_path,
            )
            runtime.world = ExternalFailureWorld()

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                summary = runtime.run(cycles=1, verbose=True)

            output = stdout.getvalue()
            self.assertEqual(summary["termination_reason"], "exception:HTTPTimeout")
            self.assertIn("classification=world_error", output)
            self.assertIn("surprise_source=exteroceptive", output)

            trace_records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            error_record = trace_records[-1]
            self.assertEqual(error_record["event"], "error")
            self.assertEqual(error_record["error_type"], "HTTPTimeout")
            self.assertEqual(
                error_record["self_model"]["classification"],
                "world_error",
            )

    def test_runtime_logs_self_error_classification_for_internal_failures(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=7, reset=True)
        runtime.world = InternalFailureWorld()
        runtime.host_state.internal_energy = 0.0

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            summary = runtime.run(cycles=1, verbose=True)

        output = stdout.getvalue()
        self.assertEqual(summary["termination_reason"], "exception:TokenLimitExceeded")
        self.assertIn("classification=self_error", output)
        self.assertIn("surprise_source=interoceptive", output)
        self.assertIn("detected_threats=['token_exhaustion']", output)

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
