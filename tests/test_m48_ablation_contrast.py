from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.agent import SegmentAgent
from segmentum.m48_audit import _gate_ablation_contrast, build_m48_ablation_evidence
from segmentum.runtime import SegmentRuntime


def _helper_query_inputs(runtime: SegmentRuntime) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, object]]:
    observed = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_decision_observation).items()
    }
    baseline_prediction = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_memory_context.get("prediction_before_memory", {})).items()
    }
    errors = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_memory_context.get("errors", {})).items()
    }
    current_state_snapshot = {
        "observation": dict(observed),
        "prediction": dict(baseline_prediction),
        "errors": dict(errors),
        "body_state": runtime.agent._current_body_state(),
    }
    return observed, baseline_prediction, errors, current_state_snapshot


class TestM48AblationContrast(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ablation = build_m48_ablation_evidence(seed=42, cycles=20)

    def test_retrieve_decision_memories_returns_empty_when_disabled_directly(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=42, reset=True, memory_enabled=False)
        runtime.run(cycles=8, verbose=False)
        observed, baseline_prediction, errors, current_state_snapshot = _helper_query_inputs(runtime)

        retrieved = runtime.agent._retrieve_decision_memories(
            observed=observed,
            baseline_prediction=baseline_prediction,
            baseline_errors=errors,
            current_state_snapshot=current_state_snapshot,
            k=3,
        )

        self.assertEqual(retrieved, [])
        self.assertGreater(len(runtime.agent.long_term_memory.episodes), 0)

    def test_build_memory_context_returns_explicit_zero_structure_when_disabled(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=42, reset=True, memory_enabled=False)
        runtime.run(cycles=4, verbose=False)
        observed, baseline_prediction, errors, _ = _helper_query_inputs(runtime)

        context = runtime.agent._build_memory_context(
            observed=observed,
            baseline_prediction=baseline_prediction,
            errors=errors,
            similar_memories=[{"episode_id": "synthetic-memory"}],
        )

        aggregate = dict(context["aggregate"])
        self.assertFalse(context["memory_hit"])
        self.assertTrue(context["state_delta"])
        self.assertTrue(all(abs(float(value)) <= 1e-9 for value in context["state_delta"].values()))
        self.assertEqual(float(aggregate["chronic_threat_bias"]), 0.0)
        self.assertEqual(float(aggregate["protected_anchor_bias"]), 0.0)

    def test_runtime_load_or_create_respects_memory_enabled_across_fresh_restored_and_recovered(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"

            fresh = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=42,
                reset=True,
                memory_enabled=False,
            )
            fresh.save_snapshot()
            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=99,
                memory_enabled=False,
            )

        with TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "segment_state.json"
            corrupt_path.write_text("{not-json", encoding="utf-8")
            recovered = SegmentRuntime.load_or_create(
                state_path=corrupt_path,
                seed=17,
                memory_enabled=False,
            )

        agent = SegmentAgent(memory_enabled=False)
        restored_agent = SegmentAgent.from_dict(agent.to_dict())

        self.assertFalse(agent.memory_enabled)
        self.assertFalse(restored_agent.memory_enabled)
        self.assertFalse(fresh.agent.memory_enabled)
        self.assertFalse(fresh.export_snapshot()["agent"]["memory_enabled"])
        self.assertFalse(restored.agent.memory_enabled)
        self.assertFalse(recovered.agent.memory_enabled)

    def test_ablation_evidence_contains_per_cycle_trace(self) -> None:
        enabled = self.ablation["enabled"]
        disabled = self.ablation["disabled"]

        self.assertEqual(len(enabled["trace"]), 20)
        self.assertEqual(len(disabled["trace"]), 20)
        self.assertIn("memory_bias", enabled["trace"][0])
        self.assertIn("pattern_bias", enabled["trace"][0])
        self.assertIn("threat_memory_bias", enabled["trace"][0])
        self.assertIn("state_delta", enabled["trace"][0])

    def test_same_seed_negative_control_is_identical(self) -> None:
        enabled = self.ablation["enabled"]["decision_sequence"]
        control = self.ablation["negative_control"]["decision_sequence"]

        self.assertEqual(enabled, control)
        self.assertTrue(self.ablation["comparison"]["decision_sequences_identical_for_negative_control"])

    def test_enabled_vs_disabled_diverge_and_entropy_differs(self) -> None:
        comparison = self.ablation["comparison"]
        enabled = self.ablation["enabled"]
        disabled = self.ablation["disabled"]

        self.assertGreater(len(comparison["differing_cycles"]), 0)
        self.assertNotEqual(enabled["decision_sequence"], disabled["decision_sequence"])
        self.assertNotAlmostEqual(
            float(enabled["decision_entropy"]),
            float(disabled["decision_entropy"]),
            places=6,
        )

    def test_disabled_run_zeroes_state_delta_and_bias_every_cycle(self) -> None:
        disabled_rows = self.ablation["disabled"]["trace"]

        self.assertTrue(
            all(float(row["max_abs_state_delta"]) <= 1e-9 for row in disabled_rows)
        )
        self.assertTrue(
            all(
                abs(float(row["memory_bias"])) <= 1e-9
                and abs(float(row["pattern_bias"])) <= 1e-9
                for row in disabled_rows
            )
        )

    def test_enabled_run_has_nonzero_state_delta_and_bias_signal(self) -> None:
        enabled_rows = self.ablation["enabled"]["trace"]

        self.assertTrue(
            any(float(row["max_abs_state_delta"]) > 0.05 for row in enabled_rows)
        )
        self.assertTrue(
            any(
                abs(float(row["memory_bias"])) > 1e-9
                or abs(float(row["pattern_bias"])) > 1e-9
                for row in enabled_rows
            )
        )

    def test_avoidance_bias_increases_on_enabled_threat_cycles(self) -> None:
        comparison = self.ablation["comparison"]

        self.assertGreater(int(comparison["enabled_threat_cycle_count"]), 0)
        self.assertGreater(
            float(comparison["enabled_avoidance_ratio_under_threat"]),
            float(comparison["disabled_avoidance_ratio_under_threat"]),
        )

    def test_ablation_gate_fails_when_avoidance_direction_reverses(self) -> None:
        tampered = {
            "enabled": dict(self.ablation["enabled"]),
            "disabled": dict(self.ablation["disabled"]),
            "negative_control": dict(self.ablation["negative_control"]),
            "comparison": {
                **dict(self.ablation["comparison"]),
                "enabled_avoidance_ratio_under_threat": 0.20,
                "disabled_avoidance_ratio_under_threat": 0.40,
            },
        }

        record = _gate_ablation_contrast(tampered)
        self.assertFalse(record["criteria_checks"]["avoidance_changes_under_threat"])
        self.assertEqual(record["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
