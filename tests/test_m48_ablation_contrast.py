from __future__ import annotations

import math
import unittest
from collections import Counter

from segmentum.runtime import SegmentRuntime


ABLATION_CYCLES = 20
ABLATION_SEED = 42


def _run_rollout(*, memory_enabled: bool, seed: int = ABLATION_SEED, cycles: int = ABLATION_CYCLES):
    rt = SegmentRuntime.load_or_create(seed=seed, reset=True, memory_enabled=memory_enabled)
    rt.run(cycles=cycles, verbose=False)
    return rt


def _action_sequence(rt: SegmentRuntime) -> list[str]:
    return list(rt.agent.action_history[-ABLATION_CYCLES:])


def _action_entropy(actions: list[str]) -> float:
    counts = Counter(actions)
    total = len(actions)
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


def _avoidance_ratio(actions: list[str]) -> float:
    avoidance = {"hide", "rest"}
    if not actions:
        return 0.0
    return sum(1 for a in actions if a in avoidance) / len(actions)


class TestM48AblationContrast(unittest.TestCase):

    def test_negative_control_same_seed_identical(self) -> None:
        rt1 = _run_rollout(memory_enabled=True, seed=ABLATION_SEED)
        rt2 = _run_rollout(memory_enabled=True, seed=ABLATION_SEED)
        seq1 = _action_sequence(rt1)
        seq2 = _action_sequence(rt2)
        self.assertEqual(seq1, seq2, "Two memory-enabled runs with the same seed must be identical")

    def test_ablation_produces_divergent_decisions(self) -> None:
        rt_on = _run_rollout(memory_enabled=True)
        rt_off = _run_rollout(memory_enabled=False)
        seq_on = _action_sequence(rt_on)
        seq_off = _action_sequence(rt_off)
        diffs = sum(1 for a, b in zip(seq_on, seq_off) if a != b)
        self.assertGreater(diffs, 0, "Enabled vs disabled must produce at least one different action")

    def test_ablation_entropy_differs(self) -> None:
        rt_on = _run_rollout(memory_enabled=True)
        rt_off = _run_rollout(memory_enabled=False)
        ent_on = _action_entropy(_action_sequence(rt_on))
        ent_off = _action_entropy(_action_sequence(rt_off))
        self.assertNotAlmostEqual(
            ent_on, ent_off, places=4,
            msg="Decision entropy should differ between enabled and disabled memory",
        )

    def test_memory_bias_nonzero_when_enabled(self) -> None:
        rt = _run_rollout(memory_enabled=True)
        ctx = rt.agent.last_memory_context
        self.assertIn("memory_bias", ctx)
        self.assertIn("pattern_bias", ctx)
        self.assertTrue(ctx["memory_enabled"])
        has_nonzero_bias = any(
            abs(float(rt.agent.last_memory_context.get("memory_bias", 0))) > 1e-9
            for _ in [None]
        )
        self.assertTrue(
            has_nonzero_bias or abs(float(ctx.get("pattern_bias", 0))) > 1e-9,
            "At least one of memory_bias or pattern_bias should be nonzero in an enabled run",
        )

    def test_memory_bias_zero_when_disabled(self) -> None:
        rt = _run_rollout(memory_enabled=False)
        ctx = rt.agent.last_memory_context
        self.assertIn("memory_bias", ctx)
        self.assertIn("pattern_bias", ctx)
        self.assertFalse(ctx["memory_enabled"])
        self.assertAlmostEqual(float(ctx["memory_bias"]), 0.0)
        self.assertAlmostEqual(float(ctx["pattern_bias"]), 0.0)

    def test_state_delta_zero_when_disabled(self) -> None:
        rt = _run_rollout(memory_enabled=False)
        ctx = rt.agent.last_memory_context
        state_delta = ctx.get("state_delta", {})
        if isinstance(state_delta, dict):
            for key, value in state_delta.items():
                self.assertAlmostEqual(
                    float(value), 0.0, places=6,
                    msg=f"state_delta[{key}] should be zero when memory is disabled",
                )

    def test_state_delta_nonzero_when_enabled(self) -> None:
        rt = _run_rollout(memory_enabled=True)
        ctx = rt.agent.last_memory_context
        state_delta = ctx.get("state_delta", {})
        max_delta = max(
            (abs(float(v)) for v in state_delta.values()),
            default=0.0,
        )
        self.assertGreater(max_delta, 0.05, "Enabled run should have at least one state_delta > 0.05")

    def test_episodes_still_recorded_when_disabled(self) -> None:
        rt = _run_rollout(memory_enabled=False)
        self.assertGreater(
            len(rt.agent.long_term_memory.episodes), 0,
            "Episodes should still be recorded even when memory is disabled",
        )

    def test_memory_enabled_persists_in_snapshot(self) -> None:
        rt = _run_rollout(memory_enabled=False, cycles=3)
        snapshot = rt.export_snapshot()
        agent_payload = snapshot["agent"]
        self.assertFalse(agent_payload["memory_enabled"])

    def test_memory_enabled_flag_roundtrips(self) -> None:
        rt = _run_rollout(memory_enabled=False, cycles=3)
        snapshot = rt.export_snapshot()
        from segmentum.agent import SegmentAgent
        restored = SegmentAgent.from_dict(snapshot["agent"])
        self.assertFalse(restored.memory_enabled)

    def test_valence_alignment_avoidance_bias(self) -> None:
        rt_on = _run_rollout(memory_enabled=True)
        rt_off = _run_rollout(memory_enabled=False)
        ctx_on = rt_on.agent.last_memory_context
        agg = ctx_on.get("aggregate", {})
        chronic_threat = float(agg.get("chronic_threat_bias", 0.0))
        if chronic_threat <= 0.1:
            self.skipTest("chronic_threat_bias too low to test avoidance alignment")
        avoid_on = _avoidance_ratio(_action_sequence(rt_on))
        avoid_off = _avoidance_ratio(_action_sequence(rt_off))
        self.assertGreater(
            avoid_on, avoid_off,
            f"When chronic_threat_bias={chronic_threat:.3f}, enabled run should have higher avoidance ratio",
        )
