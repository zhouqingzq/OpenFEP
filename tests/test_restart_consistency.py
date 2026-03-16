from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import pytest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.runtime import SegmentRuntime
from tests.test_counterfactual import OBS_DANGEROUS, _populate_dangerous_episodes


FIXED_OBSERVATION = Observation(
    food=0.16,
    danger=0.80,
    novelty=0.24,
    shelter=0.12,
    temperature=0.46,
    social=0.18,
)


def _distribution(actions: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in sorted(counts.items())}


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    labels = sorted(set(left) | set(right))
    midpoint = {
        label: (left.get(label, 0.0) + right.get(label, 0.0)) / 2.0
        for label in labels
    }

    def _kl(first: dict[str, float], second: dict[str, float]) -> float:
        total = 0.0
        for label in labels:
            p = max(first.get(label, 0.0), 1e-12)
            q = max(second.get(label, 0.0), 1e-12)
            total += p * math.log(p / q, 2)
        return total

    return (_kl(left, midpoint) + _kl(right, midpoint)) / 2.0


def _round_trip_agent(agent: SegmentAgent, seed: int = 999) -> SegmentAgent:
    payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
    return SegmentAgent.from_dict(payload, rng=random.Random(seed))


def _run_until_sleeps(runtime: SegmentRuntime, target_sleep_count: int, max_cycles: int) -> None:
    for _ in range(max_cycles):
        if len(runtime.agent.sleep_history) >= target_sleep_count:
            return
        runtime.step(verbose=False)
    raise AssertionError(f"expected at least {target_sleep_count} sleep cycles")


class TestRestartConsistency(unittest.TestCase):
    """Validate M2 continuity across persistence and restart."""

    @pytest.mark.stress
    def test_action_distribution_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"

            continuous = SegmentRuntime.load_or_create(seed=31, reset=True)
            continuous.run(cycles=700, verbose=False)
            continuous_distribution = _distribution(continuous.agent.action_history[-200:])

            split = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=31,
                reset=True,
            )
            split.run(cycles=500, verbose=False)
            split.save_snapshot()
            restored = SegmentRuntime.load_or_create(state_path=state_path, seed=999)
            restored.run(cycles=200, verbose=False)
            post_restart_distribution = _distribution(restored.agent.action_history[-200:])

            fresh = SegmentRuntime.load_or_create(seed=31, reset=True)
            fresh.run(cycles=200, verbose=False)
            fresh_distribution = _distribution(fresh.agent.action_history[-200:])

            restart_divergence = _js_divergence(
                continuous_distribution,
                post_restart_distribution,
            )
            fresh_divergence = _js_divergence(
                post_restart_distribution,
                fresh_distribution,
            )

            self.assertLess(restart_divergence, 0.05)
            # The restarted agent should remain noticeably closer to a continuous run than a fresh agent is.
            self.assertGreater(fresh_divergence, restart_divergence + 0.01)

    @pytest.mark.stress
    def test_identity_narrative_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                seed=17,
                reset=True,
            )
            _run_until_sleeps(runtime, target_sleep_count=2, max_cycles=260)

            narrative_before = runtime.agent.self_model.identity_narrative
            assert narrative_before is not None
            dominant_strategy = runtime.agent.self_model.preferred_policies.dominant_strategy
            snapshot_before = narrative_before.to_dict()

            runtime.save_snapshot()
            restored = SegmentRuntime.load_or_create(state_path=state_path, seed=77)
            narrative_after = restored.agent.self_model.identity_narrative
            assert narrative_after is not None
            self.assertEqual(snapshot_before, narrative_after.to_dict())

            restored.run(cycles=100, verbose=False)
            self.assertIn(
                dominant_strategy,
                restored.agent.self_model.identity_narrative.core_summary,
            )

    def test_preferred_policies_persistence(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
        _run_until_sleeps(runtime, target_sleep_count=1, max_cycles=180)

        before = runtime.agent.self_model.preferred_policies.to_dict()
        restored = _round_trip_agent(runtime.agent)
        after = restored.self_model.preferred_policies.to_dict()

        self.assertEqual(before, after)

    def test_goal_stack_continuity(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=29, reset=True)
        runtime.run(cycles=120, verbose=False)

        before = runtime.agent.goal_stack.to_dict()
        restored = _round_trip_agent(runtime.agent)
        after = restored.goal_stack.to_dict()

        self.assertEqual(before, after)

    def test_memory_influenced_decision_after_restart(self) -> None:
        baseline_agent = SegmentAgent(rng=random.Random(123))
        baseline = baseline_agent.decision_cycle(FIXED_OBSERVATION)["diagnostics"]
        baseline_forage = next(
            option.policy_score
            for option in baseline.ranked_options
            if option.choice == "forage"
        )

        agent = SegmentAgent(rng=random.Random(123))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=6)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()

        restored = _round_trip_agent(agent)
        diagnostics = restored.decision_cycle(FIXED_OBSERVATION)["diagnostics"]
        forage_score = next(
            option.policy_score
            for option in diagnostics.ranked_options
            if option.choice == "forage"
        )
        explanation = restored.explain_decision_details()

        self.assertLess(forage_score, baseline_forage)
        self.assertNotEqual(diagnostics.chosen.choice, "forage")
        self.assertTrue(
            "memory_bias" in explanation["text"]
            or "pattern_bias" in explanation["text"]
        )

    def test_counterfactual_insights_survive_restart(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.70
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        sleep_summary = agent.sleep()

        self.assertGreater(len(agent.counterfactual_insights), 0)
        self.assertGreater(len(sleep_summary.counterfactual_log), 0)
        absorbed = [insight for insight in agent.counterfactual_insights if insight.absorbed]
        self.assertGreater(len(absorbed), 0)

        reference = absorbed[0]
        cluster_id = reference.cluster_id
        assert cluster_id is not None
        original_bias_before = agent.world_model.get_policy_bias(cluster_id, reference.original_action)
        counterfactual_bias_before = agent.world_model.get_policy_bias(
            cluster_id,
            reference.counterfactual_action,
        )

        restored = _round_trip_agent(agent)
        original_bias_after = restored.world_model.get_policy_bias(cluster_id, reference.original_action)
        counterfactual_bias_after = restored.world_model.get_policy_bias(
            cluster_id,
            reference.counterfactual_action,
        )

        self.assertEqual(original_bias_before, original_bias_after)
        self.assertEqual(counterfactual_bias_before, counterfactual_bias_after)
        self.assertEqual(
            sleep_summary.counterfactual_log,
            restored.sleep_history[-1].counterfactual_log,
        )

    def test_explain_consistency_across_restart(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=19, reset=True)
        _run_until_sleeps(runtime, target_sleep_count=1, max_cycles=180)
        payload = json.loads(json.dumps(runtime.agent.to_dict(), ensure_ascii=True))

        first = SegmentAgent.from_dict(payload, rng=random.Random(7))
        second = SegmentAgent.from_dict(payload, rng=random.Random(7))

        first.decision_cycle(OBS_DANGEROUS)
        second.decision_cycle(OBS_DANGEROUS)
        first_details = first.explain_decision_details()
        second_details = second.explain_decision_details()

        self.assertEqual(
            first.self_model.preferred_policies.dominant_strategy,
            second.self_model.preferred_policies.dominant_strategy,
        )
        self.assertEqual(
            first_details["consistency"]["historical_action_frequency"],
            second_details["consistency"]["historical_action_frequency"],
        )
        self.assertEqual(
            first_details["consistency"]["consistency_statement"],
            second_details["consistency"]["consistency_statement"],
        )


if __name__ == "__main__":
    unittest.main()




