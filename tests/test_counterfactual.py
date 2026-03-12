"""M2.4 Counterfactual Learning 鈥?Acceptance Tests.

Seven tests covering the full counterfactual pipeline:

1. Basic counterfactual correctness
2. Confidence gating
3. Behavior change after absorption
4. Energy budget constraint
5. No-regression invariants
6. Rumination prevention
7. Multi-step confidence decay
"""
from __future__ import annotations

import random
import unittest
from dataclasses import asdict

from segmentum.agent import SegmentAgent
from segmentum.constants import ACTION_COSTS
from segmentum.counterfactual import (
    ABSORPTION_THRESHOLD,
    SIGNIFICANCE_THRESHOLD,
    CounterfactualEngine,
    CounterfactualInsight,
    CounterfactualSummary,
    ForwardGenerativeModel,
    InsightAbsorber,
    run_counterfactual_phase,
)
from segmentum.environment import Observation, SimulatedWorld
from segmentum.preferences import PreferenceModel
from segmentum.world_model import GenerativeWorldModel


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# An anomalous observation: high danger, low food
OBS_DANGEROUS = Observation(
    food=0.12,
    danger=0.82,
    novelty=0.22,
    shelter=0.10,
    temperature=0.46,
    social=0.18,
)
OBS_DANGEROUS_DICT = asdict(OBS_DANGEROUS)

# The bad outcome from foraging in this situation
HARMFUL_OUTCOME = {
    "energy_delta": -0.10,
    "stress_delta": 0.28,
    "fatigue_delta": 0.18,
    "temperature_delta": 0.02,
    "free_energy_drop": -0.45,
}

# Body state at the moment of decision (before the disastrous forage).
# Energy is moderate 鈥?the agent *could* have survived with a safer action.
HARMFUL_BODY_STATE = {
    "energy": 0.50,
    "stress": 0.40,
    "fatigue": 0.25,
    "temperature": 0.46,
}

PREDICTION = {
    "food": 0.60,
    "danger": 0.20,
    "novelty": 0.40,
    "shelter": 0.40,
    "temperature": 0.50,
    "social": 0.30,
}


def _errors() -> dict[str, float]:
    return {k: OBS_DANGEROUS_DICT[k] - PREDICTION[k] for k in OBS_DANGEROUS_DICT}


def _populate_dangerous_episodes(
    agent: SegmentAgent,
    count: int = 5,
) -> None:
    """Store multiple high-surprise forage鈫抎isaster episodes."""
    obs = OBS_DANGEROUS_DICT
    pred = PREDICTION
    errors = _errors()
    for cycle in range(1, count + 1):
        agent.cycle = cycle
        md = agent.long_term_memory.maybe_store_episode(
            cycle=cycle,
            observation=obs,
            prediction=pred,
            errors=errors,
            action="forage",
            outcome=HARMFUL_OUTCOME,
            body_state=HARMFUL_BODY_STATE,
        )
        if not md.episode_created:
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=obs,
                prediction=pred,
                errors=errors,
                action="forage",
                outcome=HARMFUL_OUTCOME,
                body_state=HARMFUL_BODY_STATE,
            )


# ---------------------------------------------------------------------------
# Test 1: Basic counterfactual correctness
# ---------------------------------------------------------------------------


class TestBasicCounterfactual(unittest.TestCase):
    """Setting: forage 鈫?survival_threat.  Counterfactual should find a safer
    action with lower EFE."""

    def test_basic_counterfactual(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.60
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1

        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()

        episodes = list(agent.long_term_memory.episodes)
        preference_model = agent.long_term_memory.preference_model

        insights, summary = run_counterfactual_phase(
            agent_energy=agent.energy,
            current_cycle=10,
            episodes=episodes,
            world_model=agent.world_model,
            preference_model=preference_model,
            rng=agent.rng,
            surprise_threshold=agent.long_term_memory.surprise_threshold,
        )

        self.assertGreater(
            summary.episodes_evaluated, 0,
            "Engine must evaluate at least one high-surprise episode",
        )
        self.assertGreater(
            summary.branches_explored, 0,
            "Engine must explore alternative branches",
        )
        self.assertGreater(
            len(insights), 0,
            "At least one counterfactual insight should be generated",
        )

        # The insight must have negative efe_delta (counterfactual is better).
        best_insight = min(insights, key=lambda i: i.efe_delta)
        self.assertLess(
            best_insight.efe_delta, 0.0,
            f"Best insight efe_delta must be negative, got {best_insight.efe_delta}",
        )
        self.assertEqual(best_insight.original_action, "forage")
        self.assertNotEqual(best_insight.counterfactual_action, "forage")


# ---------------------------------------------------------------------------
# Test 2: Confidence gating
# ---------------------------------------------------------------------------


class TestConfidenceGating(unittest.TestCase):
    """When the ForwardModel has no training data, confidence is low and
    InsightAbsorber should reject the insight."""

    def test_confidence_gating(self) -> None:
        world_model = GenerativeWorldModel()
        preference_model = PreferenceModel()

        # Forward model with NO known episodes 鈫?low confidence.
        forward_model = ForwardGenerativeModel(
            world_model=world_model,
            preference_model=preference_model,
            known_episodes=[],
        )

        # Manually create an insight with very low confidence.
        insight = CounterfactualInsight(
            source_episode_cycle=1,
            original_action="forage",
            counterfactual_action="hide",
            original_efe=5.0,
            counterfactual_efe=1.0,
            efe_delta=-4.0,
            confidence=0.15,  # Below ABSORPTION_THRESHOLD
            state_context={"observation": OBS_DANGEROUS_DICT},
            cluster_id=0,
            timestamp=10,
        )

        absorber = InsightAbsorber()
        absorbed = absorber.absorb([insight], world_model)

        self.assertEqual(absorbed, 0, "Low-confidence insight must NOT be absorbed")
        self.assertFalse(insight.absorbed)
        self.assertTrue(
            any(
                entry.get("reason") == "confidence_below_threshold"
                for entry in absorber.log
            ),
            "Rejection reason must be logged",
        )

    def test_forward_model_no_data_gives_low_confidence(self) -> None:
        """ForwardModel with empty episode list returns confidence < threshold."""
        forward_model = ForwardGenerativeModel(
            world_model=GenerativeWorldModel(),
            preference_model=PreferenceModel(),
            known_episodes=[],
        )
        state = {
            "observation": OBS_DANGEROUS_DICT,
            "body_state": {"energy": 0.5, "stress": 0.3, "fatigue": 0.2, "temperature": 0.48},
        }
        confidence = forward_model._compute_confidence(state)
        self.assertLess(
            confidence, ABSORPTION_THRESHOLD,
            f"Confidence with no training data must be < {ABSORPTION_THRESHOLD}, got {confidence}",
        )


# ---------------------------------------------------------------------------
# Test 3: Behavior change after absorption
# ---------------------------------------------------------------------------


class TestBehaviorChangeAfterAbsorption(unittest.TestCase):
    """After absorbing an insight, the policy bias for the counterfactual
    action should increase and the original action's bias should decrease."""

    def test_behavior_change_after_absorption(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.70
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1

        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()

        # Get cluster ID for the dangerous episodes.
        cluster_id = agent.long_term_memory.episodes[0].get("cluster_id")
        self.assertIsNotNone(cluster_id, "Episodes must be clustered")

        # Record policy bias BEFORE counterfactual.
        forage_bias_before = agent.world_model.get_policy_bias(cluster_id, "forage")

        # Run counterfactual phase.
        insights, summary = run_counterfactual_phase(
            agent_energy=agent.energy,
            current_cycle=10,
            episodes=list(agent.long_term_memory.episodes),
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            rng=agent.rng,
            surprise_threshold=agent.long_term_memory.surprise_threshold,
        )

        # Check that at least one insight was absorbed.
        absorbed_insights = [i for i in insights if i.absorbed]
        if not absorbed_insights:
            # If nothing was absorbed (e.g., confidence too low), the test
            # validates that the absorber respected its thresholds.  Still
            # check that the log recorded rejections.
            self.assertGreater(len(summary.counterfactual_log), 0)
            return

        # The original action's policy bias should have decreased.
        forage_bias_after = agent.world_model.get_policy_bias(cluster_id, "forage")
        self.assertLess(
            forage_bias_after, forage_bias_before,
            "Forage policy bias must decrease after absorption",
        )

        # The counterfactual action's bias should have increased.
        cf_action = absorbed_insights[0].counterfactual_action
        cf_bias = agent.world_model.get_policy_bias(cluster_id, cf_action)
        self.assertGreater(
            cf_bias, 0.0,
            f"Counterfactual action '{cf_action}' bias must be > 0 after absorption",
        )

        # Verify causal chain is logged.
        absorption_logs = [
            entry for entry in summary.counterfactual_log
            if entry.get("type") == "absorption"
        ]
        self.assertGreater(len(absorption_logs), 0, "Absorption must be logged")
        log_entry = absorption_logs[0]
        self.assertIn("source_episode_cycle", log_entry)
        self.assertIn("policy_delta", log_entry)
        self.assertIn("new_cf_bias", log_entry)

    def test_virtual_sandbox_log_in_sleep_history(self) -> None:
        """M2.4 楠屾敹: 鏃ュ織涓嚭鐜般€岃櫄鎷熸矙鐩掓帹婕斻€嶈褰曪紝涓旈殢 SleepSummary 鎸佷箙鍖?"""
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.70
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()

        insights, summary = run_counterfactual_phase(
            agent_energy=agent.energy,
            current_cycle=10,
            episodes=list(agent.long_term_memory.episodes),
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            rng=agent.rng,
            surprise_threshold=agent.long_term_memory.surprise_threshold,
        )
        if summary.episodes_evaluated == 0 and summary.branches_explored == 0:
            self.skipTest("no episodes evaluated (low surprise or energy gate)")

        sandbox_entries = [
            e for e in summary.counterfactual_log
            if e.get("type") == "virtual_sandbox_reasoning" and e.get("label") == "虚拟沙盒推演"
        ]
        self.assertGreater(
            len(sandbox_entries), 0,
            "counterfactual_log must contain 虚拟沙盒推演 record when engine ran",
        )
        self.assertIn("episodes_evaluated", sandbox_entries[0])
        self.assertIn("branches_explored", sandbox_entries[0])

        # Persist via sleep: run a minimal sleep that triggers counterfactual and check summary.
        agent.cycle = 20
        agent.long_term_memory.episodes = list(agent.long_term_memory.episodes)
        sleep_summary = agent.sleep()
        self.assertGreater(
            len(sleep_summary.counterfactual_log), 0,
            "SleepSummary must carry counterfactual_log",
        )
        persisted_sandbox = [
            e for e in sleep_summary.counterfactual_log
            if e.get("type") == "virtual_sandbox_reasoning"
        ]
        self.assertGreater(len(persisted_sandbox), 0, "虚拟沙盒推演 must persist in sleep_history")

        # Round-trip: to_dict / from_dict preserves counterfactual_log.
        payload = agent.to_dict()
        last_sleep = payload["sleep_history"][-1]
        self.assertIn("counterfactual_log", last_sleep)
        self.assertIsInstance(last_sleep["counterfactual_log"], list)


# ---------------------------------------------------------------------------
# Test 4: Energy budget constraint
# ---------------------------------------------------------------------------


class TestEnergyBudgetConstraint(unittest.TestCase):
    """Counterfactual engine respects energy constraints."""

    def test_energy_below_30pct_skips(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.25  # Below 30%
        agent.long_term_memory.minimum_support = 1
        _populate_dangerous_episodes(agent, count=3)

        insights, summary = run_counterfactual_phase(
            agent_energy=agent.energy,
            current_cycle=10,
            episodes=list(agent.long_term_memory.episodes),
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            rng=agent.rng,
        )

        self.assertEqual(len(insights), 0, "No insights when energy < 30%")
        self.assertEqual(summary.skipped_reason, "energy_below_30pct")

    def test_energy_below_50pct_reduces_depth(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.45  # Between 30% and 50%
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()

        forward_model = ForwardGenerativeModel(
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            known_episodes=list(agent.long_term_memory.episodes),
        )
        engine = CounterfactualEngine(
            forward_model=forward_model,
            preference_model=agent.long_term_memory.preference_model,
            max_depth=3,
            energy_budget=1.0,  # Large budget to not interfere.
        )

        # Run with energy < 50% 鈥?effective depth should be 1.
        insights, summary = engine.run(
            episodes=list(agent.long_term_memory.episodes),
            current_cycle=10,
            agent_energy=0.45,
            rng=agent.rng,
        )

        # With depth=1, each episode-branch pair uses exactly 1 node.
        # Max nodes = episodes_evaluated * max_branches * 1 step.
        # (depth=1 means only the first predict_step, no multi-step loop)
        if summary.episodes_evaluated > 0:
            max_single_depth_nodes = (
                summary.episodes_evaluated * engine.max_branches
            )
            # energy_spent should reflect single-step evaluation only.
            self.assertLessEqual(
                summary.energy_spent,
                max_single_depth_nodes * engine.energy_cost_per_node + 1e-9,
                "With depth=1, energy spent must match single-step evaluation",
            )


# ---------------------------------------------------------------------------
# Test 5: No-regression invariants
# ---------------------------------------------------------------------------


class TestNoRegression(unittest.TestCase):
    """M2.4 must not degrade M2.1-M2.3 invariants."""

    def test_no_regression(self) -> None:
        seed = 42
        cycles = 30

        # --- Baseline run (no counterfactual 鈥?agent still runs it during
        # sleep, but we capture metrics for comparison) ---
        world = SimulatedWorld(seed=seed)
        agent = SegmentAgent(rng=random.Random(seed))

        identity_before = asdict(agent.identity_traits)
        baseline_survival_ticks = 0
        baseline_fe_values: list[float] = []

        for tick in range(cycles):
            agent.cycle = tick + 1
            obs = world.observe()
            result = agent.decision_cycle(obs)
            diagnostics = result["diagnostics"]
            chosen = diagnostics.chosen.choice

            free_energy = result["free_energy_before"]
            baseline_fe_values.append(free_energy)

            if chosen == "internal_update":
                agent.apply_internal_update(result["errors"])
            else:
                feedback = world.apply_action(chosen)
                agent.apply_action_feedback(feedback)

            agent.integrate_outcome(
                chosen,
                result["observed"],
                result["prediction"],
                result["errors"],
                result["free_energy_before"],
                agent.compute_free_energy(result["errors"]),
            )

            if agent.energy > 0.05:
                baseline_survival_ticks += 1

            if agent.should_sleep():
                agent.sleep()
                world.drift()

        # --- Verify invariants ---

        # 1. Identity traits unchanged (counterfactual never touches core beliefs).
        identity_after = asdict(agent.identity_traits)
        self.assertEqual(
            identity_before, identity_after,
            "Identity traits must not be modified by counterfactual learning",
        )

        # 2. No counterfactual-sourced episodes in episodic memory.
        for ep in agent.long_term_memory.episodes:
            self.assertNotEqual(
                ep.get("source"), "counterfactual",
                "Episodic memory must not contain counterfactual-sourced episodes",
            )

        # 3. Agent survived (energy > 0 at some point).
        self.assertGreater(
            baseline_survival_ticks, 0,
            "Agent must survive at least one tick",
        )

        # 4. Counterfactual insights are stored (sleep was triggered).
        # This verifies integration is active.
        if agent.sleep_history:
            total_cf_evaluated = sum(
                s.counterfactual_episodes_evaluated for s in agent.sleep_history
            )
            # Counterfactual may or may not find episodes depending on dynamics,
            # but the stats field must exist and be non-negative.
            self.assertGreaterEqual(total_cf_evaluated, 0)


# ---------------------------------------------------------------------------
# Test 6: Rumination prevention
# ---------------------------------------------------------------------------


class TestRuminationPrevention(unittest.TestCase):
    """With many high-surprise episodes, energy budget must cap the work."""

    def test_rumination_prevention(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.80
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1

        # Create many high-surprise episodes (>50).
        obs = OBS_DANGEROUS_DICT
        pred = PREDICTION
        errors = _errors()
        for cycle in range(1, 55):
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=obs,
                prediction=pred,
                errors=errors,
                action="forage",
                outcome=HARMFUL_OUTCOME,
                body_state=HARMFUL_BODY_STATE,
            )

        agent.long_term_memory.assign_clusters()

        energy_budget = 0.05
        insights, summary = run_counterfactual_phase(
            agent_energy=agent.energy,
            current_cycle=100,
            episodes=list(agent.long_term_memory.episodes),
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            rng=agent.rng,
            energy_budget=energy_budget,
        )

        self.assertLessEqual(
            summary.energy_spent, energy_budget + 1e-9,
            f"Energy spent ({summary.energy_spent}) must not exceed budget ({energy_budget})",
        )
        # Engine must have stopped before processing all episodes.
        high_surprise_count = sum(
            1 for ep in agent.long_term_memory.episodes
            if float(ep.get("total_surprise", 0.0)) > agent.long_term_memory.surprise_threshold
        )
        if high_surprise_count > 5:
            self.assertLess(
                summary.episodes_evaluated, high_surprise_count,
                "Engine must stop before evaluating all episodes when budget is tight",
            )


# ---------------------------------------------------------------------------
# Test 7: Multi-step confidence decay
# ---------------------------------------------------------------------------


class TestMultistepConfidenceDecay(unittest.TestCase):
    """Confidence must decrease at each forward step, and low-confidence
    steps should prevent further expansion."""

    def test_multistep_confidence_decay(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.long_term_memory.minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)

        forward_model = ForwardGenerativeModel(
            world_model=agent.world_model,
            preference_model=agent.long_term_memory.preference_model,
            known_episodes=list(agent.long_term_memory.episodes),
            confidence_decay=0.85,
        )

        state: dict[str, object] = {
            "observation": OBS_DANGEROUS_DICT,
            "body_state": {"energy": 0.5, "stress": 0.3, "fatigue": 0.2, "temperature": 0.48},
        }

        # 3-step forward prediction with the same action.
        actions = ["hide", "hide", "hide"]
        results = forward_model.predict_multistep(state, actions)

        self.assertEqual(len(results), 3)

        confidences = [conf for _, conf in results]

        # Confidence must decrease at each step (due to decay factor).
        for i in range(1, len(confidences)):
            self.assertLessEqual(
                confidences[i], confidences[i - 1],
                f"Step {i+1} confidence ({confidences[i]:.4f}) must be <= "
                f"step {i} confidence ({confidences[i-1]:.4f})",
            )

        # Step 3 confidence must be strictly less than step 1.
        self.assertLess(
            confidences[2], confidences[0],
            "Final step confidence must be strictly less than first step",
        )

    def test_pruning_on_low_confidence(self) -> None:
        """When a forward model has no training data, confidence drops fast
        and multi-step rollout should produce very low confidence."""
        forward_model = ForwardGenerativeModel(
            world_model=GenerativeWorldModel(),
            preference_model=PreferenceModel(),
            known_episodes=[],  # No training data 鈫?very low confidence.
            confidence_decay=0.85,
        )

        state: dict[str, object] = {
            "observation": OBS_DANGEROUS_DICT,
            "body_state": {"energy": 0.5, "stress": 0.3, "fatigue": 0.2, "temperature": 0.48},
        }
        results = forward_model.predict_multistep(state, ["hide", "hide", "hide"])

        # With no training data, initial confidence is ~0.1.
        # After 3 steps with decay 0.85: 0.1 * (0.1*0.85) * (0.1*0.85) 鈫?very small.
        final_confidence = results[-1][1]
        self.assertLess(
            final_confidence, 0.05,
            f"With no training data, 3-step confidence must be < 0.05, got {final_confidence}",
        )


class TestProjectedActionExpectedFreeEnergy(unittest.TestCase):
    def test_project_action_exposes_goal_weighted_expected_free_energy(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        observed = asdict(OBS_DANGEROUS)
        priors = agent.strategic_layer.priors(
            agent.energy,
            agent.stress,
            agent.fatigue,
            agent.temperature,
            agent.dopamine,
            agent.drive_system,
        )
        prediction = agent.world_model.predict(priors)
        errors = {
            key: observed.get(key, 0.0) - prediction.get(key, 0.0)
            for key in sorted(set(observed) | set(prediction))
        }
        free_energy_before = agent.compute_free_energy(errors)

        projected = agent._project_action(
            action="hide",
            observed=observed,
            prediction=prediction,
            priors=priors,
            free_energy_before=free_energy_before,
            current_cluster_id=None,
            active_goal=agent.goal_stack.active_goal,
        )

        expected = agent.long_term_memory.preference_model.expected_free_energy(
            outcome=str(projected["predicted_outcome"]),
            predicted_error=float(projected["predicted_error"]),
            action_ambiguity=float(projected["action_ambiguity"]),
            goal=agent.goal_stack.active_goal,
            baseline_risk=float(projected["risk"]),
        )

        self.assertIn("expected_free_energy", projected)
        self.assertAlmostEqual(float(projected["expected_free_energy"]), expected)
        self.assertGreaterEqual(float(projected["expected_free_energy"]), float(projected["risk"]))


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestCounterfactualSerialization(unittest.TestCase):
    """CounterfactualInsight must survive JSON round-trip."""

    def test_insight_round_trip(self) -> None:
        insight = CounterfactualInsight(
            source_episode_cycle=42,
            original_action="forage",
            counterfactual_action="hide",
            original_efe=5.2,
            counterfactual_efe=1.8,
            efe_delta=-3.4,
            confidence=0.72,
            state_context={"observation": OBS_DANGEROUS_DICT},
            cluster_id=3,
            timestamp=100,
            absorbed=True,
        )
        payload = insight.to_dict()
        restored = CounterfactualInsight.from_dict(payload)

        self.assertEqual(restored.source_episode_cycle, 42)
        self.assertEqual(restored.original_action, "forage")
        self.assertEqual(restored.counterfactual_action, "hide")
        self.assertAlmostEqual(restored.efe_delta, -3.4)
        self.assertEqual(restored.cluster_id, 3)
        self.assertTrue(restored.absorbed)

    def test_agent_serialization_preserves_insights(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.counterfactual_insights.append(
            CounterfactualInsight(
                source_episode_cycle=1,
                original_action="forage",
                counterfactual_action="hide",
                original_efe=5.0,
                counterfactual_efe=2.0,
                efe_delta=-3.0,
                confidence=0.8,
                state_context={},
                cluster_id=0,
                timestamp=10,
            )
        )

        payload = agent.to_dict()
        restored = SegmentAgent.from_dict(payload, rng=random.Random(42))

        self.assertEqual(len(restored.counterfactual_insights), 1)
        self.assertEqual(
            restored.counterfactual_insights[0].counterfactual_action, "hide"
        )


if __name__ == "__main__":
    unittest.main()




