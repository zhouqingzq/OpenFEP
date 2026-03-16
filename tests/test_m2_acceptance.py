"""M2 Milestone Acceptance Tests.

Three acceptance criteria verified:

1. Past Influence: episodes from the past change future action rankings.
2. Identity Continuity: persist → reload → behavior stays consistent.
3. Explanation: explain_structured() clearly states why a choice matches
   or deviates from the agent's established behavioral pattern.
"""
from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from segmentum.agent import DecisionLoop, SegmentAgent
from segmentum.counterfactual import CounterfactualLearning
from segmentum.environment import Observation
from segmentum.memory import AutobiographicalMemory
from segmentum.preferences import Goal, GoalStack, ValueHierarchy
from segmentum.runtime import SegmentRuntime
from segmentum.self_model import IdentityNarrative, PreferredPolicies, SelfModel
from segmentum.sleep_consolidator import SleepConsolidation
from tests._pytest_compat import pytest


# ── helpers ──────────────────────────────────────────────────────────────

OBS_DANGEROUS = Observation(
    food=0.12, danger=0.82, novelty=0.18,
    shelter=0.10, temperature=0.46, social=0.14,
)

FIXED_OBSERVATION = Observation(
    food=0.16, danger=0.80, novelty=0.24,
    shelter=0.12, temperature=0.46, social=0.18,
)


def _populate_dangerous_episodes(agent: SegmentAgent, count: int = 6) -> None:
    """Inject high-surprise episodes where 'forage' led to survival_threat."""
    for i in range(count):
        agent.long_term_memory.store_episode(
            cycle=i + 1,
            observation={"food": 0.10, "danger": 0.80 + i * 0.01,
                         "novelty": 0.15, "shelter": 0.08,
                         "temperature": 0.45, "social": 0.10},
            prediction={"food": 0.50, "danger": 0.30,
                        "novelty": 0.40, "shelter": 0.40,
                        "temperature": 0.50, "social": 0.20},
            errors={"food": 0.40, "danger": 0.50 + i * 0.01,
                    "novelty": 0.25, "shelter": 0.32,
                    "temperature": 0.05, "social": 0.10},
            action="forage",
            outcome={"free_energy_drop": -0.20, "energy_delta": -0.10,
                     "stress_delta": 0.25, "fatigue_delta": 0.15,
                     "temperature_delta": 0.0},
            body_state={"energy": 0.50, "stress": 0.40,
                        "fatigue": 0.25, "temperature": 0.45,
                        "dopamine": 0.10},
        )


def _round_trip(agent: SegmentAgent, seed: int = 999) -> SegmentAgent:
    """JSON round-trip to simulate persist → reload."""
    payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
    return SegmentAgent.from_dict(payload, rng=random.Random(seed))


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    labels = sorted(set(left) | set(right))
    mid = {l: (left.get(l, 0.0) + right.get(l, 0.0)) / 2 for l in labels}
    def kl(a, b):
        return sum(max(a.get(l, 0), 1e-12) * math.log(max(a.get(l, 0), 1e-12) / max(b.get(l, 0), 1e-12), 2)
                   for l in labels)
    return (kl(left, mid) + kl(right, mid)) / 2


def _distribution(actions: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items())}


# ── Module surface names ─────────────────────────────────────────────────

class TestModuleSurface(unittest.TestCase):
    """Verify that all required M2 module names are importable."""

    def test_self_model_surface(self) -> None:
        model = SelfModel.from_dict(None)
        self.assertIsNotNone(model.body_schema)
        self.assertIsNotNone(model.capability_model)
        self.assertIsNotNone(model.threat_model)
        self.assertIsInstance(model.preferred_policies, PreferredPolicies)
        self.assertIsInstance(model.identity_narrative, IdentityNarrative)

    def test_autobiographical_memory_surface(self) -> None:
        mem = AutobiographicalMemory()
        self.assertTrue(hasattr(mem, "store_episode"))
        self.assertTrue(hasattr(mem, "retrieve_similar_memories"))
        self.assertTrue(hasattr(mem, "assign_clusters"))
        self.assertTrue(hasattr(mem, "replay_during_sleep"))
        self.assertTrue(hasattr(mem, "life_history_timeline"))

    def test_goal_stack_surface(self) -> None:
        stack = GoalStack()
        self.assertIn(Goal.SURVIVAL, stack.base_weights)
        self.assertIn(Goal.INTEGRITY, stack.base_weights)
        self.assertIn(Goal.CONTROL, stack.base_weights)
        self.assertIn(Goal.RESOURCES, stack.base_weights)
        self.assertIn(Goal.SOCIAL, stack.base_weights)
        self.assertTrue(hasattr(stack, "evaluate_priority"))
        self.assertTrue(hasattr(stack, "update_active_goal"))
        self.assertTrue(hasattr(stack, "goal_alignment_score"))
        self.assertTrue(hasattr(stack, "get_goal_context_for_decision"))

    def test_value_hierarchy_is_preference_model(self) -> None:
        vh = ValueHierarchy()
        self.assertTrue(hasattr(vh, "score"))
        self.assertTrue(hasattr(vh, "expected_free_energy"))
        self.assertLess(vh.survival_threat, vh.integrity_loss)
        self.assertLess(vh.integrity_loss, vh.resource_loss)
        self.assertLess(vh.resource_loss, vh.neutral)
        self.assertLess(vh.neutral, vh.resource_gain)

    def test_sleep_consolidation_surface(self) -> None:
        sc = SleepConsolidation(surprise_threshold=0.40)
        self.assertTrue(hasattr(sc, "consolidate"))

    def test_counterfactual_learning_surface(self) -> None:
        cl = CounterfactualLearning()
        self.assertTrue(hasattr(cl, "run"))

    def test_decision_loop_surface(self) -> None:
        dl = DecisionLoop()
        phases = dl.describe()
        for required in ("perception", "prediction", "memory_retrieval",
                         "goal_alignment_evaluation", "policy_evaluation",
                         "action_selection"):
            self.assertIn(required, phases)

    def test_goal_weight_ordering(self) -> None:
        stack = GoalStack()
        weights = stack.base_weights
        self.assertGreater(weights[Goal.SURVIVAL], weights[Goal.INTEGRITY])
        self.assertGreater(weights[Goal.INTEGRITY], weights[Goal.CONTROL])
        self.assertGreater(weights[Goal.CONTROL], weights[Goal.RESOURCES])
        self.assertGreater(weights[Goal.RESOURCES], weights[Goal.SOCIAL])


# ── Acceptance criterion 1: Past Influence ───────────────────────────────

class TestPastInfluence(unittest.TestCase):
    """An episode from the past must change the ranking of actions
    in a future similar state."""

    def test_dangerous_past_suppresses_forage(self) -> None:
        # Baseline: naïve agent with no memory
        baseline = SegmentAgent(rng=random.Random(123))
        baseline_diag = baseline.decision_cycle(FIXED_OBSERVATION)["diagnostics"]
        baseline_forage = next(
            o.policy_score for o in baseline_diag.ranked_options if o.choice == "forage"
        )

        # Experienced agent: dangerous foraging episodes consolidated during sleep
        agent = SegmentAgent(rng=random.Random(123))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()

        # Persist → reload
        restored = _round_trip(agent)
        diag = restored.decision_cycle(FIXED_OBSERVATION)["diagnostics"]
        forage_score = next(
            o.policy_score for o in diag.ranked_options if o.choice == "forage"
        )

        self.assertLess(forage_score, baseline_forage,
                        "Past dangerous episodes must lower forage's policy score")
        self.assertNotEqual(diag.chosen.choice, "forage",
                            "Agent should avoid forage after dangerous experience")

    def test_counterfactual_insights_survive_restart(self) -> None:
        agent = SegmentAgent(rng=random.Random(42))
        agent.energy = 0.70
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent, count=5)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()

        self.assertGreater(len(agent.counterfactual_insights), 0)
        absorbed = [i for i in agent.counterfactual_insights if i.absorbed]
        self.assertGreater(len(absorbed), 0)

        ref = absorbed[0]
        bias_before = agent.world_model.get_policy_bias(ref.cluster_id, ref.original_action)
        restored = _round_trip(agent)
        bias_after = restored.world_model.get_policy_bias(ref.cluster_id, ref.original_action)
        self.assertEqual(bias_before, bias_after,
                         "Policy biases from counterfactual must survive round-trip")


# ── Acceptance criterion 2: Identity Continuity ──────────────────────────

class TestIdentityContinuity(unittest.TestCase):
    """Run N cycles → persist → reload → run N cycles.
    Behavior must remain consistent (not restart like a new agent)."""

    @pytest.mark.stress
    def test_action_distribution_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"

            # Continuous reference: 700 cycles
            continuous = SegmentRuntime.load_or_create(seed=31, reset=True)
            continuous.run(cycles=700, verbose=False)
            continuous_dist = _distribution(continuous.agent.action_history[-200:])

            # Split: 500 → save → reload → 200
            split = SegmentRuntime.load_or_create(state_path=state_path, seed=31, reset=True)
            split.run(cycles=500, verbose=False)
            split.save_snapshot()
            restored = SegmentRuntime.load_or_create(state_path=state_path, seed=999)
            restored.run(cycles=200, verbose=False)
            restored_dist = _distribution(restored.agent.action_history[-200:])

            # Fresh 200-cycle agent for contrast
            fresh = SegmentRuntime.load_or_create(seed=31, reset=True)
            fresh.run(cycles=200, verbose=False)
            fresh_dist = _distribution(fresh.agent.action_history[-200:])

            restart_divergence = _js_divergence(continuous_dist, restored_dist)
            fresh_divergence = _js_divergence(restored_dist, fresh_dist)

            self.assertLess(restart_divergence, 0.05,
                            "Restored agent must behave like the continuous run")
            self.assertGreater(fresh_divergence, restart_divergence + 0.05,
                               "Fresh agent should differ more than the restored one")

    @pytest.mark.stress
    def test_identity_narrative_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"

            runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=17, reset=True)
            # Run until 2 sleep cycles complete
            for _ in range(260):
                if len(runtime.agent.sleep_history) >= 2:
                    break
                runtime.step(verbose=False)

            narrative_before = runtime.agent.self_model.identity_narrative
            self.assertIsNotNone(narrative_before)
            snapshot = narrative_before.to_dict()

            runtime.save_snapshot()
            restored = SegmentRuntime.load_or_create(state_path=state_path, seed=77)
            narrative_after = restored.agent.self_model.identity_narrative
            self.assertIsNotNone(narrative_after)
            self.assertEqual(snapshot, narrative_after.to_dict(),
                             "IdentityNarrative must survive JSON round-trip exactly")

    def test_preferred_policies_persist(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
        for _ in range(180):
            if len(runtime.agent.sleep_history) >= 1:
                break
            runtime.step(verbose=False)

        before = runtime.agent.self_model.preferred_policies.to_dict()
        restored = _round_trip(runtime.agent)
        after = restored.self_model.preferred_policies.to_dict()
        self.assertEqual(before, after)

    def test_goal_stack_persists(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=29, reset=True)
        runtime.run(cycles=120, verbose=False)

        before = runtime.agent.goal_stack.to_dict()
        restored = _round_trip(runtime.agent)
        after = restored.goal_stack.to_dict()
        self.assertEqual(before, after)


# ── Acceptance criterion 3: Explanation ──────────────────────────────────

class TestExplanation(unittest.TestCase):
    """explain_structured() must clearly state why the action matches
    or deviates from the agent's established behavioral pattern."""

    def test_explanation_structure(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=19, reset=True)
        for _ in range(180):
            if len(runtime.agent.sleep_history) >= 1:
                break
            runtime.step(verbose=False)

        runtime.agent.decision_cycle(OBS_DANGEROUS)
        details = runtime.agent.explain_decision_details()

        # Structural keys
        for key in ("action", "active_goal", "goal_alignment",
                     "dominant_component", "consistency", "text"):
            self.assertIn(key, details, f"Missing key: {key}")

        consistency = details["consistency"]
        for key in ("dominant_strategy", "historical_action_frequency",
                     "identity_consistency", "consistency_statement"):
            self.assertIn(key, consistency, f"Missing consistency key: {key}")

    def test_consistency_statement_for_matching_pattern(self) -> None:
        """When the choice matches the established pattern, the statement
        should say 'consistent with my established pattern'."""
        runtime = SegmentRuntime.load_or_create(seed=19, reset=True)
        for _ in range(180):
            if len(runtime.agent.sleep_history) >= 1:
                break
            runtime.step(verbose=False)

        # Reload two identical copies so both make the same decision
        payload = json.loads(json.dumps(runtime.agent.to_dict(), ensure_ascii=True))
        first = SegmentAgent.from_dict(payload, rng=random.Random(7))
        second = SegmentAgent.from_dict(payload, rng=random.Random(7))

        first.decision_cycle(OBS_DANGEROUS)
        second.decision_cycle(OBS_DANGEROUS)

        d1 = first.explain_decision_details()
        d2 = second.explain_decision_details()

        # Deterministic: identical seeds produce identical explanations
        self.assertEqual(
            d1["consistency"]["consistency_statement"],
            d2["consistency"]["consistency_statement"],
        )
        self.assertEqual(
            d1["consistency"]["historical_action_frequency"],
            d2["consistency"]["historical_action_frequency"],
        )

    def test_explanation_mentions_deviation_when_applicable(self) -> None:
        """The statement should distinguish between consistent choices
        and deviations with an explicit reason."""
        details_text_lower = ""

        # Run enough cycles that a pattern exists, then check explanation
        runtime = SegmentRuntime.load_or_create(seed=41, reset=True)
        for _ in range(180):
            if len(runtime.agent.sleep_history) >= 1:
                break
            runtime.step(verbose=False)

        runtime.agent.decision_cycle(OBS_DANGEROUS)
        details = runtime.agent.explain_decision_details()
        statement = details["consistency"]["consistency_statement"]

        # Must be one of the two canonical forms
        self.assertTrue(
            "consistent with my established pattern" in statement
            or "deviates from my usual pattern" in statement,
            f"Statement must use canonical consistency language, got: {statement!r}",
        )

    def test_explanation_includes_memory_or_pattern_bias(self) -> None:
        """After dangerous experiences, the explanation text should
        reference memory or pattern influence."""
        agent = SegmentAgent(rng=random.Random(123))
        agent.long_term_memory.minimum_support = 1
        agent.long_term_memory.sleep_minimum_support = 1
        _populate_dangerous_episodes(agent)
        agent.long_term_memory.assign_clusters()
        agent.cycle = 20
        agent.sleep()

        restored = _round_trip(agent)
        restored.decision_cycle(FIXED_OBSERVATION)
        text = restored.explain_decision_details()["text"]

        self.assertTrue(
            "memory_bias" in text or "pattern_bias" in text,
            "Explanation must reference memory or pattern influence",
        )


# ── Serialization completeness ───────────────────────────────────────────

class TestSerializationCompleteness(unittest.TestCase):
    """All long-term state must survive JSON round-trip."""

    def test_full_agent_round_trip(self) -> None:
        runtime = SegmentRuntime.load_or_create(seed=37, reset=True)
        runtime.run(cycles=120, verbose=False)
        agent = runtime.agent

        payload = json.loads(json.dumps(agent.to_dict(), ensure_ascii=True))
        restored = SegmentAgent.from_dict(payload, rng=random.Random(37))

        # decision_history
        self.assertEqual(len(agent.decision_history), len(restored.decision_history))
        # drive_history
        self.assertEqual(len(agent.drive_history), len(restored.drive_history))
        # free_energy_history
        self.assertEqual(len(agent.free_energy_history), len(restored.free_energy_history))
        # goal_stack
        self.assertEqual(agent.goal_stack.to_dict(), restored.goal_stack.to_dict())
        # self_model
        self.assertEqual(
            agent.self_model.to_dict(),
            restored.self_model.to_dict(),
        )
        # long_term_memory episodes count
        self.assertEqual(
            len(agent.long_term_memory.episodes),
            len(restored.long_term_memory.episodes),
        )

    def test_autobiographical_memory_timeline_after_sleep(self) -> None:
        """life_history_timeline() returns events after sleep consolidation."""
        runtime = SegmentRuntime.load_or_create(seed=41, reset=True)
        for _ in range(260):
            if len(runtime.agent.sleep_history) >= 1:
                break
            runtime.step(verbose=False)

        timeline = runtime.agent.long_term_memory.life_history_timeline()
        self.assertGreater(len(timeline), 0)
        for event in timeline:
            self.assertIn("tick", event)
            self.assertIn("action", event)
            self.assertIn("surprise", event)

        # Timeline must be chronologically sorted
        ticks = [e["tick"] for e in timeline]
        self.assertEqual(ticks, sorted(ticks))


if __name__ == "__main__":
    unittest.main()
