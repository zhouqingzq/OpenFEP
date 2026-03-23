from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.prediction_ledger import PredictionHypothesis
from segmentum.slow_learning import SlowVariableLearner
from segmentum.types import MemoryEpisode


def _replay_episode(
    tick: int,
    *,
    action: str = "hide",
    outcome: str = "survival_threat",
    danger: float = 0.86,
    stress: float = 0.74,
    fatigue: float = 0.71,
) -> dict[str, object]:
    return {
        "timestamp": tick,
        "cycle": tick,
        "cluster_id": 1,
        "action_taken": action,
        "predicted_outcome": outcome,
        "total_surprise": 0.92,
        "prediction_error": 0.58,
        "observation": {"danger": danger, "social": 0.2},
        "body_state": {"stress": stress, "fatigue": fatigue, "energy": 0.34},
        "errors": {"danger": 0.44},
    }


def _decision(tick: int, action: str) -> dict[str, object]:
    return {"tick": tick, "action": action, "risk": 1.6, "dominant_component": "identity_bias"}


def _prediction() -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id="pred:env:danger",
        created_tick=1,
        last_updated_tick=1,
        source_module="test",
        prediction_type="environment_state",
        target_channels=("danger",),
        expected_state={"danger": 0.2},
        confidence=0.72,
        expected_horizon=1,
    )


class TestM230SlowVariableLearning(unittest.TestCase):
    def test_repeated_experience_gradually_changes_slow_variables(self) -> None:
        agent = SegmentAgent(rng=random.Random(7))
        replay_one = [_replay_episode(tick) for tick in range(1, 6)]
        audit_one = agent.slow_variable_learner.apply_sleep_cycle(
            sleep_cycle_id=1,
            tick=5,
            replay_batch=replay_one,
            decision_history=[_decision(tick, "hide") for tick in range(1, 6)],
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            social_memory=agent.social_memory,
            identity_tension_history=[],
            self_model=agent.self_model,
            body_state={"stress": 0.76, "fatigue": 0.72},
        )
        caution_after_first = agent.slow_variable_learner.state.traits.caution_bias
        replay_two = [_replay_episode(tick) for tick in range(6, 11)]
        audit_two = agent.slow_variable_learner.apply_sleep_cycle(
            sleep_cycle_id=2,
            tick=10,
            replay_batch=replay_two,
            decision_history=[_decision(tick, "hide") for tick in range(6, 11)],
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            social_memory=agent.social_memory,
            identity_tension_history=[],
            self_model=agent.self_model,
            body_state={"stress": 0.78, "fatigue": 0.74},
        )

        self.assertGreater(caution_after_first, 0.5)
        self.assertGreater(agent.slow_variable_learner.state.traits.caution_bias, caution_after_first)
        self.assertTrue(any(item.status in {"accepted", "clipped"} for item in audit_one.updates))
        self.assertTrue(any(item.status in {"accepted", "clipped"} for item in audit_two.updates))

    def test_single_event_does_not_create_oversized_drift(self) -> None:
        learner = SlowVariableLearner()
        audit = learner.apply_sleep_cycle(
            sleep_cycle_id=1,
            tick=1,
            replay_batch=[_replay_episode(1)],
            decision_history=[_decision(1, "hide")],
            prediction_ledger=SegmentAgent().prediction_ledger,
            verification_loop=SegmentAgent().verification_loop,
            social_memory=SegmentAgent().social_memory,
            identity_tension_history=[],
            self_model=SegmentAgent().self_model,
            body_state={"stress": 0.74, "fatigue": 0.71},
        )

        self.assertLessEqual(abs(learner.state.traits.caution_bias - 0.5), 0.05)
        self.assertTrue(any(item.clipped_reason for item in audit.updates))

    def test_protected_anchors_block_abrupt_erosion(self) -> None:
        learner = SlowVariableLearner()
        learner.state.identity.commitment_stability = 0.5
        replay_batch = [_replay_episode(tick, stress=0.82, fatigue=0.8) for tick in range(1, 4)]
        audit = learner.apply_sleep_cycle(
            sleep_cycle_id=1,
            tick=3,
            replay_batch=replay_batch,
            decision_history=[_decision(tick, "forage") for tick in range(1, 4)],
            prediction_ledger=SegmentAgent().prediction_ledger,
            verification_loop=SegmentAgent().verification_loop,
            social_memory=SegmentAgent().social_memory,
            identity_tension_history=[{"tick": 3, "identity_tension": 0.9} for _ in range(3)],
            self_model=SegmentAgent().self_model,
            body_state={"stress": 0.86, "fatigue": 0.82},
        )

        commitment_updates = [
            item for item in audit.updates if item.variable_path == "identity.commitment_stability"
        ]
        self.assertTrue(commitment_updates)
        self.assertGreaterEqual(learner.state.identity.commitment_stability, 0.48)
        self.assertTrue(any(item.protected_anchor == "commitment-continuity" for item in commitment_updates))

    def test_drift_budget_and_anti_collapse_limit_total_update(self) -> None:
        learner = SlowVariableLearner()
        replay_batch = [_replay_episode(tick) for tick in range(1, 15)]
        audit = learner.apply_sleep_cycle(
            sleep_cycle_id=1,
            tick=14,
            replay_batch=replay_batch,
            decision_history=[_decision(tick, "hide") for tick in range(1, 15)],
            prediction_ledger=SegmentAgent().prediction_ledger,
            verification_loop=SegmentAgent().verification_loop,
            social_memory=SegmentAgent().social_memory,
            identity_tension_history=[{"tick": 12, "identity_tension": 0.8}],
            self_model=SegmentAgent().self_model,
            body_state={"stress": 0.84, "fatigue": 0.8},
        )

        total_delta = sum(abs(item.delta) for item in audit.updates)
        self.assertLessEqual(total_delta, learner.drift_budget.max_total_delta_per_cycle + 1e-6)
        self.assertTrue(audit.anti_collapse_triggered)

    def test_sleep_integration_and_snapshot_roundtrip_preserve_slow_learning(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        agent.episodes = [
            MemoryEpisode(
                cycle=1,
                choice="hide",
                free_energy_before=1.0,
                free_energy_after=0.6,
                dopamine_gain=0.1,
                observation={"danger": 0.82},
                prediction={"danger": 0.4},
                errors={"danger": 0.42},
                body_state={"stress": 0.78, "fatigue": 0.74},
            )
        ]
        replay_batch = [_replay_episode(tick) for tick in range(1, 6)]
        agent.long_term_memory.episodes = list(replay_batch)
        agent.long_term_memory.replay_during_sleep = lambda rng=None: list(replay_batch)
        agent.long_term_memory.assign_clusters = lambda: 1
        agent.long_term_memory.reconstruct_transitions = lambda batch: []
        agent.long_term_memory.compress_episodes = lambda current_cycle=None: 0
        agent._update_transition_model_from_replay = lambda dataset: 0
        agent._mine_sleep_patterns = lambda replay: (0, 0, 0, 0)
        agent._apply_sleep_consolidation = lambda consolidation: (0, 0, 0)
        agent._apply_narrative_sleep_updates = lambda replay: []
        agent._replay_action_prediction_error = lambda replay: 0.5
        agent._grouped_replay_pe = lambda replay: {}
        agent._conditioned_consolidation_metrics = lambda *args, **kwargs: None
        agent.dream_replay = lambda recent: []
        agent._surprise_based_forgetting = lambda replay: (0, 0)
        agent.goal_stack.review_conflicts = lambda tick: []

        class _CFSummary:
            episodes_evaluated = 0
            insights_generated = 0
            insights_absorbed = 0
            energy_spent = 0.0
            counterfactual_log = []

        with patch("segmentum.agent.run_counterfactual_phase", return_value=([], _CFSummary())):
            summary = agent.sleep()

        self.assertGreater(summary.slow_learning_updates, 0)
        self.assertTrue(agent.narrative_trace[-1]["slow_learning"]["updates"])
        restored = SegmentAgent.from_dict(agent.to_dict(), rng=random.Random(13))
        self.assertAlmostEqual(
            restored.slow_variable_learner.state.traits.caution_bias,
            agent.slow_variable_learner.state.traits.caution_bias,
            places=6,
        )

    def test_slow_variables_affect_policy_and_verification(self) -> None:
        base = SegmentAgent(rng=random.Random(21))
        adapted = SegmentAgent(rng=random.Random(21))
        adapted.slow_variable_learner.state.traits.caution_bias = 0.86
        adapted.slow_variable_learner.state.traits.threat_sensitivity = 0.84
        adapted.slow_variable_learner.state.identity.continuity_resilience = 0.38

        observation = Observation(
            food=0.55,
            danger=0.78,
            novelty=0.22,
            shelter=0.34,
            temperature=0.5,
            social=0.2,
        )
        base_diag = base.decision_cycle(observation)["diagnostics"]
        adapted_diag = adapted.decision_cycle(observation)["diagnostics"]
        base_scores = {item.choice: item for item in base_diag.ranked_options}
        adapted_scores = {item.choice: item for item in adapted_diag.ranked_options}

        self.assertGreater(adapted_scores["hide"].policy_score, base_scores["hide"].policy_score)
        self.assertLess(adapted_scores["forage"].policy_score, base_scores["forage"].policy_score)

        base.prediction_ledger.predictions.append(_prediction())
        adapted.prediction_ledger.predictions.append(_prediction())
        base_refresh = base.verification_loop.refresh_targets(
            tick=2,
            ledger=base.prediction_ledger,
            subject_state=base.subject_state,
        )
        adapted_refresh = adapted.verification_loop.refresh_targets(
            tick=2,
            ledger=adapted.prediction_ledger,
            subject_state=adapted.subject_state,
        )
        self.assertTrue(base_refresh.created_targets)
        self.assertTrue(adapted_refresh.created_targets)
        self.assertGreater(
            adapted.verification_loop.active_targets[0].priority_score,
            base.verification_loop.active_targets[0].priority_score,
        )
        self.assertIn("slow_learning", adapted.explain_decision_details())


if __name__ == "__main__":
    unittest.main()
