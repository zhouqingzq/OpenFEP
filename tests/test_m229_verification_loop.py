from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation, SimulatedWorld
from segmentum.prediction_ledger import PredictionHypothesis
from segmentum.runtime import SegmentRuntime
from segmentum.verification import VerificationOutcome


def _observation() -> Observation:
    return Observation(
        food=0.66,
        danger=0.64,
        novelty=0.20,
        shelter=0.28,
        temperature=0.49,
        social=0.22,
    )


def _prediction(
    *,
    prediction_id: str = "pred:env:danger",
    target_channels: tuple[str, ...] = ("danger",),
    expected_state: dict[str, float] | None = None,
    expected_horizon: int = 1,
    linked_identity: bool = False,
) -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id=prediction_id,
        created_tick=1,
        last_updated_tick=1,
        source_module="test",
        prediction_type="environment_state",
        target_channels=target_channels,
        expected_state=expected_state or {"danger": 0.18},
        confidence=0.72,
        expected_horizon=expected_horizon,
        linked_identity_anchors=("anchor",) if linked_identity else (),
        linked_commitments=("stay_consistent",) if linked_identity else (),
        linked_goal="SURVIVAL",
    )


class TestM229VerificationLoop(unittest.TestCase):
    def test_creates_explicit_verification_targets_and_plan(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        agent.prediction_ledger.predictions.append(_prediction())

        update = agent.verification_loop.refresh_targets(
            tick=2,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )

        self.assertTrue(update.created_targets)
        self.assertEqual(len(agent.verification_loop.active_targets), 1)
        target = agent.verification_loop.active_targets[0]
        self.assertEqual(target.prediction_id, "pred:env:danger")
        self.assertEqual(target.plan.prediction_id, "pred:env:danger")
        self.assertTrue(target.plan.evidence_sought)
        self.assertTrue(target.plan.support_criteria)
        self.assertTrue(target.plan.falsification_criteria)
        self.assertEqual(target.plan.status, "active")
        self.assertIn(VerificationOutcome.CONFIRMED.value, {item.value for item in VerificationOutcome})
        self.assertIn(VerificationOutcome.EXPIRED_UNVERIFIED.value, {item.value for item in VerificationOutcome})

    def test_verification_bias_changes_action_workspace_memory_and_maintenance(self) -> None:
        base = SegmentAgent(rng=random.Random(9))
        with_verification = SegmentAgent(rng=random.Random(9))
        with_verification.prediction_ledger.predictions.append(_prediction())

        base_decision = base.decision_cycle(_observation())["diagnostics"]
        verification_decision = with_verification.decision_cycle(_observation())["diagnostics"]

        base_scores = {item.choice: item for item in base_decision.ranked_options}
        verification_scores = {item.choice: item for item in verification_decision.ranked_options}

        self.assertGreater(verification_scores["scan"].policy_score, base_scores["scan"].policy_score)
        self.assertGreater(verification_scores["scan"].verification_bias, 0.0)
        self.assertTrue(with_verification.verification_loop.workspace_focus())
        self.assertLess(with_verification.verification_loop.memory_threshold_delta(), 0.0)
        self.assertTrue(with_verification.verification_loop.maintenance_signal()["active_tasks"])
        self.assertIn("currently trying to verify", verification_decision.verification_summary)

    def test_confirming_and_falsifying_evidence_updates_ledger_and_history(self) -> None:
        confirmed_agent = SegmentAgent(rng=random.Random(5))
        confirmed_agent.prediction_ledger.predictions.append(
            _prediction(expected_state={"danger": 0.64})
        )
        confirmed_agent.verification_loop.refresh_targets(
            tick=2,
            ledger=confirmed_agent.prediction_ledger,
            subject_state=confirmed_agent.subject_state,
        )
        confirm_update = confirmed_agent.verification_loop.process_observation(
            tick=2,
            observation={"danger": 0.64},
            ledger=confirmed_agent.prediction_ledger,
            source="runtime_observation",
            subject_state=confirmed_agent.subject_state,
        )
        self.assertIn(VerificationOutcome.CONFIRMED.value, confirm_update.outcomes)
        self.assertFalse(confirmed_agent.verification_loop.active_targets)
        self.assertTrue(
            any(item.prediction_id == "pred:env:danger" for item in confirmed_agent.prediction_ledger.archived_predictions)
        )

        falsified_agent = SegmentAgent(rng=random.Random(6))
        falsified_agent.prediction_ledger.predictions.append(
            _prediction(prediction_id="pred:identity:danger", linked_identity=True)
        )
        falsified_agent.verification_loop.refresh_targets(
            tick=2,
            ledger=falsified_agent.prediction_ledger,
            subject_state=falsified_agent.subject_state,
        )
        falsify_update = falsified_agent.verification_loop.process_observation(
            tick=2,
            observation={"danger": 0.94},
            ledger=falsified_agent.prediction_ledger,
            source="runtime_observation",
            subject_state=falsified_agent.subject_state,
        )
        self.assertTrue(
            set(falsify_update.outcomes)
            & {
                VerificationOutcome.FALSIFIED.value,
                VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value,
            }
        )
        self.assertTrue(falsified_agent.verification_loop.falsification_history)
        self.assertTrue(falsified_agent.prediction_ledger.active_discrepancies())
        self.assertTrue(falsified_agent.prediction_ledger.active_discrepancies()[0].identity_relevant)

    def test_missing_evidence_expires_target_and_escalates_prediction(self) -> None:
        agent = SegmentAgent(rng=random.Random(12))
        agent.prediction_ledger.predictions.append(
            _prediction(
                prediction_id="pred:env:novelty",
                target_channels=("novelty",),
                expected_state={"novelty": 0.15},
                expected_horizon=1,
            )
        )
        agent.verification_loop.refresh_targets(
            tick=1,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )

        update = agent.verification_loop.process_observation(
            tick=3,
            observation={"danger": 0.50},
            ledger=agent.prediction_ledger,
            source="runtime_observation",
            subject_state=agent.subject_state,
        )

        self.assertIn(VerificationOutcome.EXPIRED_UNVERIFIED.value, update.outcomes)
        self.assertTrue(update.expired_targets)
        self.assertTrue(
            any(item.discrepancy_type == "verification_timeout" for item in agent.prediction_ledger.active_discrepancies())
        )
        self.assertEqual(agent.prediction_ledger.predictions[0].status, "escalated")

    def test_snapshot_roundtrip_and_trace_preserve_verification_state(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        agent.prediction_ledger.predictions.append(_prediction())
        agent.verification_loop.refresh_targets(
            tick=2,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )

        restored = SegmentAgent.from_dict(agent.to_dict(), rng=random.Random(13))
        self.assertEqual(len(restored.verification_loop.active_targets), 1)
        self.assertEqual(
            restored.verification_loop.active_targets[0].prediction_id,
            "pred:env:danger",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            runtime = SegmentRuntime(
                agent=restored,
                world=SimulatedWorld(seed=17),
                trace_path=trace_path,
            )
            runtime.run(cycles=1, verbose=False)
            trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(trace_lines)
            record = json.loads(trace_lines[-1])
            self.assertIn("verification_loop", record)
            self.assertIn(
                "verification_payload",
                record["decision_loop"],
            )


if __name__ == "__main__":
    unittest.main()
