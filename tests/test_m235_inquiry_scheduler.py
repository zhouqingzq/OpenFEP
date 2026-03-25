from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.inquiry_scheduler import (
    InquiryBudgetScheduler,
    InquirySchedulingDecision,
)
from segmentum.narrative_experiment import ExperimentDesignResult, ExperimentPlan
from segmentum.narrative_uncertainty import (
    DecisionRelevanceMap,
    NarrativeUnknown,
    UncertaintyDecompositionResult,
)
from segmentum.prediction_ledger import PredictionHypothesis, PredictionLedger
from segmentum.runtime import SegmentRuntime
from segmentum.subject_state import SubjectState
from segmentum.verification import (
    VerificationLoop,
    VerificationOutcome,
    VerificationPlan,
    VerificationTarget,
)


def _unknown(
    unknown_id: str,
    *,
    unknown_type: str,
    uncertainty: float,
    total_score: float,
    verification_urgency: float,
    continuity_impact: float = 0.0,
    risk_level: float = 0.0,
) -> NarrativeUnknown:
    return NarrativeUnknown(
        unknown_id=unknown_id,
        unknown_type=unknown_type,
        source_episode_id="ep:test",
        source_span="span",
        unresolved_reason=f"{unknown_type} unresolved",
        uncertainty_level=uncertainty,
        action_relevant=True,
        decision_relevance=DecisionRelevanceMap(
            verification_urgency=verification_urgency,
            continuity_impact=continuity_impact,
            risk_level=risk_level,
            total_score=total_score,
        ),
        competing_hypothesis_ids=(f"hyp:{unknown_id}",),
        promotion_reason="retained for bounded inquiry",
    )


def _plan(
    plan_id: str,
    *,
    action: str,
    status: str = "queued_experiment",
    score: float = 0.8,
    informative_value: float = 0.8,
    inconclusive_count: int = 0,
) -> ExperimentPlan:
    return ExperimentPlan(
        plan_id=plan_id,
        candidate_id=f"cand:{plan_id}",
        target_unknown_id="unk:hi",
        target_hypothesis_ids=("hyp:hi",),
        selected_action=action,
        selected_reason="high information gain and decision relevance",
        evidence_sought=("observe:danger",),
        outcome_differences=("hyp:hi:danger persists",),
        fallback_behavior="rest",
        expected_horizon=1,
        status=status,
        score=score,
        informative_value=informative_value,
        inconclusive_count=inconclusive_count,
    )


def _prediction(
    prediction_id: str,
    *,
    confidence: float = 0.75,
    decision_relevance: float = 0.8,
    attempts: int = 0,
) -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id=prediction_id,
        created_tick=1,
        last_updated_tick=1,
        source_module="narrative_experiment",
        prediction_type="danger_probe",
        target_channels=("danger",),
        expected_state={"danger": 0.8},
        confidence=confidence,
        expected_horizon=2,
        linked_unknown_ids=("unk:hi",),
        linked_hypothesis_ids=("hyp:hi",),
        linked_experiment_plan_id="plan:high",
        decision_relevance=decision_relevance,
        verification_attempts=attempts,
    )


def _archived_deferred_target(prediction_id: str, suffix: str) -> VerificationTarget:
    plan = VerificationPlan(
        prediction_id=prediction_id,
        selected_reason="prior probe",
        evidence_sought=("observe:danger",),
        support_criteria=("danger high",),
        falsification_criteria=("danger low",),
        expected_horizon=1,
        created_tick=1,
        expires_tick=2,
        attention_channels=("danger",),
    )
    return VerificationTarget(
        target_id=f"vt:{prediction_id}:{suffix}",
        prediction_id=prediction_id,
        created_tick=1,
        priority_score=0.5,
        selected_reason="prior probe",
        plan=plan,
        outcome=VerificationOutcome.DEFERRED.value,
        status="deferred",
    )


class TestM235InquiryScheduler(unittest.TestCase):
    def test_scheduler_prefers_high_value_over_merely_uncertain_candidate(self) -> None:
        scheduler = InquiryBudgetScheduler()
        uncertainty = UncertaintyDecompositionResult(
            unknowns=(
                _unknown(
                    "unk:low-value",
                    unknown_type="general",
                    uncertainty=0.98,
                    total_score=0.08,
                    verification_urgency=0.06,
                ),
            )
        )
        experiment = ExperimentDesignResult(plans=(_plan("plan:high", action="scan"),))

        state = scheduler.schedule(
            tick=3,
            narrative_uncertainty=uncertainty,
            experiment_design=experiment,
            subject_state=SubjectState(),
        )

        high_value_decision = state.decision_for_plan("plan:high")
        low_value_decision = next(
            item for item in state.decisions if item.linked_unknown_id == "unk:low-value"
        )
        self.assertIsNotNone(high_value_decision)
        self.assertIn(
            high_value_decision.decision,
            {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            },
        )
        self.assertIn(
            low_value_decision.decision,
            {
                InquirySchedulingDecision.SUPPRESS.value,
                InquirySchedulingDecision.DEFER.value,
            },
        )
        self.assertGreater(high_value_decision.priority_score, low_value_decision.priority_score)

    def test_scheduler_limits_verification_slots_and_cools_down_low_yield_predictions(self) -> None:
        scheduler = InquiryBudgetScheduler(max_verification_slots=1)
        ledger = PredictionLedger(
            predictions=[
                _prediction("pred:cooldown", confidence=0.82, decision_relevance=0.9, attempts=2),
                _prediction("pred:active", confidence=0.84, decision_relevance=0.88, attempts=0),
            ]
        )
        verification = VerificationLoop(
            archived_targets=[
                _archived_deferred_target("pred:cooldown", "a"),
                _archived_deferred_target("pred:cooldown", "b"),
            ]
        )

        state = scheduler.schedule(
            tick=5,
            prediction_ledger=ledger,
            verification_loop=verification,
            subject_state=SubjectState(),
        )

        cooled = state.decision_for_prediction("pred:cooldown")
        active = state.decision_for_prediction("pred:active")
        self.assertIsNotNone(cooled)
        self.assertEqual(cooled.decision, InquirySchedulingDecision.COOLDOWN.value)
        self.assertIsNotNone(active)
        self.assertEqual(len(state.verification_assignments), 1)
        self.assertEqual(state.verification_assignments[0].prediction_id, "pred:active")

    def test_scheduler_drives_workspace_and_action_bias(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        agent.cycle = 7
        agent.latest_narrative_uncertainty = UncertaintyDecompositionResult(
            unknowns=(
                _unknown(
                    "unk:danger",
                    unknown_type="threat_persistence",
                    uncertainty=0.74,
                    total_score=0.82,
                    verification_urgency=0.86,
                    continuity_impact=0.72,
                ),
            )
        )
        agent.latest_narrative_experiment = ExperimentDesignResult(
            plans=(_plan("plan:high", action="scan", score=0.84, informative_value=0.88),)
        )
        agent.prediction_ledger.predictions.append(_prediction("pred:high"))

        agent._refresh_inquiry_budget()
        inquiry_state = agent.inquiry_budget_scheduler.state
        focus = inquiry_state.workspace_focus()
        self.assertIn("danger", focus)
        self.assertGreater(inquiry_state.action_bias("scan"), inquiry_state.action_bias("forage"))

        workspace = agent.global_workspace.broadcast(
            tick=agent.cycle,
            observation={"food": 0.2, "danger": 0.7},
            prediction={"food": 0.3, "danger": 0.2},
            errors={"food": -0.1, "danger": 0.5},
            attention_trace=None,
            ledger_focus=focus,
        )
        self.assertIsNotNone(workspace)
        self.assertIn("danger", [item.channel for item in workspace.broadcast_contents])

    def test_scheduler_state_survives_snapshot_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path,
                trace_path=trace_path,
                seed=13,
                reset=True,
            )
            runtime.agent.latest_narrative_uncertainty = UncertaintyDecompositionResult(
                unknowns=(
                    _unknown(
                        "unk:trace",
                        unknown_type="threat_persistence",
                        uncertainty=0.68,
                        total_score=0.78,
                        verification_urgency=0.82,
                        continuity_impact=0.6,
                    ),
                )
            )
            runtime.agent.prediction_ledger.predictions.append(_prediction("pred:trace"))

            runtime.step(verbose=False)
            runtime.save_snapshot()

            trace_records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(trace_records)
            self.assertIn("inquiry_scheduler", trace_records[-1])
            self.assertIn(
                "inquiry_scheduler_payload",
                trace_records[-1]["decision_loop"],
            )

            restored = SegmentRuntime.load_or_create(
                state_path,
                trace_path=trace_path,
                seed=13,
                reset=False,
            )
            self.assertTrue(restored.agent.inquiry_budget_scheduler.state.decisions)
            self.assertEqual(
                restored.agent.inquiry_budget_scheduler.state.active_candidate_ids,
                runtime.agent.inquiry_budget_scheduler.state.active_candidate_ids,
            )


if __name__ == "__main__":
    unittest.main()
