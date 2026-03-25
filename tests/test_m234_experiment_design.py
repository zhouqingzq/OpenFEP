from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.action_registry import ActionRegistry, build_default_action_registry
from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_experiment import InquiryPlanStatus, NarrativeExperimentDesigner
from segmentum.narrative_types import NarrativeEpisode
from segmentum.narrative_uncertainty import UncertaintyDecompositionResult
from segmentum.runtime import SegmentRuntime
from segmentum.subject_state import SubjectState, derive_subject_state


def _compiler() -> NarrativeCompiler:
    return NarrativeCompiler()


def _social_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-social",
        timestamp=4,
        source="user_diary",
        raw_text=(
            "My counterpart said they would meet me, but left me outside and later sent a vague apology. "
            "I do not know whether this was betrayal, temporary constraint, or a misunderstanding."
        ),
        tags=["social"],
        metadata={"counterpart_id": "counterpart-x", "chapter_id": 4},
    )


def _threat_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-threat",
        timestamp=5,
        source="user_diary",
        raw_text=(
            "A predator attacked near the shelter, but I still cannot tell whether it was a stable threat source "
            "or a local accident that will fade if I wait."
        ),
        tags=["threat"],
        metadata={"environment_id": "shelter-edge", "chapter_id": 5},
    )


def _noise_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-noise",
        timestamp=6,
        source="user_diary",
        raw_text="Everything sounded dramatic, but it was only a slogan on a wall and nothing actionable happened.",
        tags=["noise"],
        metadata={},
    )


class TestM234ExperimentDesign(unittest.TestCase):
    def test_competing_hypotheses_become_predictions_and_candidates(self) -> None:
        compiler = _compiler()
        agent = SegmentAgent()
        compiled = compiler.compile_episode(_social_episode())
        agent.ingest_narrative_episode(compiled)

        design = agent.latest_narrative_experiment
        self.assertTrue(design.predictions)
        self.assertTrue(design.candidates)
        self.assertGreaterEqual(len({item.parent_hypothesis_id for item in design.predictions}), 2)
        self.assertTrue(any(item.action_name == "seek_contact" for item in design.candidates))
        self.assertTrue(any(item.action_name == "scan" for item in design.candidates))
        contact = next(item for item in design.candidates if item.action_name == "seek_contact")
        scan = next(item for item in design.candidates if item.action_name == "scan")
        self.assertNotEqual(contact.information_gain.score, scan.information_gain.score)
        self.assertTrue(any(plan.status in {InquiryPlanStatus.ACTIVE_EXPERIMENT.value, InquiryPlanStatus.QUEUED_EXPERIMENT.value} for plan in design.plans))

    def test_risk_and_cost_tradeoff_defers_contact_when_socially_destabilized(self) -> None:
        compiler = _compiler()
        uncertainty = compiler.compile_episode(_social_episode()).uncertainty_decomposition
        agent = SegmentAgent()
        subject_state = SubjectState(status_flags={"socially_destabilized": True, "threatened": False})
        design = NarrativeExperimentDesigner().design(
            tick=4,
            uncertainty=UncertaintyDecompositionResult.from_dict(uncertainty),
            action_registry=agent.action_registry,
            subject_state=subject_state,
            active_goal="SOCIAL",
        )
        seek_contact = next(item for item in design.candidates if item.action_name == "seek_contact")
        scan = next(item for item in design.candidates if item.action_name == "scan")
        self.assertGreater(seek_contact.risk_profile.score, scan.risk_profile.score)
        self.assertTrue(any(plan.selected_action == "seek_contact" and plan.status == InquiryPlanStatus.DEFERRED_FOR_RISK.value for plan in design.plans))
        self.assertTrue(any(plan.selected_action == "scan" and plan.status in {InquiryPlanStatus.ACTIVE_EXPERIMENT.value, InquiryPlanStatus.QUEUED_EXPERIMENT.value} for plan in design.plans))

    def test_social_goal_prefers_contact_probe_before_passive_scan_when_stable(self) -> None:
        compiler = _compiler()
        uncertainty = compiler.compile_episode(_social_episode()).uncertainty_decomposition
        agent = SegmentAgent()
        design = NarrativeExperimentDesigner().design(
            tick=4,
            uncertainty=UncertaintyDecompositionResult.from_dict(uncertainty),
            action_registry=agent.action_registry,
            subject_state=SubjectState(status_flags={"socially_destabilized": False, "threatened": False}),
            active_goal="SOCIAL",
        )

        self.assertGreaterEqual(len(design.candidates), 2)
        self.assertEqual(design.candidates[0].action_name, "seek_contact")
        self.assertEqual(design.plans[0].selected_action, "seek_contact")
        self.assertIn(design.plans[0].status, {InquiryPlanStatus.ACTIVE_EXPERIMENT.value, InquiryPlanStatus.QUEUED_EXPERIMENT.value})

    def test_safety_goal_prefers_scan_probe_for_threat_ambiguity(self) -> None:
        compiler = _compiler()
        uncertainty = compiler.compile_episode(_threat_episode()).uncertainty_decomposition
        agent = SegmentAgent()
        design = NarrativeExperimentDesigner().design(
            tick=5,
            uncertainty=UncertaintyDecompositionResult.from_dict(uncertainty),
            action_registry=agent.action_registry,
            active_goal="SAFETY",
        )

        self.assertTrue(design.candidates)
        self.assertEqual(design.candidates[0].action_name, "scan")
        self.assertEqual(design.plans[0].selected_action, "scan")

    def test_experiment_design_changes_ledger_verification_subject_state_and_explanations(self) -> None:
        compiler = _compiler()
        agent = SegmentAgent()
        compiled = compiler.compile_episode(_threat_episode())
        agent.ingest_narrative_episode(compiled)
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)

        result = agent.decision_cycle(
            Observation(
                food=0.32,
                danger=0.68,
                novelty=0.18,
                shelter=0.64,
                temperature=0.48,
                social=0.14,
            )
        )
        diagnostics = result["diagnostics"]

        self.assertTrue(agent.latest_narrative_experiment.plans)
        self.assertTrue(any(item.source_module == "narrative_experiment" for item in agent.prediction_ledger.active_predictions()))
        self.assertTrue(any(item.linked_experiment_plan_id for item in agent.verification_loop.active_targets))
        self.assertTrue(agent.subject_state.active_inquiries)
        ranked = {item.choice: item for item in diagnostics.ranked_options}
        self.assertGreater(ranked["scan"].experiment_bias, 0.0)
        self.assertIn("experiment design", diagnostics.explanation.lower())
        self.assertIn("narrative_experiment", agent.explain_decision_details())

    def test_surface_only_noise_does_not_create_active_experiments(self) -> None:
        compiler = _compiler()
        agent = SegmentAgent()
        compiled = compiler.compile_episode(_noise_episode())
        agent.ingest_narrative_episode(compiled)

        design = agent.latest_narrative_experiment
        self.assertFalse(design.active_plans())
        self.assertTrue(
            "low-value" in design.summary.lower()
            or "low-gain" in design.summary.lower()
            or "no bounded experiment" in design.summary.lower()
            or "no narrative discrimination target" in design.summary.lower()
            or "rejected" in design.summary.lower()
        )

    def test_governed_action_constraints_limit_candidates_to_registered_actions(self) -> None:
        compiler = _compiler()
        compiled = compiler.compile_episode(_social_episode())
        uncertainty = UncertaintyDecompositionResult.from_dict(compiled.uncertainty_decomposition)
        base_registry = build_default_action_registry()
        registry = ActionRegistry()
        for action_name in ("scan", "rest"):
            action = base_registry.get(action_name)
            assert action is not None
            registry.register(action, action.cost_estimate)

        design = NarrativeExperimentDesigner().design(
            tick=7,
            uncertainty=uncertainty,
            action_registry=registry,
            active_goal="SOCIAL",
        )

        self.assertTrue(design.candidates)
        self.assertTrue(all(item.action_name in {"scan", "rest"} for item in design.candidates))
        self.assertFalse(any(item.action_name == "seek_contact" for item in design.candidates))

    def test_snapshot_roundtrip_preserves_experiment_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=234,
                reset=True,
            )
            compiled = runtime.narrative_ingestion_service.compiler.compile_episode(_threat_episode())
            runtime.agent.ingest_narrative_episode(compiled)
            runtime.subject_state = derive_subject_state(runtime.agent, previous_state=runtime.subject_state)
            runtime.agent.subject_state = runtime.subject_state
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=234,
                reset=False,
            )

            self.assertEqual(
                restored.agent.latest_narrative_experiment.summary,
                runtime.agent.latest_narrative_experiment.summary,
            )
            self.assertEqual(
                [item.to_dict() for item in restored.agent.latest_narrative_experiment.plans],
                [item.to_dict() for item in runtime.agent.latest_narrative_experiment.plans],
            )
            self.assertTrue(
                restored.subject_state.active_inquiries or restored.agent.subject_state.active_inquiries
            )


if __name__ == "__main__":
    unittest.main()
