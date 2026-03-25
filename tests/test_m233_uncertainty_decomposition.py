from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from segmentum.agent import SegmentAgent
from segmentum.narrative_compiler import NarrativeCompiler
from segmentum.narrative_types import NarrativeEpisode
from segmentum.prediction_ledger import PredictionLedger
from segmentum.runtime import SegmentRuntime
from segmentum.subject_state import derive_subject_state
from segmentum.verification import VerificationLoop


def _diagnostics() -> SimpleNamespace:
    return SimpleNamespace(
        chosen=SimpleNamespace(
            choice="scan",
            predicted_effects={"danger_delta": -0.05, "social_delta": 0.03},
            preferred_probability=0.62,
        ),
        workspace_broadcast_channels=["danger", "social"],
        commitment_focus=["protect continuity"],
        active_goal="SURVIVAL",
        social_focus=["counterpart-x"],
        social_alerts=["rupture"],
        identity_tension=0.0,
        self_inconsistency_error=0.0,
        violated_commitments=[],
        conflict_type="none",
        repair_triggered=False,
    )


class TestM233UncertaintyDecomposition(unittest.TestCase):
    def test_social_rupture_extracts_unknowns_competing_hypotheses_and_surface_cues(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m233-social",
            timestamp=1,
            source="user_diary",
            raw_text=(
                "My counterpart promised to meet me, but left me outside and gave mixed signals. "
                "I still do not know whether it was betrayal, pressure, or a misunderstanding."
            ),
            tags=["social"],
            metadata={"counterpart_id": "counterpart-x", "chapter_id": 3},
        )

        compiled = compiler.compile_episode(episode)
        uncertainty = compiled.uncertainty_decomposition
        unknowns = uncertainty["unknowns"]
        hypotheses = uncertainty["competing_hypotheses"]
        surface_cues = uncertainty["surface_cues"]

        self.assertTrue(unknowns)
        self.assertEqual(unknowns[0]["unknown_type"], "trust")
        self.assertTrue(unknowns[0]["action_relevant"])
        self.assertGreaterEqual(len(hypotheses), 2)
        self.assertIn("counterpart-x", unknowns[0]["linked_entities"])
        self.assertTrue(surface_cues)
        self.assertLessEqual(len(unknowns), 3)
        self.assertIn("trust unresolved", uncertainty["summary"])

    def test_surface_only_noise_is_not_overpromoted(self) -> None:
        compiler = NarrativeCompiler()
        episode = NarrativeEpisode(
            episode_id="m233-noise",
            timestamp=2,
            source="user_diary",
            raw_text="Very dramatic!!! It felt huge, but nothing happened and it was just a slogan on a poster.",
            tags=["noise"],
            metadata={},
        )

        compiled = compiler.compile_episode(episode)
        uncertainty = compiled.uncertainty_decomposition

        self.assertGreaterEqual(len(uncertainty["surface_cues"]), 1)
        self.assertEqual(uncertainty["profile"]["decision_relevant_unknown_count"], 0)
        self.assertIn("surface", uncertainty["summary"])

    def test_uncertainty_changes_subject_state_ledger_and_verification(self) -> None:
        compiler = NarrativeCompiler()
        agent = SegmentAgent()
        episode = NarrativeEpisode(
            episode_id="m233-causal",
            timestamp=3,
            source="user_diary",
            raw_text=(
                "A predator attacked near the camp, but I do not know whether it was a one-off accident "
                "or a stable threat source that will return."
            ),
            tags=["threat"],
            metadata={"environment_id": "camp-edge", "chapter_id": 4},
        )
        compiled = compiler.compile_episode(episode)
        agent.ingest_narrative_episode(compiled)
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)

        self.assertTrue(agent.subject_state.narrative_uncertainties)
        self.assertTrue(agent.subject_state.status_flags["narrative_ambiguity_active"])

        ledger = PredictionLedger()
        update = ledger.seed_predictions(
            tick=3,
            diagnostics=_diagnostics(),
            prediction={"danger": 0.55, "social": 0.20},
            subject_state=agent.subject_state,
            narrative_uncertainty=agent.latest_narrative_uncertainty,
        )
        self.assertTrue(update.created_predictions)
        narrative_predictions = [
            item for item in ledger.active_predictions() if item.source_module == "narrative_uncertainty"
        ]
        self.assertTrue(narrative_predictions)

        verification = VerificationLoop()
        verification.refresh_targets(
            tick=4,
            ledger=ledger,
            diagnostics=_diagnostics(),
            subject_state=agent.subject_state,
            narrative_uncertainty=agent.latest_narrative_uncertainty,
            workspace_channels=("danger",),
        )
        self.assertTrue(verification.active_targets)
        self.assertTrue(
            any(
                "narrative ambiguity" in target.selected_reason
                for target in verification.active_targets
            )
        )

    def test_snapshot_roundtrip_preserves_latest_uncertainty_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=91,
                reset=True,
            )
            episode = NarrativeEpisode(
                episode_id="m233-snapshot",
                timestamp=5,
                source="user_diary",
                raw_text=(
                    "A predator attacked near the bridge, but I still cannot tell whether it was "
                    "a persistent threat source or a local accident."
                ),
                tags=["mixed"],
                metadata={"environment_id": "bridge-edge", "chapter_id": 7},
            )
            compiled = runtime.narrative_ingestion_service.compiler.compile_episode(episode)
            runtime.agent.ingest_narrative_episode(compiled)
            runtime.subject_state = derive_subject_state(
                runtime.agent,
                previous_state=runtime.subject_state,
            )
            runtime.agent.subject_state = runtime.subject_state
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=91,
                reset=False,
            )

            self.assertEqual(
                restored.agent.latest_narrative_uncertainty.summary,
                runtime.agent.latest_narrative_uncertainty.summary,
            )
            self.assertTrue(restored.agent.latest_narrative_uncertainty.unknowns)
            self.assertTrue(
                restored.subject_state.narrative_uncertainties
                or restored.agent.subject_state.narrative_uncertainties
            )


if __name__ == "__main__":
    unittest.main()
