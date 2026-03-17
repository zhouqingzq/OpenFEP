from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.runtime import SegmentRuntime
from segmentum.self_model import build_default_self_model


class TestM218ContinuityPreservation(unittest.TestCase):
    def test_personality_drift_guard_stabilizes_traits(self) -> None:
        model = build_default_self_model()
        model.update_continuity_audit(
            episodic_memory=[],
            archived_memory=[],
            action_history=["scan", "forage", "hide", "rest"],
            current_tick=1,
        )

        model.personality_profile.openness = 1.0
        model.personality_profile.conscientiousness = 1.0
        model.personality_profile.extraversion = 0.0
        model.personality_profile.agreeableness = 0.0
        model.personality_profile.neuroticism = 1.0
        model.personality_profile.meaning_construction_tendency = 1.0
        model.personality_profile.emotional_regulation_style = 0.0

        audit = model.update_continuity_audit(
            episodic_memory=[],
            archived_memory=[],
            action_history=["scan", "forage", "hide", "rest"],
            current_tick=2,
        )

        self.assertIn("personality_drift_guard", audit.interventions)
        self.assertLess(model.personality_profile.openness, 1.0)
        self.assertGreater(model.personality_profile.extraversion, 0.0)

    def test_restart_consistency_stays_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            trace_path = root / "trace.jsonl"

            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=17,
                reset=True,
            )
            runtime.run(cycles=80, verbose=False)
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=99,
                reset=False,
            )

            self.assertLessEqual(
                restored.agent.self_model.continuity_audit.restart_divergence,
                restored.agent.self_model.drift_budget.restart_tolerance,
            )
            self.assertIn("m218", restored.export_snapshot())


if __name__ == "__main__":
    unittest.main()
