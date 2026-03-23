from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.self_model import IdentityNarrative, NarrativeChapter
from segmentum.subject_state import derive_subject_state


def _diagnostics(
    *,
    conflict_type: str = "temporary_deviation",
    violated_commitments: tuple[str, ...] = ("adaptive_exploration",),
    relevant_commitments: tuple[str, ...] = ("adaptive_exploration", "core_survival"),
    repair_policy: str = "",
    repair_result: dict[str, object] | None = None,
) -> object:
    class _Payload:
        severity_level = "medium"
        identity_tension = 0.36
        self_inconsistency_error = 0.38
        social_alerts: list[str] = []
        commitment_compatibility_score = 0.52

    payload = _Payload()
    payload.conflict_type = conflict_type
    payload.violated_commitments = list(violated_commitments)
    payload.relevant_commitments = list(relevant_commitments)
    payload.repair_policy = repair_policy
    payload.repair_result = repair_result or {}
    return payload


class TestM231NarrativeWriteback(unittest.TestCase):
    def test_writeback_updates_narrative_fields_and_survives_roundtrip(self) -> None:
        agent = SegmentAgent(rng=random.Random(2311))
        agent.self_model.identity_narrative = IdentityNarrative(
            chapters=[NarrativeChapter(chapter_id=1, tick_range=(0, 2), dominant_theme="strain")],
            current_chapter=NarrativeChapter(chapter_id=2, tick_range=(3, 5), dominant_theme="carryover"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
        )
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
        diagnostics = _diagnostics()

        agent.reconciliation_engine.observe_runtime(
            tick=5,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.72,
            slow_biases={},
        )
        agent.reconciliation_engine.observe_runtime(
            tick=6,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.70,
            slow_biases={},
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertIn("Reconciliation:", narrative.core_summary)
        self.assertIn("reconciliation", narrative.current_chapter.state_summary)
        self.assertTrue(narrative.chapter_transition_evidence)
        self.assertIn("reconciliation", narrative.contradiction_summary)
        self.assertTrue(
            any(key.startswith("reconciliation:") for key in narrative.evidence_provenance)
        )

        restored = SegmentAgent.from_dict(json.loads(json.dumps(agent.to_dict())), rng=random.Random(2312))
        restored_narrative = restored.self_model.identity_narrative
        assert restored_narrative is not None
        self.assertEqual(narrative.core_summary, restored_narrative.core_summary)
        self.assertEqual(
            narrative.contradiction_summary["reconciliation"]["summary"],
            restored_narrative.contradiction_summary["reconciliation"]["summary"],
        )

    def test_writeback_replaces_reconciliation_clause_instead_of_appending_forever(self) -> None:
        agent = SegmentAgent(rng=random.Random(2312))
        agent.self_model.identity_narrative = IdentityNarrative(
            current_chapter=NarrativeChapter(chapter_id=1, tick_range=(0, 2), dominant_theme="strain"),
            core_summary="I am an adaptive subject. Reconciliation: old clause.",
            autobiographical_summary="I am an adaptive subject. Reconciliation: old clause.",
        )
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
        diagnostics = _diagnostics()

        for tick in (3, 4):
            agent.reconciliation_engine.observe_runtime(
                tick=tick,
                diagnostics=diagnostics,
                narrative=agent.self_model.identity_narrative,
                prediction_ledger=agent.prediction_ledger,
                verification_loop=agent.verification_loop,
                subject_state=agent.subject_state,
                continuity_score=0.69,
                slow_biases={},
            )

        summary = agent.self_model.identity_narrative.core_summary
        self.assertEqual(summary.count("Reconciliation:"), 1)


if __name__ == "__main__":
    unittest.main()
