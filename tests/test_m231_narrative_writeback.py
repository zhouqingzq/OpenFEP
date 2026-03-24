from __future__ import annotations

import json
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.reconciliation import (
    ConflictOrigin,
    ConflictThread,
    ReconciliationOutcome,
    ReconciliationStatus,
)
from segmentum.self_model import IdentityCommitment, IdentityNarrative, NarrativeChapter, NarrativeClaim
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

    def test_writeback_targets_reconciled_cross_chapter_thread_over_unresolved_noise(self) -> None:
        agent = SegmentAgent(rng=random.Random(2313))
        agent.self_model.identity_narrative = IdentityNarrative(
            chapters=[NarrativeChapter(chapter_id=1, tick_range=(0, 2), dominant_theme="strain")],
            current_chapter=NarrativeChapter(chapter_id=3, tick_range=(6, 9), dominant_theme="repair"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
        )
        target_thread = ConflictThread(
            thread_id="conflict:identity_action:adaptive_exploration:5",
            signature="identity_action:adaptive_exploration",
            title="temporary deviation",
            created_tick=5,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=5,
                chapter_id=2,
            ),
            linked_chapter_ids=[2, 3],
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            recurrence_count=2,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=3,
            verification_evidence_ids=["verify:a:6", "verify:b:6"],
        )
        distractor = ConflictThread(
            thread_id="conflict:self_expectation_falsification:none:1",
            signature="self_expectation_falsification:none",
            title="none",
            created_tick=1,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="self_expectation_falsification:none",
                source_category="self_expectation_falsification",
                created_tick=1,
                chapter_id=3,
            ),
            linked_chapter_ids=[3],
            recurrence_count=4,
            status=ReconciliationStatus.ACTIVE.value,
            current_outcome=ReconciliationOutcome.NONE.value,
            protected=True,
        )
        agent.reconciliation_engine.active_threads.extend([distractor, target_thread])
        agent.reconciliation_engine._write_back_to_narrative(
            narrative=agent.self_model.identity_narrative,
            tick=9,
            reason="sleep_review:3",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        contradiction_summary = narrative.contradiction_summary["reconciliation"]
        current_chapter_summary = narrative.current_chapter.state_summary["reconciliation"]
        transition_entry = narrative.chapter_transition_evidence[-1]
        self.assertEqual(contradiction_summary["dominant_thread_id"], target_thread.thread_id)
        self.assertEqual(current_chapter_summary["dominant_thread_id"], target_thread.thread_id)
        self.assertEqual(transition_entry["dominant_thread_id"], target_thread.thread_id)
        self.assertEqual(contradiction_summary["linked_chapter_ids"], [2, 3])
        self.assertEqual(contradiction_summary["dominant_outcome"], ReconciliationOutcome.DEEP_REPAIR.value)
        self.assertIn("reconciled across 2 chapter(s)", narrative.core_summary)

    def test_writeback_updates_claims_and_recalibrates_commitments(self) -> None:
        agent = SegmentAgent(rng=random.Random(2314))
        claim = NarrativeClaim(
            claim_id="claim-01-trait-aggressive",
            claim_type="trait",
            text="I am generally aggressive under pressure.",
            claim_key="aggressive",
            contradicted_by=["ev-1-rest"],
            contradiction_score=1.0,
            contradict_count=1,
            confidence=0.25,
        )
        commitment = IdentityCommitment(
            commitment_id="commitment-exploration-drive",
            commitment_type="behavioral_style",
            statement="When conditions are stable, reduce uncertainty through active exploration.",
            target_actions=["scan", "seek_contact"],
            discouraged_actions=["rest"],
            confidence=0.3,
            priority=0.55,
            source_claim_ids=[claim.claim_id],
        )
        agent.self_model.identity_narrative = IdentityNarrative(
            current_chapter=NarrativeChapter(chapter_id=3, tick_range=(6, 9), dominant_theme="repair"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
            claims=[claim],
            commitments=[commitment],
        )
        target_thread = ConflictThread(
            thread_id="conflict:identity_action:adaptive_exploration:5",
            signature="identity_action:adaptive_exploration",
            title="temporary deviation",
            created_tick=5,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=5,
                chapter_id=2,
            ),
            linked_chapter_ids=[2, 3],
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            recurrence_count=2,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=3,
            verification_evidence_ids=["verify:a:6", "verify:b:6"],
            supporting_evidence=["adaptive_exploration", "medium"],
        )
        agent.reconciliation_engine.active_threads.append(target_thread)
        agent.reconciliation_engine._write_back_to_narrative(
            narrative=agent.self_model.identity_narrative,
            tick=9,
            reason="sleep_review:3",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        updated_claim = narrative.claims[0]
        updated_commitment = narrative.commitments[0]
        self.assertEqual(updated_claim.reconciliation_thread_id, target_thread.thread_id)
        self.assertEqual(updated_claim.reconciliation_status, ReconciliationStatus.RECONCILED.value)
        self.assertFalse(updated_claim.reconciliation_contested)
        self.assertGreaterEqual(updated_claim.confidence, 0.72)
        self.assertIn("thread:conflict:identity_action:adaptive_exploration:5", updated_claim.reconciliation_evidence_ids)
        self.assertIn(updated_claim.claim_id, narrative.contradiction_summary["reconciled_claim_ids"])
        self.assertGreater(updated_commitment.confidence, 0.3)
        self.assertIn(target_thread.thread_id, updated_commitment.evidence_ids)

    def test_writeback_does_not_rewrite_unrelated_claim_without_anchor_overlap(self) -> None:
        agent = SegmentAgent(rng=random.Random(2315))
        unrelated_claim = NarrativeClaim(
            claim_id="claim-social-withdrawal",
            claim_type="trait",
            text="I avoid strangers and stay still.",
            claim_key="cautious_social_withdrawal",
            confidence=0.8,
        )
        agent.self_model.identity_narrative = IdentityNarrative(
            current_chapter=NarrativeChapter(chapter_id=3, tick_range=(6, 9), dominant_theme="repair"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
            claims=[unrelated_claim],
        )
        target_thread = ConflictThread(
            thread_id="conflict:identity_action:adaptive_exploration:5",
            signature="identity_action:adaptive_exploration",
            title="temporary deviation",
            created_tick=5,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=5,
                chapter_id=2,
            ),
            linked_chapter_ids=[2, 3],
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            recurrence_count=2,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=3,
            verification_evidence_ids=["verify:a:6"],
            supporting_evidence=["adaptive_exploration"],
        )
        agent.reconciliation_engine.active_threads.append(target_thread)

        agent.reconciliation_engine._write_back_to_narrative(
            narrative=agent.self_model.identity_narrative,
            tick=9,
            reason="sleep_review:3",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertEqual(narrative.claims[0].claim_id, unrelated_claim.claim_id)
        self.assertEqual(narrative.claims[0].reconciliation_thread_id, "")
        self.assertEqual(narrative.claims[0].reconciliation_status, "")
        self.assertEqual(len(narrative.claims), 2)
        synthetic = narrative.claims[1]
        self.assertEqual(synthetic.claim_type, "reconciliation")
        self.assertEqual(synthetic.reconciliation_thread_id, target_thread.thread_id)

    def test_writeback_only_updates_anchor_matched_claims(self) -> None:
        agent = SegmentAgent(rng=random.Random(2316))
        matched_claim = NarrativeClaim(
            claim_id="claim-01-trait-aggressive",
            claim_type="trait",
            text="I am generally aggressive under pressure.",
            claim_key="aggressive",
            confidence=0.25,
        )
        unrelated_claim = NarrativeClaim(
            claim_id="claim-02-trait-withdrawn",
            claim_type="trait",
            text="I remain withdrawn around strangers.",
            claim_key="withdrawn",
            confidence=0.66,
        )
        agent.self_model.identity_narrative = IdentityNarrative(
            current_chapter=NarrativeChapter(chapter_id=3, tick_range=(6, 9), dominant_theme="repair"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
            claims=[matched_claim, unrelated_claim],
        )
        target_thread = ConflictThread(
            thread_id="conflict:identity_action:adaptive_exploration:5",
            signature="identity_action:adaptive_exploration",
            title="temporary deviation",
            created_tick=5,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=5,
                chapter_id=2,
            ),
            linked_chapter_ids=[2, 3],
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            recurrence_count=2,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=3,
            verification_evidence_ids=["verify:a:6"],
            supporting_evidence=["adaptive_exploration"],
        )
        agent.reconciliation_engine.active_threads.append(target_thread)

        agent.reconciliation_engine._write_back_to_narrative(
            narrative=agent.self_model.identity_narrative,
            tick=9,
            reason="sleep_review:3",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertEqual(narrative.claims[0].reconciliation_thread_id, target_thread.thread_id)
        self.assertEqual(narrative.claims[1].reconciliation_thread_id, "")
        self.assertEqual(narrative.contradiction_summary["reconciled_claim_ids"], [matched_claim.claim_id])

    def test_writeback_can_resume_explicitly_bound_reconciliation_claim_without_anchor_overlap(self) -> None:
        agent = SegmentAgent(rng=random.Random(2317))
        reconciliation_claim = NarrativeClaim(
            claim_id="claim-reconciliation-conflict-identity_action-adaptive_exploration-5",
            claim_type="reconciliation",
            text="Old reconciliation state.",
            claim_key="reconciliation",
            reconciliation_thread_id="conflict:identity_action:adaptive_exploration:5",
        )
        unrelated_claim = NarrativeClaim(
            claim_id="claim-02-trait-withdrawn",
            claim_type="trait",
            text="I remain withdrawn around strangers.",
            claim_key="withdrawn",
            confidence=0.66,
        )
        agent.self_model.identity_narrative = IdentityNarrative(
            current_chapter=NarrativeChapter(chapter_id=3, tick_range=(6, 9), dominant_theme="repair"),
            core_summary="I remain adaptive under uncertainty.",
            autobiographical_summary="I remain adaptive under uncertainty.",
            claims=[reconciliation_claim, unrelated_claim],
        )
        target_thread = ConflictThread(
            thread_id="conflict:identity_action:adaptive_exploration:5",
            signature="identity_action:adaptive_exploration",
            title="temporary deviation",
            created_tick=5,
            latest_tick=8,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=5,
                chapter_id=2,
            ),
            linked_chapter_ids=[2, 3],
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            recurrence_count=2,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=3,
            verification_evidence_ids=["verify:a:6"],
        )
        agent.reconciliation_engine.active_threads.append(target_thread)

        agent.reconciliation_engine._write_back_to_narrative(
            narrative=agent.self_model.identity_narrative,
            tick=9,
            reason="sleep_review:3",
        )

        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertEqual(narrative.claims[0].reconciliation_thread_id, target_thread.thread_id)
        self.assertEqual(narrative.claims[0].claim_id, reconciliation_claim.claim_id)
        self.assertEqual(narrative.claims[1].reconciliation_thread_id, "")


if __name__ == "__main__":
    unittest.main()
