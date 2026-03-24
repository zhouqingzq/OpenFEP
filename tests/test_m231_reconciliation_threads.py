from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from segmentum.agent import SegmentAgent
from segmentum.reconciliation import (
    ConflictOrigin,
    ConflictSeverity,
    ConflictThread,
    ReconciliationEngine,
    ReconciliationOutcome,
    ReconciliationStatus,
)
from segmentum.runtime import SegmentRuntime
from segmentum.self_model import IdentityNarrative, NarrativeChapter
from segmentum.subject_state import derive_subject_state


def _diagnostics(
    *,
    conflict_type: str = "temporary_deviation",
    severity_level: str = "medium",
    identity_tension: float = 0.34,
    self_inconsistency_error: float = 0.36,
    violated_commitments: tuple[str, ...] = ("adaptive_exploration",),
    relevant_commitments: tuple[str, ...] = ("adaptive_exploration", "core_survival"),
    social_alerts: tuple[str, ...] = (),
    repair_policy: str = "",
    repair_result: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        conflict_type=conflict_type,
        severity_level=severity_level,
        identity_tension=identity_tension,
        self_inconsistency_error=self_inconsistency_error,
        violated_commitments=list(violated_commitments),
        relevant_commitments=list(relevant_commitments),
        social_alerts=list(social_alerts),
        repair_policy=repair_policy,
        repair_result=repair_result or {},
        commitment_compatibility_score=0.48,
    )


class TestM231ReconciliationThreads(unittest.TestCase):
    def test_repeated_conflicts_promote_to_thread_and_span_chapters(self) -> None:
        agent = SegmentAgent(rng=random.Random(231))
        agent.self_model.identity_narrative = IdentityNarrative(
            chapters=[NarrativeChapter(chapter_id=1, tick_range=(0, 3), dominant_theme="strain")],
            current_chapter=NarrativeChapter(chapter_id=1, tick_range=(0, 3), dominant_theme="strain"),
        )
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
        diagnostics = _diagnostics()

        agent.reconciliation_engine.observe_runtime(
            tick=1,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.74,
            slow_biases={},
        )
        report = agent.reconciliation_engine.observe_runtime(
            tick=2,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.72,
            slow_biases={},
        )

        self.assertTrue(report["created_threads"])
        thread = next(
            item
            for item in agent.reconciliation_engine.active_threads
            if "adaptive_exploration" in item.signature
        )
        self.assertEqual(thread.status, ReconciliationStatus.ACTIVE.value)
        self.assertEqual(thread.recurrence_count, 1)
        self.assertEqual(thread.linked_chapter_ids, [1])
        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertIn("reconciliation", narrative.current_chapter.state_summary)
        self.assertIn("reconciliation", narrative.contradiction_summary)
        self.assertTrue(narrative.chapter_transition_evidence)
        self.assertIn("Reconciliation:", narrative.core_summary)

        agent.self_model.identity_narrative.current_chapter = NarrativeChapter(
            chapter_id=2,
            tick_range=(4, 7),
            dominant_theme="carryover",
        )
        agent.reconciliation_engine.observe_runtime(
            tick=4,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.70,
            slow_biases={},
        )

        self.assertIn(2, thread.linked_chapter_ids)
        self.assertGreaterEqual(len(thread.chapter_bridges), 1)
        self.assertEqual(thread.persistence_class, "long_horizon")

    def test_local_patch_requires_sleep_and_evidence_before_reconciliation(self) -> None:
        agent = SegmentAgent(rng=random.Random(232))
        agent.cycle = 5
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
        diagnostics = _diagnostics(
            repair_policy="metacognitive_review+policy_rebias",
            repair_result={
                "success": True,
                "policy": "metacognitive_review+policy_rebias",
                "target_action": "forage",
                "repaired_action": "scan",
                "pre_alignment": 0.38,
                "post_alignment": 0.72,
            },
        )

        for tick in (4, 5):
            agent.reconciliation_engine.observe_runtime(
                tick=tick,
                diagnostics=diagnostics,
                narrative=agent.self_model.identity_narrative,
                prediction_ledger=agent.prediction_ledger,
                verification_loop=agent.verification_loop,
                subject_state=agent.subject_state,
                continuity_score=0.79,
                slow_biases={},
            )

        thread = agent.reconciliation_engine.active_threads[0]
        self.assertEqual(thread.status, ReconciliationStatus.PATCHED.value)
        self.assertNotEqual(thread.current_outcome, ReconciliationOutcome.DEEP_REPAIR.value)

        agent.verification_loop.archived_targets.extend(
            [
                SimpleNamespace(
                    target_id="verify:a",
                    prediction_id="pred:a",
                    outcome="confirmed",
                    outcome_tick=5,
                    linked_commitments=("adaptive_exploration",),
                    linked_identity_anchors=(),
                    target_channels=("continuity", "conflict"),
                    prediction_type="action_consequence",
                ),
                SimpleNamespace(
                    target_id="verify:b",
                    prediction_id="pred:b",
                    outcome="confirmed",
                    outcome_tick=5,
                    linked_commitments=("adaptive_exploration",),
                    linked_identity_anchors=(),
                    target_channels=("continuity", "conflict"),
                    prediction_type="action_consequence",
                ),
            ]
        )
        for _ in range(3):
            agent.sleep()

        self.assertEqual(thread.status, ReconciliationStatus.RECONCILED.value)
        self.assertEqual(thread.current_outcome, ReconciliationOutcome.DEEP_REPAIR.value)
        self.assertTrue(thread.verification_evidence_ids)
        self.assertIn("reconciliation_sleep", agent.narrative_trace[-1])
        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        self.assertEqual(
            narrative.contradiction_summary["reconciliation"]["dominant_status"],
            ReconciliationStatus.RECONCILED.value,
        )
        self.assertEqual(
            narrative.contradiction_summary["reconciliation"]["dominant_thread_id"],
            thread.thread_id,
        )
        if narrative.current_chapter is not None:
            self.assertEqual(
                narrative.current_chapter.state_summary["reconciliation"]["dominant_thread_id"],
                thread.thread_id,
            )
        self.assertEqual(
            narrative.chapter_transition_evidence[-1]["dominant_thread_id"],
            thread.thread_id,
        )
        self.assertIn("Reconciliation:", narrative.autobiographical_summary)

    def test_verification_evidence_binds_only_to_matching_thread(self) -> None:
        engine = ReconciliationEngine()
        matching = ConflictThread(
            thread_id="conflict:match:1",
            signature="identity_action:adaptive_exploration",
            title="matching conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            severity=ConflictSeverity.HIGH.value,
            recurrence_count=2,
            status=ReconciliationStatus.PARTIALLY_RECONCILED.value,
            current_outcome=ReconciliationOutcome.PARTIAL_REPAIR.value,
        )
        unrelated = ConflictThread(
            thread_id="conflict:other:1",
            signature="identity_action:core_survival",
            title="other conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="identity_action:core_survival",
                source_category="identity_action",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["core_survival"],
            linked_identity_elements=["core_survival"],
            severity=ConflictSeverity.HIGH.value,
            recurrence_count=2,
            status=ReconciliationStatus.PARTIALLY_RECONCILED.value,
            current_outcome=ReconciliationOutcome.PARTIAL_REPAIR.value,
        )
        engine.active_threads.extend([matching, unrelated])

        verification_loop = SimpleNamespace(
            archived_targets=[
                SimpleNamespace(
                    target_id="verify:adaptive",
                    prediction_id="pred:adaptive",
                    outcome="confirmed",
                    outcome_tick=6,
                    linked_commitments=("adaptive_exploration",),
                    linked_identity_anchors=(),
                    target_channels=("continuity", "conflict"),
                    prediction_type="action_consequence",
                )
            ]
        )

        engine._attach_verification_evidence(tick=6, verification_loop=verification_loop)

        self.assertEqual(matching.verification_evidence_ids, ["verify:adaptive:6"])
        self.assertEqual(unrelated.verification_evidence_ids, [])

    def test_verification_evidence_with_weak_anchors_does_not_bind(self) -> None:
        engine = ReconciliationEngine()
        left = ConflictThread(
            thread_id="conflict:left:1",
            signature="identity_action:adaptive_exploration",
            title="left conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            status=ReconciliationStatus.PARTIALLY_RECONCILED.value,
        )
        right = ConflictThread(
            thread_id="conflict:right:1",
            signature="self_expectation_falsification:adaptive_exploration",
            title="right conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="self_expectation_falsification:adaptive_exploration",
                source_category="self_expectation_falsification",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            status=ReconciliationStatus.PARTIALLY_RECONCILED.value,
        )
        engine.active_threads.extend([left, right])

        verification_loop = SimpleNamespace(
            archived_targets=[
                SimpleNamespace(
                    target_id="verify:weak",
                    prediction_id="pred:weak",
                    outcome="confirmed",
                    outcome_tick=6,
                    linked_commitments=("adaptive_exploration",),
                    linked_identity_anchors=(),
                    target_channels=(),
                    prediction_type="",
                )
            ]
        )

        engine._attach_verification_evidence(tick=6, verification_loop=verification_loop)

        self.assertEqual(left.verification_evidence_ids, [])
        self.assertEqual(right.verification_evidence_ids, [])

    def test_unmatched_repair_attempt_does_not_bind_to_any_thread(self) -> None:
        engine = ReconciliationEngine()
        left = ConflictThread(
            thread_id="conflict:left:1",
            signature="identity_action:adaptive_exploration",
            title="left conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            severity=ConflictSeverity.HIGH.value,
            recurrence_count=2,
            status=ReconciliationStatus.ACTIVE.value,
        )
        right = ConflictThread(
            thread_id="conflict:right:1",
            signature="self_expectation_falsification:adaptive_exploration",
            title="right conflict",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="self_expectation_falsification:adaptive_exploration",
                source_category="self_expectation_falsification",
                created_tick=1,
                chapter_id=1,
            ),
            linked_commitments=["adaptive_exploration"],
            linked_identity_elements=["adaptive_exploration"],
            severity=ConflictSeverity.HIGH.value,
            recurrence_count=2,
            status=ReconciliationStatus.ACTIVE.value,
        )
        engine.active_threads.extend([left, right])

        diagnostics = _diagnostics(
            conflict_type="temporary_deviation",
            violated_commitments=("novelty_seek",),
            relevant_commitments=("novelty_seek",),
            repair_policy="metacognitive_review+policy_rebias",
            repair_result={
                "success": True,
                "policy": "metacognitive_review+policy_rebias",
                "target_action": "forage",
                "repaired_action": "scan",
                "pre_alignment": 0.35,
                "post_alignment": 0.78,
            },
        )

        engine._attach_repair_attempts(tick=7, diagnostics=diagnostics)

        self.assertEqual(left.repair_attempt_history, [])
        self.assertEqual(right.repair_attempt_history, [])
        self.assertEqual(left.status, ReconciliationStatus.ACTIVE.value)
        self.assertEqual(right.status, ReconciliationStatus.ACTIVE.value)

    def test_reconciled_thread_reopens_when_same_conflict_returns(self) -> None:
        engine = ReconciliationEngine()
        thread = ConflictThread(
            thread_id="conflict:test:1",
            signature="identity_action:adaptive_exploration",
            title="exploration contradiction",
            created_tick=1,
            latest_tick=4,
            origin=ConflictOrigin(
                signature="identity_action:adaptive_exploration",
                source_category="identity_action",
                created_tick=1,
                chapter_id=1,
            ),
            linked_chapter_ids=[1],
            severity=ConflictSeverity.HIGH.value,
            recurrence_count=3,
            status=ReconciliationStatus.RECONCILED.value,
            current_outcome=ReconciliationOutcome.DEEP_REPAIR.value,
            stable_confirmations=4,
            protected=True,
        )
        engine.active_threads.append(thread)
        diagnostics = _diagnostics(violated_commitments=("adaptive_exploration",))

        engine.observe_runtime(
            tick=8,
            diagnostics=diagnostics,
            narrative=IdentityNarrative(
                current_chapter=NarrativeChapter(chapter_id=3, tick_range=(8, 10), dominant_theme="return")
            ),
            prediction_ledger=None,
            verification_loop=SimpleNamespace(archived_targets=[]),
            subject_state=SimpleNamespace(status_flags={"continuity_fragile": True}, continuity_anchors=("adaptive_exploration",)),
            continuity_score=0.68,
            slow_biases={},
        )

        self.assertEqual(thread.status, ReconciliationStatus.REOPENED.value)
        self.assertEqual(thread.current_outcome, ReconciliationOutcome.UNRESOLVED_CHRONIC.value)
        self.assertEqual(thread.last_reopened_tick, 8)

    def test_reconciliation_state_is_causal_and_survives_snapshot_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=231,
                reset=True,
            )

            runtime.agent.reconciliation_engine.active_threads.append(
                ConflictThread(
                    thread_id="conflict:seeded:1",
                    signature="continuity_anchor:core_survival",
                    title="seeded long conflict",
                    created_tick=1,
                    latest_tick=1,
                    origin=ConflictOrigin(
                        signature="continuity_anchor:core_survival",
                        source_category="continuity_anchor",
                        created_tick=1,
                        chapter_id=1,
                    ),
                    linked_chapter_ids=[1, 2],
                    linked_commitments=["core_survival"],
                    severity=ConflictSeverity.HIGH.value,
                    recurrence_count=3,
                    status=ReconciliationStatus.ACTIVE.value,
                    current_outcome=ReconciliationOutcome.UNRESOLVED_CHRONIC.value,
                    protected=True,
                )
            )

            self.assertNotEqual(runtime.agent.reconciliation_engine.action_bias("scan"), 0.0)
            self.assertTrue(runtime.agent.reconciliation_engine.workspace_focus())
            self.assertLess(runtime.agent.reconciliation_engine.memory_threshold_delta(), 0.0)
            self.assertTrue(runtime.agent.reconciliation_engine.maintenance_signal()["active_tasks"])

            runtime.subject_state = derive_subject_state(
                runtime.agent,
                previous_state=runtime.subject_state,
            )
            runtime.agent.subject_state = runtime.subject_state
            self.assertTrue(runtime.subject_state.status_flags["long_horizon_conflict"])

            runtime.step(verbose=False)
            runtime.save_snapshot()

            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=231,
                reset=False,
            )
            self.assertTrue(restored.agent.reconciliation_engine.active_threads)
            self.assertEqual(
                restored.agent.reconciliation_engine.active_threads[0].thread_id,
                runtime.agent.reconciliation_engine.active_threads[0].thread_id,
            )

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("reconciliation_engine", payload["agent"])

            trace_lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            trace_record = json.loads(trace_lines[-1])
            self.assertIn("reconciliation", trace_record)
            self.assertIn("reconciliation_payload", trace_record["decision_loop"])
            self.assertIn("reconciliation", trace_record["decision_loop"]["explanation_details"])


if __name__ == "__main__":
    unittest.main()
