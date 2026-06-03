from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.memory_consolidation import MemoryReuseEvent
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_store import MemoryStore


def _state() -> dict[str, object]:
    return {
        "active_goals": ["keep promises"],
        "identity_themes": ["reliable mentor"],
        "threat_level": 0.25,
        "recent_mood_baseline": "reflective",
        "cognitive_style": {
            "update_rigidity": 0.3,
            "error_aversion": 0.4,
            "uncertainty_sensitivity": 0.4,
        },
    }


def _entry(
    *,
    entry_id: str,
    content: str,
    semantic_tags: list[str],
    context_tags: list[str],
    accessibility: float = 0.4,
    abstractness: float = 0.2,
    retrieval_count: int = 0,
    relevance_self: float = 0.2,
    source_confidence: float = 0.9,
    mood_context: str = "reflective",
    anchor_slots: dict[str, str | None] | None = None,
    compression_metadata: dict[str, object] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        memory_class=MemoryClass.EPISODIC,
        store_level=StoreLevel.SHORT,
        source_type=SourceType.EXPERIENCE,
        created_at=1,
        last_accessed=1,
        valence=0.0,
        arousal=0.3,
        encoding_attention=0.4,
        novelty=0.3,
        relevance_goal=0.3,
        relevance_threat=0.2,
        relevance_self=relevance_self,
        relevance_social=0.2,
        relevance_reward=0.2,
        relevance=0.3,
        salience=0.5,
        trace_strength=0.5,
        accessibility=accessibility,
        abstractness=abstractness,
        source_confidence=source_confidence,
        reality_confidence=0.7,
        semantic_tags=semantic_tags,
        context_tags=context_tags,
        anchor_slots=anchor_slots or {
            "time": None,
            "place": None,
            "agents": "lin",
            "action": "mentor_checkin",
            "outcome": "commitment_kept",
        },
        anchor_strengths={"agents": "strong", "action": "strong", "outcome": "strong"},
        mood_context=mood_context,
        retrieval_count=retrieval_count,
        support_count=1,
        compression_metadata=compression_metadata,
    )


def _reuse_event(
    *,
    event_id: str,
    memory_id: str,
    reuse_prediction_error: float,
    reuse_free_energy_delta: float,
    recall_confidence: float,
    contradiction_detected: bool = False,
) -> MemoryReuseEvent:
    return MemoryReuseEvent(
        reuse_event_id=event_id,
        memory_id=memory_id,
        prediction_before_reuse={"danger": 0.4},
        observation_after_reuse={"danger": 0.8},
        reuse_prediction_error=reuse_prediction_error,
        reuse_free_energy_delta=reuse_free_energy_delta,
        recall_confidence=recall_confidence,
        contradiction_detected=contradiction_detected,
        live_reuse=True,
    )


class TestM177SurpriseGatedReconsolidation(unittest.TestCase):
    def test_agent_apply_reuse_reconsolidation_records_live_reports(self) -> None:
        agent = SegmentAgent(rng=random.Random(31))
        payload = agent.long_term_memory.store_episode(
            cycle=1,
            observation={
                "food": 0.30,
                "danger": 0.70,
                "novelty": 0.20,
                "shelter": 0.45,
                "temperature": 0.50,
                "social": 0.20,
            },
            prediction={
                "food": 0.55,
                "danger": 0.20,
                "novelty": 0.30,
                "shelter": 0.35,
                "temperature": 0.50,
                "social": 0.22,
            },
            errors={
                "food": -0.25,
                "danger": 0.50,
                "novelty": -0.10,
                "shelter": 0.10,
                "temperature": 0.0,
                "social": -0.02,
            },
            action="hide",
            outcome={"energy_delta": 0.02, "stress_delta": -0.03, "free_energy_drop": 0.08},
            body_state={"energy": 0.82, "stress": 0.15, "fatigue": 0.18, "temperature": 0.50},
        )
        agent.sync_memory_awareness_to_long_term_memory()
        entry_id = str(payload["episode_id"])
        agent.last_memory_context = {
            "prediction_before_memory": {"danger": 0.40},
            "recall_hypothesis": {"primary_entry_id": entry_id, "confidence": 0.82},
        }

        reports = agent.apply_reuse_reconsolidation(
            [
                {
                    "linked_prediction_id": "pred:test:reuse",
                    "linked_memory_ids": [entry_id],
                    "linked_path_ids": [],
                    "outcome": "confirmed",
                    "support_score": 0.92,
                    "contradiction_score": 0.08,
                    "prediction_error_delta": 0.20,
                    "free_energy_delta": 0.20,
                    "confidence_weight": 0.82,
                    "source_module": "test",
                    "settlement_version": 1,
                }
            ],
            observation={"danger": 0.68, "shelter": 0.45},
            tick=2,
        )

        self.assertTrue(reports)
        self.assertIn("reuse_reconsolidation_reports", agent.last_memory_context)
        self.assertEqual(
            agent.last_memory_context["reuse_reconsolidation_reports"][0]["reason_code"],
            "low_surprise_reinforcement",
        )

    def test_low_surprise_reuse_reinforces_without_rewrite(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(
                    entry_id="target",
                    content="thin",
                    semantic_tags=["mentor"],
                    context_tags=["lab"],
                    abstractness=0.85,
                    retrieval_count=2,
                ),
                _entry(
                    entry_id="donor",
                    content="Mentor promise happened in the community lab.",
                    semantic_tags=["mentor", "care"],
                    context_tags=["lab", "community"],
                    anchor_slots={
                        "time": "cycle-12",
                        "place": "community_lab",
                        "agents": "lin",
                        "action": "mentor_checkin",
                        "outcome": "commitment_kept",
                    },
                ),
            ]
        )

        before = store.get("target")
        assert before is not None
        before_version = before.version
        report = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:low",
                memory_id="target",
                reuse_prediction_error=0.05,
                reuse_free_energy_delta=0.10,
                recall_confidence=0.92,
            ),
        )

        after = store.get("target")
        assert after is not None
        self.assertEqual(report.update_type, "reinforcement_only")
        self.assertFalse(report.version_changed)
        self.assertEqual(after.version, before_version)
        self.assertFalse(report.fields_reconstructed)
        self.assertEqual(report.reason_code, "low_surprise_reinforcement")

    def test_medium_surprise_reuse_rebinds_context(self) -> None:
        store = MemoryStore(entries=[_entry(entry_id="target", content="stable", semantic_tags=["mentor"], context_tags=["lab"])])
        report = store.reconsolidate_entry(
            "target",
            current_mood="anxious",
            current_context_tags=["storm"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:medium",
                memory_id="target",
                reuse_prediction_error=0.30,
                reuse_free_energy_delta=-0.10,
                recall_confidence=0.80,
            ),
        )
        entry = store.get("target")
        assert entry is not None
        self.assertEqual(report.update_type, "contextual_rebinding")
        self.assertIn("mood_context", report.fields_rebound)
        self.assertEqual(entry.mood_context, "anxious")
        self.assertIn("storm", entry.context_tags)
        self.assertFalse(report.version_changed)

    def test_high_surprise_reuse_creates_versioned_reconstruction(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(
                    entry_id="target",
                    content="thin",
                    semantic_tags=["mentor"],
                    context_tags=["lab"],
                    abstractness=0.85,
                    retrieval_count=2,
                ),
                _entry(
                    entry_id="donor",
                    content="Mentor promise happened in the community lab.",
                    semantic_tags=["mentor", "care"],
                    context_tags=["lab", "community"],
                    anchor_slots={
                        "time": "cycle-12",
                        "place": "community_lab",
                        "agents": "lin",
                        "action": "mentor_checkin",
                        "outcome": "commitment_kept",
                    },
                ),
            ]
        )
        before = store.get("target")
        assert before is not None
        before_version = before.version
        report = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:high",
                memory_id="target",
                reuse_prediction_error=0.62,
                reuse_free_energy_delta=-0.52,
                recall_confidence=0.60,
            ),
        )
        after = store.get("target")
        assert after is not None
        self.assertEqual(report.update_type, "structural_reconstruction")
        self.assertTrue(report.version_changed)
        self.assertTrue(report.fields_reconstructed)
        self.assertGreater(after.version, before_version)

    def test_explicit_conflict_marks_even_with_high_recall_confidence(self) -> None:
        store = MemoryStore(entries=[_entry(entry_id="target", content="stable", semantic_tags=["mentor"], context_tags=["lab"])])
        report = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:conflict",
                memory_id="target",
                reuse_prediction_error=0.40,
                reuse_free_energy_delta=-0.45,
                recall_confidence=0.95,
                contradiction_detected=True,
            ),
        )
        entry = store.get("target")
        assert entry is not None
        self.assertEqual(report.update_type, "conflict_marking")
        self.assertIn("factual", report.conflict_flags)
        self.assertGreater(entry.counterevidence_count, 0)
        self.assertFalse(report.suppressed)

    def test_identity_critical_memory_requires_higher_rewrite_threshold(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(
                    entry_id="target",
                    content="thin",
                    semantic_tags=["mentor"],
                    context_tags=["lab"],
                    abstractness=0.85,
                    retrieval_count=2,
                    relevance_self=0.90,
                    compression_metadata={"legacy_template": {"identity_critical": True}},
                ),
                _entry(
                    entry_id="donor",
                    content="Mentor promise happened in the community lab.",
                    semantic_tags=["mentor", "care"],
                    context_tags=["lab", "community"],
                ),
            ]
        )
        report = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:identity",
                memory_id="target",
                reuse_prediction_error=0.60,
                reuse_free_energy_delta=-0.50,
                recall_confidence=0.70,
            ),
        )
        self.assertNotEqual(report.update_type, "structural_reconstruction")
        self.assertFalse(report.version_changed)
        self.assertEqual(report.reason_code, "identity_rewrite_floor_not_met")

    def test_same_reuse_event_is_suppressed(self) -> None:
        store = MemoryStore(entries=[_entry(entry_id="target", content="stable", semantic_tags=["mentor"], context_tags=["lab"])])
        first = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:dup",
                memory_id="target",
                reuse_prediction_error=0.10,
                reuse_free_energy_delta=0.05,
                recall_confidence=0.90,
            ),
        )
        entry = store.get("target")
        assert entry is not None
        retrieval_after_first = entry.retrieval_count
        second = store.reconsolidate_entry(
            "target",
            current_mood="reflective",
            current_context_tags=["lab"],
            current_cycle=30,
            current_state=_state(),
            reuse_event=_reuse_event(
                event_id="reuse:dup",
                memory_id="target",
                reuse_prediction_error=0.10,
                reuse_free_energy_delta=0.05,
                recall_confidence=0.90,
            ),
        )
        refreshed = store.get("target")
        assert refreshed is not None
        self.assertFalse(first.suppressed)
        self.assertTrue(second.suppressed)
        self.assertEqual(second.reason_code, "duplicate_reuse_event")
        self.assertEqual(refreshed.retrieval_count, retrieval_after_first)


if __name__ == "__main__":
    unittest.main()
