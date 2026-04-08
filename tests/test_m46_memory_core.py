from __future__ import annotations

import random
import unittest

from segmentum.memory import LongTermMemory
from segmentum.memory_consolidation import (
    ConflictType,
    ReconsolidationUpdateType,
    ReconstructionConfig,
    compress_episodic_cluster_to_semantic_skeleton,
    maybe_reconstruct,
    reconsolidate,
    validate_inference,
)
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_retrieval import RetrievalQuery, compete_candidates
from segmentum.memory_store import MemoryStore


def _state() -> dict[str, object]:
    return {
        "active_goals": ["keep promises", "protect mentees"],
        "identity_themes": ["reliable mentor", "care continuity"],
        "threat_level": 0.2,
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
    memory_class: MemoryClass = MemoryClass.EPISODIC,
    store_level: StoreLevel = StoreLevel.SHORT,
    source_type: SourceType = SourceType.EXPERIENCE,
    valence: float = 0.0,
    accessibility: float = 0.6,
    abstractness: float = 0.2,
    reality_confidence: float = 0.85,
    retrieval_count: int = 0,
    support_count: int = 1,
    mood_context: str = "",
    created_at: int = 1,
    last_accessed: int = 1,
    is_dormant: bool = False,
    counterevidence_count: int = 0,
    compression_metadata: dict[str, object] | None = None,
    procedure_steps: list[str] | None = None,
    execution_contexts: list[str] | None = None,
    anchor_strengths: dict[str, str] | None = None,
    anchor_slots: dict[str, str | None] | None = None,
) -> MemoryEntry:
    kwargs = {
        "id": entry_id,
        "content": content,
        "memory_class": memory_class,
        "store_level": store_level,
        "source_type": source_type,
        "created_at": created_at,
        "last_accessed": last_accessed,
        "valence": valence,
        "arousal": 0.3,
        "encoding_attention": 0.4,
        "novelty": 0.3,
        "relevance_goal": 0.3,
        "relevance_threat": 0.2,
        "relevance_self": 0.2,
        "relevance_social": 0.2,
        "relevance_reward": 0.2,
        "relevance": 0.3,
        "salience": 0.5,
        "trace_strength": 0.5,
        "accessibility": accessibility,
        "abstractness": abstractness,
        "source_confidence": 0.9,
        "reality_confidence": reality_confidence,
        "semantic_tags": semantic_tags,
        "context_tags": context_tags,
        "anchor_slots": anchor_slots or {"action": "mentor_checkin", "outcome": "commitment_kept", "agents": "lin"},
        "anchor_strengths": anchor_strengths or {"agents": "strong", "action": "strong", "outcome": "strong"},
        "mood_context": mood_context,
        "retrieval_count": retrieval_count,
        "support_count": support_count,
        "counterevidence_count": counterevidence_count,
        "compression_metadata": compression_metadata,
        "is_dormant": is_dormant,
    }
    if memory_class is MemoryClass.PROCEDURAL:
        kwargs["procedure_steps"] = procedure_steps or ["scan gauges", "vent pressure", "log readings"]
        kwargs["step_confidence"] = [0.9 for _ in kwargs["procedure_steps"]]
        kwargs["execution_contexts"] = execution_contexts or ["reactor_room"]
    return MemoryEntry(**kwargs)


class TestM46MemoryCore(unittest.TestCase):
    def test_retrieve_supports_multi_cue_scoring_and_recall_artifact(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(
                    entry_id="tag-primary",
                    content="I kept the mentor promise to Lin in the lab.",
                    semantic_tags=["mentor", "promise", "care"],
                    context_tags=["lab", "weekly"],
                    accessibility=0.78,
                    mood_context="reflective",
                    last_accessed=48,
                    created_at=40,
                    compression_metadata={"state_vector": [0.9, 0.2]},
                ),
                _entry(
                    entry_id="context-rich",
                    content="We reviewed the care plan in the lab during a weekly sync.",
                    semantic_tags=["care", "plan"],
                    context_tags=["lab", "weekly", "team"],
                    accessibility=0.74,
                    last_accessed=47,
                    created_at=39,
                    compression_metadata={"state_vector": [0.88, 0.25]},
                ),
                _entry(
                    entry_id="negative-mood",
                    content="I worried the promise might fail under pressure.",
                    semantic_tags=["mentor", "promise", "pressure"],
                    context_tags=["storm"],
                    valence=-0.6,
                    accessibility=0.62,
                    mood_context="anxious",
                    last_accessed=46,
                    created_at=38,
                ),
                _entry(
                    entry_id="low-access",
                    content="The exact mentor promise detail is stored but hard to reach.",
                    semantic_tags=["mentor", "promise", "care"],
                    context_tags=["lab"],
                    accessibility=0.01,
                    last_accessed=49,
                    created_at=45,
                ),
                _entry(
                    entry_id="dormant",
                    content="Dormant mentor promise trace.",
                    semantic_tags=["mentor", "promise"],
                    context_tags=["lab"],
                    accessibility=0.95,
                    is_dormant=True,
                ),
                _entry(
                    entry_id="procedure",
                    content="Reactor calming routine for emergencies.",
                    semantic_tags=["reactor", "procedure"],
                    context_tags=["maintenance"],
                    memory_class=MemoryClass.PROCEDURAL,
                    accessibility=0.72,
                ),
            ]
        )

        tag_query = RetrievalQuery(
            semantic_tags=["mentor", "promise"],
            context_tags=["lab"],
            content_keywords=["lin", "promise"],
            state_vector=[0.91, 0.22],
            reference_cycle=50,
        )
        tag_result = store.retrieve(tag_query, current_mood="reflective", k=4)
        self.assertEqual(tag_result.candidates[0].entry_id, "tag-primary")
        self.assertNotIn("dormant", [candidate.entry_id for candidate in tag_result.candidates])
        self.assertIsNotNone(tag_result.recall_hypothesis)
        self.assertNotEqual(tag_result.recall_hypothesis.content, store.get("tag-primary").content)
        self.assertIn("primary:tag-primary", tag_result.source_trace)
        self.assertTrue(tag_result.candidates[0].score_breakdown["tag_overlap"] > 0.0)

        context_query = RetrievalQuery(
            semantic_tags=["plan"],
            context_tags=["lab", "weekly", "team"],
            state_vector=[0.88, 0.24],
            reference_cycle=50,
        )
        context_result = store.retrieve(context_query, current_mood="reflective", k=3)
        self.assertEqual(context_result.candidates[0].entry_id, "context-rich")

        mood_query = RetrievalQuery(
            semantic_tags=["promise"],
            context_tags=["storm"],
            reference_cycle=50,
        )
        mood_result = store.retrieve(mood_query, current_mood="anxious", k=3)
        self.assertEqual(mood_result.candidates[0].entry_id, "negative-mood")

        accessibility_query = RetrievalQuery(
            semantic_tags=["mentor", "promise", "care"],
            context_tags=["lab"],
            reference_cycle=50,
        )
        accessibility_result = store.retrieve(accessibility_query, current_mood="reflective", k=4)
        candidate_ids = [candidate.entry_id for candidate in accessibility_result.candidates]
        self.assertLess(candidate_ids.index("low-access"), candidate_ids.index("procedure") if "procedure" in candidate_ids else 99)
        self.assertNotEqual(accessibility_result.candidates[0].entry_id, "low-access")

        procedural_query = RetrievalQuery(
            semantic_tags=["reactor", "procedure"],
            context_tags=["maintenance"],
            reference_cycle=50,
            target_memory_class=MemoryClass.PROCEDURAL,
        )
        procedural_result = store.retrieve(procedural_query, current_mood="calm", k=1)
        self.assertEqual(procedural_result.candidates[0].entry_id, "procedure")
        self.assertTrue(procedural_result.recall_hypothesis.procedure_step_outline)

    def test_candidate_competition_handles_clear_and_close_rankings(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(entry_id="dominant", content="dominant", semantic_tags=["mentor", "promise"], context_tags=["lab"], accessibility=0.9),
                _entry(entry_id="runner-up", content="runner", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4),
                _entry(entry_id="close-a", content="close-a", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.8),
                _entry(entry_id="close-b", content="close-b", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.78),
            ]
        )
        dominant_result = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=10),
            k=2,
        )
        dominant_competition = compete_candidates(dominant_result.candidates)
        self.assertEqual(dominant_competition.confidence, "high")
        self.assertFalse(dominant_competition.interference_risk)

        close_result = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "care"], context_tags=["lab"], reference_cycle=10),
            k=3,
        )
        close_competition = compete_candidates(close_result.candidates, dominance_threshold=0.15)
        self.assertEqual(close_competition.confidence, "low")
        self.assertTrue(close_competition.interference_risk)
        self.assertTrue(close_competition.competitors)
        self.assertTrue(close_competition.competing_interpretations)

    def test_reconstruction_triggers_a_b_c_and_protects_strong_anchors(self) -> None:
        base = _entry(
            entry_id="base",
            content="Thin recall.",
            semantic_tags=["mentor", "promise"],
            context_tags=["lab"],
            abstractness=0.75,
            reality_confidence=0.6,
            retrieval_count=2,
            anchor_slots={"time": None, "place": None, "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"},
            anchor_strengths={"time": "weak", "place": "weak", "agents": "locked", "action": "strong", "outcome": "strong"},
        )
        semantic = _entry(
            entry_id="semantic",
            content="Semantic trust pattern.",
            semantic_tags=["mentor", "trust"],
            context_tags=["community"],
            memory_class=MemoryClass.SEMANTIC,
            abstractness=0.8,
            reality_confidence=0.7,
            retrieval_count=1,
        )
        low_conf = _entry(
            entry_id="low-conf",
            content="Uncertain memory trace with weak grounding but recurring access.",
            semantic_tags=["mentor", "promise"],
            context_tags=["lab"],
            abstractness=0.3,
            reality_confidence=0.2,
            retrieval_count=3,
        )
        donor = _entry(
            entry_id="donor",
            content="Mentor promise happened in the community lab.",
            semantic_tags=["mentor", "promise", "care"],
            context_tags=["lab", "community"],
            anchor_slots={"time": "cycle-12", "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"},
        )
        procedural = _entry(
            entry_id="procedural",
            content="Reactor procedure summary.",
            semantic_tags=["reactor", "procedure"],
            context_tags=["maintenance"],
            memory_class=MemoryClass.PROCEDURAL,
            abstractness=0.8,
            reality_confidence=0.7,
            retrieval_count=2,
            procedure_steps=["scan gauges", "vent pressure", "log readings"],
        )
        donor_procedure = _entry(
            entry_id="procedure-donor",
            content="Secondary reactor procedure support.",
            semantic_tags=["reactor", "procedure"],
            context_tags=["maintenance", "night"],
            memory_class=MemoryClass.PROCEDURAL,
            procedure_steps=["scan gauges", "vent pressure", "log readings"],
            execution_contexts=["night_shift"],
        )
        store = MemoryStore(entries=[base, semantic, low_conf, donor, procedural, donor_procedure])

        result_a = maybe_reconstruct(base, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
        self.assertTrue(result_a.triggered)
        self.assertEqual(result_a.entry.source_type, SourceType.RECONSTRUCTION)
        self.assertEqual(result_a.entry.anchor_slots["agents"], "lin")
        self.assertEqual(result_a.entry.anchor_slots["action"], "mentor_checkin")
        self.assertEqual(result_a.entry.anchor_slots["place"], "community_lab")
        self.assertLess(result_a.entry.reality_confidence, base.reality_confidence)

        result_b = maybe_reconstruct(semantic, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
        self.assertTrue(result_b.triggered)
        self.assertEqual(result_b.trigger_reason, "semantic_abstractness")

        result_c = maybe_reconstruct(low_conf, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
        self.assertTrue(result_c.triggered)
        self.assertEqual(result_c.trigger_reason, "low_reality_after_retrieval")

        procedural_result = maybe_reconstruct(procedural, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
        self.assertTrue(procedural_result.triggered)
        self.assertEqual(procedural_result.entry.procedure_steps, procedural.procedure_steps)
        self.assertIn("night_shift", procedural_result.entry.execution_contexts)

    def test_reconsolidation_distinguishes_four_update_types(self) -> None:
        store = MemoryStore(
            entries=[
                _entry(entry_id="reinforce", content="stable", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2),
                _entry(entry_id="rebind", content="rebind", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2, mood_context="reflective"),
                _entry(entry_id="reconstruct", content="thin", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.8, retrieval_count=2),
                _entry(entry_id="conflict", content="conflict", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2),
                _entry(entry_id="donor", content="donor", semantic_tags=["mentor", "care"], context_tags=["lab"], anchor_slots={"time": None, "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"}),
            ]
        )
        reinforce_report = reconsolidate(store.get("reinforce"), None, None, store=store, current_cycle=30, current_state=_state())
        self.assertEqual(reinforce_report.update_type, ReconsolidationUpdateType.REINFORCEMENT_ONLY.value)

        rebind_report = reconsolidate(store.get("rebind"), "anxious", ["storm"], store=store, current_cycle=30, current_state=_state())
        self.assertEqual(rebind_report.update_type, ReconsolidationUpdateType.CONTEXTUAL_REBINDING.value)
        self.assertIn("mood_context", rebind_report.fields_rebound)

        reconstruct_report = reconsolidate(store.get("reconstruct"), "reflective", ["lab"], store=store, current_cycle=30, current_state=_state())
        self.assertEqual(reconstruct_report.update_type, ReconsolidationUpdateType.STRUCTURAL_RECONSTRUCTION.value)
        self.assertTrue(reconstruct_report.version_changed)

        conflict_artifact = store.retrieve(RetrievalQuery(semantic_tags=["mentor"], context_tags=["lab"], reference_cycle=30), k=1).recall_hypothesis
        conflict_report = reconsolidate(
            store.get("conflict"),
            "reflective",
            ["lab"],
            store=store,
            current_cycle=30,
            current_state=_state(),
            recall_artifact=conflict_artifact,
            conflict_type=ConflictType.FACTUAL,
        )
        self.assertEqual(conflict_report.update_type, ReconsolidationUpdateType.CONFLICT_MARKING.value)
        self.assertIn("factual", conflict_report.conflict_flags)

    def test_offline_consolidation_cycle_and_bridge_work(self) -> None:
        shared_entries = [
            _entry(
                entry_id=f"ep-{index}",
                content=f"Mentor promise episode {index}",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab", "weekly"],
                accessibility=0.5,
                support_count=2,
                retrieval_count=2,
                created_at=index,
                last_accessed=index + 10,
            )
            for index in range(1, 6)
        ]
        shared_entries.append(
            _entry(
                entry_id="cleanup-short",
                content="cleanup",
                semantic_tags=["noise", "flash"],
                context_tags=["roof"],
                store_level=StoreLevel.SHORT,
                accessibility=0.02,
                created_at=1,
                last_accessed=1,
            )
        )
        shared_entries[-1].trace_strength = 0.01
        store = MemoryStore(entries=shared_entries)
        report = store.run_consolidation_cycle(current_cycle=40, rng=random.Random(0), current_state=_state())
        self.assertTrue(report.upgrade.promoted_ids)
        self.assertTrue(report.extracted_patterns)
        self.assertTrue(report.replay_created_ids)
        self.assertIn("cleanup-short", report.cleanup.deleted_ids)
        extracted_entries = [store.get(entry_id) for entry_id in report.extracted_patterns]
        self.assertTrue(any(entry.memory_class is MemoryClass.SEMANTIC for entry in extracted_entries if entry is not None))
        self.assertTrue(any(entry.memory_class is MemoryClass.INFERRED for entry in extracted_entries if entry is not None))

        memory = LongTermMemory()
        memory.episodes = store.to_legacy_episodes()
        memory.ensure_memory_store()
        replay_batch = memory.replay_during_sleep(rng=random.Random(1), limit=2)
        self.assertEqual(len(replay_batch), 2)
        bridge_report = memory.run_memory_consolidation_cycle(
            current_cycle=60,
            rng=random.Random(1),
            current_state=_state(),
        )
        self.assertTrue(bridge_report.to_dict())
        self.assertEqual(len(memory.episodes), len(memory.memory_store.entries))

    def test_validate_inference_gates_unvalidated_entries_from_factual_donation(self) -> None:
        validated = _entry(
            entry_id="validated",
            content="validated pattern",
            semantic_tags=["mentor", "pattern"],
            context_tags=["lab", "weekly", "community"],
            memory_class=MemoryClass.INFERRED,
            source_type=SourceType.INFERENCE,
            support_count=5,
            retrieval_count=4,
            compression_metadata={"predictive_gain": 0.8, "cross_context_consistency": 0.9},
        )
        unvalidated = _entry(
            entry_id="unvalidated",
            content="weak hypothesis",
            semantic_tags=["mentor", "pattern"],
            context_tags=["lab"],
            memory_class=MemoryClass.INFERRED,
            source_type=SourceType.INFERENCE,
            support_count=1,
            retrieval_count=0,
            counterevidence_count=2,
            compression_metadata={"predictive_gain": 0.1, "cross_context_consistency": 0.2},
        )
        validated_result = validate_inference(validated)
        unvalidated_result = validate_inference(unvalidated)
        self.assertTrue(validated_result.passed)
        self.assertEqual(validated_result.validation_status, "validated")
        self.assertFalse(unvalidated_result.passed)
        self.assertIn(unvalidated_result.validation_status, {"unvalidated", "contradicted"})

        store = MemoryStore(entries=[validated, unvalidated, _entry(entry_id="base", content="base", semantic_tags=["mentor"], context_tags=["lab"])])
        result = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=10),
            k=3,
        )
        hypothesis = result.recall_hypothesis
        self.assertIsNotNone(hypothesis)
        self.assertNotIn("unvalidated", hypothesis.auxiliary_entry_ids)

    def test_semantic_skeleton_records_lineage_metadata(self) -> None:
        entries = [
            _entry(
                entry_id=f"cluster-{index}",
                content=f"cluster {index}",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab", "weekly"],
                support_count=2,
            )
            for index in range(3)
        ]
        skeleton = compress_episodic_cluster_to_semantic_skeleton(entries)
        metadata = dict(skeleton.compression_metadata or {})
        self.assertEqual(metadata["lineage_type"], "episodic_compression")
        self.assertEqual(metadata["support_entry_ids"], [entry.id for entry in entries])
        self.assertTrue(metadata["stable_structure"])


if __name__ == "__main__":
    unittest.main()
