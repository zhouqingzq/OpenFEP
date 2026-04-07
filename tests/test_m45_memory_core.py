from __future__ import annotations

import unittest

from segmentum.m45_acceptance_shared import (
    baseline_body_state,
    baseline_errors,
    baseline_observation,
    baseline_prediction,
)
from segmentum.memory import LongTermMemory
from segmentum.memory_decay import (
    LONG_DORMANT_ACCESS_THRESHOLD,
    LONG_DORMANT_TRACE_THRESHOLD,
    decay_accessibility,
    decay_accessibility_for_level,
    decay_trace_strength,
    decay_trace_strength_for_level,
    trace_decay_rate,
)
from segmentum.memory_encoding import SalienceConfig, encode_memory
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_store import MemoryStore


def _identity_state() -> dict[str, object]:
    return {
        "active_goals": ["keep promises", "protect mentees"],
        "goal_keywords": ["promise", "mentor", "care"],
        "identity_roles": ["mentor", "caregiver"],
        "important_relationships": ["lin"],
        "active_commitments": ["weekly mentor checkin"],
        "identity_commitments": ["weekly mentor checkin"],
        "identity_themes": ["reliable mentor", "care continuity"],
        "identity_active_themes": ["mentor", "care continuity"],
        "self_narrative_keywords": ["mentor", "promise", "care"],
        "recent_mood_baseline": "reflective",
        "social_context_active": True,
    }


class TestM45MemoryCore(unittest.TestCase):
    def test_memory_entry_round_trip_and_version_tracking(self) -> None:
        entry = MemoryEntry(
            content="Anchored memory entry",
            created_at=1,
            last_accessed=1,
            arousal=0.2,
            encoding_attention=0.3,
            novelty=0.1,
            relevance_goal=0.2,
            relevance_threat=0.1,
            relevance_self=0.7,
            relevance_social=0.4,
            relevance_reward=0.1,
            relevance=0.3,
            salience=0.25,
            trace_strength=0.25,
            accessibility=0.25,
            abstractness=0.2,
            source_confidence=0.9,
            reality_confidence=0.85,
            semantic_tags=["mentor"],
            context_tags=["community"],
            anchor_slots={"action": "checkin", "outcome": "trust", "agents": "lin"},
            anchor_strengths={"agents": "strong", "action": "strong", "outcome": "strong"},
            competing_interpretations=["alt-1"],
            compression_metadata={"abstraction_reason": "test"},
            support_count=1,
        )

        restored = MemoryEntry.from_dict(entry.to_dict())
        self.assertEqual(restored.to_dict(), entry.to_dict())
        stable_hash = restored.content_hash
        restored.sync_content_hash()
        self.assertEqual(stable_hash, restored.content_hash)
        version_before = restored.version
        restored.content = "Anchored memory entry updated"
        restored.sync_content_hash()
        self.assertEqual(restored.version, version_before + 1)

    def test_guardrails_block_all_weak_anchors_and_missing_procedural_steps(self) -> None:
        with self.assertRaises(ValueError):
            MemoryEntry(
                content="bad episodic entry",
                anchor_strengths={key: "weak" for key in ("time", "place", "agents", "action", "outcome")},
                support_count=1,
            )

        with self.assertRaises(ValueError):
            MemoryEntry(
                content="bad procedural entry",
                memory_class=MemoryClass.PROCEDURAL,
                support_count=1,
            )

    def test_encode_memory_respects_self_relevance_constraints(self) -> None:
        config = SalienceConfig()
        state = _identity_state()
        identity_entry = encode_memory(
            {
                "content": "I kept the weekly mentor promise to Lin.",
                "action": "mentor_checkin",
                "outcome": "commitment_kept",
                "semantic_tags": ["mentor", "promise", "care"],
                "roles": ["mentor"],
                "relationships": ["lin"],
                "commitments": ["weekly mentor checkin"],
                "narrative_nodes": ["reliable mentor"],
                "arousal": 0.18,
                "novelty": 0.10,
                "encoding_attention": 0.45,
            },
            state,
            config,
        )
        first_person_only = encode_memory(
            {
                "content": "I noticed the hallway light during my current task.",
                "action": "observe_light",
                "outcome": "light_flicker",
                "semantic_tags": ["light", "task"],
                "arousal": 0.25,
                "novelty": 0.20,
                "encoding_attention": 0.45,
            },
            state,
            config,
        )
        procedural = encode_memory(
            {
                "memory_class": "procedural",
                "content": "Reactor calming routine",
                "procedure_steps": ["scan gauges", "vent pressure", "log readings"],
                "step_confidence": [0.9, 0.85, 0.88],
                "execution_contexts": ["reactor_room"],
                "semantic_tags": ["reactor", "procedure"],
                "encoding_attention": 0.75,
                "arousal": 0.2,
                "novelty": 0.2,
            },
            state,
            config,
        )

        self.assertGreater(identity_entry.relevance_self, 0.7)
        self.assertLess(first_person_only.relevance_self, 0.2)
        self.assertTrue(procedural.procedure_steps)
        self.assertEqual(len(procedural.procedure_steps), len(procedural.step_confidence))
        first_person_audit = dict(dict(first_person_only.compression_metadata or {}).get("encoding_audit", {}))
        self.assertIn("guard:first_person_not_identity", list(first_person_audit.get("self_evidence", [])))

    def test_semantic_and_inferred_entries_record_lineage_metadata(self) -> None:
        semantic = encode_memory(
            {
                "content": "Repeated mentor check-ins stabilize trust over time.",
                "memory_class": "semantic",
                "semantic_pattern": True,
                "supporting_episode_ids": ["ep-a", "ep-b", "ep-c"],
                "semantic_tags": ["mentor", "trust", "pattern"],
                "context_tags": ["community"],
                "encoding_attention": 0.65,
                "arousal": 0.25,
                "novelty": 0.25,
            },
            _identity_state(),
            SalienceConfig(),
        )
        inferred = encode_memory(
            {
                "content": "Lin may trust weekly check-ins because consistency signals safety.",
                "memory_class": "inferred",
                "inferred": True,
                "supporting_episode_ids": ["ep-a", "ep-b"],
                "semantic_tags": ["mentor", "trust", "safety"],
                "context_tags": ["community"],
                "encoding_attention": 0.55,
                "arousal": 0.20,
                "novelty": 0.40,
            },
            _identity_state(),
            SalienceConfig(),
        )

        semantic_metadata = dict(semantic.compression_metadata or {})
        inferred_metadata = dict(inferred.compression_metadata or {})
        self.assertEqual(semantic_metadata["lineage_type"], "episodic_compression")
        self.assertTrue(semantic_metadata["predictive_use_cases"])
        self.assertEqual(inferred_metadata["lineage_type"], "pattern_extraction")
        self.assertTrue(inferred_metadata["predictive_use_cases"])
        self.assertEqual(semantic.derived_from, ["ep-a", "ep-b", "ep-c"])
        self.assertEqual(inferred.derived_from, ["ep-a", "ep-b"])

    def test_source_and_reality_confidence_can_vary_independently(self) -> None:
        entries = [
            MemoryEntry(
                content=f"pair-{index}",
                created_at=index,
                last_accessed=index,
                arousal=0.1,
                encoding_attention=0.1,
                novelty=0.1,
                relevance_goal=0.1,
                relevance_threat=0.1,
                relevance_self=0.1,
                relevance_social=0.1,
                relevance_reward=0.1,
                relevance=0.1,
                salience=0.1,
                trace_strength=0.1,
                accessibility=0.1,
                abstractness=0.1,
                source_confidence=source,
                reality_confidence=reality,
                support_count=1,
            )
            for index, (source, reality) in enumerate(
                ((0.95, 0.20), (0.20, 0.95), (0.90, 0.90), (0.30, 0.30)),
                start=1,
            )
        ]

        self.assertEqual(len({(entry.source_confidence, entry.reality_confidence) for entry in entries}), 4)

    def test_long_term_memory_bridge_syncs_store_after_append_merge_delete_and_compress(self) -> None:
        memory = LongTermMemory(surprise_threshold=0.2, sleep_minimum_support=2, max_active_age=1)
        memory.ensure_memory_store()
        self.assertEqual(len(memory.memory_store.entries), 0)

        memory.store_episode(
            cycle=1,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="hide",
            outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.45},
            body_state=baseline_body_state(),
        )
        self.assertEqual(len(memory.episodes), 1)
        self.assertEqual(len(memory.memory_store.entries), 1)

        merge_decision = memory.maybe_store_episode(
            cycle=2,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="hide",
            outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.45},
            body_state=baseline_body_state(),
        )
        self.assertFalse(merge_decision.episode_created)
        self.assertEqual(merge_decision.support_delta, 1)
        self.assertEqual(len(memory.episodes), 1)
        self.assertEqual(len(memory.memory_store.entries), 1)
        self.assertGreaterEqual(int(memory.episodes[0]["support_count"]), 2)

        payload = memory.episodes[0]
        self.assertTrue(memory.delete_episode(payload))
        self.assertEqual(len(memory.episodes), 0)
        self.assertEqual(len(memory.memory_store.entries), 0)

        for cycle, action in ((3, "hide"), (4, "forage")):
            memory.store_episode(
                cycle=cycle,
                observation=baseline_observation(),
                prediction=baseline_prediction(),
                errors=baseline_errors(),
                action=action,
                outcome={"energy_delta": -0.05, "stress_delta": 0.10, "free_energy_drop": -0.40},
                body_state=baseline_body_state(),
            )
        removed = memory.compress_episodes(current_cycle=10)
        self.assertGreaterEqual(removed, 0)
        self.assertEqual(len(memory.memory_store.entries), len(memory.episodes))

    def test_legacy_bridge_round_trip_reflects_entry_mutations_and_preserves_unknown_fields(self) -> None:
        memory = LongTermMemory()
        payload = memory.store_episode(
            cycle=31,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="hide",
            outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.5},
            body_state=baseline_body_state(),
        )
        legacy_payload = dict(payload)
        legacy_payload["custom_flag"] = "keep-me"
        legacy_payload["custom_nested"] = {"note": "preserved"}

        store = MemoryStore.from_legacy_episodes([legacy_payload])
        entry = store.entries[0]
        entry.anchor_slots["action"] = "hide_revised"
        entry.anchor_slots["outcome"] = "mutated_outcome"
        entry.novelty = 0.99
        entry.salience = 0.77
        entry.support_count = 5

        restored_payload = store.to_legacy_episodes()[0]
        self.assertEqual(restored_payload["action"], "hide_revised")
        self.assertEqual(restored_payload["predicted_outcome"], "mutated_outcome")
        self.assertEqual(restored_payload["value_label"], "mutated_outcome")
        self.assertEqual(restored_payload["prediction_error"], 0.99)
        self.assertEqual(restored_payload["total_surprise"], 0.77)
        self.assertEqual(restored_payload["support_count"], 5)
        self.assertEqual(restored_payload["custom_flag"], "keep-me")
        self.assertEqual(restored_payload["custom_nested"], {"note": "preserved"})

    def test_memory_store_incremental_legacy_sync_preserves_unchanged_entries(self) -> None:
        payload_a = {
            "episode_id": "ep-a",
            "timestamp": 1,
            "action": "hide",
            "predicted_outcome": "safe",
            "prediction_error": 0.2,
            "total_surprise": 0.2,
            "support_count": 1,
        }
        payload_b = {
            "episode_id": "ep-b",
            "timestamp": 2,
            "action": "forage",
            "predicted_outcome": "gain",
            "prediction_error": 0.3,
            "total_surprise": 0.3,
            "support_count": 1,
        }
        store = MemoryStore.from_legacy_episodes([payload_a, payload_b])
        original_a = store.get("ep-a")
        original_b = store.get("ep-b")

        payload_b_updated = dict(payload_b)
        payload_b_updated["support_count"] = 3
        sync_report = store.replace_legacy_group([payload_a, payload_b_updated])

        self.assertEqual(store.get("ep-a"), original_a)
        self.assertIs(store.get("ep-a"), original_a)
        self.assertIsNot(store.get("ep-b"), original_b)
        self.assertIn("ep-a", sync_report["reused_ids"])
        self.assertIn("ep-b", sync_report["upserted_ids"])

        self.assertTrue(store.remove_legacy_episode("ep-b"))
        self.assertIsNone(store.get("ep-b"))

    def test_negative_reward_language_does_not_raise_reward_score(self) -> None:
        entry = encode_memory(
            {
                "content": "Severe penalty and loss recorded.",
                "action": "audit",
                "outcome": "loss_recorded",
                "semantic_tags": ["penalty", "loss"],
                "encoding_attention": 0.5,
                "arousal": 0.4,
                "novelty": 0.2,
            },
            _identity_state(),
            SalienceConfig(),
        )
        audit = dict(entry.compression_metadata or {}).get("encoding_audit", {})

        self.assertLessEqual(entry.relevance_reward, 0.05)
        self.assertGreater(entry.relevance_threat, 0.0)
        self.assertTrue(dict(audit).get("threat_evidence"))

    def test_dual_decay_and_non_deletion_forgetting(self) -> None:
        short_compare = MemoryEntry(
            content="short-compare",
            store_level=StoreLevel.SHORT,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        )
        short_entry = MemoryEntry(
            content="short",
            store_level=StoreLevel.SHORT,
            created_at=1,
            last_accessed=1,
            trace_strength=0.04,
            accessibility=0.20,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        )
        mid_entry = MemoryEntry(
            content="mid",
            store_level=StoreLevel.MID,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        )
        long_entry = MemoryEntry(
            content="long",
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        )
        procedural_long = MemoryEntry(
            content="procedure",
            memory_class=MemoryClass.PROCEDURAL,
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            procedure_steps=["a", "b"],
            step_confidence=[0.9, 0.9],
            support_count=1,
        )
        source_conflict_entry = MemoryEntry(
            content="source-conflict",
            store_level=StoreLevel.LONG,
            source_type=SourceType.RECONSTRUCTION,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.3,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=1,
            compression_metadata={"source_conflict": True},
        )
        reality_conflict_entry = MemoryEntry(
            content="reality-conflict",
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=1.0,
            accessibility=1.0,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.3,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=2,
            counterevidence_count=3,
            competing_interpretations=["alt-hypothesis"],
            compression_metadata={"factual_conflict": True},
        )
        dormant_long = MemoryEntry(
            content="dormant",
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=LONG_DORMANT_TRACE_THRESHOLD,
            accessibility=LONG_DORMANT_ACCESS_THRESHOLD,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        )
        store = MemoryStore(
            entries=[
                short_compare,
                short_entry,
                mid_entry,
                long_entry,
                procedural_long,
                source_conflict_entry,
                reality_conflict_entry,
                dormant_long,
            ]
        )

        for elapsed in (5, 20, 80):
            self.assertGreater(
                decay_trace_strength_for_level(
                    short_compare.trace_strength,
                    StoreLevel.SHORT,
                    elapsed,
                    memory_class=short_compare.memory_class,
                ),
                decay_accessibility_for_level(short_compare.accessibility, StoreLevel.SHORT, elapsed),
            )
            self.assertLess(
                decay_trace_strength_for_level(
                    short_entry.trace_strength,
                    StoreLevel.SHORT,
                    elapsed,
                    memory_class=short_entry.memory_class,
                ),
                decay_trace_strength_for_level(
                    mid_entry.trace_strength,
                    StoreLevel.MID,
                    elapsed,
                    memory_class=mid_entry.memory_class,
                ),
            )
            self.assertLess(
                decay_trace_strength_for_level(
                    mid_entry.trace_strength,
                    StoreLevel.MID,
                    elapsed,
                    memory_class=mid_entry.memory_class,
                ),
                decay_trace_strength_for_level(
                    long_entry.trace_strength,
                    StoreLevel.LONG,
                    elapsed,
                    memory_class=long_entry.memory_class,
                ),
            )
            self.assertGreater(
                decay_trace_strength_for_level(
                    procedural_long.trace_strength,
                    StoreLevel.LONG,
                    elapsed,
                    memory_class=procedural_long.memory_class,
                ),
                decay_trace_strength_for_level(
                    long_entry.trace_strength,
                    StoreLevel.LONG,
                    elapsed,
                    memory_class=long_entry.memory_class,
                ),
            )

        source_before = source_conflict_entry.source_confidence
        reality_before = reality_conflict_entry.reality_confidence
        report = store.apply_decay(current_cycle=21)
        self.assertIn(short_entry.id, report.deleted_short_residue)
        self.assertIn(dormant_long.id, report.dormant_marked)
        self.assertTrue(report.abstracted_entries or report.confidence_drifted)
        self.assertIn(source_conflict_entry.id, report.source_confidence_drifted)
        self.assertIn(reality_conflict_entry.id, report.reality_confidence_drifted)
        self.assertLess(source_conflict_entry.source_confidence, source_before)
        self.assertLess(reality_conflict_entry.reality_confidence, reality_before)

    def test_long_term_memory_incremental_sync_does_not_churn_unmodified_entries(self) -> None:
        memory = LongTermMemory()
        memory.ensure_memory_store()
        first_payload = memory.store_episode(
            cycle=1,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="hide",
            outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.45},
            body_state=baseline_body_state(),
        )
        first_entry = memory.memory_store.get(first_payload["episode_id"])

        memory.store_episode(
            cycle=2,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="forage",
            outcome={"energy_delta": 0.1, "stress_delta": -0.1, "free_energy_drop": -0.2},
            body_state=baseline_body_state(),
        )

        self.assertIs(memory.memory_store.get(first_payload["episode_id"]), first_entry)

    def test_promotion_recomputes_trace_and_accessibility_using_new_level_rates(self) -> None:
        state = _identity_state()
        identity_entry = encode_memory(
            {
                "content": "I kept the weekly mentor promise to Lin.",
                "action": "mentor_checkin",
                "outcome": "commitment_kept",
                "semantic_tags": ["mentor", "promise", "care"],
                "roles": ["mentor"],
                "relationships": ["lin"],
                "commitments": ["weekly mentor checkin"],
                "narrative_nodes": ["reliable mentor"],
                "arousal": 0.18,
                "novelty": 0.10,
                "encoding_attention": 0.45,
                "created_at": 12,
            },
            state,
            SalienceConfig(),
        )
        identity_entry.retrieval_count = 1
        identity_entry.last_accessed = 20
        identity_entry.compression_metadata = {
            "m45_internal": {
                "last_decay_cycle": 12,
                "decay_base_trace_strength": 1.0,
                "decay_base_accessibility": 1.0,
            }
        }

        semantic_entry = encode_memory(_semantic_event := {
            "content": "Repeated mentor check-ins stabilize trust over time.",
            "memory_class": "semantic",
            "semantic_pattern": True,
            "supporting_episode_ids": ["ep-a", "ep-b", "ep-c"],
            "semantic_tags": ["mentor", "trust", "pattern"],
            "context_tags": ["community"],
            "encoding_attention": 0.65,
            "arousal": 0.25,
            "novelty": 0.25,
            "created_at": 16,
        }, state, SalienceConfig())
        semantic_entry.store_level = StoreLevel.MID
        semantic_entry.retrieval_count = 3
        semantic_entry.salience = 0.88
        semantic_entry.last_accessed = 26
        semantic_entry.compression_metadata = {
            "m45_internal": {
                "last_decay_cycle": 16,
                "decay_base_trace_strength": 1.0,
                "decay_base_accessibility": 1.0,
            }
        }

        store = MemoryStore()
        store.add(identity_entry)
        store.add(semantic_entry)

        promoted_mid = store.get(identity_entry.id)
        promoted_long = store.get(semantic_entry.id)
        self.assertIsNotNone(promoted_mid)
        self.assertIsNotNone(promoted_long)

        promoted_mid = promoted_mid
        promoted_long = promoted_long
        mid_internal = dict((promoted_mid.compression_metadata or {}).get("m45_internal", {}))
        long_internal = dict((promoted_long.compression_metadata or {}).get("m45_internal", {}))
        mid_promotion = dict(mid_internal.get("last_promotion", {}))
        long_promotion = dict(long_internal.get("last_promotion", {}))

        self.assertEqual(promoted_mid.store_level, StoreLevel.MID)
        self.assertEqual(mid_promotion.get("old_level"), "short")
        self.assertEqual(mid_promotion.get("new_level"), "mid")
        self.assertGreater(float(mid_promotion.get("trace_after", 0.0)), float(mid_promotion.get("trace_before", 1.0)))
        self.assertGreater(
            float(mid_promotion.get("accessibility_after", 0.0)),
            float(mid_promotion.get("accessibility_before", 1.0)),
        )
        self.assertLess(
            float(mid_promotion.get("trace_rate_after", 1.0)),
            float(mid_promotion.get("trace_rate_before", 0.0)),
        )
        self.assertLess(
            float(mid_promotion.get("access_rate_after", 1.0)),
            float(mid_promotion.get("access_rate_before", 0.0)),
        )
        self.assertEqual(mid_internal.get("last_decay_cycle"), promoted_mid.last_accessed)
        self.assertAlmostEqual(float(mid_internal.get("decay_base_trace_strength", 0.0)), promoted_mid.trace_strength)
        self.assertAlmostEqual(
            float(mid_internal.get("decay_base_accessibility", 0.0)),
            promoted_mid.accessibility,
        )
        self.assertEqual(mid_promotion.get("trace_rate_after"), trace_decay_rate(StoreLevel.MID))

        self.assertEqual(promoted_long.store_level, StoreLevel.LONG)
        self.assertEqual(long_promotion.get("old_level"), "mid")
        self.assertEqual(long_promotion.get("new_level"), "long")
        self.assertGreater(float(long_promotion.get("trace_after", 0.0)), float(long_promotion.get("trace_before", 1.0)))
        self.assertGreater(
            float(long_promotion.get("accessibility_after", 0.0)),
            float(long_promotion.get("accessibility_before", 1.0)),
        )
        self.assertLess(
            float(long_promotion.get("trace_rate_after", 1.0)),
            float(long_promotion.get("trace_rate_before", 0.0)),
        )
        self.assertLess(
            float(long_promotion.get("access_rate_after", 1.0)),
            float(long_promotion.get("access_rate_before", 0.0)),
        )
        self.assertEqual(long_internal.get("last_decay_cycle"), promoted_long.last_accessed)


if __name__ == "__main__":
    unittest.main()
