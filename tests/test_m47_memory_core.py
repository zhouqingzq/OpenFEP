from __future__ import annotations

import unittest

from segmentum.m47_runtime import build_m47_runtime_snapshot
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_retrieval import RetrievalQuery
from segmentum.memory_store import MemoryStore


class TestM47MemoryCore(unittest.TestCase):
    def test_runtime_snapshot_exposes_state_dynamic_and_cognitive_probes(self) -> None:
        snapshot = build_m47_runtime_snapshot()

        self.assertIn("state_vector", snapshot["probes"])
        self.assertIn("dynamic_salience", snapshot["probes"])
        self.assertIn("cognitive_style", snapshot["probes"])
        self.assertEqual(len(snapshot["short_seed_groups"]), 3)
        self.assertEqual(len(snapshot["long_seed_groups"]), 2)

    def test_promotion_audit_uses_unified_score_breakdown(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        long_entries = [
            entry
            for group in snapshot["long_seed_groups"]
            for entry in group["entries"]
            if entry["memory_class"] == "episodic"
        ]
        audited = next(entry for entry in long_entries if "m47_promotion_audit" in dict(entry.get("compression_metadata") or {}))
        audit = dict(dict(audited.get("compression_metadata") or {}).get("m47_promotion_audit", {}))

        self.assertIn("short_to_mid_score", audit)
        self.assertIn("mid_to_long_score", audit)
        self.assertIn("score_breakdown", audit)
        self.assertIn("novelty_noise_penalty", dict(audit["score_breakdown"]))

    def test_recall_reconstruction_tracks_candidates_and_donor_blocks(self) -> None:
        store = MemoryStore(
            entries=[
                MemoryEntry(
                    id="primary",
                    content="Mentor meeting in the archive.",
                    memory_class=MemoryClass.EPISODIC,
                    store_level=StoreLevel.MID,
                    source_type=SourceType.EXPERIENCE,
                    created_at=1,
                    last_accessed=3,
                    valence=0.1,
                    arousal=0.2,
                    encoding_attention=0.5,
                    novelty=0.2,
                    relevance_goal=0.2,
                    relevance_threat=0.0,
                    relevance_self=0.3,
                    relevance_social=0.1,
                    relevance_reward=0.1,
                    relevance=0.3,
                    salience=0.5,
                    trace_strength=0.6,
                    accessibility=0.65,
                    abstractness=0.2,
                    source_confidence=0.9,
                    reality_confidence=0.8,
                    semantic_tags=["mentor", "meeting", "continuity"],
                    context_tags=["archive"],
                    anchor_slots={"time": "monday", "place": "archive", "agents": "mentor_lin", "action": "meet", "outcome": "report_reviewed"},
                    anchor_strengths={"time": "weak", "place": "weak", "agents": "strong", "action": "strong", "outcome": "strong"},
                ),
                MemoryEntry(
                    id="aux",
                    content="Mentor meeting near the river annex.",
                    memory_class=MemoryClass.EPISODIC,
                    store_level=StoreLevel.MID,
                    source_type=SourceType.EXPERIENCE,
                    created_at=2,
                    last_accessed=3,
                    valence=0.1,
                    arousal=0.2,
                    encoding_attention=0.5,
                    novelty=0.2,
                    relevance_goal=0.2,
                    relevance_threat=0.0,
                    relevance_self=0.2,
                    relevance_social=0.1,
                    relevance_reward=0.1,
                    relevance=0.25,
                    salience=0.45,
                    trace_strength=0.55,
                    accessibility=0.62,
                    abstractness=0.2,
                    source_confidence=0.9,
                    reality_confidence=0.8,
                    semantic_tags=["mentor", "meeting", "continuity"],
                    context_tags=["archive"],
                    anchor_slots={"time": "tuesday", "place": "river_annex", "agents": "mentor_lin", "action": "meet", "outcome": "report_reviewed"},
                    anchor_strengths={"time": "weak", "place": "weak", "agents": "strong", "action": "strong", "outcome": "strong"},
                ),
            ]
        )
        result = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "meeting"], context_tags=["archive"], reference_cycle=4),
            k=2,
        )

        self.assertIsNotNone(result.recall_hypothesis)
        recall = result.recall_hypothesis
        self.assertTrue(recall.candidate_ids)
        self.assertIn("anchor_contributions", result.reconstruction_trace)
        self.assertIn("competition_snapshot", result.reconstruction_trace)
        primary_entry = store.get(recall.primary_entry_id)
        self.assertIsNotNone(primary_entry)
        self.assertIn("m47_recall_audit", dict(primary_entry.compression_metadata or {}))

    def test_identity_noise_boundary_survives_in_runtime_snapshot(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        probe = snapshot["probes"]["dynamic_salience"]["identity_vs_noise_control"]

        self.assertGreater(probe["identity_event"]["relevance_self"], probe["novelty_noise"]["relevance_self"])
        self.assertGreater(probe["identity_event"]["salience"], 0.0)


if __name__ == "__main__":
    unittest.main()
