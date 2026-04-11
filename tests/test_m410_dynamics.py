from __future__ import annotations

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m410_audit import (
    GATE_ORDER,
    _default_path_entry_evidence,
    build_budget_competition_evidence,
    build_m410_acceptance_report,
    build_replay_drift_evidence,
    write_m410_acceptance_artifacts,
)
from segmentum.memory_consolidation import (
    compress_episodic_cluster_to_semantic_skeleton,
    constrained_replay,
)
from segmentum.memory_encoding import EncodingDynamics, EncodingDynamicsInput, SalienceConfig, encode_memory
from segmentum.memory_model import MemoryClass, MemoryEntry, StoreLevel
from segmentum.memory_store import MemoryStore


class TestM410Dynamics(unittest.TestCase):
    def test_encoding_dynamics_budget_competition_and_fallback_tags(self) -> None:
        evidence = build_budget_competition_evidence()
        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["runtime_competition"]["passed"])
        self.assertGreaterEqual(evidence["runtime_competition"]["event_count"], 2)
        self.assertNotEqual(
            evidence["runtime_competition"]["constrained_retention_curve"],
            evidence["runtime_competition"]["unlimited_retention_curve"],
        )
        constrained = list(evidence["constrained_strengths"])
        unlimited = list(evidence["unlimited_strengths"])
        self.assertGreater(min(constrained[2:]), max(constrained[:2]))
        self.assertLess(max(constrained[:2]), max(unlimited[:2]))

        dynamic = encode_memory(
            {
                "content": "sensor mismatch",
                "prediction_error": 0.7,
                "total_surprise": 0.8,
                "arousal": 0.6,
                "attention_budget": 0.5,
                "encoding_attention": 1.0,
            },
            {},
            SalienceConfig(),
        )
        heuristic = encode_memory(
            {
                "content": "I noticed the current task light",
                "arousal": 0.2,
                "encoding_attention": 0.4,
            },
            {},
            SalienceConfig(),
        )
        self.assertEqual(dynamic.compression_metadata["encoding_source"], "dynamics")
        self.assertEqual(heuristic.compression_metadata["encoding_source"], "heuristic")
        self.assertIn("m410_encoding_dynamics", dynamic.compression_metadata)
        self.assertEqual(
            dynamic.compression_metadata["encoding_strength"],
            dynamic.salience,
        )

    def test_semantic_memory_is_vector_centroid_and_replay_updates_it(self) -> None:
        entries = [
            MemoryEntry(
                id=f"m410-ep-{index}",
                content=f"episode {index}",
                semantic_tags=["m410", "cluster"],
                context_tags=["lab"],
                state_vector=[float(index), float(index + 1)],
                salience=0.4,
                arousal=0.5,
                encoding_attention=0.9,
                novelty=0.25,
            )
            for index in range(5)
        ]
        semantic = compress_episodic_cluster_to_semantic_skeleton(entries)
        self.assertEqual(semantic.memory_class, MemoryClass.SEMANTIC)
        self.assertEqual(semantic.consolidation_source, "dynamics")
        self.assertTrue(semantic.centroid)
        self.assertTrue(semantic.support_ids)
        self.assertIsNotNone(entries[0].semantic_reconstruction_error)
        self.assertNotIn("Semantic skeleton from", semantic.content)

        before = list(semantic.centroid or [])
        entries[-1].state_vector = [10.0, 11.0]
        entries[-1].salience = 0.99
        store = MemoryStore(entries=[*entries, semantic])
        touched = constrained_replay(store, random.Random(2), batch_size=1)
        after = list(semantic.centroid or [])
        shift = sum(abs(left - right) for left, right in zip(before, after))
        self.assertGreater(shift, 0.1)
        self.assertTrue(touched)
        self.assertIsNotNone(touched[0].replay_second_pass_error)
        self.assertIsNotNone(touched[0].salience_delta)
        refresh = dict((semantic.compression_metadata or {}).get("m410_replay_refresh", {}))
        self.assertTrue(refresh)
        self.assertNotEqual(refresh["residual_norm_mean_before"], refresh["residual_norm_mean_after"])

    def test_replay_drift_evidence_passes(self) -> None:
        evidence = build_replay_drift_evidence()
        self.assertTrue(evidence["passed"])
        self.assertGreater(evidence["centroid_shift_l1"], 0.1)
        self.assertTrue(evidence["replay_trail"])

    def test_acceptance_report_and_artifact_writer(self) -> None:
        report, evidence = build_m410_acceptance_report(seed=4, cycles=20)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_acceptance_conclusion"], "ACCEPT")
        self.assertEqual(tuple(report["gate_order"]), GATE_ORDER)
        self.assertEqual(report["failed_gates"], [])
        self.assertEqual(evidence["default_path"]["invalid_encoding_source_ids"], [])
        self.assertGreater(evidence["default_path"]["encoded_episode_count"], 0)
        self.assertNotIn("missing", evidence["default_path"]["encoding_source_histogram"])
        self.assertTrue(evidence["default_path"]["semantic_dynamic_ids"])
        self.assertTrue(evidence["default_path"]["replay_touched_ids"])
        self.assertTrue(evidence["default_path"]["replay_semantic_refresh_updated_ids"])
        self.assertTrue(evidence["budget"]["runtime_competition"]["passed"])
        self.assertTrue(
            report["gate_summaries"]["replay_reencoding"]["synthetic_drift_support_passed"]
        )

        with TemporaryDirectory() as tmp_dir:
            outputs = write_m410_acceptance_artifacts(output_root=tmp_dir, seed=4, cycles=20)
            written_report = json.loads(Path(outputs["report"]).read_text(encoding="utf-8"))
            written_evidence = json.loads(Path(outputs["evidence"]).read_text(encoding="utf-8"))
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")
        self.assertEqual(written_report["status"], "PASS")
        self.assertTrue(written_evidence["default_path"]["semantic_dynamic_ids"])
        self.assertIn("M4.10 Acceptance Summary", summary)

    def test_formula_is_exposed_on_entry_point(self) -> None:
        result = EncodingDynamics.score(
            EncodingDynamicsInput(
                prediction_error=0.8,
                surprise=0.7,
                arousal=0.6,
                attention_budget=0.5,
                requested_budget=1.0,
            )
        )
        self.assertAlmostEqual(result.raw_drive, 0.8 * 0.7 * 0.6)
        self.assertAlmostEqual(result.raw_strength, 0.45 * 0.8 + 0.35 * 0.7 + 0.20 * 0.6)
        self.assertAlmostEqual(result.attention_budget_granted, 0.5)
        self.assertAlmostEqual(result.encoding_strength, result.raw_strength * 0.5)

    def test_default_path_negative_cases_fail(self) -> None:
        valid_episode = MemoryEntry(
            id="ep-valid",
            content="valid encoded episode",
            memory_class=MemoryClass.EPISODIC,
            compression_metadata={"encoding_source": "dynamics"},
            replay_second_pass_error=0.2,
            salience_delta=0.1,
        )
        missing_source = MemoryEntry(
            id="ep-missing",
            content="missing encoded source",
            memory_class=MemoryClass.EPISODIC,
            compression_metadata={},
        )
        semantic = MemoryEntry(
            id="sem-valid",
            content="semantic centroid display",
            memory_class=MemoryClass.SEMANTIC,
            store_level=StoreLevel.MID,
            consolidation_source="dynamics",
            centroid=[0.1, 0.2],
            residual_norm_mean=0.1,
            residual_norm_var=0.01,
            support_ids=["ep-valid"],
            compression_metadata={
                "content_role": "metadata_display_only",
                "m410_replay_refresh": {
                    "touched_source_ids": ["ep-valid"],
                    "centroid_before": [0.1, 0.2],
                    "centroid_after": [0.2, 0.3],
                    "residual_norm_mean_before": 0.2,
                    "residual_norm_mean_after": 0.1,
                    "residual_norm_var_before": 0.02,
                    "residual_norm_var_after": 0.01,
                },
            },
        )
        missing_source_evidence = _default_path_entry_evidence(
            [valid_episode, missing_source, semantic],
            seed=0,
            cycles=0,
            sleep_trace_has_m410=True,
        )
        self.assertFalse(missing_source_evidence["passed"])
        self.assertEqual(missing_source_evidence["invalid_encoding_source_ids"], ["ep-missing"])
        self.assertEqual(missing_source_evidence["encoding_source_histogram"]["missing"], 1)

        missing_centroid = MemoryEntry(
            id="sem-no-centroid",
            content="semantic centroid display",
            memory_class=MemoryClass.SEMANTIC,
            consolidation_source="dynamics",
            residual_norm_mean=0.1,
            residual_norm_var=0.01,
            support_ids=["ep-valid"],
            compression_metadata={"content_role": "metadata_display_only"},
        )
        semantic_evidence = _default_path_entry_evidence(
            [valid_episode, missing_centroid],
            seed=0,
            cycles=0,
            sleep_trace_has_m410=True,
        )
        self.assertFalse(semantic_evidence["passed"])
        self.assertEqual(semantic_evidence["semantic_missing_centroid_ids"], ["sem-no-centroid"])

        replay_without_refresh = MemoryEntry(
            id="sem-no-refresh",
            content="semantic centroid display",
            memory_class=MemoryClass.SEMANTIC,
            consolidation_source="dynamics",
            centroid=[0.1, 0.2],
            residual_norm_mean=0.1,
            residual_norm_var=0.01,
            support_ids=["ep-valid"],
            compression_metadata={"content_role": "metadata_display_only"},
        )
        replay_evidence = _default_path_entry_evidence(
            [valid_episode, replay_without_refresh],
            seed=0,
            cycles=0,
            sleep_trace_has_m410=True,
        )
        self.assertFalse(replay_evidence["passed"])
        self.assertEqual(replay_evidence["replay_semantic_refresh_updated_ids"], [])


if __name__ == "__main__":
    unittest.main()
