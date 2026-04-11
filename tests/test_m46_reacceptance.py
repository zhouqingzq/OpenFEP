from __future__ import annotations

from copy import deepcopy
import json
import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from segmentum.m46_reacceptance import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    GATE_HONESTY,
    GATE_LEGACY,
    GATE_RETRIEVAL,
    STATUS_NOT_RUN,
    _build_consolidation_entries,
    _build_honesty_record,
    _entry,
    _reconstruction_store,
    _retrieval_store,
    _state,
    build_m46_reacceptance_report,
    write_m46_reacceptance_artifacts,
)
from segmentum.memory import LongTermMemory
from segmentum.memory_consolidation import (
    ConflictType,
    ReconstructionConfig,
    maybe_reconstruct,
    reconsolidate,
    validate_inference,
)
from segmentum.memory_model import MemoryClass, SourceType, StoreLevel
from segmentum.memory_retrieval import RecallArtifact, RetrievalQuery, compete_candidates
from segmentum.memory_store import MemoryStore


class TestM46Reacceptance(unittest.TestCase):
    def test_g1_retrieval_rebuild_uses_real_retrieve_api(self) -> None:
        store = _retrieval_store()
        raw_contents = [entry.content for entry in store.entries]

        tag_result = store.retrieve(
            RetrievalQuery(
                semantic_tags=["mentor", "promise"],
                context_tags=["lab"],
                content_keywords=["lin", "promise"],
                state_vector=[0.91, 0.22],
                reference_cycle=50,
            ),
            current_mood="reflective",
            k=4,
        )
        tag_candidate_ids = [candidate.entry_id for candidate in tag_result.candidates]
        self.assertEqual(tag_result.candidates[0].entry_id, "tag-primary")
        self.assertEqual(
            set(tag_result.candidates[0].score_breakdown),
            {"tag_overlap", "context_overlap", "mood_match", "accessibility", "recency"},
        )
        self.assertNotIn("dormant", tag_candidate_ids)
        self.assertIsInstance(tag_result.recall_hypothesis, RecallArtifact)
        self.assertEqual(tag_result.recall_hypothesis.primary_entry_id, "tag-primary")
        self.assertTrue(tag_result.recall_hypothesis.auxiliary_entry_ids)
        self.assertIn("primary:tag-primary", tag_result.source_trace)
        self.assertNotIn(tag_result.recall_hypothesis.content, raw_contents)

        context_result = store.retrieve(
            RetrievalQuery(
                semantic_tags=["plan"],
                context_tags=["lab", "weekly", "team"],
                state_vector=[0.88, 0.24],
                reference_cycle=50,
            ),
            current_mood="reflective",
            k=3,
        )
        self.assertEqual(context_result.candidates[0].entry_id, "context-rich")
        self.assertGreater(context_result.candidates[0].score_breakdown["context_overlap"], 0.0)

        mood_result = store.retrieve(
            RetrievalQuery(
                semantic_tags=["promise"],
                context_tags=["storm"],
                reference_cycle=50,
            ),
            current_mood="anxious",
            k=3,
        )
        self.assertEqual(mood_result.candidates[0].entry_id, "negative-mood")
        self.assertGreater(mood_result.candidates[0].score_breakdown["mood_match"], 0.0)

        accessibility_result = store.retrieve(
            RetrievalQuery(
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab"],
                reference_cycle=50,
            ),
            current_mood="reflective",
            k=4,
        )
        accessibility_ids = [candidate.entry_id for candidate in accessibility_result.candidates]
        self.assertNotEqual(accessibility_result.candidates[0].entry_id, "low-access")
        self.assertGreater(accessibility_ids.index("low-access"), 0)

        procedural_result = store.retrieve(
            RetrievalQuery(
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance"],
                reference_cycle=50,
                target_memory_class=MemoryClass.PROCEDURAL,
            ),
            current_mood="calm",
            k=1,
        )
        self.assertEqual(procedural_result.candidates[0].entry_id, "procedure")
        self.assertTrue(procedural_result.recall_hypothesis.procedure_step_outline)
        self.assertNotIn(procedural_result.recall_hypothesis.content, raw_contents)

    def test_g2_candidate_competition_rebuild_uses_real_competition_api(self) -> None:
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
        dominant = compete_candidates(dominant_result.candidates)
        self.assertEqual(dominant.confidence, "high")
        self.assertFalse(dominant.interference_risk)
        self.assertFalse(dominant.competitors)
        self.assertGreater(dominant.dominance_margin, 0.15)

        close_result = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "care"], context_tags=["lab"], reference_cycle=10),
            k=3,
        )
        close = compete_candidates(close_result.candidates, dominance_threshold=0.15)
        self.assertEqual(close.confidence, "low")
        self.assertTrue(close.interference_risk)
        self.assertTrue(close.competitors)
        self.assertTrue(close.competing_interpretations)

    def test_g3_reconstruction_rebuild_uses_real_reconstruction_api(self) -> None:
        store = _reconstruction_store()
        config = ReconstructionConfig(current_cycle=20, current_state=_state())

        base = store.get("base")
        semantic = store.get("semantic")
        low_conf = store.get("low-conf")
        procedural = store.get("procedural")
        assert base is not None
        assert semantic is not None
        assert low_conf is not None
        assert procedural is not None

        base_before = (base.reality_confidence, base.version, base.content_hash)
        result_a = maybe_reconstruct(base, store.entries, store, config)
        self.assertTrue(result_a.triggered)
        self.assertEqual(result_a.trigger_reason, "abstract_short_content")
        self.assertLessEqual(len(result_a.borrowed_source_ids), 2)
        self.assertIs(result_a.entry.source_type, SourceType.RECONSTRUCTION)
        self.assertLess(result_a.entry.reality_confidence, base_before[0])
        self.assertNotEqual(result_a.entry.content_hash, base_before[2])
        self.assertGreater(result_a.entry.version, base_before[1])
        self.assertEqual(result_a.entry.anchor_slots["agents"], "lin")
        self.assertEqual(result_a.entry.anchor_slots["action"], "mentor_checkin")
        self.assertEqual(result_a.entry.anchor_slots["place"], "community_lab")
        self.assertTrue(result_a.reconstruction_trace["borrowed_source_ids"])
        self.assertTrue(result_a.reconstruction_trace["protected_fields"])

        result_b = maybe_reconstruct(semantic, store.entries, store, config)
        self.assertTrue(result_b.triggered)
        self.assertEqual(result_b.trigger_reason, "semantic_abstractness")
        self.assertIs(result_b.entry.source_type, SourceType.RECONSTRUCTION)

        result_c = maybe_reconstruct(low_conf, store.entries, store, config)
        self.assertTrue(result_c.triggered)
        self.assertEqual(result_c.trigger_reason, "low_reality_after_retrieval")
        self.assertIs(result_c.entry.source_type, SourceType.RECONSTRUCTION)
        self.assertLess(result_c.entry.reality_confidence, low_conf.reality_confidence)

        procedural_result = maybe_reconstruct(procedural, store.entries, store, config)
        self.assertTrue(procedural_result.triggered)
        self.assertEqual(procedural_result.entry.procedure_steps, procedural.procedure_steps)
        self.assertIn("night_shift", procedural_result.entry.execution_contexts)
        self.assertNotIn("procedure_steps", procedural_result.reconstructed_fields)

    def test_g4_reconsolidation_rebuild_uses_real_reconsolidation_api(self) -> None:
        reinforce = _entry(entry_id="reinforce", content="stable", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2)
        reinforce_before = (reinforce.accessibility, reinforce.trace_strength, reinforce.retrieval_count, reinforce.abstractness)
        reinforce_report = reconsolidate(reinforce, None, None, current_cycle=30)
        self.assertEqual(reinforce_report.update_type, "reinforcement_only")
        self.assertGreater(reinforce.accessibility, reinforce_before[0])
        self.assertGreater(reinforce.trace_strength, reinforce_before[1])
        self.assertEqual(reinforce.retrieval_count, reinforce_before[2] + 1)
        self.assertGreater(reinforce.abstractness, reinforce_before[3])
        self.assertFalse(reinforce_report.version_changed)

        rebind = _entry(entry_id="rebind", content="rebind", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2, mood_context="reflective")
        rebind_report = reconsolidate(rebind, "anxious", ["storm"], current_cycle=30)
        self.assertEqual(rebind_report.update_type, "contextual_rebinding")
        self.assertEqual(rebind.mood_context, "anxious")
        self.assertIn("storm", rebind.context_tags)

        reconstruction_store = MemoryStore(
            entries=[
                _entry(entry_id="reconstruct", content="thin", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.8, retrieval_count=2),
                _entry(entry_id="donor", content="donor", semantic_tags=["mentor", "care"], context_tags=["lab"], anchor_slots={"time": None, "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"}),
            ]
        )
        reconstruct = reconstruction_store.get("reconstruct")
        assert reconstruct is not None
        reconstruct_before = reconstruct.version
        reconstruct_report = reconsolidate(
            reconstruct,
            "reflective",
            ["lab"],
            store=reconstruction_store,
            current_cycle=30,
            current_state=_state(),
        )
        self.assertEqual(reconstruct_report.update_type, "structural_reconstruction")
        self.assertTrue(reconstruct_report.fields_reconstructed)
        self.assertTrue(reconstruct_report.version_changed)
        self.assertGreater(reconstruct.version, reconstruct_before)

        conflict_store = _retrieval_store()
        conflict_artifact = conflict_store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=50),
            current_mood="reflective",
            k=3,
        ).recall_hypothesis
        self.assertIsNotNone(conflict_artifact)

        factual = _entry(entry_id="factual", content="conflict", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2)
        factual_report = reconsolidate(
            factual,
            "reflective",
            ["lab"],
            current_cycle=30,
            recall_artifact=conflict_artifact,
            conflict_type=ConflictType.FACTUAL,
        )
        self.assertEqual(factual_report.update_type, "conflict_marking")
        self.assertIn("factual", factual_report.conflict_flags)
        self.assertGreater(factual.counterevidence_count, 0)
        self.assertLess(factual_report.confidence_delta["reality_confidence"], 0.0)

        source = _entry(entry_id="source", content="conflict", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2)
        source_report = reconsolidate(
            source,
            "reflective",
            ["lab"],
            current_cycle=30,
            recall_artifact=conflict_artifact,
            conflict_type=ConflictType.SOURCE,
        )
        self.assertIn("source", source_report.conflict_flags)
        self.assertLess(source_report.confidence_delta["source_confidence"], 0.0)

        interpretive = _entry(entry_id="interpretive", content="conflict", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2)
        interpretive_report = reconsolidate(
            interpretive,
            "reflective",
            ["lab"],
            current_cycle=30,
            recall_artifact=conflict_artifact,
            conflict_type=ConflictType.INTERPRETIVE,
        )
        self.assertIn("interpretive", interpretive_report.conflict_flags)
        self.assertEqual(interpretive_report.confidence_delta["source_confidence"], 0.0)
        self.assertEqual(interpretive_report.confidence_delta["reality_confidence"], 0.0)

        procedural_store = MemoryStore(
            entries=[
                _entry(
                    entry_id="proc-recall",
                    content="Reactor procedure summary.",
                    semantic_tags=["reactor", "procedure"],
                    context_tags=["maintenance"],
                    memory_class=MemoryClass.PROCEDURAL,
                    abstractness=0.8,
                    retrieval_count=2,
                    execution_contexts=["reactor_room"],
                ),
                _entry(
                    entry_id="proc-donor",
                    content="Secondary reactor procedure support.",
                    semantic_tags=["reactor", "procedure"],
                    context_tags=["maintenance", "night"],
                    memory_class=MemoryClass.PROCEDURAL,
                    execution_contexts=["night_shift"],
                ),
            ]
        )
        procedural = procedural_store.get("proc-recall")
        assert procedural is not None
        procedural_steps_before = list(procedural.procedure_steps)
        procedural_report = reconsolidate(
            procedural,
            "calm",
            ["maintenance"],
            store=procedural_store,
            current_cycle=30,
            current_state=_state(),
        )
        self.assertEqual(procedural.procedure_steps, procedural_steps_before)
        self.assertNotIn("procedure_steps", procedural_report.fields_reconstructed)
        self.assertIn("night_shift", procedural.execution_contexts)

    def test_g5_offline_consolidation_rebuild_uses_real_consolidation_api(self) -> None:
        store = MemoryStore(entries=_build_consolidation_entries())
        before_entries = {entry.id: deepcopy(entry.to_dict()) for entry in store.entries if entry.id.startswith("ep-")}

        report = store.run_consolidation_cycle(current_cycle=40, rng=random.Random(0), current_state=_state())

        self.assertTrue(report.upgrade.promoted_ids)
        self.assertTrue(report.extracted_patterns)
        self.assertTrue(report.replay_reencoded_ids)
        self.assertTrue(report.cleanup.deleted_ids or report.cleanup.dormant_ids or report.cleanup.absorbed_ids)
        self.assertIn("cleanup-short", report.cleanup.deleted_ids)
        self.assertEqual(
            set(report.to_dict()),
            {"upgrade", "extracted_patterns", "replay_reencoded_ids", "validated_inference_ids", "cleanup"},
        )

        extracted_entries = [store.get(entry_id) for entry_id in report.extracted_patterns]
        semantic_entry = next(entry for entry in extracted_entries if entry is not None and entry.memory_class is MemoryClass.SEMANTIC)
        inferred_entry = next(entry for entry in extracted_entries if entry is not None and entry.memory_class is MemoryClass.INFERRED)
        self.assertIsNotNone(semantic_entry)
        self.assertIsNotNone(inferred_entry)
        semantic_metadata = dict(semantic_entry.compression_metadata or {})
        self.assertTrue(semantic_metadata.get("support_entry_ids"))
        self.assertTrue(semantic_metadata.get("stable_structure"))
        self.assertTrue(semantic_metadata.get("lineage_type"))
        support_ids = semantic_metadata["support_entry_ids"]
        self.assertTrue(all(store.get(entry_id) is not None for entry_id in support_ids))
        self.assertGreater(
            semantic_entry.abstractness,
            max(before_entries[entry_id]["abstractness"] for entry_id in before_entries),
        )
        self.assertGreaterEqual(
            len([entry_id for entry_id in before_entries if store.get(entry_id) is not None]),
            len(before_entries),
        )

    def test_g6_inference_gate_rebuild_uses_real_validation_api(self) -> None:
        validated = _entry(
            entry_id="validated",
            content="validated pattern",
            semantic_tags=["mentor", "pattern"],
            context_tags=["lab", "weekly", "community"],
            memory_class=MemoryClass.INFERRED,
            store_level=StoreLevel.MID,
            source_type=SourceType.INFERENCE,
            support_count=5,
            retrieval_count=4,
            reality_confidence=0.4,
            compression_metadata={"predictive_gain": 0.8, "cross_context_consistency": 0.9},
        )
        validated_result = validate_inference(validated)
        self.assertTrue(validated_result.passed)
        self.assertEqual(validated_result.validation_status, "validated")
        self.assertGreaterEqual(validated_result.score, validated_result.threshold)
        self.assertIs(validated.store_level, StoreLevel.LONG)

        unvalidated = _entry(
            entry_id="unvalidated",
            content="weak hypothesis",
            semantic_tags=["mentor", "pattern"],
            context_tags=["lab"],
            memory_class=MemoryClass.INFERRED,
            store_level=StoreLevel.MID,
            source_type=SourceType.INFERENCE,
            support_count=1,
            retrieval_count=0,
            counterevidence_count=2,
            compression_metadata={"predictive_gain": 0.1, "cross_context_consistency": 0.2},
        )
        unvalidated_result = validate_inference(unvalidated)
        self.assertFalse(unvalidated_result.passed)
        self.assertLess(unvalidated_result.score, unvalidated_result.threshold)
        self.assertIn(unvalidated_result.validation_status, {"unvalidated", "contradicted"})
        self.assertIs(unvalidated.store_level, StoreLevel.MID)

        store = MemoryStore(
            entries=[
                validated,
                unvalidated,
                _entry(entry_id="base", content="base", semantic_tags=["mentor"], context_tags=["lab"]),
            ]
        )
        retrieval = store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=10),
            k=3,
        )
        candidate_ids = [candidate.entry_id for candidate in retrieval.candidates]
        self.assertIn("unvalidated", candidate_ids)
        self.assertNotIn("unvalidated", retrieval.recall_hypothesis.auxiliary_entry_ids)

    def test_g7_legacy_bridge_rebuild_uses_real_bridge_apis(self) -> None:
        bridge_store = MemoryStore(entries=_build_consolidation_entries()[:5])
        memory = LongTermMemory()
        memory.episodes = bridge_store.to_legacy_episodes()
        memory.ensure_memory_store()

        replay_batch = memory.replay_during_sleep(rng=random.Random(0), limit=2)
        self.assertGreaterEqual(len(replay_batch), 1)
        self.assertTrue(hasattr(memory.memory_store, "run_consolidation_cycle"))

        report = memory.run_memory_consolidation_cycle(
            current_cycle=60,
            rng=random.Random(0),
            current_state=_state(),
        )
        self.assertTrue(report.to_dict())
        self.assertEqual(len(memory.episodes), len(memory.memory_store.entries))

    def test_g8_report_honesty_and_summary_reflect_independent_rebuild(self) -> None:
        report = build_m46_reacceptance_report()
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertGreaterEqual(report["gate_summaries"][GATE_RETRIEVAL]["counts"]["total"], 5)
        self.assertEqual(report["gate_summaries"][GATE_LEGACY]["status"], STATUS_NOT_RUN)
        self.assertGreaterEqual(report["gate_summaries"]["offline_consolidation_pipeline"]["counts"]["total"], 3)
        self.assertGreaterEqual(report["gate_summaries"]["inference_validation_gate"]["counts"]["total"], 3)
        regression_record = next(
            record
            for record in report["evidence_records"]
            if record["scenario_id"] == "legacy_regression_prereq"
        )
        self.assertEqual(regression_record["status"], STATUS_NOT_RUN)

        for record in report["evidence_records"]:
            self.assertTrue(
                {"gate", "scenario_id", "api", "input_summary", "observed", "criteria_checks", "status", "notes"}.issubset(record)
            )
            self.assertTrue(record["observed"])
            self.assertIn(record["status"], {"PASS", "FAIL", "NOT_RUN"})

        raw_records = [
            record for record in deepcopy(report["evidence_records"]) if record["gate"] != GATE_HONESTY
        ]
        tampered = next(record for record in raw_records if record["scenario_id"] == "legacy_regression_prereq")
        tampered["status"] = "PASS"
        honesty_record = _build_honesty_record(raw_records, include_regressions=False)
        self.assertEqual(honesty_record["status"], "FAIL")
        self.assertIn("legacy_regression_prereq", honesty_record["observed"]["mismatched_status_records"])

        missing_integration = [
            record for record in deepcopy(raw_records) if record["scenario_id"] != "consolidation_validation_linkage"
        ]
        honesty_record = _build_honesty_record(missing_integration, include_regressions=False)
        self.assertEqual(honesty_record["status"], "FAIL")
        self.assertIn(
            "consolidation_validation_linkage",
            honesty_record["observed"]["missing_integration_scenarios"],
        )

    def test_g8_report_honesty_fail_closes_fake_regression_pass_with_regressions_enabled(self) -> None:
        realistic_regression_summary = {
            "executed": True,
            "command": [sys.executable, "-m", "pytest", "tests/test_m41_acceptance.py", "-q"],
            "files": ["tests/test_m41_acceptance.py"],
            "returncode": 0,
            "passed": True,
            "duration_seconds": 1.25,
            "stdout_tail": [
                "============================= test session starts =============================",
                "1 passed in 1.25s (0:00:01)",
            ],
            "summary_line": "1 passed in 1.25s (0:00:01)",
        }
        with patch("segmentum.m46_reacceptance.REGRESSION_TARGETS", ["tests/test_m41_acceptance.py"]):
            with patch("segmentum.m46_reacceptance._run_regressions", return_value=realistic_regression_summary):
                report = build_m46_reacceptance_report(include_regressions=True)

            raw_records = [
                record for record in deepcopy(report["evidence_records"]) if record["gate"] != GATE_HONESTY
            ]
            tampered = next(record for record in raw_records if record["scenario_id"] == "legacy_regression_prereq")
            tampered["status"] = "PASS"
            tampered["observed"] = {
                "executed": True,
                "files": ["tests/test_m41_acceptance.py"],
                "returncode": 0,
                "passed": True,
                "summary_line": "synthetic pass",
                "stdout_tail": ["synthetic pass"],
            }

            honesty_record = _build_honesty_record(raw_records, include_regressions=True)

        self.assertEqual(honesty_record["status"], "FAIL")
        self.assertNotIn("legacy_regression_prereq", honesty_record["observed"]["mismatched_status_records"])
        self.assertEqual(
            set(honesty_record["observed"]["external_check_failures"]["legacy_regression_prereq"]),
            {
                "regression_command_matches_expected",
                "regression_duration_recorded",
                "regression_summary_looks_like_pytest_output",
            },
        )

    def test_artifact_writer_emits_g1_to_g8_human_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m46_reacceptance_artifacts(reports_dir=tmpdir)
            evidence = json.loads(Path(outputs["evidence"]).read_text(encoding="utf-8"))
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(evidence["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertIn("M4.6 Reacceptance Summary", summary)
        self.assertIn("G1 `retrieval_multi_cue`: `PASS`", summary)
        self.assertIn("G7 `legacy_integration`: `NOT_RUN`", summary)
        self.assertIn("G8 `report_honesty`: `PASS`", summary)
        self.assertIn("not a formal acceptance pass", summary)


if __name__ == "__main__":
    unittest.main()
