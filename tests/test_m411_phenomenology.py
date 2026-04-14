from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pytest

from segmentum.m4_acceptance import final_conclusion, memory_milestone_layer_status
from segmentum.m411_phenomenology import (
    M411RolloutConfig,
    build_m411_acceptance_report,
    evaluate_m411_phenomenology,
    run_m411_rollout_pair,
    write_m411_acceptance_artifacts,
)


SMOKE_CONFIG = M411RolloutConfig(
    seed=411,
    ticks=200,
    recall_probe_interval=50,
    perturbation_tick=100,
    sleep_interval=50,
    min_acceptance_ticks=200,
)

pytestmark = pytest.mark.m411


class TestM411Phenomenology(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pair = run_m411_rollout_pair(SMOKE_CONFIG)
        cls.evaluation = evaluate_m411_phenomenology(cls.pair, SMOKE_CONFIG)

    def test_rollout_is_free_and_negative_control_is_paired(self) -> None:
        default = self.pair["default"]
        control = self.pair["negative_control"]

        self.assertEqual(default["curated_corpus_paths_read"], [])
        self.assertEqual(default["curated_corpus_prohibited_path"], "data/m47_corpus.json")
        self.assertEqual(default["seed"], control["seed"])
        self.assertTrue(control["negative_control_interventions"])
        self.assertTrue(
            all(
                intervention["mutates_existing_memory"] is False
                for intervention in control["negative_control_interventions"]
            )
        )
        self.assertTrue(default["encoded_events"])
        self.assertTrue(default["replay_events"])
        self.assertTrue(
            all(event["source"] == "live_sleep_consolidation" for event in default["replay_events"])
        )

    def test_smoke_rollout_is_pipeline_only_and_cannot_accept(self) -> None:
        report, _ = build_m411_acceptance_report(config=SMOKE_CONFIG, pair=self.pair)

        self.assertEqual(report["status"], "NOT_ISSUED")
        self.assertEqual(report["formal_acceptance_conclusion"], "PARTIAL_ACCEPT")
        self.assertFalse(report["phenomenological_pass"])
        self.assertFalse(report["official_acceptance_config"])
        self.assertIn("long_horizon_free_rollout", self.evaluation["failed_gates"])
        self.assertTrue(
            self.evaluation["gate_summaries"]["long_horizon_free_rollout"][
                "budget_competition_observed"
            ]
        )
        self.assertEqual(
            self.evaluation["gate_summaries"]["long_horizon_free_rollout"][
                "budget_evidence_source"
            ],
            "encoded_event_metadata",
        )

    def test_tick_level_budget_competition_is_logged(self) -> None:
        events = list(self.pair["default"]["budget_events"])
        self.assertTrue(events)
        competition_events = [
            event
            for event in events
            if any(float(item["attention_budget_denied"]) > 0.0 for item in event["candidates"])
            and any(float(item["attention_budget_granted"]) > 0.0 for item in event["candidates"])
        ]
        self.assertTrue(competition_events)
        first = competition_events[0]
        self.assertIn("winner_choice", first)
        self.assertIn("granted_requested_histogram", first)
        histogram = self.pair["default"]["budget_ratio_histogram"]
        self.assertEqual(sum(histogram.values()), len(self.pair["default"]["encoded_events"]))

    def test_200_tick_smoke_keeps_four_effects_and_control_collapse(self) -> None:
        summaries = self.evaluation["gate_summaries"]

        self.assertTrue(summaries["serial_position_effect"]["default"]["passed"])
        self.assertTrue(summaries["retention_curve_fit"]["default"]["passed"])
        self.assertTrue(summaries["schema_intrusion"]["default"]["passed"])
        self.assertTrue(summaries["identity_continuity"]["default"]["matched_baseline_sufficient"])
        self.assertTrue(summaries["negative_controls"]["comparisons"]["serial_position_collapse"])
        self.assertTrue(summaries["negative_controls"]["comparisons"]["retention_curve_collapse"])
        self.assertTrue(summaries["negative_controls"]["comparisons"]["schema_intrusion_collapse"])
        self.assertTrue(summaries["negative_controls"]["comparisons"]["identity_continuity_collapse"])

    def test_tampered_negative_control_fails(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["negative_control"] = deepcopy(tampered["default"])
        tampered["negative_control"]["negative_control_interventions"] = [
            {"tick": 1, "mode": "tampered_same_as_default"}
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        self.assertIn("negative_controls", evaluation["failed_gates"])
        self.assertEqual(evaluation["gate_summaries"]["honesty_safety_net"]["status"], "FAIL")

    def test_destructive_negative_control_fails_honesty_audit(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["negative_control"]["negative_control_interventions"] = [
            {
                "tick": 1,
                "mode": "salience_zeroed",
                "touched_count": 4,
                "mutates_existing_memory": True,
            }
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        self.assertIn("negative_controls", evaluation["failed_gates"])
        self.assertFalse(
            evaluation["gate_summaries"]["honesty_safety_net"]["checks"][
                "negative_control_not_destructive_memory_mutation"
            ]
        )

    def test_out_of_band_replay_fails_closed(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["default"]["replay_events"] = [
            {"tick": 999, "source": "out_of_band_constrained_replay", "replay_reencoded_ids": ["fake"]}
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        self.assertIn("long_horizon_free_rollout", evaluation["failed_gates"])
        self.assertEqual(evaluation["gate_summaries"]["honesty_safety_net"]["status"], "FAIL")

    def test_schema_intrusion_rejects_keyword_only_evidence(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["default"]["final_entries"] = [
            {
                "entry_id": "episodic-support",
                "memory_class": "episodic",
                "created_at": 1,
                "salience": 0.8,
                "accessibility": 0.8,
                "trace_strength": 0.8,
                "relevance_self": 0.0,
                "encoding_strength": 0.8,
                "semantic_tags": ["shelter"],
                "state_vector": [0.0, 1.0, 0.0],
            },
            {
                "entry_id": "keyword-schema",
                "memory_class": "semantic",
                "support_ids": ["episodic-support"],
                "semantic_tags": ["shelter"],
                "has_centroid": False,
                "keyword_match_only": True,
                "schema_intrusion_evidence_mode": "keyword_only",
            },
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        schema = evaluation["gate_summaries"]["schema_intrusion"]["default"]
        self.assertEqual(schema["status"], "FAIL")
        self.assertEqual(schema["keyword_only_cluster_count"], 1)
        self.assertIn("schema_intrusion", evaluation["failed_gates"])

    def test_default_schema_degeneracy_does_not_pass_intrusion_gate(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["default"]["final_entries"] = [
            entry for entry in tampered["default"]["final_entries"]
            if entry["memory_class"] == "episodic"
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        schema = evaluation["gate_summaries"]["schema_intrusion"]
        self.assertEqual(schema["default"]["status"], "FAIL")
        self.assertTrue(schema["default"]["degenerate_cluster_formation"])
        self.assertFalse(schema["default_intrusion_present"])
        self.assertIn("schema_intrusion", evaluation["failed_gates"])

    def test_identity_tag_without_structured_self_relevance_is_not_self_related(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["default"]["encoded_events"] = []
        tampered["default"]["final_entries"] = [
            {
                "entry_id": "semantic-tag-only",
                "memory_class": "episodic",
                "created_at": 1,
                "salience": 0.8,
                "accessibility": 0.8,
                "trace_strength": 0.8,
                "relevance_self": 0.0,
                "encoding_strength": 0.8,
                "semantic_tags": ["identity"],
            },
            {
                "entry_id": "matched-baseline",
                "memory_class": "episodic",
                "created_at": 2,
                "salience": 0.8,
                "accessibility": 0.8,
                "trace_strength": 0.8,
                "relevance_self": 0.0,
                "encoding_strength": 0.8,
                "semantic_tags": [],
            },
        ]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        identity = evaluation["gate_summaries"]["identity_continuity"]["default"]
        self.assertEqual(identity["self_related_count"], 0)
        self.assertEqual(identity["self_related_source"], "relevance_self_structured_field")
        self.assertEqual(identity["status"], "FAIL")

    def test_identity_matching_uses_structured_arousal_novelty_and_created_at(self) -> None:
        tampered = deepcopy(self.pair)
        tampered["default"]["encoded_events"] = [
            {
                "entry_id": "self-1",
                "memory_class": "episodic",
                "created_at": 10,
                "tick": 10,
                "salience": 0.8,
                "accessibility": 0.8,
                "trace_strength": 0.8,
                "relevance_self": 0.8,
                "encoding_strength": 0.7,
                "arousal": 0.42,
                "novelty": 0.31,
            },
            {
                "entry_id": "baseline-1",
                "memory_class": "episodic",
                "created_at": 12,
                "tick": 12,
                "salience": 0.6,
                "accessibility": 0.4,
                "trace_strength": 0.4,
                "relevance_self": 0.0,
                "encoding_strength": 0.2,
                "arousal": 0.43,
                "novelty": 0.33,
            },
        ]
        tampered["default"]["final_entries"] = deepcopy(tampered["default"]["encoded_events"])
        tampered["default"]["perturbation_tick"] = 20
        tampered["default"]["recall_events"] = [{"candidate_ids": ["self-1"], "tick": 20}]

        evaluation = evaluate_m411_phenomenology(tampered, SMOKE_CONFIG)

        identity = evaluation["gate_summaries"]["identity_continuity"]["default"]
        self.assertEqual(identity["baseline_count"], 1)
        self.assertTrue(identity["matched_baseline_sufficient"])
        self.assertEqual(
            identity["baseline_match_fields"],
            ["arousal", "novelty", "created_at", "encoding_strength_tiebreak"],
        )

    def test_report_uses_three_layer_acceptance_without_smoke_acceptance(self) -> None:
        report, evidence = build_m411_acceptance_report(config=SMOKE_CONFIG, pair=self.pair)

        self.assertEqual(report["formal_acceptance_conclusion"], "PARTIAL_ACCEPT")
        self.assertTrue(report["structural_pass"])
        self.assertTrue(report["behavioral_pass"])
        self.assertFalse(report["phenomenological_pass"])
        self.assertFalse(report["three_layer_accept_ready"])
        self.assertEqual(report["honesty_audit_role"], "upper_safety_net_not_primary_grader")
        self.assertIn("evaluation", evidence)
        self.assertEqual(
            evidence["evaluation"]["default_metrics"]["schema_intrusion"]["identification_criterion"],
            "representational_centroid_not_keyword",
        )

    def test_three_layer_taxonomy_and_docs_are_declared(self) -> None:
        partial = final_conclusion(
            structural_pass=True,
            behavioral_pass=True,
            phenomenological_pass=False,
        )
        self.assertEqual(partial.formal_acceptance_conclusion, "PARTIAL_ACCEPT")
        self.assertEqual(partial.missing_layers, ("phenomenological_pass",))

        status_table = memory_milestone_layer_status()
        self.assertIn("Layer (a) only", status_table["M4.5"]["status_note"])
        self.assertEqual(status_table["M4.8"]["behavioral_pass"], True)
        self.assertEqual(status_table["M4.10"]["phenomenological_pass"], "pending(M4.11)")
        self.assertIn("Layer (c)", status_table["M4.11"]["status_note"])

        readme = Path("README.md").read_text(encoding="utf-8")
        boundaries = Path("reports/m4_milestone_boundaries.md").read_text(encoding="utf-8")

        self.assertIn("`M4.11`: natural-rollout phenomenology supplies layer (c)", readme)
        self.assertIn("honesty audit remains a safety net", readme)
        self.assertIn("M4.5-M4.7 currently satisfy only layer (a)", boundaries)
        self.assertIn("M4.11 targets layer (c)", boundaries)
        self.assertIn("honesty / fail-closed audit remains a safety net", boundaries)

    def test_writer_emits_all_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            outputs = write_m411_acceptance_artifacts(
                output_root=tmp_dir,
                config=SMOKE_CONFIG,
            )
            self.assertTrue(Path(outputs["default_rollout"]).exists())
            self.assertTrue(Path(outputs["negative_control_rollout"]).exists())
            self.assertTrue(Path(outputs["evidence"]).exists())
            report = json.loads(Path(outputs["report"]).read_text(encoding="utf-8"))
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(report["formal_acceptance_conclusion"], "PARTIAL_ACCEPT")
        self.assertIn("M4.11 Acceptance Summary", summary)
        self.assertIn("phenomenological_pass=False", summary)


if __name__ == "__main__":
    unittest.main()
