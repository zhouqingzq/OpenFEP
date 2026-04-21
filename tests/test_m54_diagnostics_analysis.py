from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_m54_diagnostics import build_diagnosis
from segmentum.dialogue.validation.pipeline import ValidationReport
from segmentum.dialogue.validation.report import generate_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


class TestM54DiagnosticsAnalysis(unittest.TestCase):
    def test_artifact_sanity_and_baseline_c_win_slicing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "aggregate_report.json",
                {
                    "comparisons": {
                        "semantic_similarity": {
                            "baseline_c_mean": 0.55,
                        }
                    },
                    "behavioral_majority_baseline_gate": {
                        "majority_behavioral_strategy_mean": 0.90,
                        "personality_behavioral_strategy_mean": 0.20,
                    },
                    "classifier_gate": {"cue_override_rate": 0.10},
                    "reports": [
                        {
                            "per_strategy": {
                                "random": {
                                    "skipped": False,
                                    "state_calibration_summary": {
                                        "strategy_counts": {"escape": 9, "explore": 1},
                                        "policy_dominant_strategy": "escape",
                                    },
                                }
                            }
                        }
                    ],
                },
            )
            _write_json(
                root / "ablation_summary.json",
                {
                    "no_surface_profile": {
                        "semantic_mean": 0.45,
                        "semantic_vs_baseline_c_diff_mean": -0.10,
                    },
                    "no_policy_trait_bias": {
                        "semantic_mean": 0.50,
                        "semantic_vs_baseline_c_diff_mean": -0.05,
                    },
                    "surface_only_default_agent": {
                        "semantic_mean": 0.52,
                        "semantic_vs_baseline_c_diff_mean": -0.03,
                    },
                },
            )
            _write_json(
                root / "baseline_c_behavioral_failure_audit.json",
                {
                    "rows": [
                        {
                            "majority_strategy": "escape",
                            "real_strategy_distribution": {"escape": 10},
                        }
                    ]
                },
            )
            _write_jsonl(
                root / "diagnostic_trace.jsonl",
                [
                    {
                        "user_uid": 1,
                        "strategy": "random",
                        "pair_index": 0,
                        "personality_semantic_pair_score": 0.20,
                        "baseline_c_semantic_pair_score": 0.70,
                        "personality_vs_c_pair_delta": -0.50,
                        "baseline_c_action": "minimal_response",
                        "baseline_c_strategy": "escape",
                        "baseline_c_template_id": "minimal_response:0",
                        "baseline_c_rhetorical_move": "guarded_short",
                        "baseline_c_surface_source": "population_average",
                        "baseline_c_profile_degraded_reason": "anchor_mismatch",
                        "baseline_c_generated_chars": 2,
                        "real_strategy": "escape",
                        "personality_strategy": "exploit",
                    },
                    {
                        "user_uid": 1,
                        "strategy": "random",
                        "pair_index": 1,
                        "personality_semantic_pair_score": 0.80,
                        "baseline_c_semantic_pair_score": 0.30,
                        "personality_vs_c_pair_delta": 0.50,
                        "baseline_c_action": "disengage",
                        "baseline_c_strategy": "escape",
                        "baseline_c_generated_chars": 4,
                        "real_strategy": "escape",
                        "personality_strategy": "exploit",
                    },
                ],
            )
            diagnosis = build_diagnosis(root)
            self.assertTrue(diagnosis["artifact_sanity"]["passed"])
            baseline = diagnosis["baseline_c_diagnosis"]
            self.assertEqual(baseline["baseline_c_win_turns"], 1)
            self.assertEqual(
                baseline["win_slice"]["baseline_c_action"],
                {"minimal_response": 1},
            )
            self.assertEqual(
                diagnosis["behavior_diagnosis"]["diagnosis_tags"],
                ["classifier_definition_issue", "state_modeling_issue"],
            )

    def test_stale_ablation_summary_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "aggregate_report.json",
                {"comparisons": {"semantic_similarity": {"baseline_c_mean": 0.55}}},
            )
            _write_json(
                root / "ablation_summary.json",
                {
                    "no_surface_profile": {
                        "semantic_mean": 0.45,
                        "semantic_vs_baseline_c_diff_mean": 0.10,
                    }
                },
            )
            _write_json(root / "baseline_c_behavioral_failure_audit.json", {"rows": []})
            _write_jsonl(root / "diagnostic_trace.jsonl", [{"personality_vs_c_pair_delta": 0.1}])
            diagnosis = build_diagnosis(root)
            self.assertFalse(diagnosis["artifact_sanity"]["passed"])
            self.assertTrue(diagnosis["artifact_sanity"]["stale_ablation_baseline_c_warning"])

    def test_ablation_trace_lifts_and_close_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "aggregate_report.json",
                {"comparisons": {"semantic_similarity": {"baseline_c_mean": 0.50}}},
            )
            _write_json(
                root / "ablation_summary.json",
                {
                    name: {
                        "semantic_mean": 0.40,
                        "semantic_vs_baseline_c_diff_mean": -0.10,
                    }
                    for name in (
                        "no_surface_profile",
                        "no_policy_trait_bias",
                        "surface_only_default_agent",
                    )
                },
            )
            _write_json(root / "baseline_c_behavioral_failure_audit.json", {"rows": []})
            _write_jsonl(root / "diagnostic_trace.jsonl", [{"personality_vs_c_pair_delta": 0.1}])
            _write_jsonl(
                root / "ablation_trace.jsonl",
                [
                    {
                        "user_uid": 1,
                        "strategy": "random",
                        "pair_index": 0,
                        "full_personality_semantic_pair_score": 0.60,
                        "no_policy_trait_bias_semantic_pair_score": 0.59,
                        "no_surface_profile_semantic_pair_score": 0.45,
                        "surface_only_default_agent_semantic_pair_score": 0.58,
                        "baseline_c_semantic_pair_score": 0.50,
                    }
                ],
            )
            diagnosis = build_diagnosis(root)
            ablation = diagnosis["ablation_diagnosis"]
            self.assertTrue(ablation["trace_available"])
            self.assertAlmostEqual(ablation["lift_means"]["trait_policy_lift_full_minus_no_policy"], 0.01)
            self.assertEqual(
                ablation["close_or_beating_full_cases"]["surface_only_default_agent"][0]["ablation_minus_full"],
                -0.02,
            )


class TestM54TraceSchema(unittest.TestCase):
    def test_report_writes_real_labels_and_ablation_trace(self) -> None:
        strategy = {
            "skipped": False,
            "eligible_for_hard_gate": True,
            "split_metadata": {"strategy": "random"},
            "personality_metrics": {
                "semantic_similarity": 0.8,
                "behavioral_similarity_strategy": 0.8,
                "agent_state_similarity": 0.9,
            },
            "baseline_a_metrics": {"semantic_similarity": 0.3, "behavioral_similarity_strategy": 0.3},
            "baseline_b_metrics": {"semantic_similarity": 0.3, "behavioral_similarity_strategy": 0.3},
            "baseline_c_metrics": {"semantic_similarity": 0.4, "behavioral_similarity_strategy": 0.4},
            "personality_metric_details": {"semantic_similarity": {"method": "sentence_embedding_cosine"}},
            "classifier_validation": {
                "passed_3class_gate": False,
                "formal_gate_eligible": False,
                "classifier_evidence_tier": "repo_fixture_smoke",
            },
            "baseline_c_builder": "population_average_full_implant",
            "baseline_c_leave_one_out": True,
            "baseline_c_population_excluded_uid": 42,
            "baseline_c_population_user_count": 1,
            "majority_baseline_metrics": {
                "majority_strategy": "exploit",
                "behavioral_similarity_strategy": 0.2,
                "real_strategy_distribution": {"exploit": 1},
            },
            "state_distance_diagnostics": {
                "train_full": {"l2": 0.1},
                "train_default": {"l2": 0.5},
            },
            "diagnostic_trace": [
                {
                    "pair_index": 0,
                    "real_text": "real",
                    "real_action": "elaborate",
                    "real_strategy": "exploit",
                    "personality_action": "elaborate",
                    "personality_strategy": "exploit",
                    "baseline_c_action": "minimal_response",
                    "baseline_c_strategy": "escape",
                }
            ],
            "ablation_summary": [
                {
                    "name": "no_surface_profile",
                    "pair_count": 1,
                    "semantic_mean": 0.3,
                    "semantic_vs_baseline_c_diff": -0.1,
                },
                {
                    "name": "no_policy_trait_bias",
                    "pair_count": 1,
                    "semantic_mean": 0.3,
                    "semantic_vs_baseline_c_diff": -0.1,
                },
                {
                    "name": "surface_only_default_agent",
                    "pair_count": 1,
                    "semantic_mean": 0.3,
                    "semantic_vs_baseline_c_diff": -0.1,
                },
            ],
            "ablation_trace": [
                {
                    "pair_index": 0,
                    "real_action": "elaborate",
                    "real_strategy": "exploit",
                    "full_personality_semantic_pair_score": 0.8,
                    "no_policy_trait_bias_semantic_pair_score": 0.7,
                    "no_surface_profile_semantic_pair_score": 0.6,
                    "surface_only_default_agent_semantic_pair_score": 0.5,
                    "baseline_c_semantic_pair_score": 0.4,
                }
            ],
        }
        report = ValidationReport(
            user_uid=42,
            per_strategy={"random": strategy},
            aggregate={"formal_requested": False, "required_users": 1},
            conclusion="completed",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_report([report], root)
            diagnostic_rows = [json.loads(line) for line in (root / "diagnostic_trace.jsonl").read_text().splitlines()]
            ablation_rows = [json.loads(line) for line in (root / "ablation_trace.jsonl").read_text().splitlines()]
            aggregate = json.loads((root / "aggregate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostic_rows[0]["real_action"], "elaborate")
            self.assertEqual(diagnostic_rows[0]["real_strategy"], "exploit")
            self.assertEqual(ablation_rows[0]["real_strategy"], "exploit")
            self.assertEqual(aggregate["ablation_trace_rows"], 1)
            self.assertTrue(aggregate["ablation_trace_jsonl"].endswith("ablation_trace.jsonl"))


if __name__ == "__main__":
    unittest.main()
