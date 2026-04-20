"""Aggregate report hard_pass rules (M5.4)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from segmentum.dialogue.validation.constants import M54_ACCEPTANCE_RULES_VERSION
from segmentum.dialogue.validation.pipeline import ValidationReport
from segmentum.dialogue.validation.report import generate_report


def _bundle(
    *,
    sem: float,
    beh: float,
    ast: float,
    base_sem: float,
    base_beh: float,
    base_sem_c: float | None = None,
    base_beh_c: float | None = None,
    classifier_pass: bool = True,
    topic_not_applicable: bool = False,
) -> dict[str, dict]:
    """Baseline dicts omit agent_state_like the real pipeline (train-only vs full agent only)."""
    keys = (
        "semantic_similarity",
        "behavioral_similarity_strategy",
        "behavioral_similarity_action11",
        "stylistic_similarity",
        "personality_similarity",
        "agent_state_similarity",
    )
    c_sem = float(base_sem_c) if base_sem_c is not None else base_sem
    c_beh = float(base_beh_c) if base_beh_c is not None else base_beh
    p = {
        "semantic_similarity": sem,
        "behavioral_similarity_strategy": beh,
        "behavioral_similarity_action11": 0.5,
        "stylistic_similarity": 0.4,
        "personality_similarity": 0.7,
        "agent_state_similarity": ast,
    }
    a = {
        "semantic_similarity": base_sem,
        "behavioral_similarity_strategy": base_beh,
        "behavioral_similarity_action11": 0.4,
        "stylistic_similarity": 0.35,
        "personality_similarity": 0.65,
    }
    b = dict(a)
    c = {
        "semantic_similarity": c_sem,
        "behavioral_similarity_strategy": c_beh,
        "behavioral_similarity_action11": 0.4,
        "stylistic_similarity": 0.35,
        "personality_similarity": 0.65,
    }
    assert set(p.keys()) == set(keys)
    return {
        "skipped": bool(topic_not_applicable),
        "reason": "topic_split_not_applicable" if topic_not_applicable else None,
        "eligible_for_hard_gate": not bool(topic_not_applicable),
        "split_metadata": {"topic_split_not_applicable": bool(topic_not_applicable)},
        "personality_metrics": p,
        "baseline_a_metrics": a,
        "baseline_b_metrics": b,
        "baseline_c_metrics": c,
        "personality_metric_details": {
            "semantic_similarity": {"method": "sentence_embedding_cosine"},
        },
        "classifier_validation": {
            "passed_3class_gate": classifier_pass,
            "formal_gate_eligible": classifier_pass,
            "classifier_evidence_tier": (
                "external_human_labeled" if classifier_pass else "repo_fixture_smoke"
            ),
            "dataset_origin": "external_human_labeled_unit_tests",
            "classifier_provenance_ok": classifier_pass,
            "cue_override_gate_passed": classifier_pass,
            "cue_override_rate": 0.10 if classifier_pass else 1.0,
            "without_cue_3class_gate_passed": classifier_pass,
            "macro_f1_3class_without_cue": 0.82 if classifier_pass else 0.0,
            "formal_engine": classifier_pass,
            "engine": "sentence_embedding_nearest_centroid" if classifier_pass else "keyword_debug",
            "macro_f1_3class": 0.85,
        },
    }


def _all_strategies(bundle: dict[str, dict]) -> dict[str, dict]:
    return {
        "random": dict(bundle, split_metadata={"strategy": "random"}),
        "temporal": dict(bundle, split_metadata={"strategy": "temporal"}),
        "partner": dict(bundle, split_metadata={"strategy": "partner"}),
        "topic": dict(bundle, split_metadata={"strategy": "topic", "topic_split_not_applicable": False}),
    }


def _formal_strategy(
    bundle: dict[str, dict],
    *,
    uid: int,
    required_users: int = 8,
    leave_one_out: bool = True,
    population_user_count: int | None = None,
    excluded_uid: int | None = None,
    metrics_present: bool = True,
    diagnostic_trace_rows: int = 1,
    train_full_l2: float = 0.10,
    train_default_l2: float = 0.60,
    train_wrong_l2: float = 0.70,
    majority_behavioral: float = 0.30,
    baseline_c_builder: str = "population_average_full_implant",
    surface_only_diff: float = -0.10,
    no_surface_diff: float = -0.10,
    no_policy_diff: float = -0.10,
    surface_only_c_diff: float | None = None,
    no_surface_c_diff: float | None = None,
    no_policy_c_diff: float | None = None,
) -> dict[str, dict]:
    row = dict(bundle)
    if not metrics_present:
        row["baseline_c_metrics"] = {}
    trace = []
    for idx in range(diagnostic_trace_rows):
        trace.append(
            {
                "pair_index": idx,
                "real_text": "real response",
                "personality_action": "elaborate",
                "personality_strategy": "exploit",
                "baseline_c_action": "ask_question",
                "baseline_c_strategy": "explore",
            }
        )
    row.update(
        {
            "baseline_c_leave_one_out": bool(leave_one_out),
            "baseline_c_builder": baseline_c_builder,
            "baseline_c_input_scope": "leave_one_out_population_train_and_profile_data",
            "baseline_c_population_excluded_uid": int(uid if excluded_uid is None else excluded_uid),
            "baseline_c_population_user_count": int(
                required_users - 1 if population_user_count is None else population_user_count
            ),
            "diagnostic_trace": trace,
            "state_distance_diagnostics": {
                "train_full": {"l2": float(train_full_l2)},
                "train_default": {"l2": float(train_default_l2)},
                "train_wrong_user": {"l2": float(train_wrong_l2)},
            },
            "majority_baseline_metrics": {
                "majority_action": "elaborate",
                "majority_strategy": "exploit",
                "behavioral_similarity_strategy": float(majority_behavioral),
                "balanced_behavioral_similarity_strategy": 0.50,
                "real_strategy_distribution": {"exploit": 2, "explore": 1},
                "real_action_distribution": {"elaborate": 2, "ask_question": 1},
            },
            "ablation_summary": [
                {
                    "name": "no_surface_profile",
                    "pair_count": 4,
                    "semantic_mean": 0.10,
                    "semantic_vs_baseline_a_diff": float(no_surface_diff),
                    "semantic_vs_baseline_c_diff": float(
                        no_surface_diff if no_surface_c_diff is None else no_surface_c_diff
                    ),
                    "action_agreement_vs_personality": 0.25,
                    "strategy_agreement_vs_personality": 0.25,
                    "text_similarity_vs_personality": 0.20,
                },
                {
                    "name": "no_policy_trait_bias",
                    "pair_count": 4,
                    "semantic_mean": 0.10,
                    "semantic_vs_baseline_a_diff": float(no_policy_diff),
                    "semantic_vs_baseline_c_diff": float(
                        no_policy_diff if no_policy_c_diff is None else no_policy_c_diff
                    ),
                    "action_agreement_vs_personality": 0.25,
                    "strategy_agreement_vs_personality": 0.25,
                    "text_similarity_vs_personality": 0.20,
                },
                {
                    "name": "surface_only_default_agent",
                    "pair_count": 4,
                    "semantic_mean": 0.10,
                    "semantic_vs_baseline_a_diff": float(surface_only_diff),
                    "semantic_vs_baseline_c_diff": float(
                        surface_only_diff if surface_only_c_diff is None else surface_only_c_diff
                    ),
                    "action_agreement_vs_personality": 0.25,
                    "strategy_agreement_vs_personality": 0.25,
                    "text_similarity_vs_personality": 0.20,
                },
            ],
        }
    )
    return row


def _all_formal_strategies(uid: int, bundle: dict[str, dict], **kwargs) -> dict[str, dict]:
    return {
        "random": dict(_formal_strategy(bundle, uid=uid, **kwargs), split_metadata={"strategy": "random"}),
        "temporal": dict(_formal_strategy(bundle, uid=uid, **kwargs), split_metadata={"strategy": "temporal"}),
        "partner": dict(_formal_strategy(bundle, uid=uid, **kwargs), split_metadata={"strategy": "partner"}),
        "topic": dict(
            _formal_strategy(bundle, uid=uid, **kwargs),
            split_metadata={"strategy": "topic", "topic_split_not_applicable": False},
        ),
    }


def _mark_llm_generated_provisional(strategies: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, row in strategies.items():
        item = dict(row)
        cv = dict(item.get("classifier_validation", {}))
        cv.update(
            {
                "passed_3class_gate": False,
                "formal_gate_eligible": False,
                "classifier_evidence_tier": "llm_generated_provisional",
                "dataset_origin": "codex_authored_realistic_zh_fixture_v4",
                "classifier_provenance_ok": False,
                "classifier_provenance_failure_reason": "codex_authored",
                "behavioral_hard_metric_enabled": False,
                "degradation_required": True,
            }
        )
        item["classifier_validation"] = cv
        out[name] = item
    return out


def _aggregate(
    required_users: int = 8,
    *,
    skip_population_average_implant: bool = False,
    formal_requested: bool = False,
) -> dict[str, object]:
    return {
        "required_users": int(required_users),
        "skip_population_average_implant": bool(skip_population_average_implant),
        "formal_requested": bool(formal_requested),
        "pilot": {
            "pilot_user_count": 3,
            "required_users": int(required_users),
            "suggested_min_users": int(required_users),
            "semantic_diff_sd": 0.01,
        },
        "requested_strategies": ["random", "temporal", "partner", "topic"],
    }


class TestM54ReportAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        def fake_paired(personality, baseline, *, test: str = "wilcoxon", alpha: float = 0.05):
            n = min(len(personality), len(baseline))
            if n == 0:
                return 1.0, False, 0.0, False
            diffs = [float(personality[idx]) - float(baseline[idx]) for idx in range(n)]
            mean_diff = sum(diffs) / float(n)
            better = mean_diff > 1e-12
            p = 0.01 if better else 1.0
            return p, bool(better and p < alpha), mean_diff, better

        self._patches = [
            patch("segmentum.dialogue.validation.report.scipy_wilcoxon_available", return_value=True),
            patch("segmentum.dialogue.validation.report.paired_comparison", side_effect=fake_paired),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()

    def test_hard_pass_when_sem_beh_sig_and_agent_state(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_strategies(
                    _bundle(
                            sem=0.80 + 0.001 * uid,
                            beh=0.70 + 0.001 * uid,
                            ast=0.90,
                            base_sem=0.50,
                            base_beh=0.40,
                            base_beh_c=0.35,
                            classifier_pass=True,
                    )
                ),
                aggregate=_aggregate(required_users=8),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["metric_version"], M54_ACCEPTANCE_RULES_VERSION)
        self.assertIn("hard_pass_breakdown", payload)
        self.assertTrue(payload["hard_pass_breakdown"]["classifier_3class_gate_passed"])
        self.assertTrue(payload["hard_pass"])
        self.assertEqual(payload["overall_conclusion"], "pass")
        self.assertTrue(payload["pilot_gate"]["passed"])
        self.assertTrue(payload["split_gate"]["passed"])
        self.assertTrue(payload["partner_gate"]["passed"])
        self.assertTrue(payload["topic_gate"]["passed"])
        self.assertEqual(payload["users_tested"], 8)
        self.assertEqual(payload["users_skipped_no_strategy"], 0)
        self.assertEqual(payload["agent_state_users_tested"], 8)
        self.assertEqual(payload["agent_state_users_skipped_no_metric"], 0)
        ast = payload["comparisons"]["agent_state_similarity"]
        self.assertIsNone(ast["baseline_a_mean"])
        self.assertIn("interpretation_notes", ast)
        self.assertFalse(payload["behavioral_hard_metric_degraded"])

    def test_hard_fail_when_agent_state_low(self) -> None:
        reports = [
            ValidationReport(
                user_uid=1,
                per_strategy=_all_strategies(
                    _bundle(
                        sem=0.9,
                        beh=0.85,
                        ast=0.5,
                        base_sem=0.2,
                        base_beh=0.2,
                        classifier_pass=False,
                    )
                ),
                aggregate=_aggregate(required_users=2),
                conclusion="completed",
            ),
            ValidationReport(
                user_uid=2,
                per_strategy=_all_strategies(
                    _bundle(
                        sem=0.91,
                        beh=0.86,
                        ast=0.55,
                        base_sem=0.21,
                        base_beh=0.21,
                        classifier_pass=False,
                    )
                ),
                aggregate=_aggregate(required_users=2),
                conclusion="completed",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertEqual(payload["overall_conclusion"], "fail")
        self.assertTrue(payload["behavioral_hard_metric_degraded"])

    def test_hard_fail_when_classifier_gate_passed_but_behavioral_not_vs_c(self) -> None:
        """
        When gate passes, behavioral must beat baseline C. Here personality behavioral
        equals baseline C (zero paired diff), so vs_c is not significant.
        """
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_strategies(
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.70,
                        classifier_pass=True,
                    )
                ),
                aggregate=_aggregate(required_users=8),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["hard_pass_breakdown"]["behavioral_similarity_strategy_vs_baseline_c_significant_better"])

    def test_formal_hard_fail_when_semantic_not_better_than_baseline_c(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_sem_c=0.90,
                        base_beh_c=0.35,
                    ),
                    no_surface_c_diff=-0.50,
                    no_policy_c_diff=-0.50,
                    surface_only_c_diff=-0.50,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass_breakdown"]["semantic_similarity_vs_baseline_c_significant_better"])
        self.assertIn("semantic_vs_baseline_c_failed", payload["formal_blockers"])
        self.assertFalse(payload["hard_pass"])

    def test_random_only_cannot_pass_formal_acceptance(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy={
                    "random": _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    )
                },
                aggregate=_aggregate(required_users=8),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertEqual(payload["overall_conclusion"], "partial")
        self.assertFalse(payload["split_gate"]["passed"])

    def test_topic_not_applicable_is_excluded_but_min_users_blocks(self) -> None:
        reports = []
        for uid in range(8):
            strategies = _all_strategies(
                _bundle(
                    sem=0.80 + 0.001 * uid,
                    beh=0.70 + 0.001 * uid,
                    ast=0.90,
                    base_sem=0.50,
                    base_beh=0.40,
                    base_beh_c=0.35,
                )
            )
            if uid >= 6:
                strategies["topic"] = _bundle(
                    sem=0.80 + 0.001 * uid,
                    beh=0.70 + 0.001 * uid,
                    ast=0.90,
                    base_sem=0.50,
                    base_beh=0.40,
                    base_beh_c=0.35,
                    topic_not_applicable=True,
                )
            reports.append(
                ValidationReport(
                    user_uid=uid,
                    per_strategy=strategies,
                    aggregate=_aggregate(required_users=8),
                    conclusion="completed",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["topic_gate"]["passed"])
        self.assertEqual(payload["topic_gate"]["valid_topic_users"], 6)

    def test_partner_failure_blocks_overall_pass(self) -> None:
        reports = []
        for uid in range(8):
            good = _bundle(
                sem=0.80 + 0.001 * uid,
                beh=0.70 + 0.001 * uid,
                ast=0.90,
                base_sem=0.50,
                base_beh=0.40,
                base_beh_c=0.35,
            )
            strategies = _all_strategies(good)
            strategies["partner"] = _bundle(
                sem=0.50,
                beh=0.70 + 0.001 * uid,
                ast=0.90,
                base_sem=0.50,
                base_beh=0.40,
                base_beh_c=0.35,
            )
            reports.append(
                ValidationReport(
                    user_uid=uid,
                    per_strategy=strategies,
                    aggregate=_aggregate(required_users=8),
                    conclusion="completed",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["partner_gate"]["passed"])

    def test_classifier_failure_degrades_behavioral_without_blocking(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_strategies(
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.10,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.90,
                        base_beh_c=0.90,
                        classifier_pass=False,
                    )
                ),
                aggregate=_aggregate(required_users=8),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["formal_acceptance_eligible"])
        self.assertEqual(payload["overall_conclusion"], "partial")
        self.assertTrue(payload["behavioral_hard_metric_degraded"])
        self.assertFalse(payload["hard_pass_breakdown"]["behavioral_hard_metric_required"])

    def test_formal_classifier_failure_blocks_behavioral_hard_metric(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                        classifier_pass=False,
                    ),
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["hard_pass_breakdown"]["behavioral_hard_metric_required"])
        self.assertFalse(payload["hard_pass_breakdown"]["classifier_3class_gate_passed"])
        self.assertFalse(payload["hard_pass_breakdown"]["metric_hard_pass"])
        self.assertIn("classifier_fixture_only", payload["formal_blockers"])
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["partial_acceptance_eligible"])
        self.assertEqual(payload["overall_conclusion"], "fail")

    def test_formal_llm_generated_labels_can_receive_partial_acceptance(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_mark_llm_generated_provisional(
                    _all_formal_strategies(
                        uid,
                        _bundle(
                            sem=0.80 + 0.001 * uid,
                            beh=0.70 + 0.001 * uid,
                            ast=0.90,
                            base_sem=0.50,
                            base_beh=0.40,
                            base_beh_c=0.35,
                            classifier_pass=False,
                        ),
                    )
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["formal_acceptance_eligible"])
        self.assertTrue(payload["partial_acceptance_eligible"])
        self.assertTrue(payload["partial_acceptance_gate"]["llm_generated_provisional_partial_allowed"])
        self.assertEqual(payload["overall_conclusion"], "partial")
        self.assertIn("classifier_provisional_llm_labels", payload["formal_blockers"])

    def test_test_only_baseline_c_blocks_formal_pass(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_strategies(
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    )
                ),
                aggregate=_aggregate(required_users=8, skip_population_average_implant=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["hard_pass"])
        self.assertFalse(payload["baseline_c_gate"]["passed"])

    def test_formal_report_passes_new_repair_gates_when_evidence_is_present(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    required_users=8,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["baseline_c_gate"]["passed"])
        self.assertTrue(payload["diagnostic_trace_gate"]["passed"])
        self.assertTrue(payload["agent_state_differentiation_gate"]["passed"])
        self.assertTrue(payload["behavioral_majority_baseline_gate"]["passed"])
        self.assertTrue(payload["surface_ablation_gate"]["passed"])
        self.assertEqual(payload["surface_ablation_gate"]["aggregation"], "per_user_mean_across_strategies")
        self.assertEqual(payload["formal_blockers"], [])
        self.assertEqual(payload["artifact_rules_version"], M54_ACCEPTANCE_RULES_VERSION)
        self.assertGreater(payload["baseline_c_behavioral_failure_audit"]["row_count"], 0)
        self.assertEqual(payload["overall_conclusion"], "pass")

    def test_formal_baseline_c_gate_rejects_bad_population_metadata(self) -> None:
        cases = [
            {"population_user_count": 0},
            {"population_user_count": 6},
            {"excluded_uid": 99999},
            {"metrics_present": False},
            {"leave_one_out": False},
            {"baseline_c_builder": "profile_only_average_fallback"},
        ]
        for case in cases:
            with self.subTest(case=case):
                reports = [
                    ValidationReport(
                        user_uid=uid,
                        per_strategy=_all_formal_strategies(
                            uid,
                            _bundle(
                                sem=0.80 + 0.001 * uid,
                                beh=0.70 + 0.001 * uid,
                                ast=0.90,
                                base_sem=0.50,
                                base_beh=0.40,
                                base_beh_c=0.35,
                            ),
                            required_users=8,
                            **case,
                        ),
                        aggregate=_aggregate(required_users=8, formal_requested=True),
                        conclusion="completed",
                    )
                    for uid in range(8)
                ]
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp)
                    generate_report(reports, p)
                    payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
                self.assertFalse(payload["baseline_c_gate"]["passed"])
                self.assertFalse(payload["hard_pass"])

    def test_formal_surface_ablation_gate_blocks_phrase_only_win(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    required_users=8,
                    surface_only_diff=0.35,
                    no_surface_diff=0.35,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["surface_ablation_gate"]["passed"])
        self.assertEqual(payload["surface_ablation_gate"]["aggregation"], "per_user_mean_across_strategies")
        self.assertIn("surface_ablation_failed", payload["formal_blockers"])
        self.assertFalse(payload["formal_acceptance_eligible"])
        self.assertFalse(payload["hard_pass"])

    def test_formal_surface_ablation_gate_blocks_no_policy_win_vs_baseline_c(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    no_policy_c_diff=0.60,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["surface_ablation_gate"]["passed"])
        self.assertFalse(payload["surface_ablation_gate"]["checks"]["full_significant_better_than_no_policy"])
        self.assertIn("no_policy_ablation_failed", payload["formal_blockers"])
        self.assertFalse(payload["hard_pass"])

    def test_formal_surface_ablation_gate_blocks_surface_only_win_vs_baseline_c(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    surface_only_c_diff=0.60,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["surface_ablation_gate"]["passed"])
        self.assertFalse(payload["surface_ablation_gate"]["checks"]["full_significant_better_than_surface_only"])
        self.assertIn("surface_only_ablation_failed", payload["formal_blockers"])
        self.assertFalse(payload["hard_pass"])

    def test_surface_ablation_gate_uses_user_level_pairs(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80,
                        beh=0.70,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    required_users=2,
                    surface_only_diff=-0.10,
                    no_surface_diff=-0.10,
                ),
                aggregate=_aggregate(required_users=2, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        gate = payload["surface_ablation_gate"]
        self.assertEqual(gate["aggregation"], "per_user_mean_across_strategies")
        self.assertEqual(gate["comparisons"]["no_surface_profile"]["pairs"], 2)
        self.assertEqual(gate["comparisons"]["no_policy_trait_bias"]["pairs"], 2)
        self.assertEqual(gate["comparisons"]["surface_only_default_agent"]["pairs"], 2)

    def test_surface_ablation_gate_blocks_strategy_weighted_pseudo_replication(self) -> None:
        user_one = _all_formal_strategies(
            1,
            _bundle(
                sem=0.90,
                beh=0.70,
                ast=0.90,
                base_sem=0.50,
                base_beh=0.40,
                base_beh_c=0.35,
            ),
            required_users=2,
            surface_only_diff=0.00,
            no_surface_diff=0.00,
        )
        user_two = {
            "random": _formal_strategy(
                _bundle(
                    sem=0.50,
                    beh=0.70,
                    ast=0.90,
                    base_sem=0.50,
                    base_beh=0.40,
                    base_beh_c=0.35,
                ),
                uid=2,
                required_users=2,
                surface_only_diff=0.50,
                no_surface_diff=0.50,
            )
        }
        reports = [
            ValidationReport(
                user_uid=1,
                per_strategy=user_one,
                aggregate=_aggregate(required_users=2, formal_requested=True),
                conclusion="completed",
            ),
            ValidationReport(
                user_uid=2,
                per_strategy=user_two,
                aggregate=_aggregate(required_users=2, formal_requested=True),
                conclusion="completed",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        gate = payload["surface_ablation_gate"]
        self.assertEqual(gate["comparisons"]["no_surface_profile"]["pairs"], 2)
        self.assertFalse(gate["passed"])
        self.assertIn("surface_ablation_failed", payload["formal_blockers"])

    def test_formal_diagnostic_trace_absence_blocks_hard_pass(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    diagnostic_trace_rows=0,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["diagnostic_trace_gate"]["passed"])
        self.assertFalse(payload["hard_pass"])

    def test_formal_agent_state_gate_rejects_wrong_user_closer_than_full(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70 + 0.001 * uid,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    train_full_l2=0.50,
                    train_default_l2=0.70,
                    train_wrong_l2=0.40,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["agent_state_differentiation_gate"]["passed"])
        self.assertFalse(payload["agent_state_differentiation_gate"]["train_full_closer_than_wrong_user"])
        self.assertFalse(payload["hard_pass"])

    def test_formal_majority_baseline_warning_blocks_hard_pass(self) -> None:
        reports = [
            ValidationReport(
                user_uid=uid,
                per_strategy=_all_formal_strategies(
                    uid,
                    _bundle(
                        sem=0.80 + 0.001 * uid,
                        beh=0.70,
                        ast=0.90,
                        base_sem=0.50,
                        base_beh=0.40,
                        base_beh_c=0.35,
                    ),
                    majority_behavioral=0.70,
                ),
                aggregate=_aggregate(required_users=8, formal_requested=True),
                conclusion="completed",
            )
            for uid in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            generate_report(reports, p)
            payload = json.loads((p / "aggregate_report.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["behavioral_majority_baseline_gate"]["behavioral_metric_majority_warning"])
        self.assertFalse(payload["behavioral_majority_baseline_gate"]["passed"])
        self.assertFalse(payload["hard_pass"])


if __name__ == "__main__":
    unittest.main()
