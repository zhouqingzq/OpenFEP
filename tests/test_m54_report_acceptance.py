"""Aggregate report hard_pass rules (M5.4)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

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
        "classifier_validation": {
            "passed_3class_gate": classifier_pass,
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


def _aggregate(required_users: int = 8, *, skip_population_average_implant: bool = False) -> dict[str, object]:
    return {
        "required_users": int(required_users),
        "skip_population_average_implant": bool(skip_population_average_implant),
        "pilot": {
            "pilot_user_count": 3,
            "required_users": int(required_users),
            "suggested_min_users": int(required_users),
            "semantic_diff_sd": 0.01,
        },
        "requested_strategies": ["random", "temporal", "partner", "topic"],
    }


class TestM54ReportAcceptance(unittest.TestCase):
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
        self.assertEqual(payload["metric_version"], "m54_v3")
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
        self.assertTrue(payload["hard_pass"])
        self.assertTrue(payload["behavioral_hard_metric_degraded"])
        self.assertFalse(payload["hard_pass_breakdown"]["behavioral_hard_metric_required"])

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


if __name__ == "__main__":
    unittest.main()
