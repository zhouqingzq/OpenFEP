"""Per-user aggregation for M5.4 aggregate report statistics."""

from __future__ import annotations

import unittest

from segmentum.dialogue.validation.pipeline import ValidationReport
from segmentum.dialogue.validation.report import collect_per_user_metric_vectors


def _mini_strategy(sem_p: float, sem_a: float, sem_c: float) -> dict[str, object]:
    return {
        "skipped": False,
        "personality_metrics": {"semantic_similarity": sem_p},
        "baseline_a_metrics": {"semantic_similarity": sem_a},
        "baseline_b_metrics": {"semantic_similarity": sem_a},
        "baseline_c_metrics": {"semantic_similarity": sem_c},
    }


class TestM54ReportAggregation(unittest.TestCase):
    def test_two_strategies_per_user_yield_two_wilcoxon_rows_not_four(self) -> None:
        r1 = ValidationReport(
            user_uid=1,
            per_strategy={
                "random": _mini_strategy(0.8, 0.5, 0.45),
                "topic": _mini_strategy(0.82, 0.51, 0.46),
            },
            aggregate={},
            conclusion="completed",
        )
        r2 = ValidationReport(
            user_uid=2,
            per_strategy={
                "random": _mini_strategy(0.75, 0.48, 0.44),
                "topic": _mini_strategy(0.77, 0.49, 0.43),
            },
            aggregate={},
            conclusion="completed",
        )
        p, a, _, c, users_used, skipped = collect_per_user_metric_vectors(
            [r1, r2], "semantic_similarity"
        )
        self.assertEqual(users_used, 2)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(p), 2)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(p[0], (0.8 + 0.82) / 2.0)
        self.assertAlmostEqual(p[1], (0.75 + 0.77) / 2.0)

    def test_skipped_user_when_no_strategy_data(self) -> None:
        empty = ValidationReport(
            user_uid=99,
            per_strategy={"random": {"skipped": True, "reason": "x"}},
            aggregate={},
            conclusion="completed",
        )
        r = ValidationReport(
            user_uid=1,
            per_strategy={"random": _mini_strategy(0.8, 0.5, 0.45)},
            aggregate={},
            conclusion="completed",
        )
        p, _, _, _, users_used, skipped = collect_per_user_metric_vectors(
            [r, empty], "semantic_similarity"
        )
        self.assertEqual(users_used, 1)
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
