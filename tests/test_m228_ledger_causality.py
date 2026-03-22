from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.prediction_ledger import LedgerDiscrepancy


def _observation() -> Observation:
    return Observation(
        food=0.68,
        danger=0.62,
        novelty=0.18,
        shelter=0.22,
        temperature=0.47,
        social=0.24,
    )


class TestM228LedgerCausality(unittest.TestCase):
    def test_ledger_changes_action_ranking_memory_and_explanation(self) -> None:
        base = SegmentAgent(rng=random.Random(9))
        with_ledger = SegmentAgent(rng=random.Random(9))
        with_ledger.prediction_ledger.discrepancies.append(
            LedgerDiscrepancy(
                discrepancy_id="disc:danger_mismatch",
                label="repeated danger mismatch",
                source="prediction_error",
                discrepancy_type="danger_mismatch",
                created_tick=0,
                last_seen_tick=0,
                severity=0.95,
                recurrence_count=4,
                chronic=True,
                subject_critical=True,
                target_channels=("danger",),
            )
        )

        base_decision = base.decision_cycle(_observation())["diagnostics"]
        ledger_decision = with_ledger.decision_cycle(_observation())["diagnostics"]

        base_scores = {item.choice: item for item in base_decision.ranked_options}
        ledger_scores = {item.choice: item for item in ledger_decision.ranked_options}

        self.assertGreater(ledger_scores["hide"].policy_score, base_scores["hide"].policy_score)
        self.assertLess(ledger_scores["forage"].policy_score, base_scores["forage"].policy_score)
        self.assertLess(with_ledger.prediction_ledger.memory_threshold_delta(), 0.0)
        self.assertTrue(with_ledger.prediction_ledger.workspace_focus())
        self.assertTrue(with_ledger.prediction_ledger.maintenance_signal()["active_tasks"])
        self.assertIn("unresolved discrepancy", ledger_decision.ledger_summary)


if __name__ == "__main__":
    unittest.main()
