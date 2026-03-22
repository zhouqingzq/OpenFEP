from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.prediction_ledger import LedgerDiscrepancy, VerificationStatus


class TestM228SleepLedgerIntegration(unittest.TestCase):
    def test_sleep_archives_resolved_items_but_keeps_identity_critical_tension(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        agent.prediction_ledger.discrepancies.extend(
            [
                LedgerDiscrepancy(
                    discrepancy_id="disc:identity_tension",
                    label="identity critical discrepancy",
                    source="identity",
                    discrepancy_type="identity_tension",
                    created_tick=1,
                    last_seen_tick=2,
                    severity=0.82,
                    status=VerificationStatus.DISCHARGED.value,
                    recurrence_count=4,
                    chronic=True,
                    identity_relevant=True,
                    subject_critical=True,
                    target_channels=("continuity",),
                ),
                LedgerDiscrepancy(
                    discrepancy_id="disc:minor",
                    label="minor discrepancy",
                    source="prediction_error",
                    discrepancy_type="minor_mismatch",
                    created_tick=1,
                    last_seen_tick=2,
                    severity=0.20,
                    status=VerificationStatus.DISCHARGED.value,
                    recurrence_count=1,
                    target_channels=("food",),
                ),
            ]
        )

        agent.sleep()

        remaining_ids = {item.discrepancy_id for item in agent.prediction_ledger.discrepancies}
        archived_ids = {item.discrepancy_id for item in agent.prediction_ledger.archived_discrepancies}

        self.assertIn("disc:identity_tension", remaining_ids)
        self.assertIn("disc:minor", archived_ids)
        self.assertTrue(agent.narrative_trace)
        self.assertIn("prediction_ledger_sleep", agent.narrative_trace[-1])

    def test_sleep_archives_old_discharged_noncritical_items_before_chronic_escalation(self) -> None:
        agent = SegmentAgent(rng=random.Random(17))
        agent.prediction_ledger.discrepancies.append(
            LedgerDiscrepancy(
                discrepancy_id="disc:minor_old",
                label="minor old discrepancy",
                source="prediction_error",
                discrepancy_type="minor_mismatch",
                created_tick=1,
                last_seen_tick=2,
                severity=0.20,
                status=VerificationStatus.DISCHARGED.value,
                recurrence_count=1,
                target_channels=("food",),
            )
        )
        agent.cycle = 5

        review = agent.prediction_ledger.sleep_review(tick=agent.cycle)

        remaining_ids = {item.discrepancy_id for item in agent.prediction_ledger.discrepancies}
        archived = {item.discrepancy_id: item for item in agent.prediction_ledger.archived_discrepancies}

        self.assertNotIn("disc:minor_old", remaining_ids)
        self.assertIn("disc:minor_old", review["archived_discrepancies"])
        self.assertNotIn("disc:minor_old", review["escalated_discrepancies"])
        self.assertEqual(archived["disc:minor_old"].status, VerificationStatus.ARCHIVED.value)
        self.assertEqual(archived["disc:minor_old"].last_seen_tick, agent.cycle)

    def test_sleep_refreshes_last_seen_when_escalating_chronic_items(self) -> None:
        agent = SegmentAgent(rng=random.Random(19))
        agent.prediction_ledger.discrepancies.append(
            LedgerDiscrepancy(
                discrepancy_id="disc:stale",
                label="stale discrepancy",
                source="memory",
                discrepancy_type="surprise_burden",
                created_tick=1,
                last_seen_tick=1,
                severity=0.45,
                status=VerificationStatus.ACTIVE.value,
                recurrence_count=1,
                target_channels=("novelty",),
            )
        )
        agent.prediction_ledger.last_tick = 5

        review = agent.prediction_ledger.sleep_review(tick=5)

        self.assertIn("disc:stale", review["escalated_discrepancies"])
        refreshed = next(
            item for item in agent.prediction_ledger.discrepancies if item.discrepancy_id == "disc:stale"
        )
        self.assertEqual(refreshed.last_seen_tick, 5)
        self.assertEqual(refreshed.to_dict(reference_tick=5)["age"], 4)


if __name__ == "__main__":
    unittest.main()
