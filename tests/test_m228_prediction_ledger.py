from __future__ import annotations

import unittest
from types import SimpleNamespace

from segmentum.prediction_ledger import (
    LedgerDiscrepancy,
    PredictionHypothesis,
    PredictionLedger,
    VerificationStatus,
)


def _diagnostics() -> SimpleNamespace:
    return SimpleNamespace(
        identity_tension=0.42,
        self_inconsistency_error=0.36,
        violated_commitments=["keep continuity"],
        conflict_type="commitment_conflict",
        commitment_focus=["keep continuity"],
        active_goal="SURVIVAL",
        repair_triggered=False,
        social_alerts=["counterpart_withdrawal"],
    )


def _agenda(policy_shift_strength: float, chronic_debt_pressure: float) -> SimpleNamespace:
    return SimpleNamespace(
        policy_shift_strength=policy_shift_strength,
        chronic_debt_pressure=chronic_debt_pressure,
        active_tasks=("stress_relief", "continuity_guard"),
        protected_mode=policy_shift_strength >= 0.3,
    )


def _subject_state() -> SimpleNamespace:
    return SimpleNamespace(
        status_flags={"continuity_fragile": True},
        continuity_anchors=("anchor-a",),
        identity_tension_level=0.4,
    )


class TestM228PredictionLedger(unittest.TestCase):
    def test_predictions_escalate_into_chronic_discrepancy_and_discharge(self) -> None:
        ledger = PredictionLedger()
        ledger.predictions.append(
            PredictionHypothesis(
                prediction_id="pred:env:danger",
                created_tick=1,
                last_updated_tick=1,
                source_module="test",
                prediction_type="environment_state",
                target_channels=("danger",),
                expected_state={"danger": 0.10},
                confidence=0.8,
                expected_horizon=1,
            )
        )

        verification = ledger.verify_predictions(tick=2, observation={"danger": 0.82})
        self.assertIn("pred:env:danger", verification.falsified_predictions)
        self.assertTrue(ledger.active_discrepancies())

        for tick in (2, 3, 4):
            ledger.record_runtime_discrepancies(
                tick=tick,
                diagnostics=_diagnostics(),
                errors={"danger": 0.55},
                maintenance_agenda=_agenda(0.34, 0.22),
                continuity_score=0.61,
                subject_state=_subject_state(),
                memory_surprise=0.91,
            )

        top = ledger.top_discrepancies(limit=1)[0]
        self.assertGreaterEqual(top.recurrence_count, 3)
        self.assertTrue(top.chronic)
        self.assertIn(top.priority, {"high", "critical"})

        discharge = ledger.record_runtime_discrepancies(
            tick=5,
            diagnostics=SimpleNamespace(
                identity_tension=0.0,
                self_inconsistency_error=0.0,
                violated_commitments=[],
                conflict_type="none",
                commitment_focus=[],
                active_goal="SURVIVAL",
                repair_triggered=True,
                social_alerts=[],
            ),
            errors={"danger": 0.01},
            maintenance_agenda=_agenda(0.05, 0.01),
            continuity_score=0.95,
            subject_state=SimpleNamespace(
                status_flags={"continuity_fragile": False},
                continuity_anchors=("anchor-a",),
                identity_tension_level=0.0,
            ),
            memory_surprise=0.05,
        )
        self.assertTrue(discharge.discharged_discrepancies)
        self.assertGreaterEqual(len(ledger.archived_discrepancies), 1)

    def test_explanation_payload_lists_all_unresolved_discrepancies_with_source_age_and_status(self) -> None:
        ledger = PredictionLedger(last_tick=7)
        for index in range(6):
            ledger.discrepancies.append(
                LedgerDiscrepancy(
                    discrepancy_id=f"disc:{index}",
                    label=f"discrepancy {index}",
                    source="prediction_error" if index % 2 == 0 else "identity",
                    discrepancy_type=f"type_{index}",
                    created_tick=index,
                    last_seen_tick=index + 1,
                    severity=0.20 + 0.08 * index,
                    status=VerificationStatus.ACTIVE.value,
                    recurrence_count=1 + index,
                    target_channels=("danger",) if index % 2 == 0 else ("conflict",),
                    identity_relevant=bool(index % 2),
                )
            )

        payload = ledger.explanation_payload()

        self.assertEqual(payload["counts"]["active_discrepancies"], 6)
        self.assertEqual(len(payload["unresolved_discrepancies"]), 6)
        self.assertLessEqual(len(payload["top_discrepancies"]), 4)

        by_id = {item["discrepancy_id"]: item for item in payload["unresolved_discrepancies"]}
        self.assertEqual(by_id["disc:0"]["source"], "prediction_error")
        self.assertEqual(by_id["disc:0"]["status"], VerificationStatus.ACTIVE.value)
        self.assertEqual(by_id["disc:0"]["age"], 7)
        self.assertEqual(by_id["disc:5"]["source"], "identity")
        self.assertEqual(by_id["disc:5"]["status"], VerificationStatus.ACTIVE.value)
        self.assertEqual(by_id["disc:5"]["age"], 2)

    def test_discrepancy_to_dict_uses_reference_tick_for_age(self) -> None:
        discrepancy = LedgerDiscrepancy(
            discrepancy_id="disc:age",
            label="age test",
            source="memory",
            discrepancy_type="surprise_burden",
            created_tick=2,
            last_seen_tick=3,
            severity=0.4,
        )

        self.assertEqual(discrepancy.to_dict()["age"], 1)
        self.assertEqual(discrepancy.to_dict(reference_tick=8)["age"], 6)


if __name__ == "__main__":
    unittest.main()
