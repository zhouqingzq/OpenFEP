from __future__ import annotations

import random
import unittest

from segmentum.action_schema import ActionSchema
from segmentum.agent import SegmentAgent
from segmentum.prediction_ledger import PredictionHypothesis


def _store_episode(agent: SegmentAgent, *, cycle: int = 1) -> str:
    payload = agent.long_term_memory.store_episode(
        cycle=cycle,
        observation={
            "food": 0.30,
            "danger": 0.70,
            "novelty": 0.20,
            "shelter": 0.45,
            "temperature": 0.50,
            "social": 0.20,
        },
        prediction={
            "food": 0.55,
            "danger": 0.20,
            "novelty": 0.30,
            "shelter": 0.35,
            "temperature": 0.50,
            "social": 0.22,
        },
        errors={
            "food": -0.25,
            "danger": 0.50,
            "novelty": -0.10,
            "shelter": 0.10,
            "temperature": 0.0,
            "social": -0.02,
        },
        action=ActionSchema(name="hide"),
        outcome={"energy_delta": 0.02, "stress_delta": -0.03, "free_energy_drop": 0.08},
        body_state={"energy": 0.82, "stress": 0.15, "fatigue": 0.18, "temperature": 0.50},
    )
    agent.sync_memory_awareness_to_long_term_memory()
    return str(payload["episode_id"])


def _prediction(entry_id: str, *, prediction_id: str = "pred:test:danger") -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id=prediction_id,
        created_tick=1,
        last_updated_tick=1,
        source_module="test",
        prediction_type="environment_state",
        target_channels=("danger",),
        expected_state={"danger": 0.70},
        confidence=0.72,
        expected_horizon=1,
        semantic_provenance={
            "linked_memory_ids": [entry_id],
            "committed_memory_ids": [entry_id],
            "auxiliary_memory_ids": [],
            "linked_path_ids": [],
        },
    )


class TestM176MemoryCredit(unittest.TestCase):
    def test_decision_cycle_seeds_prediction_with_memory_provenance(self) -> None:
        agent = SegmentAgent(rng=random.Random(20))
        entry_id = _store_episode(agent)

        agent.decision_cycle(
            {
                "food": 0.30,
                "danger": 0.70,
                "novelty": 0.20,
                "shelter": 0.45,
                "temperature": 0.50,
                "social": 0.20,
            }
        )

        active_predictions = agent.prediction_ledger.active_predictions()
        self.assertTrue(active_predictions)
        self.assertTrue(
            any(
                entry_id in list(item.semantic_provenance.get("linked_memory_ids", []))
                for item in active_predictions
            )
        )
        self.assertTrue(
            any(
                list(item.semantic_provenance.get("committed_memory_ids", []))
                for item in active_predictions
            )
        )

    def test_confirmed_verification_strengthens_linked_memory(self) -> None:
        agent = SegmentAgent(rng=random.Random(21))
        entry_id = _store_episode(agent)
        before = agent.memory_store.get(entry_id)
        assert before is not None
        before_access = before.accessibility
        before_trace = before.trace_strength

        agent.prediction_ledger.predictions.append(_prediction(entry_id))
        agent.verification_loop.refresh_targets(
            tick=2,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )
        update = agent.verification_loop.process_observation(
            tick=2,
            observation={"danger": 0.70},
            ledger=agent.prediction_ledger,
            source="runtime_observation",
            subject_state=agent.subject_state,
        )

        self.assertTrue(update.memory_credit_signals)
        reports = agent.apply_memory_credit_signals(update.memory_credit_signals, tick=2)
        self.assertTrue(reports)

        after = agent.memory_store.get(entry_id)
        assert after is not None
        metadata = dict(after.compression_metadata or {}).get("m17_memory_credit", {})
        self.assertGreater(after.accessibility, before_access)
        self.assertGreater(after.trace_strength, before_trace)
        self.assertGreater(float(metadata.get("predictive_reliability", 0.0)), 0.5)
        self.assertEqual(int(metadata.get("confirmed_count", 0)), 1)

    def test_violated_verification_adds_contradiction_burden(self) -> None:
        agent = SegmentAgent(rng=random.Random(22))
        entry_id = _store_episode(agent)
        before = agent.memory_store.get(entry_id)
        assert before is not None
        before_counterevidence = before.counterevidence_count

        agent.prediction_ledger.predictions.append(
            PredictionHypothesis(
                prediction_id="pred:test:violate",
                created_tick=1,
                last_updated_tick=1,
                source_module="test",
                prediction_type="environment_state",
                target_channels=("danger",),
                expected_state={"danger": 0.18},
                confidence=0.72,
                expected_horizon=1,
                semantic_provenance={
                    "linked_memory_ids": [entry_id],
                    "committed_memory_ids": [entry_id],
                    "auxiliary_memory_ids": [],
                    "linked_path_ids": [],
                },
            )
        )
        agent.verification_loop.refresh_targets(
            tick=2,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )
        update = agent.verification_loop.process_observation(
            tick=2,
            observation={"danger": 0.98},
            ledger=agent.prediction_ledger,
            source="runtime_observation",
            subject_state=agent.subject_state,
        )
        reports = agent.apply_memory_credit_signals(update.memory_credit_signals, tick=2)
        self.assertTrue(reports)

        after = agent.memory_store.get(entry_id)
        assert after is not None
        metadata = dict(after.compression_metadata or {}).get("m17_memory_credit", {})
        self.assertGreater(after.counterevidence_count, before_counterevidence)
        self.assertGreater(float(metadata.get("contradiction_burden", 0.0)), 0.0)
        self.assertEqual(int(metadata.get("violated_count", 0)), 1)

    def test_unlinked_memory_receives_no_credit(self) -> None:
        agent = SegmentAgent(rng=random.Random(23))
        linked_id = _store_episode(agent, cycle=1)
        unlinked_id = _store_episode(agent, cycle=2)

        agent.prediction_ledger.predictions.append(_prediction(linked_id, prediction_id="pred:test:linked"))
        agent.verification_loop.refresh_targets(
            tick=3,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )
        update = agent.verification_loop.process_observation(
            tick=3,
            observation={"danger": 0.70},
            ledger=agent.prediction_ledger,
            source="runtime_observation",
            subject_state=agent.subject_state,
        )
        agent.apply_memory_credit_signals(update.memory_credit_signals, tick=3)

        linked = agent.memory_store.get(linked_id)
        unlinked = agent.memory_store.get(unlinked_id)
        assert linked is not None and unlinked is not None
        linked_meta = dict(linked.compression_metadata or {}).get("m17_memory_credit", {})
        unlinked_meta = dict(unlinked.compression_metadata or {}).get("m17_memory_credit", {})
        self.assertEqual(int(linked_meta.get("confirmed_count", 0)), 1)
        self.assertEqual(int(unlinked_meta.get("confirmed_count", 0)), 0)
        self.assertEqual(int(unlinked_meta.get("violated_count", 0)), 0)

    def test_duplicate_credit_signal_is_idempotent(self) -> None:
        agent = SegmentAgent(rng=random.Random(24))
        entry_id = _store_episode(agent)

        agent.prediction_ledger.predictions.append(_prediction(entry_id, prediction_id="pred:test:idempotent"))
        agent.verification_loop.refresh_targets(
            tick=2,
            ledger=agent.prediction_ledger,
            subject_state=agent.subject_state,
        )
        update = agent.verification_loop.process_observation(
            tick=2,
            observation={"danger": 0.70},
            ledger=agent.prediction_ledger,
            source="runtime_observation",
            subject_state=agent.subject_state,
        )
        first = agent.apply_memory_credit_signals(update.memory_credit_signals, tick=2)
        second = agent.apply_memory_credit_signals(update.memory_credit_signals, tick=2)

        self.assertEqual(first[0]["applied_ids"], [entry_id])
        self.assertIn(entry_id, second[0]["skipped_ids"])
        self.assertEqual(
            second[0]["skipped_reasons"][entry_id],
            "duplicate_credit_application",
        )


if __name__ == "__main__":
    unittest.main()
