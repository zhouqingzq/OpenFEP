from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.memory_credit import MemoryCreditSignal
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_store import MemoryStore


def _entry(
    entry_id: str,
    *,
    cycle: int,
    action: str,
    outcome: str,
    semantic_tags: list[str],
    context_tags: list[str],
    salience: float = 0.55,
    novelty: float = 0.22,
    risk: float = 0.18,
    preferred_probability: float = 0.72,
    predicted_effects: dict[str, float] | None = None,
    state_vector: list[float] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"{action} tends toward {outcome}",
        memory_class=MemoryClass.EPISODIC,
        store_level=StoreLevel.MID,
        source_type=SourceType.EXPERIENCE,
        created_at=cycle,
        last_accessed=cycle,
        valence=0.25,
        arousal=0.30,
        encoding_attention=0.45,
        novelty=novelty,
        relevance_goal=0.52,
        relevance_threat=0.48,
        relevance_self=0.18,
        relevance_social=0.08,
        relevance_reward=0.32,
        relevance=0.54,
        salience=salience,
        trace_strength=0.52,
        accessibility=0.51,
        abstractness=0.12,
        source_confidence=0.88,
        reality_confidence=0.82,
        semantic_tags=list(semantic_tags),
        context_tags=list(context_tags),
        anchor_slots={
            "time": str(cycle),
            "place": "cave",
            "agents": "self",
            "action": action,
            "outcome": outcome,
        },
        mood_context="alert",
        state_vector=list(state_vector or [0.2, 0.8, 0.1, 0.7]),
        compression_metadata={
            "legacy_template": {
                "action": action,
                "predicted_outcome": outcome,
                "preferred_probability": preferred_probability,
                "risk": risk,
                "observation": {
                    "food": 0.20,
                    "danger": 0.78,
                    "novelty": 0.15,
                    "shelter": 0.72,
                    "temperature": 0.50,
                    "social": 0.18,
                },
                "errors": {
                    "danger": 0.34,
                    "shelter": 0.20,
                    "food": -0.12,
                },
                "outcome": dict(
                    predicted_effects
                    or {
                        "energy_delta": 0.04,
                        "stress_delta": -0.12,
                        "free_energy_delta": 0.18,
                    }
                ),
            }
        },
    )


def _apply_signal(
    store: MemoryStore,
    entry_id: str,
    *,
    prediction_id: str,
    outcome: str,
    free_energy_delta: float,
    contradiction_score: float = 0.0,
) -> None:
    store.apply_memory_credit(
        MemoryCreditSignal(
            linked_prediction_id=prediction_id,
            linked_memory_ids=(entry_id,),
            linked_path_ids=(),
            outcome=outcome,
            support_score=0.92 if outcome == "confirmed" else 0.18,
            contradiction_score=contradiction_score,
            prediction_error_delta=free_energy_delta,
            free_energy_delta=free_energy_delta,
            confidence_weight=0.78,
            source_module="test",
        ),
        tick=10,
    )


class TestM178PathFirstMemorySubstrate(unittest.TestCase):
    def test_repeated_confirmed_episodes_create_memory_path(self) -> None:
        store = MemoryStore()
        a = _entry("ep:a", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "shelter"], context_tags=["danger", "shelter"])
        b = _entry("ep:b", cycle=2, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "cover"], context_tags=["danger", "cover"], state_vector=[0.22, 0.78, 0.12, 0.69])
        store.add(a)
        store.add(b)
        _apply_signal(store, a.id, prediction_id="pred:a", outcome="confirmed", free_energy_delta=0.36)
        _apply_signal(store, b.id, prediction_id="pred:b", outcome="confirmed", free_energy_delta=0.30)

        self.assertEqual(len(store.memory_paths), 1)
        path = store.memory_paths[0]
        self.assertEqual(path.dominant_action, "hide")
        self.assertGreater(path.path_quality, 0.45)
        self.assertEqual(path.confirmation_count, 2)
        self.assertCountEqual(path.source_episode_ids, ["ep:a", "ep:b"])

    def test_single_episode_does_not_create_path_without_repeated_support(self) -> None:
        store = MemoryStore()
        single = _entry("ep:solo", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger"], context_tags=["danger"])
        store.add(single)
        _apply_signal(store, single.id, prediction_id="pred:solo", outcome="confirmed", free_energy_delta=0.40)
        self.assertEqual(store.memory_paths, [])

    def test_path_quality_tracks_confirmation_and_violation_history(self) -> None:
        store = MemoryStore()
        a = _entry("ep:q1", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "shelter"], context_tags=["danger", "shelter"])
        b = _entry("ep:q2", cycle=2, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "cover"], context_tags=["danger", "cover"])
        store.add(a)
        store.add(b)
        _apply_signal(store, a.id, prediction_id="pred:q1", outcome="confirmed", free_energy_delta=0.38)
        _apply_signal(store, b.id, prediction_id="pred:q2", outcome="confirmed", free_energy_delta=0.34)
        before = store.memory_paths[0].path_quality

        _apply_signal(store, a.id, prediction_id="pred:q1b", outcome="violated", free_energy_delta=-0.30, contradiction_score=0.85)
        after = store.memory_paths[0].path_quality

        self.assertLess(after, before)
        self.assertGreater(store.memory_paths[0].violation_count, 0)

    def test_path_can_be_negative_or_cautionary_not_only_positive(self) -> None:
        store = MemoryStore()
        a = _entry(
            "ep:n1",
            cycle=1,
            action="forage",
            outcome="predator_contact",
            semantic_tags=["forage", "danger", "food"],
            context_tags=["danger", "food"],
            risk=0.72,
            predicted_effects={"energy_delta": -0.08, "stress_delta": 0.22, "free_energy_delta": -0.30},
        )
        b = _entry(
            "ep:n2",
            cycle=2,
            action="forage",
            outcome="predator_contact",
            semantic_tags=["forage", "danger", "food"],
            context_tags=["danger", "exposed"],
            risk=0.76,
            predicted_effects={"energy_delta": -0.06, "stress_delta": 0.25, "free_energy_delta": -0.24},
            state_vector=[0.45, 0.85, 0.12, 0.20],
        )
        store.add(a)
        store.add(b)
        _apply_signal(store, a.id, prediction_id="pred:n1", outcome="violated", free_energy_delta=-0.28, contradiction_score=0.90)
        _apply_signal(store, b.id, prediction_id="pred:n2", outcome="violated", free_energy_delta=-0.26, contradiction_score=0.88)

        self.assertEqual(len(store.memory_paths), 1)
        self.assertIn(store.memory_paths[0].path_polarity, {"negative", "cautionary"})

    def test_live_runtime_consumer_can_use_path_without_single_raw_episode(self) -> None:
        agent = SegmentAgent(rng=random.Random(31))
        store = agent.memory_store
        assert store is not None
        a = _entry("ep:p1", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "shelter"], context_tags=["danger", "shelter"])
        b = _entry("ep:p2", cycle=2, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "cover"], context_tags=["danger", "cover"])
        store.add(a)
        store.add(b)
        _apply_signal(store, a.id, prediction_id="pred:p1", outcome="confirmed", free_energy_delta=0.36)
        _apply_signal(store, b.id, prediction_id="pred:p2", outcome="confirmed", free_energy_delta=0.34)
        agent.sync_memory_awareness_to_long_term_memory()

        observed = {"food": 0.24, "danger": 0.76, "novelty": 0.18, "shelter": 0.74, "temperature": 0.50, "social": 0.15}
        baseline_prediction = {"food": 0.30, "danger": 0.40, "novelty": 0.22, "shelter": 0.48, "temperature": 0.50, "social": 0.20}
        baseline_errors = {
            key: observed.get(key, 0.0) - baseline_prediction.get(key, 0.0)
            for key in observed
        }
        query = agent._decision_retrieval_query(observed, baseline_prediction, baseline_errors)
        active_paths = store.retrieve_paths(query, k=3)
        agent.last_retrieval_result = {"active_paths": active_paths}

        memory_context = agent._build_memory_context(
            observed=observed,
            baseline_prediction=baseline_prediction,
            errors=baseline_errors,
            similar_memories=[],
        )

        self.assertTrue(memory_context["memory_hit"])
        self.assertEqual(memory_context["retrieved_episode_ids"], [])
        self.assertTrue(memory_context["active_path_ids"])
        self.assertIn("hide", memory_context["actions"])

        refined = agent.world_model.refine_action_prediction(
            action="hide",
            projected_snapshot={
                "observation": dict(observed),
                "prediction": dict(baseline_prediction),
                "errors": dict(baseline_errors),
                "body_state": {"energy": 0.80, "stress": 0.22, "fatigue": 0.18, "temperature": 0.50},
            },
            predicted_effects={"energy_delta": 0.0, "stress_delta": 0.0},
            predicted_outcome="neutral",
            preferred_probability=0.40,
            risk=0.35,
            predicted_error=0.30,
            memory_context=memory_context,
        )
        self.assertTrue(refined["applied_memory"])

    def test_path_preserves_source_episode_ids_for_audit(self) -> None:
        store = MemoryStore()
        a = _entry("ep:audit1", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "shelter"], context_tags=["danger", "shelter"])
        b = _entry("ep:audit2", cycle=2, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "cover"], context_tags=["danger", "cover"])
        store.add(a)
        store.add(b)
        _apply_signal(store, a.id, prediction_id="pred:audit1", outcome="confirmed", free_energy_delta=0.32)
        _apply_signal(store, b.id, prediction_id="pred:audit2", outcome="confirmed", free_energy_delta=0.31)

        path = store.memory_paths[0]
        self.assertCountEqual(path.source_episode_ids, ["ep:audit1", "ep:audit2"])
        self.assertCountEqual(path.source_memory_ids, ["ep:audit1", "ep:audit2"])

    def test_high_salience_episode_does_not_alone_dominate_path_quality(self) -> None:
        store = MemoryStore()
        loud = _entry(
            "ep:loud",
            cycle=1,
            action="forage",
            outcome="resource_gain",
            semantic_tags=["forage", "food", "reward"],
            context_tags=["food", "open"],
            salience=0.98,
        )
        c1 = _entry(
            "ep:c1",
            cycle=2,
            action="forage",
            outcome="predator_contact",
            semantic_tags=["forage", "danger", "food"],
            context_tags=["danger", "food"],
            risk=0.70,
            predicted_effects={"energy_delta": -0.05, "stress_delta": 0.20, "free_energy_delta": -0.22},
        )
        c2 = _entry(
            "ep:c2",
            cycle=3,
            action="forage",
            outcome="predator_contact",
            semantic_tags=["forage", "danger", "food"],
            context_tags=["danger", "exposed"],
            risk=0.74,
            predicted_effects={"energy_delta": -0.06, "stress_delta": 0.24, "free_energy_delta": -0.20},
            state_vector=[0.44, 0.82, 0.18, 0.18],
        )
        store.add(loud)
        store.add(c1)
        store.add(c2)
        _apply_signal(store, loud.id, prediction_id="pred:loud", outcome="confirmed", free_energy_delta=0.35)
        _apply_signal(store, c1.id, prediction_id="pred:c1", outcome="violated", free_energy_delta=-0.20, contradiction_score=0.84)
        _apply_signal(store, c2.id, prediction_id="pred:c2", outcome="violated", free_energy_delta=-0.18, contradiction_score=0.82)

        path = store.memory_paths[0]
        self.assertLess(path.path_quality, 0.45)
        self.assertIn(path.path_polarity, {"negative", "cautionary"})

    def test_old_snapshots_without_paths_remain_loadable(self) -> None:
        store = MemoryStore()
        store.add(_entry("ep:snap1", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger"], context_tags=["danger"]))
        payload = store.to_dict()
        payload.pop("memory_paths", None)

        restored = MemoryStore.from_dict(payload)
        self.assertEqual(len(restored.entries), 1)
        self.assertEqual(restored.memory_paths, [])


if __name__ == "__main__":
    unittest.main()
