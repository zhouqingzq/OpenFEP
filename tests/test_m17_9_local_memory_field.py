from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.memory_credit import MemoryCreditSignal
from segmentum.memory_field import build_local_memory_field
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_store import MemoryStore
from segmentum.memory_retrieval import RetrievalQuery


def _entry(
    entry_id: str,
    *,
    cycle: int,
    action: str,
    outcome: str,
    semantic_tags: list[str],
    context_tags: list[str],
    state_vector: list[float],
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"{action} in {outcome}",
        memory_class=MemoryClass.EPISODIC,
        store_level=StoreLevel.MID,
        source_type=SourceType.EXPERIENCE,
        created_at=cycle,
        last_accessed=cycle,
        valence=0.20,
        arousal=0.30,
        encoding_attention=0.45,
        novelty=0.20,
        relevance_goal=0.55,
        relevance_threat=0.42,
        relevance_self=0.15,
        relevance_social=0.12,
        relevance_reward=0.28,
        relevance=0.56,
        salience=0.56,
        trace_strength=0.50,
        accessibility=0.50,
        abstractness=0.12,
        source_confidence=0.86,
        reality_confidence=0.82,
        semantic_tags=list(semantic_tags),
        context_tags=list(context_tags),
        anchor_slots={
            "time": str(cycle),
            "place": "ridge",
            "agents": "self",
            "action": action,
            "outcome": outcome,
        },
        mood_context="alert",
        state_vector=list(state_vector),
        compression_metadata={
            "legacy_template": {
                "action": action,
                "predicted_outcome": outcome,
                "preferred_probability": 0.68,
                "risk": 0.22,
                "observation": {
                    "food": 0.22,
                    "danger": 0.78,
                    "novelty": 0.26,
                    "shelter": 0.62,
                    "temperature": 0.50,
                    "social": 0.18,
                },
                "errors": {"danger": 0.32, "shelter": 0.18, "food": -0.08},
                "outcome": {"energy_delta": 0.04, "stress_delta": -0.08, "free_energy_delta": 0.18},
            }
        },
    )


def _credit(
    *,
    prediction_id: str,
    entry_id: str,
    outcome: str,
    free_energy_delta: float,
    contradiction_score: float = 0.0,
) -> MemoryCreditSignal:
    return MemoryCreditSignal(
        linked_prediction_id=prediction_id,
        linked_memory_ids=(entry_id,),
        linked_path_ids=(),
        outcome=outcome,
        support_score=0.90 if outcome == "confirmed" else 0.16,
        contradiction_score=contradiction_score,
        prediction_error_delta=free_energy_delta,
        free_energy_delta=free_energy_delta,
        confidence_weight=0.76,
        source_module="test",
    )


def _field_path(
    path_id: str,
    *,
    action: str,
    proposal_score: float,
    path_quality: float,
    support_count: int,
    utility: float,
    risk: float,
    surprise: float,
    polarity: str,
    channels: list[str],
    source_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "path_id": path_id,
        "dominant_action": action,
        "source_episode_ids": list(source_ids or [path_id.replace("path", "ep")]),
        "source_memory_ids": list(source_ids or [path_id.replace("path", "ep")]),
        "proposal_score": proposal_score,
        "retrieval_score": proposal_score,
        "path_quality": path_quality,
        "path_polarity": polarity,
        "support_count": support_count,
        "cue_signature": {
            "semantic_tags": [action, *channels],
            "context_tags": list(channels),
            "sensitive_channels": list(channels),
        },
        "outcome_profile": {
            "outcome_distribution": {"safe_escape": 0.70},
            "predicted_effects": {"energy_delta": 0.02, "stress_delta": -0.10},
            "preferred_probability": 0.66,
            "future_path_utility": utility,
        },
        "risk_profile": {
            "mean_risk": risk,
            "max_risk": risk + 0.05,
            "contradiction_burden": 0.24 if polarity != "positive" else 0.02,
            "maintenance_cost": 0.12,
            "caution_score": max(risk, surprise),
        },
        "expected_surprise_profile": {
            "mean_prediction_error": surprise,
            "mean_free_energy_delta": utility - surprise,
            "error_avoidance_gain": max(0.0, utility - 0.08),
        },
        "score_breakdown": {"cue_match": proposal_score, "semantic_overlap": proposal_score * 0.8},
    }


class TestM179LocalMemoryField(unittest.TestCase):
    def test_coarse_retrieval_is_only_neighborhood_proposal(self) -> None:
        store = MemoryStore()
        first = _entry("ep:1", cycle=1, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "shelter"], context_tags=["danger", "shelter"], state_vector=[0.20, 0.82, 0.12, 0.70])
        second = _entry("ep:2", cycle=2, action="hide", outcome="safe_escape", semantic_tags=["hide", "danger", "cover"], context_tags=["danger", "cover"], state_vector=[0.22, 0.80, 0.10, 0.68])
        store.add(first)
        store.add(second)
        store.apply_memory_credit(_credit(prediction_id="pred:1", entry_id="ep:1", outcome="confirmed", free_energy_delta=0.34), tick=4)
        store.apply_memory_credit(_credit(prediction_id="pred:2", entry_id="ep:2", outcome="confirmed", free_energy_delta=0.32), tick=5)

        query = RetrievalQuery(context_tags=["danger", "shelter"], content_keywords=["danger", "shelter"], state_vector=[0.21, 0.81, 0.10, 0.69], reference_cycle=5)
        proposal = store.propose_path_neighborhood(query, k=3)
        self.assertTrue(proposal)
        self.assertNotIn("potential_by_channel", proposal[0])
        field = build_local_memory_field(
            proposal,
            baseline_prediction={"danger": 0.42, "shelter": 0.48, "food": 0.30, "novelty": 0.22, "temperature": 0.50, "social": 0.20},
            errors={"danger": 0.34, "shelter": 0.20, "food": -0.10},
            body_state={"energy": 0.78, "stress": 0.20, "fatigue": 0.18, "temperature": 0.50},
        )
        self.assertIsNotNone(field)
        assert field is not None
        self.assertEqual(field.member_path_ids, [proposal[0]["path_id"]])

    def test_local_field_builds_from_multiple_paths_without_global_search(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:a", action="hide", proposal_score=0.40, path_quality=0.44, support_count=2, utility=0.34, risk=0.16, surprise=0.12, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:b", action="scan", proposal_score=0.35, path_quality=0.48, support_count=2, utility=0.52, risk=0.14, surprise=0.10, polarity="positive", channels=["danger", "novelty"]),
                _field_path("path:c", action="forage", proposal_score=0.28, path_quality=0.32, support_count=2, utility=0.12, risk=0.44, surprise=0.30, polarity="cautionary", channels=["food", "danger"]),
            ],
            baseline_prediction={"danger": 0.48, "novelty": 0.30, "food": 0.22, "shelter": 0.55, "temperature": 0.50, "social": 0.18},
            errors={"danger": 0.28, "novelty": 0.18, "food": -0.10, "shelter": 0.14},
            body_state={"energy": 0.76, "stress": 0.24, "fatigue": 0.16, "temperature": 0.50},
        )
        assert field is not None
        self.assertEqual(len(field.member_path_ids), 3)
        self.assertGreater(field.effective_member_count, 1.5)
        self.assertIn("danger", field.potential_by_channel)

    def test_duplicate_members_do_not_fake_richer_field(self) -> None:
        duplicate = _field_path(
            "path:dup",
            action="hide",
            proposal_score=0.42,
            path_quality=0.46,
            support_count=2,
            utility=0.36,
            risk=0.12,
            surprise=0.10,
            polarity="positive",
            channels=["danger", "shelter"],
            source_ids=["ep:dup1", "ep:dup2"],
        )
        field = build_local_memory_field(
            [duplicate, dict(duplicate)],
            baseline_prediction={"danger": 0.44, "shelter": 0.50, "food": 0.28, "novelty": 0.20, "temperature": 0.50, "social": 0.18},
            errors={"danger": 0.30, "shelter": 0.18},
            body_state={"energy": 0.80, "stress": 0.18, "fatigue": 0.14, "temperature": 0.50},
        )
        assert field is not None
        self.assertEqual(len(field.member_path_ids), 1)
        self.assertLess(field.effective_member_count, 1.1)

    def test_conflicting_members_raise_conflict_density_or_ridge_strength(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:p", action="hide", proposal_score=0.34, path_quality=0.40, support_count=2, utility=0.30, risk=0.14, surprise=0.10, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:n", action="hide", proposal_score=0.33, path_quality=0.36, support_count=2, utility=0.08, risk=0.56, surprise=0.42, polarity="negative", channels=["danger", "shelter"]),
            ],
            baseline_prediction={"danger": 0.52, "shelter": 0.45, "food": 0.24, "novelty": 0.22, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.36, "shelter": 0.10},
            body_state={"energy": 0.74, "stress": 0.28, "fatigue": 0.18, "temperature": 0.50},
        )
        assert field is not None
        self.assertGreater(field.conflict_density, 0.15)
        self.assertGreater(field.ridge_strength, 0.05)

    def test_field_consumer_reads_field_summary_not_only_best_single_path(self) -> None:
        agent = SegmentAgent(rng=random.Random(44))
        active_paths = [
            _field_path("path:forage", action="forage", proposal_score=0.34, path_quality=0.30, support_count=2, utility=0.10, risk=0.52, surprise=0.36, polarity="cautionary", channels=["food", "danger"]),
            _field_path("path:hide1", action="hide", proposal_score=0.24, path_quality=0.46, support_count=2, utility=0.84, risk=0.08, surprise=0.08, polarity="positive", channels=["danger", "shelter"]),
            _field_path("path:hide2", action="hide", proposal_score=0.22, path_quality=0.44, support_count=2, utility=0.80, risk=0.10, surprise=0.10, polarity="positive", channels=["danger", "shelter"]),
            _field_path("path:scan", action="scan", proposal_score=0.28, path_quality=0.92, support_count=2, utility=0.94, risk=0.06, surprise=0.06, polarity="positive", channels=["danger", "novelty"]),
        ]
        local_field = build_local_memory_field(
            active_paths,
            baseline_prediction={"danger": 0.60, "novelty": 0.18, "food": 0.25, "shelter": 0.42, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.38, "novelty": 0.20, "food": -0.06, "shelter": 0.22},
            body_state={"energy": 0.72, "stress": 0.30, "fatigue": 0.18, "temperature": 0.50},
        )
        assert local_field is not None
        agent.last_retrieval_result = {
            "active_paths": active_paths,
            "local_field": local_field.to_dict(),
        }
        memory_context = agent._build_memory_context(
            observed={"food": 0.22, "danger": 0.76, "novelty": 0.28, "shelter": 0.64, "temperature": 0.50, "social": 0.16},
            baseline_prediction={"food": 0.28, "danger": 0.46, "novelty": 0.14, "shelter": 0.40, "temperature": 0.50, "social": 0.18},
            errors={"food": -0.06, "danger": 0.30, "novelty": 0.14, "shelter": 0.24, "social": -0.02},
            similar_memories=[],
        )
        self.assertIn("local_field", memory_context)
        self.assertIn("scan", memory_context["actions"])
        refined = agent.world_model.refine_action_prediction(
            action="scan",
            projected_snapshot={
                "observation": {"food": 0.22, "danger": 0.76, "novelty": 0.28, "shelter": 0.64, "temperature": 0.50, "social": 0.16},
                "prediction": {"food": 0.28, "danger": 0.46, "novelty": 0.14, "shelter": 0.40, "temperature": 0.50, "social": 0.18},
                "errors": {"food": -0.06, "danger": 0.30, "novelty": 0.14, "shelter": 0.24, "social": -0.02},
                "body_state": {"energy": 0.72, "stress": 0.30, "fatigue": 0.18, "temperature": 0.50},
            },
            predicted_effects={"energy_delta": 0.0, "stress_delta": 0.0},
            predicted_outcome="neutral",
            preferred_probability=0.32,
            risk=0.42,
            predicted_error=0.34,
            memory_context=memory_context,
        )
        self.assertTrue(refined["applied_field"])
        self.assertIn("field_gradient_magnitude", refined["predicted_effects"])

    def test_field_required_decision_suppressed_when_best_single_would_trigger(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:single", action="hide", proposal_score=0.60, path_quality=0.58, support_count=2, utility=0.42, risk=0.10, surprise=0.08, polarity="positive", channels=["danger", "shelter"]),
            ],
            baseline_prediction={"danger": 0.46, "shelter": 0.52, "food": 0.24, "novelty": 0.18, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.24, "shelter": 0.10},
            body_state={"energy": 0.82, "stress": 0.18, "fatigue": 0.14, "temperature": 0.50},
        )
        assert field is not None
        self.assertEqual(field.counterfactual_audit["status"], "suppressed_best_single_equivalent")

    def test_field_required_decision_suppressed_when_naive_topk_would_trigger(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:forage", action="forage", proposal_score=0.46, path_quality=0.34, support_count=2, utility=0.14, risk=0.46, surprise=0.30, polarity="cautionary", channels=["food", "danger"]),
                _field_path("path:hide1", action="hide", proposal_score=0.26, path_quality=0.40, support_count=2, utility=0.54, risk=0.12, surprise=0.10, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:hide2", action="hide", proposal_score=0.24, path_quality=0.42, support_count=2, utility=0.52, risk=0.14, surprise=0.10, polarity="positive", channels=["danger", "shelter"]),
            ],
            baseline_prediction={"danger": 0.58, "shelter": 0.42, "food": 0.20, "novelty": 0.18, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.32, "food": -0.08, "shelter": 0.18},
            body_state={"energy": 0.74, "stress": 0.24, "fatigue": 0.16, "temperature": 0.50},
        )
        assert field is not None
        self.assertEqual(field.counterfactual_audit["best_single_action"], "forage")
        self.assertEqual(field.counterfactual_audit["naive_topk_action"], "hide")
        self.assertEqual(field.counterfactual_audit["field_selected_action"], "hide")
        self.assertEqual(field.counterfactual_audit["status"], "suppressed_naive_topk_equivalent")

    def test_local_field_can_change_bounded_downstream_decision(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:forage", action="forage", proposal_score=0.65, path_quality=0.34, support_count=2, utility=0.10, risk=0.52, surprise=0.36, polarity="cautionary", channels=["food", "danger"]),
                _field_path("path:hide1", action="hide", proposal_score=0.24, path_quality=0.46, support_count=2, utility=0.84, risk=0.08, surprise=0.08, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:hide2", action="hide", proposal_score=0.22, path_quality=0.44, support_count=2, utility=0.80, risk=0.10, surprise=0.10, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:scan", action="scan", proposal_score=0.05, path_quality=0.98, support_count=2, utility=1.00, risk=0.06, surprise=0.06, polarity="positive", channels=["danger", "novelty"]),
            ],
            baseline_prediction={"danger": 0.60, "novelty": 0.18, "food": 0.25, "shelter": 0.42, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.38, "novelty": 0.20, "food": -0.06, "shelter": 0.22},
            body_state={"energy": 0.72, "stress": 0.30, "fatigue": 0.18, "temperature": 0.50},
        )
        assert field is not None
        audit = field.counterfactual_audit
        self.assertEqual(audit["best_single_action"], "forage")
        self.assertEqual(audit["naive_topk_action"], "hide")
        self.assertEqual(audit["field_selected_action"], "scan")
        self.assertEqual(audit["status"], "field_required")
        self.assertGreater(audit["fe_advantage_vs_naive_topk"], 0.0)

    def test_field_divergent_without_fe_gain_is_audited_not_claimed_required(self) -> None:
        field = build_local_memory_field(
            [
                _field_path("path:forage", action="forage", proposal_score=0.7248370855143806, path_quality=0.4351938528215114, support_count=2, utility=0.1428688263770917, risk=0.42860724602167044, surprise=0.2678254082374536, polarity="cautionary", channels=["food", "danger"]),
                _field_path("path:scan", action="scan", proposal_score=0.025199224388401928, path_quality=1.289500489011584, support_count=2, utility=0.6942132700101633, risk=0.7097550198019936, surprise=0.9885417762445252, polarity="negative", channels=["danger", "novelty"]),
                _field_path("path:hide1", action="hide", proposal_score=0.2530763920361821, path_quality=0.3661170173728088, support_count=2, utility=0.2791626001089248, risk=0.06168506420253771, surprise=0.1764984396886649, polarity="positive", channels=["danger", "shelter"]),
                _field_path("path:hide2", action="hide", proposal_score=0.1959579015508609, path_quality=0.3215348892099113, support_count=2, utility=0.1517888137086126, risk=0.15722383156928438, surprise=0.15296099583607148, polarity="positive", channels=["danger", "shelter"]),
            ],
            baseline_prediction={"danger": 0.62, "novelty": 0.20, "food": 0.22, "shelter": 0.44, "temperature": 0.50, "social": 0.16},
            errors={"danger": 0.36, "novelty": 0.20, "food": -0.08, "shelter": 0.18},
            body_state={"energy": 0.70, "stress": 0.32, "fatigue": 0.20, "temperature": 0.50},
        )
        assert field is not None
        self.assertEqual(field.counterfactual_audit["status"], "field_divergent_no_gain")
        self.assertFalse(field.counterfactual_audit["field_required"])


if __name__ == "__main__":
    unittest.main()
