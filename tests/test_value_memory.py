from __future__ import annotations

from segmentum.dialogue.runtime.manager import PersonaManager
from segmentum.memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from segmentum.memory_retrieval import RetrievalQuery
from segmentum.memory_store import MemoryStore
from segmentum.value_memory import (
    QUARANTINE_KIND,
    ValueMemoryCalibrator,
    ValueMemoryCandidate,
    ValueMemoryEvaluator,
    ValueMemoryExtractor,
    ValueMemoryQuarantinePolicy,
)


def test_structural_material_extracts_without_surface_value_keywords() -> None:
    material = "When requirements are uncertain, state a verifiable hypothesis and make the minimum implementation to test it."

    evaluations = ValueMemoryExtractor().extract(material, source_id="case")

    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation.candidate_kind == "skill_candidate"
    assert evaluation.future_path_utility > 0.35
    assert evaluation.score_breakdown.activation_clarity >= 0.5
    assert evaluation.score_breakdown.cost_reduction > 0.0
    assert evaluation.candidate.trigger_conditions
    assert evaluation.candidate.action_steps
    assert evaluation.candidate.applicability_bounds
    assert evaluation.candidate.disable_conditions


def test_chinese_structural_material_extracts_with_clear_boundaries() -> None:
    material = "当需求不确定时，先提出可验证假设，再做最小实现来验证它。"

    evaluations = ValueMemoryExtractor().extract(material, source_id="zh-case")

    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation.candidate_kind == "skill_candidate"
    assert evaluation.future_path_utility > 0.35
    assert evaluation.candidate.trigger_conditions
    assert evaluation.candidate.action_steps
    assert evaluation.candidate.applicability_bounds
    assert evaluation.candidate.disable_conditions


def test_surface_value_words_without_action_structure_are_quarantined() -> None:
    material = "Success, risk, trust, and protect are just inspiring words on a poster."

    evaluations = ValueMemoryExtractor().extract(material, source_id="poster")

    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation.candidate_kind in {QUARANTINE_KIND, "rejected_candidate"}
    assert evaluation.future_path_utility < 0.25
    assert evaluation.quarantine_reasons or evaluation.rejection_reasons


def test_missing_semantic_boundary_rejects_or_quarantines_candidate() -> None:
    candidate = ValueMemoryCandidate(
        summary="Ask better questions before coding.",
        trigger_conditions=["when requirements are uncertain"],
        action_steps=["ask one question"],
        expected_benefits=["reduce mistakes"],
        applicability_bounds=[],
        disable_conditions=[],
        evidence_refs=["requirements are uncertain"],
        source_material="requirements are uncertain",
    )

    evaluation = ValueMemoryEvaluator().evaluate(candidate)

    assert evaluation.candidate_kind == "rejected_candidate"
    assert "missing_boundary:applicability_bounds" in evaluation.rejection_reasons
    assert "missing_boundary:disable_conditions" in evaluation.rejection_reasons


def test_single_evidence_absolute_rule_raises_overgeneralization_risk() -> None:
    candidate = ValueMemoryCandidate(
        summary="Always use this one example for every task.",
        trigger_conditions=["when doing any task"],
        action_steps=["copy the example"],
        expected_benefits=["fast decision"],
        applicability_bounds=["all tasks"],
        disable_conditions=["none"],
        evidence_refs=["one example"],
        source_material="one example",
    )

    evaluation = ValueMemoryEvaluator().evaluate(candidate)

    assert evaluation.score_breakdown.overgeneralization_risk >= 0.65
    assert evaluation.candidate_kind == QUARANTINE_KIND
    assert "overgeneralization_risk_high" in evaluation.quarantine_reasons


def test_quarantine_policy_release_and_retire_rules() -> None:
    policy = ValueMemoryQuarantinePolicy()
    releasable = {
        "candidate_kind": QUARANTINE_KIND,
        "quarantine_reasons": ["evidence_strength_below_threshold"],
        "score_breakdown": {
            "evidence_strength": 0.72,
            "activation_clarity": 0.68,
            "overgeneralization_risk": 0.20,
            "maintenance_cost": 0.10,
        },
    }
    stale = {
        "candidate_kind": QUARANTINE_KIND,
        "quarantine_reasons": ["activation_clarity_below_threshold"],
        "score_breakdown": {"maintenance_cost": 0.20},
    }

    assert policy.decide(releasable).action == "release"
    retired = policy.decide(stale, observed={"no_activation_audit_count": 3})
    assert retired.action == "retire"
    assert "stale_without_activation" in retired.reasons


def test_calibrator_reports_drift_without_mutating_weights() -> None:
    report = ValueMemoryCalibrator().audit(
        [
            {
                "predicted_utility": 0.80,
                "observed_utility": 0.30,
                "score_breakdown": {
                    "future_reuse_gain": 0.90,
                    "cost_reduction": 0.80,
                    "error_avoidance_gain": 0.70,
                },
            },
            {
                "predicted_utility": 0.60,
                "observed_utility": 0.25,
                "score_breakdown": {
                    "future_reuse_gain": 0.70,
                    "cost_reduction": 0.75,
                    "error_avoidance_gain": 0.65,
                },
            },
        ]
    )

    payload = report.to_dict()
    assert payload["drift"] < 0.0
    assert payload["recommended_weight_updates"]
    assert payload["audit_record"]["requires_manual_application"] is True


def _entry_with_value_memory(
    *,
    entry_id: str,
    kind: str,
    utility: float,
    salience: float = 0.25,
    retrieval_count: int = 0,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"value memory {entry_id}",
        memory_class=MemoryClass.SEMANTIC,
        store_level=StoreLevel.SHORT,
        source_type=SourceType.INFERENCE,
        created_at=1,
        last_accessed=10,
        valence=0.0,
        arousal=0.1,
        encoding_attention=0.2,
        novelty=0.1,
        relevance_goal=0.1,
        relevance_threat=0.0,
        relevance_self=0.0,
        relevance_social=0.0,
        relevance_reward=0.0,
        relevance=0.1,
        salience=salience,
        trace_strength=0.6,
        accessibility=0.6,
        abstractness=0.75,
        source_confidence=0.9,
        reality_confidence=0.8,
        semantic_tags=["value_memory", "uncertain_requirements"],
        context_tags=["planning"],
        retrieval_count=retrieval_count,
        support_count=2,
        compression_metadata={
            "value_memory": {
                "candidate_kind": kind,
                "future_path_utility": utility,
                "score_breakdown": {
                    "future_reuse_gain": utility,
                    "activation_clarity": 0.7,
                    "evidence_strength": 0.7,
                    "overgeneralization_risk": 0.2,
                    "maintenance_cost": 0.1,
                },
            },
            "m45_internal": {
                "last_decay_cycle": 1,
                "decay_base_trace_strength": 1.0,
                "decay_base_accessibility": 1.0,
            },
        },
    )


def test_quarantined_value_memory_is_excluded_from_ordinary_retrieval() -> None:
    active = _entry_with_value_memory(entry_id="active", kind="skill_candidate", utility=0.75)
    quarantined = _entry_with_value_memory(entry_id="quarantined", kind=QUARANTINE_KIND, utility=0.90)
    store = MemoryStore(entries=[quarantined, active])

    result = store.retrieve(
        RetrievalQuery(semantic_tags=["value_memory"], context_tags=["planning"], reference_cycle=12),
        k=5,
    )

    assert [candidate.entry_id for candidate in result.candidates] == ["active"]
    assert [entry.id for entry in store.query_by_tags(["value_memory"], k=5)] == ["active"]


def test_future_path_utility_can_promote_without_old_value_score() -> None:
    high_utility = _entry_with_value_memory(
        entry_id="high-utility",
        kind="skill_candidate",
        utility=0.90,
        salience=0.42,
        retrieval_count=1,
    )
    old_value_only = _entry_with_value_memory(
        entry_id="old-value-only",
        kind=QUARANTINE_KIND,
        utility=0.0,
        salience=0.42,
        retrieval_count=1,
    )
    old_value_only.valence = 1.0

    store = MemoryStore()
    store.add(high_utility)
    store.add(old_value_only)

    promoted = store.get("high-utility")
    retained = store.get("old-value-only")
    assert promoted is not None
    assert retained is not None
    assert promoted.store_level is StoreLevel.MID
    assert retained.store_level is StoreLevel.SHORT
    audit = dict(dict(promoted.compression_metadata or {}).get("m47_promotion_audit", {}))
    assert audit["score_breakdown"]["future_path_utility_signal"] > 0.0


def test_create_from_description_writes_value_memory_candidate(tmp_path) -> None:
    manager = PersonaManager(storage_dir=tmp_path / "personas")

    agent = manager.create_from_description(
        "When requirements are uncertain, state a verifiable hypothesis and make the minimum implementation to test it."
    )

    entries = agent.long_term_memory.ensure_memory_store().entries
    value_entries = [
        entry for entry in entries
        if "value_memory" in dict(entry.compression_metadata or {})
    ]
    assert value_entries
    payload = dict(value_entries[0].compression_metadata or {})["value_memory"]
    assert payload["candidate_kind"] == "skill_candidate"
    assert payload["future_path_utility"] > 0.35
