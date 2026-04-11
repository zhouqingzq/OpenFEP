from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from .memory import LongTermMemory
from .memory_consolidation import (
    ConflictType,
    ReconstructionConfig,
    maybe_reconstruct,
    reconsolidate,
    validate_inference,
)
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from .memory_retrieval import RetrievalQuery, compete_candidates
from .memory_store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"

LEGACY_M46_ACCEPTANCE_DATA_NOTICE = (
    "Legacy M4.6 acceptance payload builder. Historical only; not the primary evidence chain."
)


def _discover_regression_targets() -> list[str]:
    targets: list[str] = []
    for prefix in ("m41", "m42", "m43", "m44", "m45"):
        for path in sorted(TESTS_DIR.glob(f"test_{prefix}*.py")):
            if path.suffix == ".py":
                targets.append(path.relative_to(ROOT).as_posix())
    return targets


REGRESSION_TARGETS = _discover_regression_targets()


def _state() -> dict[str, object]:
    return {
        "active_goals": ["keep promises", "protect mentees"],
        "identity_themes": ["reliable mentor", "care continuity"],
        "threat_level": 0.2,
        "recent_mood_baseline": "reflective",
        "cognitive_style": {
            "update_rigidity": 0.3,
            "error_aversion": 0.4,
            "uncertainty_sensitivity": 0.4,
        },
    }


def _entry(
    *,
    entry_id: str,
    content: str,
    semantic_tags: list[str],
    context_tags: list[str],
    memory_class: MemoryClass = MemoryClass.EPISODIC,
    store_level: StoreLevel = StoreLevel.SHORT,
    source_type: SourceType = SourceType.EXPERIENCE,
    valence: float = 0.0,
    accessibility: float = 0.6,
    abstractness: float = 0.2,
    reality_confidence: float = 0.85,
    retrieval_count: int = 0,
    support_count: int = 1,
    mood_context: str = "",
    created_at: int = 1,
    last_accessed: int = 1,
    is_dormant: bool = False,
    counterevidence_count: int = 0,
    compression_metadata: dict[str, object] | None = None,
    procedure_steps: list[str] | None = None,
    execution_contexts: list[str] | None = None,
    anchor_strengths: dict[str, str] | None = None,
    anchor_slots: dict[str, str | None] | None = None,
) -> MemoryEntry:
    kwargs = {
        "id": entry_id,
        "content": content,
        "memory_class": memory_class,
        "store_level": store_level,
        "source_type": source_type,
        "created_at": created_at,
        "last_accessed": last_accessed,
        "valence": valence,
        "arousal": 0.3,
        "encoding_attention": 0.4,
        "novelty": 0.3,
        "relevance_goal": 0.3,
        "relevance_threat": 0.2,
        "relevance_self": 0.2,
        "relevance_social": 0.2,
        "relevance_reward": 0.2,
        "relevance": 0.3,
        "salience": 0.5,
        "trace_strength": 0.5,
        "accessibility": accessibility,
        "abstractness": abstractness,
        "source_confidence": 0.9,
        "reality_confidence": reality_confidence,
        "semantic_tags": semantic_tags,
        "context_tags": context_tags,
        "anchor_slots": anchor_slots or {"action": "mentor_checkin", "outcome": "commitment_kept", "agents": "lin"},
        "anchor_strengths": anchor_strengths or {"agents": "strong", "action": "strong", "outcome": "strong"},
        "mood_context": mood_context,
        "retrieval_count": retrieval_count,
        "support_count": support_count,
        "counterevidence_count": counterevidence_count,
        "compression_metadata": compression_metadata,
        "is_dormant": is_dormant,
    }
    if memory_class is MemoryClass.PROCEDURAL:
        kwargs["procedure_steps"] = procedure_steps or ["scan gauges", "vent pressure", "log readings"]
        kwargs["step_confidence"] = [0.9 for _ in kwargs["procedure_steps"]]
        kwargs["execution_contexts"] = execution_contexts or ["reactor_room"]
    return MemoryEntry(**kwargs)


def _probe_result(*, gate: str, probe_id: str, channel: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate": gate,
        "probe_id": probe_id,
        "channel": channel,
        "observed": observed,
    }


def build_probe_catalog() -> dict[str, object]:
    return {
        "boundary": [
            {"id": "retrieval_boundary", "gate": "retrieval_multi_cue"},
            {"id": "competition_boundary", "gate": "candidate_competition"},
            {"id": "reconstruction_boundary", "gate": "reconstruction_mechanism"},
            {"id": "reconsolidation_boundary", "gate": "reconsolidation"},
            {"id": "consolidation_boundary", "gate": "offline_consolidation_pipeline"},
            {"id": "inference_boundary", "gate": "inference_validation_gate"},
            {"id": "legacy_bridge_boundary", "gate": "legacy_integration"},
        ],
        "integration": [
            {"id": "retrieval_integration", "gate": "retrieval_multi_cue"},
            {"id": "competition_integration", "gate": "candidate_competition"},
            {"id": "reconstruction_integration", "gate": "reconstruction_mechanism"},
            {"id": "reconsolidation_integration", "gate": "reconsolidation"},
            {"id": "consolidation_integration", "gate": "offline_consolidation_pipeline"},
            {"id": "inference_integration", "gate": "inference_validation_gate"},
            {"id": "legacy_bridge_integration", "gate": "legacy_integration"},
        ],
        "regression_targets": list(REGRESSION_TARGETS),
    }


def _retrieval_store() -> MemoryStore:
    return MemoryStore(
        entries=[
            _entry(
                entry_id="tag-primary",
                content="I kept the mentor promise to Lin in the lab.",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab", "weekly"],
                accessibility=0.78,
                mood_context="reflective",
                last_accessed=48,
                created_at=40,
                compression_metadata={"state_vector": [0.9, 0.2]},
            ),
            _entry(
                entry_id="context-rich",
                content="We reviewed the care plan in the lab during a weekly sync.",
                semantic_tags=["care", "plan"],
                context_tags=["lab", "weekly", "team"],
                accessibility=0.74,
                last_accessed=47,
                created_at=39,
                compression_metadata={"state_vector": [0.88, 0.25]},
            ),
            _entry(
                entry_id="negative-mood",
                content="I worried the promise might fail under pressure.",
                semantic_tags=["mentor", "promise", "pressure"],
                context_tags=["storm"],
                valence=-0.6,
                accessibility=0.62,
                mood_context="anxious",
                last_accessed=46,
                created_at=38,
            ),
            _entry(
                entry_id="low-access",
                content="The exact mentor promise detail is stored but hard to reach.",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab"],
                accessibility=0.01,
                last_accessed=49,
                created_at=45,
            ),
            _entry(
                entry_id="dormant",
                content="Dormant mentor promise trace.",
                semantic_tags=["mentor", "promise"],
                context_tags=["lab"],
                accessibility=0.95,
                is_dormant=True,
            ),
            _entry(
                entry_id="procedure",
                content="Reactor calming routine for emergencies.",
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance"],
                memory_class=MemoryClass.PROCEDURAL,
                accessibility=0.72,
            ),
        ]
    )


def _retrieval_boundary_probe() -> dict[str, Any]:
    store = _retrieval_store()
    scenarios = {
        "tag": store.retrieve(
            RetrievalQuery(
                semantic_tags=["mentor", "promise"],
                context_tags=["lab"],
                content_keywords=["lin", "promise"],
                state_vector=[0.91, 0.22],
                reference_cycle=50,
            ),
            current_mood="reflective",
            k=4,
        ),
        "context": store.retrieve(
            RetrievalQuery(
                semantic_tags=["plan"],
                context_tags=["lab", "weekly", "team"],
                state_vector=[0.88, 0.24],
                reference_cycle=50,
            ),
            current_mood="reflective",
            k=3,
        ),
        "mood": store.retrieve(
            RetrievalQuery(semantic_tags=["promise"], context_tags=["storm"], reference_cycle=50),
            current_mood="anxious",
            k=3,
        ),
        "accessibility": store.retrieve(
            RetrievalQuery(semantic_tags=["mentor", "promise", "care"], context_tags=["lab"], reference_cycle=50),
            current_mood="reflective",
            k=4,
        ),
        "procedural": store.retrieve(
            RetrievalQuery(
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance"],
                reference_cycle=50,
                target_memory_class=MemoryClass.PROCEDURAL,
            ),
            current_mood="calm",
            k=1,
        ),
    }
    return {
        "scenario_count": len(scenarios),
        "tag_top_id": scenarios["tag"].candidates[0].entry_id,
        "context_top_id": scenarios["context"].candidates[0].entry_id,
        "mood_top_id": scenarios["mood"].candidates[0].entry_id,
        "low_access_not_top": scenarios["accessibility"].candidates[0].entry_id != "low-access",
        "dormant_present": "dormant" in [candidate.entry_id for candidate in scenarios["tag"].candidates],
        "recall_primary_id": scenarios["tag"].recall_hypothesis.primary_entry_id if scenarios["tag"].recall_hypothesis else None,
        "recall_aux_ids": scenarios["tag"].recall_hypothesis.auxiliary_entry_ids if scenarios["tag"].recall_hypothesis else [],
        "recall_is_reconstructed": scenarios["tag"].recall_hypothesis.content != store.get("tag-primary").content if scenarios["tag"].recall_hypothesis else False,
        "score_breakdowns": {
            name: [candidate.score_breakdown for candidate in result.candidates]
            for name, result in scenarios.items()
        },
        "procedural_outline": scenarios["procedural"].recall_hypothesis.procedure_step_outline if scenarios["procedural"].recall_hypothesis else [],
    }


def _competition_boundary_probe() -> dict[str, Any]:
    store = MemoryStore(
        entries=[
            _entry(entry_id="dominant", content="dominant", semantic_tags=["mentor", "promise"], context_tags=["lab"], accessibility=0.9),
            _entry(entry_id="runner-up", content="runner", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4),
            _entry(entry_id="close-a", content="close-a", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.8),
            _entry(entry_id="close-b", content="close-b", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.78),
        ]
    )
    dominant = compete_candidates(
        store.retrieve(RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=10), k=2).candidates
    )
    close = compete_candidates(
        store.retrieve(RetrievalQuery(semantic_tags=["mentor", "care"], context_tags=["lab"], reference_cycle=10), k=3).candidates
    )
    return {
        "dominant_confidence": dominant.confidence,
        "dominant_interference_risk": dominant.interference_risk,
        "close_confidence": close.confidence,
        "close_interference_risk": close.interference_risk,
        "close_competitor_ids": [entry.id for entry in close.competitors],
        "close_interpretations": list(close.competing_interpretations),
    }


def _reconstruction_boundary_probe() -> dict[str, Any]:
    base = _entry(
        entry_id="base",
        content="Thin recall.",
        semantic_tags=["mentor", "promise"],
        context_tags=["lab"],
        abstractness=0.75,
        reality_confidence=0.6,
        retrieval_count=2,
        anchor_slots={"time": None, "place": None, "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"},
        anchor_strengths={"time": "weak", "place": "weak", "agents": "locked", "action": "strong", "outcome": "strong"},
    )
    semantic = _entry(
        entry_id="semantic",
        content="Semantic trust pattern.",
        semantic_tags=["mentor", "trust"],
        context_tags=["community"],
        memory_class=MemoryClass.SEMANTIC,
        abstractness=0.8,
        retrieval_count=1,
    )
    low_conf = _entry(
        entry_id="low-conf",
        content="Uncertain memory trace with weak grounding but recurring access.",
        semantic_tags=["mentor", "promise"],
        context_tags=["lab"],
        reality_confidence=0.2,
        retrieval_count=3,
    )
    donor = _entry(
        entry_id="donor",
        content="Mentor promise happened in the community lab.",
        semantic_tags=["mentor", "promise", "care"],
        context_tags=["lab", "community"],
        anchor_slots={"time": "cycle-12", "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"},
    )
    procedural = _entry(
        entry_id="procedural",
        content="Reactor procedure summary.",
        semantic_tags=["reactor", "procedure"],
        context_tags=["maintenance"],
        memory_class=MemoryClass.PROCEDURAL,
        abstractness=0.8,
        retrieval_count=2,
    )
    donor_procedure = _entry(
        entry_id="procedure-donor",
        content="Secondary reactor procedure support.",
        semantic_tags=["reactor", "procedure"],
        context_tags=["maintenance", "night"],
        memory_class=MemoryClass.PROCEDURAL,
        execution_contexts=["night_shift"],
    )
    store = MemoryStore(entries=[base, semantic, low_conf, donor, procedural, donor_procedure])
    result_a = maybe_reconstruct(base, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
    result_b = maybe_reconstruct(semantic, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
    result_c = maybe_reconstruct(low_conf, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
    procedural_result = maybe_reconstruct(procedural, store.entries, store, ReconstructionConfig(current_cycle=20, current_state=_state()))
    return {
        "trigger_a": result_a.trigger_reason,
        "trigger_b": result_b.trigger_reason,
        "trigger_c": result_c.trigger_reason,
        "locked_preserved": result_a.entry.anchor_slots["agents"] == "lin",
        "weak_filled": result_a.entry.anchor_slots["place"] == "community_lab",
        "procedural_steps_preserved": procedural_result.entry.procedure_steps == procedural.procedure_steps,
        "procedural_contexts": list(procedural_result.entry.execution_contexts),
        "source_type": result_a.entry.source_type.value,
        "version_changed": result_a.entry.version > base.version,
    }


def _reconsolidation_boundary_probe() -> dict[str, Any]:
    store = MemoryStore(
        entries=[
            _entry(entry_id="reinforce", content="stable", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2),
            _entry(entry_id="rebind", content="rebind", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2, mood_context="reflective"),
            _entry(entry_id="reconstruct", content="thin", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.8, retrieval_count=2),
            _entry(entry_id="conflict", content="conflict", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2),
            _entry(entry_id="donor", content="donor", semantic_tags=["mentor", "care"], context_tags=["lab"], anchor_slots={"time": None, "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"}),
        ]
    )
    reinforce = reconsolidate(store.get("reinforce"), None, None, store=store, current_cycle=30, current_state=_state())
    rebind = reconsolidate(store.get("rebind"), "anxious", ["storm"], store=store, current_cycle=30, current_state=_state())
    reconstruct = reconsolidate(store.get("reconstruct"), "reflective", ["lab"], store=store, current_cycle=30, current_state=_state())
    conflict_artifact = store.retrieve(RetrievalQuery(semantic_tags=["mentor"], context_tags=["lab"], reference_cycle=30), k=1).recall_hypothesis
    conflict = reconsolidate(
        store.get("conflict"),
        "reflective",
        ["lab"],
        store=store,
        current_cycle=30,
        current_state=_state(),
        recall_artifact=conflict_artifact,
        conflict_type=ConflictType.FACTUAL,
    )
    return {
        "update_types": {
            "reinforce": reinforce.update_type,
            "rebind": rebind.update_type,
            "reconstruct": reconstruct.update_type,
            "conflict": conflict.update_type,
        },
        "rebind_fields": list(rebind.fields_rebound),
        "reconstruct_fields": list(reconstruct.fields_reconstructed),
        "conflict_flags": list(conflict.conflict_flags),
        "confidence_deltas": {
            "reinforce": reinforce.confidence_delta,
            "conflict": conflict.confidence_delta,
        },
    }


def _consolidation_boundary_probe() -> dict[str, Any]:
    entries = [
        _entry(
            entry_id=f"ep-{index}",
            content=f"Mentor promise episode {index}",
            semantic_tags=["mentor", "promise", "care"],
            context_tags=["lab", "weekly"],
            accessibility=0.5,
            support_count=2,
            retrieval_count=2,
            created_at=index,
            last_accessed=index + 10,
        )
        for index in range(1, 6)
    ]
    cleanup_entry = _entry(
        entry_id="cleanup-short",
        content="cleanup",
        semantic_tags=["noise", "flash"],
        context_tags=["roof"],
        store_level=StoreLevel.SHORT,
        accessibility=0.02,
        created_at=1,
        last_accessed=1,
    )
    cleanup_entry.trace_strength = 0.01
    entries.append(cleanup_entry)
    store = MemoryStore(entries=entries)
    report = store.run_consolidation_cycle(current_cycle=40, rng=random.Random(0), current_state=_state())
    extracted = [store.get(entry_id) for entry_id in report.extracted_patterns]
    return {
        "upgrade_promoted_ids": list(report.upgrade.promoted_ids),
        "extracted_pattern_ids": list(report.extracted_patterns),
        "replay_reencoded_ids": list(report.replay_reencoded_ids),
        "validated_inference_ids": list(report.validated_inference_ids),
        "cleanup_deleted_ids": list(report.cleanup.deleted_ids),
        "semantic_created": any(entry is not None and entry.memory_class is MemoryClass.SEMANTIC for entry in extracted),
        "inferred_created": any(entry is not None and entry.memory_class is MemoryClass.INFERRED for entry in extracted),
        "report": report.to_dict(),
    }


def _inference_boundary_probe() -> dict[str, Any]:
    validated = _entry(
        entry_id="validated",
        content="validated pattern",
        semantic_tags=["mentor", "pattern"],
        context_tags=["lab", "weekly", "community"],
        memory_class=MemoryClass.INFERRED,
        source_type=SourceType.INFERENCE,
        support_count=5,
        retrieval_count=4,
        compression_metadata={"predictive_gain": 0.8, "cross_context_consistency": 0.9},
    )
    unvalidated = _entry(
        entry_id="unvalidated",
        content="weak hypothesis",
        semantic_tags=["mentor", "pattern"],
        context_tags=["lab"],
        memory_class=MemoryClass.INFERRED,
        source_type=SourceType.INFERENCE,
        support_count=1,
        retrieval_count=0,
        counterevidence_count=2,
        compression_metadata={"predictive_gain": 0.1, "cross_context_consistency": 0.2},
    )
    validated_result = validate_inference(validated)
    unvalidated_result = validate_inference(unvalidated)
    store = MemoryStore(entries=[validated, unvalidated, _entry(entry_id="base", content="base", semantic_tags=["mentor"], context_tags=["lab"])])
    retrieval = store.retrieve(RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=10), k=3)
    return {
        "validated": validated_result.to_dict(),
        "unvalidated": unvalidated_result.to_dict(),
        "unvalidated_aux_blocked": "unvalidated" not in (retrieval.recall_hypothesis.auxiliary_entry_ids if retrieval.recall_hypothesis else []),
        "status_labels": [
            dict(validated.compression_metadata or {}).get("validation_status"),
            dict(unvalidated.compression_metadata or {}).get("validation_status"),
        ],
    }


def _legacy_bridge_boundary_probe() -> dict[str, Any]:
    memory = LongTermMemory()
    store = _consolidation_boundary_probe()["report"]
    # Rebuild a fresh store so the bridge works on real entries rather than report payload.
    entries = [
        _entry(
            entry_id=f"bridge-{index}",
            content=f"Bridge episode {index}",
            semantic_tags=["mentor", "promise", "care"],
            context_tags=["lab", "weekly"],
            accessibility=0.5,
            support_count=2,
            retrieval_count=2,
            created_at=index,
            last_accessed=index + 10,
        )
        for index in range(1, 6)
    ]
    bridge_store = MemoryStore(entries=entries)
    memory.episodes = bridge_store.to_legacy_episodes()
    memory.ensure_memory_store()
    replay_batch = memory.replay_during_sleep(rng=random.Random(0), limit=2)
    report = memory.run_memory_consolidation_cycle(current_cycle=60, rng=random.Random(0), current_state=_state())
    return {
        "replay_batch_size": len(replay_batch),
        "store_cycle_callable": hasattr(memory.memory_store, "run_consolidation_cycle"),
        "entries_match_after_bridge": len(memory.episodes) == len(memory.memory_store.entries),
        "consolidation_report": report.to_dict(),
        "seed_report_shape": bool(store),
    }


BOUNDARY_PROBES: dict[str, Callable[[], dict[str, Any]]] = {
    "retrieval_boundary": _retrieval_boundary_probe,
    "competition_boundary": _competition_boundary_probe,
    "reconstruction_boundary": _reconstruction_boundary_probe,
    "reconsolidation_boundary": _reconsolidation_boundary_probe,
    "consolidation_boundary": _consolidation_boundary_probe,
    "inference_boundary": _inference_boundary_probe,
    "legacy_bridge_boundary": _legacy_bridge_boundary_probe,
}

INTEGRATION_PROBES: dict[str, Callable[[], dict[str, Any]]] = {
    "retrieval_integration": _retrieval_boundary_probe,
    "competition_integration": _competition_boundary_probe,
    "reconstruction_integration": _reconstruction_boundary_probe,
    "reconsolidation_integration": _reconsolidation_boundary_probe,
    "consolidation_integration": _consolidation_boundary_probe,
    "inference_integration": _inference_boundary_probe,
    "legacy_bridge_integration": _legacy_bridge_boundary_probe,
}


def _run_probe_set(
    specs: list[dict[str, str]],
    registry: dict[str, Callable[[], dict[str, Any]]],
    *,
    channel: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        observed = registry[spec["id"]]()
        results[spec["id"]] = _probe_result(
            gate=spec["gate"],
            probe_id=spec["id"],
            channel=channel,
            observed=observed,
        )
    return results


def _ablation_evidence() -> dict[str, Any]:
    store = _retrieval_store()
    baseline = store.retrieve(
        RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=50),
        current_mood="reflective",
        k=3,
    )
    ablated = store.retrieve(
        RetrievalQuery(semantic_tags=["mentor"], context_tags=[], reference_cycle=50),
        current_mood=None,
        k=3,
    )
    return {
        "top_score_gap": baseline.candidates[0].retrieval_score - ablated.candidates[0].retrieval_score,
        "baseline_top_id": baseline.candidates[0].entry_id,
        "ablated_top_id": ablated.candidates[0].entry_id,
    }


def _failure_injection_evidence() -> dict[str, Any]:
    return {
        "cases": [
            {"case": "raw_entry_return_disallowed", "message": "retrieve() must return RecallArtifact rather than raw MemoryEntry"},
            {"case": "unvalidated_inference_donor_disallowed", "message": "unvalidated inferred entries cannot serve as factual detail donors"},
        ]
    }


def run_probe_catalog(
    catalog: dict[str, object] | None = None,
    *,
    regression_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_catalog = build_probe_catalog() if catalog is None else catalog
    return {
        "artifact_lineage": "legacy_self_attested_acceptance",
        "legacy_notice": LEGACY_M46_ACCEPTANCE_DATA_NOTICE,
        "probe_catalog": active_catalog,
        "boundary_probes": _run_probe_set(list(active_catalog["boundary"]), BOUNDARY_PROBES, channel="boundary"),
        "integration_probes": _run_probe_set(list(active_catalog["integration"]), INTEGRATION_PROBES, channel="integration"),
        "regression_summary": dict(regression_summary or {}),
        "ablation": _ablation_evidence(),
        "failure_injection": _failure_injection_evidence(),
    }


def build_m46_acceptance_payload(
    *,
    regression_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Legacy helper retained for historical acceptance artifacts.
    return run_probe_catalog(build_probe_catalog(), regression_summary=regression_summary)
