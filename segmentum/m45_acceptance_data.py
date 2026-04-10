from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from .memory import Episode, LongTermMemory
from .m45_acceptance_shared import (
    baseline_body_state,
    baseline_errors,
    baseline_observation,
    baseline_prediction,
)
from .memory_decay import (
    LONG_DORMANT_ACCESS_THRESHOLD,
    LONG_DORMANT_TRACE_THRESHOLD,
    SHORT_CLEANUP_TRACE_THRESHOLD,
    decay_accessibility_for_level,
    decay_trace_strength_for_level,
)
from .memory_encoding import (
    EMERGENCY_AROUSAL_THRESHOLD,
    EMERGENCY_SALIENCE_THRESHOLD,
    SalienceConfig,
    build_salience_audit,
    compute_salience,
    encode_memory,
    format_salience_audit,
)
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from .memory_store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


def _discover_regression_targets() -> list[str]:
    targets: list[str] = []
    for prefix in ("m41", "m42", "m43", "m44"):
        pattern = f"test_{prefix}*.py"
        for path in sorted(TESTS_DIR.glob(pattern)):
            if path.suffix != ".py" or path.parent.name == "__pycache__":
                continue
            targets.append(path.relative_to(ROOT).as_posix())
    return targets


REGRESSION_TARGETS = _discover_regression_targets()


def _identity_state() -> dict[str, Any]:
    return {
        "active_goals": ["keep promises", "protect mentees"],
        "goal_keywords": ["promise", "care", "mentor", "stability"],
        "identity_roles": ["mentor", "caregiver"],
        "important_relationships": ["lin", "team"],
        "active_commitments": ["weekly mentor checkin", "care plan"],
        "identity_commitments": ["weekly mentor checkin", "care plan"],
        "identity_themes": ["reliable mentor", "care continuity"],
        "identity_active_themes": ["mentor", "care continuity", "reliable mentor"],
        "self_narrative_keywords": ["mentor", "promise", "care"],
        "recent_mood_baseline": "reflective",
        "threat_level": 0.1,
        "reward_context_active": False,
        "social_context_active": True,
    }


def _identity_event() -> dict[str, Any]:
    return {
        "content": "I kept the weekly mentor promise to Lin despite a quiet day.",
        "event_time": "cycle-12",
        "place": "community_lab",
        "action": "mentor_checkin",
        "outcome": "commitment_kept",
        "semantic_tags": ["mentor", "promise", "care", "routine"],
        "context_tags": ["community", "weekly"],
        "roles": ["mentor"],
        "relationships": ["lin"],
        "commitments": ["weekly mentor checkin"],
        "narrative_nodes": ["reliable mentor"],
        "identity_themes": ["care continuity"],
        "arousal": 0.18,
        "novelty": 0.12,
        "encoding_attention": 0.48,
        "valence": 0.25,
        "created_at": 12,
    }


def _noise_event() -> dict[str, Any]:
    return {
        "content": "A bright drone flashed over the roof and vanished.",
        "event_time": "cycle-13",
        "place": "roof",
        "action": "look_up",
        "outcome": "novel_flash",
        "semantic_tags": ["flash", "drone", "novelty"],
        "context_tags": ["weather"],
        "arousal": 0.35,
        "novelty": 0.92,
        "encoding_attention": 0.50,
        "valence": 0.0,
        "created_at": 13,
    }


def _first_person_only_event() -> dict[str, Any]:
    return {
        "content": "I noticed the corridor light flicker during my current task.",
        "action": "observe_light",
        "outcome": "light_flicker",
        "semantic_tags": ["light", "task", "corridor"],
        "context_tags": ["maintenance"],
        "arousal": 0.25,
        "novelty": 0.20,
        "encoding_attention": 0.45,
        "created_at": 14,
    }


def _procedural_event() -> dict[str, Any]:
    return {
        "content": "How to calm the reactor check routine.",
        "memory_class": "procedural",
        "action": "reactor_check",
        "outcome": "stable_temperature",
        "procedure_steps": ["scan gauges", "vent pressure", "log readings"],
        "step_confidence": [0.9, 0.85, 0.88],
        "execution_contexts": ["reactor_room", "maintenance_shift"],
        "semantic_tags": ["reactor", "check", "procedure"],
        "context_tags": ["maintenance"],
        "encoding_attention": 0.8,
        "arousal": 0.3,
        "novelty": 0.2,
        "created_at": 15,
    }


def _semantic_event() -> dict[str, Any]:
    return {
        "content": "Repeated mentor check-ins stabilize trust over time.",
        "memory_class": "semantic",
        "semantic_pattern": True,
        "supporting_episode_ids": ["ep-a", "ep-b", "ep-c"],
        "semantic_tags": ["mentor", "trust", "pattern"],
        "context_tags": ["community"],
        "encoding_attention": 0.65,
        "arousal": 0.25,
        "novelty": 0.25,
        "created_at": 16,
    }


def _inferred_event() -> dict[str, Any]:
    return {
        "content": "Lin may trust weekly check-ins because consistency signals safety.",
        "memory_class": "inferred",
        "inferred": True,
        "supporting_episode_ids": ["ep-a", "ep-b"],
        "semantic_tags": ["mentor", "trust", "safety"],
        "context_tags": ["community"],
        "encoding_attention": 0.55,
        "arousal": 0.20,
        "novelty": 0.40,
        "created_at": 17,
    }


def _probe_result(*, gate: str, probe_id: str, channel: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate": gate,
        "probe_id": probe_id,
        "channel": channel,
        "observed": observed,
    }


def _with_decay_baseline(
    entry: MemoryEntry,
    baseline_cycle: int,
    *,
    base_trace: float | None = None,
    base_accessibility: float | None = None,
) -> MemoryEntry:
    metadata = dict(entry.compression_metadata or {})
    internal = dict(metadata.get("m45_internal", {})) if isinstance(metadata.get("m45_internal"), dict) else {}
    internal["last_decay_cycle"] = baseline_cycle
    internal["decay_base_trace_strength"] = entry.trace_strength if base_trace is None else base_trace
    internal["decay_base_accessibility"] = entry.accessibility if base_accessibility is None else base_accessibility
    metadata["m45_internal"] = internal
    entry.compression_metadata = metadata
    return entry


def _confidence_entry_record(label: str, entry: MemoryEntry) -> dict[str, Any]:
    return {
        "label": label,
        "entry_id": entry.id,
        "source_type": entry.source_type.value,
        "source_confidence": entry.source_confidence,
        "reality_confidence": entry.reality_confidence,
    }


def _confidence_drift_record(label: str, before: MemoryEntry, after: MemoryEntry) -> dict[str, Any]:
    return {
        "label": label,
        "entry_id": after.id,
        "source_type": after.source_type.value,
        "before": {
            "source_confidence": before.source_confidence,
            "reality_confidence": before.reality_confidence,
        },
        "after": {
            "source_confidence": after.source_confidence,
            "reality_confidence": after.reality_confidence,
        },
        "source_changed": after.source_confidence != before.source_confidence,
        "reality_changed": after.reality_confidence != before.reality_confidence,
    }


def _manual_numeric_checks(config: SalienceConfig) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for arousal, attention, novelty, relevance in [
        (0.2, 0.4, 0.6, 0.8),
        (0.9, 0.7, 0.2, 0.1),
        (0.2, 0.5, 0.8, 0.3),
    ]:
        expected = (
            (config.w_arousal * arousal)
            + (config.w_attention * attention)
            + (config.w_novelty * novelty)
            + (config.w_relevance * relevance)
        )
        rows.append(
            {
                "expected": expected,
                "actual": compute_salience(arousal, attention, novelty, relevance, config),
            }
        )
    return rows


def build_probe_catalog() -> dict[str, object]:
    return {
        "boundary": [
            {"id": "data_model_boundary", "gate": "data_model_integrity"},
            {"id": "salience_boundary", "gate": "salience_auditability"},
            {"id": "encoding_boundary", "gate": "encoding_pipeline"},
            {"id": "decay_boundary", "gate": "dual_decay_correctness"},
            {"id": "legacy_bridge_boundary", "gate": "legacy_bridge"},
            {"id": "store_transitions_boundary", "gate": "store_level_transitions"},
        ],
        "integration": [
            {"id": "data_model_integration", "gate": "data_model_integrity"},
            {"id": "salience_integration", "gate": "salience_auditability"},
            {"id": "encoding_integration", "gate": "encoding_pipeline"},
            {"id": "decay_integration", "gate": "dual_decay_correctness"},
            {"id": "legacy_bridge_integration", "gate": "legacy_bridge"},
            {"id": "store_transitions_integration", "gate": "store_level_transitions"},
        ],
        "regression_targets": list(REGRESSION_TARGETS),
    }


def _data_model_boundary_probe() -> dict[str, Any]:
    entry = MemoryEntry(
        content="Anchored mentor event.",
        created_at=1,
        last_accessed=1,
        arousal=0.2,
        encoding_attention=0.3,
        novelty=0.1,
        relevance_goal=0.2,
        relevance_threat=0.0,
        relevance_self=0.7,
        relevance_social=0.6,
        relevance_reward=0.1,
        relevance=0.32,
        salience=0.26,
        trace_strength=0.26,
        accessibility=0.26,
        abstractness=0.2,
        source_confidence=0.9,
        reality_confidence=0.85,
        semantic_tags=["mentor"],
        context_tags=["community"],
        anchor_slots={"action": "checkin", "outcome": "trust", "agents": "lin"},
        anchor_strengths={"agents": "strong", "action": "strong", "outcome": "strong"},
        support_count=1,
    )
    round_trip = MemoryEntry.from_dict(entry.to_dict())
    version_before = entry.version
    entry.content = "Anchored mentor event updated."
    entry.sync_content_hash()
    procedural_guard = False
    anchor_guard = False
    try:
        MemoryEntry(memory_class=MemoryClass.PROCEDURAL, content="broken")
    except ValueError:
        procedural_guard = True
    try:
        MemoryEntry(
            content="all weak anchors",
            anchor_strengths={key: "weak" for key in ("time", "place", "agents", "action", "outcome")},
        )
    except ValueError:
        anchor_guard = True
    return {
        "round_trip_equal": round_trip.to_dict() == MemoryEntry.from_dict(round_trip.to_dict()).to_dict(),
        "content_hash_stable": round_trip.content_hash == MemoryEntry.from_dict(round_trip.to_dict()).content_hash,
        "version_incremented": entry.version == version_before + 1,
        "procedural_guard": procedural_guard,
        "anchor_guard": anchor_guard,
        "field_count": len(round_trip.to_dict()),
    }


def _data_model_integration_probe() -> dict[str, Any]:
    identity_entry = encode_memory(_identity_event(), _identity_state(), SalienceConfig())
    semantic_entry = encode_memory(_semantic_event(), _identity_state(), SalienceConfig())
    inferred_entry = encode_memory(_inferred_event(), _identity_state(), SalienceConfig())
    round_trip = MemoryEntry.from_dict(identity_entry.to_dict())
    return {
        "round_trip_equal": round_trip.to_dict() == identity_entry.to_dict(),
        "semantic_lineage_type": dict(semantic_entry.compression_metadata or {}).get("lineage_type"),
        "semantic_predictive_use_cases": list(dict(semantic_entry.compression_metadata or {}).get("predictive_use_cases", [])),
        "inferred_lineage_type": dict(inferred_entry.compression_metadata or {}).get("lineage_type"),
        "inferred_predictive_use_cases": list(dict(inferred_entry.compression_metadata or {}).get("predictive_use_cases", [])),
        "protected_anchor_strengths": dict(identity_entry.to_dict()["anchor_strengths"]),
    }


def _salience_boundary_probe() -> dict[str, Any]:
    config = SalienceConfig()
    checks = _manual_numeric_checks(config)
    diffs = [abs(item["expected"] - item["actual"]) for item in checks]
    entry = encode_memory(_identity_event(), _identity_state(), config)
    audit = build_salience_audit(entry, config)
    return {
        "numeric_checks": checks,
        "max_diff": max(diffs) if diffs else 1.0,
        "audit_inputs": dict(audit["inputs"]),
        "audit_string": format_salience_audit(entry, config),
    }


def _salience_integration_probe() -> dict[str, Any]:
    entry = encode_memory(_identity_event(), _identity_state(), SalienceConfig())
    encoding_audit = dict(dict(entry.compression_metadata or {}).get("encoding_audit", {}))
    signal_breakdown = dict(encoding_audit.get("signal_breakdown", {}))
    self_signal = dict(signal_breakdown.get("self", {}))
    return {
        "has_signal_breakdown": bool(signal_breakdown),
        "relevance_audit": dict(encoding_audit.get("relevance_audit", {})),
        "self_evidence": list(encoding_audit.get("self_evidence", [])),
        "self_guardrails": list(self_signal.get("guardrails", [])),
        "memory_class_reason": encoding_audit.get("memory_class_reason"),
    }


def _encoding_boundary_probe() -> dict[str, Any]:
    config = SalienceConfig()
    state = _identity_state()
    identity_entry = encode_memory(_identity_event(), state, config)
    noise_entry = encode_memory(_noise_event(), state, config)
    first_person_entry = encode_memory(_first_person_only_event(), state, config)
    procedural_entry = encode_memory(_procedural_event(), state, config)
    semantic_entry = encode_memory(_semantic_event(), state, config)
    inferred_entry = encode_memory({**_inferred_event(), "source_type": "inference"}, state, config)
    high_arousal = encode_memory(
        {
            "content": "Explosion alarm triggered.",
            "action": "evacuate",
            "outcome": "alarm",
            "arousal": 0.95,
            "encoding_attention": 0.8,
            "novelty": 0.8,
            "semantic_tags": ["alarm", "danger"],
        },
        state,
        config,
    )
    high_salience = encode_memory(
        {
            "content": "Goal-critical threat and reward conflict.",
            "action": "decide",
            "outcome": "high_priority",
            "arousal": 0.89,
            "encoding_attention": 1.0,
            "novelty": 1.0,
            "goal_cues": ["keep promises"],
            "threat_cues": ["risk", "danger"],
            "reward_signal": 1.0,
            "semantic_tags": ["promise", "danger", "reward"],
            "commitments": ["weekly mentor checkin"],
            "roles": ["mentor"],
            "relationships": ["lin"],
            "narrative_nodes": ["reliable mentor"],
        },
        state,
        config,
    )
    reward_probe = encode_memory(
        {
            "content": "Severe penalty and loss recorded.",
            "action": "audit",
            "outcome": "loss_recorded",
            "semantic_tags": ["penalty", "loss"],
            "encoding_attention": 0.5,
            "arousal": 0.4,
            "novelty": 0.2,
        },
        state,
        config,
    )
    return {
        "memory_classes": sorted(
            {
                identity_entry.memory_class.value,
                procedural_entry.memory_class.value,
                semantic_entry.memory_class.value,
                inferred_entry.memory_class.value,
            }
        ),
        "source_defaults": {
            "experience": (identity_entry.source_confidence, identity_entry.reality_confidence),
            "hearsay": tuple(
                encode_memory({**_noise_event(), "source_type": "hearsay"}, state, config).to_dict()[key]
                for key in ("source_confidence", "reality_confidence")
            ),
            "inference": (inferred_entry.source_confidence, inferred_entry.reality_confidence),
            "reconstruction": tuple(
                encode_memory({**_identity_event(), "source_type": "reconstruction"}, state, config).to_dict()[key]
                for key in ("source_confidence", "reality_confidence")
            ),
        },
        "high_arousal_store_level": high_arousal.store_level.value,
        "high_salience_store_level": high_salience.store_level.value,
        "identity_self_relevance": identity_entry.relevance_self,
        "noise_self_relevance": noise_entry.relevance_self,
        "first_person_self_relevance": first_person_entry.relevance_self,
        "procedural_steps": list(procedural_entry.procedure_steps),
        "semantic_lineage_type": dict(semantic_entry.compression_metadata or {}).get("lineage_type"),
        "inferred_lineage_type": dict(inferred_entry.compression_metadata or {}).get("lineage_type"),
        "semantic_derived_from": list(semantic_entry.derived_from),
        "inferred_derived_from": list(inferred_entry.derived_from),
        "reward_probe": {
            "relevance_reward": reward_probe.relevance_reward,
            "relevance_threat": reward_probe.relevance_threat,
            "threat_evidence": list(dict(dict(reward_probe.compression_metadata or {}).get("encoding_audit", {})).get("threat_evidence", [])),
        },
    }


def _encoding_integration_probe() -> dict[str, Any]:
    store = MemoryStore()
    config = SalienceConfig()
    state = _identity_state()
    identity_entry = _with_decay_baseline(encode_memory(_identity_event(), state, config), 12)
    noise_entry = encode_memory(_noise_event(), state, config)
    confidence_entries = [
        encode_memory(_identity_event(), state, config),
        encode_memory({**_noise_event(), "source_type": "hearsay"}, state, config),
        encode_memory({**_inferred_event(), "source_type": "inference"}, state, config),
        encode_memory({**_identity_event(), "source_type": "reconstruction"}, state, config),
    ]
    source_only_drift = _with_decay_baseline(
        MemoryEntry(
            content="source-only-drift",
            store_level=StoreLevel.LONG,
            source_type=SourceType.RECONSTRUCTION,
            created_at=1,
            last_accessed=1,
            trace_strength=0.4,
            accessibility=0.35,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.3,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=1,
            compression_metadata={"source_conflict": True},
        ),
        1,
    )
    reality_only_drift = _with_decay_baseline(
        MemoryEntry(
            content="reality-only-drift",
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=0.4,
            accessibility=0.35,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.3,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=2,
            counterevidence_count=3,
            competing_interpretations=["alt-hypothesis"],
            compression_metadata={"factual_conflict": True},
        ),
        1,
    )
    identity_entry.retrieval_count = 1
    store.add(identity_entry)
    store.add(noise_entry)
    drift_store = MemoryStore(entries=[source_only_drift, reality_only_drift])
    source_before = MemoryEntry.from_dict(source_only_drift.to_dict())
    reality_before = MemoryEntry.from_dict(reality_only_drift.to_dict())
    drift_store.apply_decay(current_cycle=21)
    source_after = drift_store.get(source_only_drift.id)
    reality_after = drift_store.get(reality_only_drift.id)
    promoted_identity = store.get(identity_entry.id)
    retained_noise = store.get(noise_entry.id)
    encoding_audit = dict(dict(promoted_identity.compression_metadata or {}).get("encoding_audit", {})) if promoted_identity is not None else {}
    return {
        "identity_store_level": promoted_identity.store_level.value if promoted_identity is not None else None,
        "noise_store_level": retained_noise.store_level.value if retained_noise is not None else None,
        "identity_retention_reasons": list(dict(encoding_audit.get("retention_priority", {})).get("reasons", [])),
        "identity_anchor_reasoning": dict(encoding_audit.get("anchor_reasoning", {})),
        "confidence_pairs": [
            _confidence_entry_record("experience", confidence_entries[0]),
            _confidence_entry_record("hearsay", confidence_entries[1]),
            _confidence_entry_record("inference", confidence_entries[2]),
            _confidence_entry_record("reconstruction", confidence_entries[3]),
        ],
        "confidence_drift_cases": [
            _confidence_drift_record("source_only_drift", source_before, source_after)
            if source_after is not None
            else {},
            _confidence_drift_record("reality_only_drift", reality_before, reality_after)
            if reality_after is not None
            else {},
        ],
    }


def _decay_boundary_probe() -> dict[str, Any]:
    short_entry = _with_decay_baseline(
        MemoryEntry(
            content="short",
            store_level=StoreLevel.SHORT,
            created_at=1,
            last_accessed=1,
            trace_strength=SHORT_CLEANUP_TRACE_THRESHOLD - 0.01,
            accessibility=0.20,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        ),
        1,
    )
    abstract_entry = _with_decay_baseline(
        MemoryEntry(
            content="abstract-me",
            store_level=StoreLevel.MID,
            created_at=1,
            last_accessed=1,
            trace_strength=0.92,
            accessibility=0.88,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.62,
            source_confidence=0.8,
            reality_confidence=0.8,
            support_count=1,
        ),
        1,
    )
    source_conflict_entry = _with_decay_baseline(
        MemoryEntry(
            content="source-conflict",
            store_level=StoreLevel.LONG,
            source_type=SourceType.RECONSTRUCTION,
            created_at=1,
            last_accessed=1,
            trace_strength=0.35,
            accessibility=0.30,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.3,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=1,
            compression_metadata={"source_conflict": True},
        ),
        1,
    )
    long_entry = _with_decay_baseline(
        MemoryEntry(
            content="long-dormant",
            store_level=StoreLevel.LONG,
            created_at=1,
            last_accessed=1,
            trace_strength=0.0201,
            accessibility=0.0102,
            arousal=0.1,
            encoding_attention=0.1,
            novelty=0.1,
            relevance_goal=0.1,
            relevance_threat=0.1,
            relevance_self=0.1,
            relevance_social=0.1,
            relevance_reward=0.1,
            relevance=0.1,
            salience=0.1,
            abstractness=0.2,
            source_confidence=0.82,
            reality_confidence=0.78,
            support_count=2,
            compression_metadata={"factual_conflict": True},
            competing_interpretations=["alt-hypothesis"],
            counterevidence_count=2,
        ),
        1,
    )
    store = MemoryStore(entries=[short_entry, abstract_entry, source_conflict_entry, long_entry])
    timepoints = [5, 20, 80]
    report = store.apply_decay(current_cycle=81)
    return {
        "timepoints": timepoints,
        "trace_curves": {
            "short": [decay_trace_strength_for_level(1.0, StoreLevel.SHORT, elapsed) for elapsed in timepoints],
            "mid": [decay_trace_strength_for_level(1.0, StoreLevel.MID, elapsed) for elapsed in timepoints],
            "long": [decay_trace_strength_for_level(1.0, StoreLevel.LONG, elapsed) for elapsed in timepoints],
            "procedural_long": [
                decay_trace_strength_for_level(1.0, StoreLevel.LONG, elapsed, memory_class=MemoryClass.PROCEDURAL)
                for elapsed in timepoints
            ],
        },
        "accessibility_curves": {
            "short": [decay_accessibility_for_level(1.0, StoreLevel.SHORT, elapsed) for elapsed in timepoints],
            "mid": [decay_accessibility_for_level(1.0, StoreLevel.MID, elapsed) for elapsed in timepoints],
            "long": [decay_accessibility_for_level(1.0, StoreLevel.LONG, elapsed) for elapsed in timepoints],
        },
        "forgetting_paths": report.to_dict(),
        "cleanup_report": dict(store.last_cleanup_report),
    }


def _decay_integration_probe() -> dict[str, Any]:
    store = MemoryStore()
    identity_entry = _with_decay_baseline(encode_memory(_identity_event(), _identity_state(), SalienceConfig()), 12)
    identity_entry.store_level = StoreLevel.MID
    identity_entry.support_count = 1
    source_entry = _with_decay_baseline(
        encode_memory({**_identity_event(), "source_type": "reconstruction"}, _identity_state(), SalienceConfig()),
        12,
    )
    source_entry.store_level = StoreLevel.LONG
    source_entry.compression_metadata = {**dict(source_entry.compression_metadata or {}), "source_conflict": True}
    source_entry.source_confidence = 0.82
    source_entry.reality_confidence = 0.78
    source_entry.counterevidence_count = 2
    source_entry.competing_interpretations = ["alt-hypothesis"]
    store.add(identity_entry)
    store.add(source_entry)
    report = store.apply_decay(current_cycle=32)
    return {
        "processed_entries": report.processed_entries,
        "abstracted_entries": list(report.abstracted_entries),
        "source_confidence_drifted": list(report.source_confidence_drifted),
        "reality_confidence_drifted": list(report.reality_confidence_drifted),
    }


def _legacy_bridge_boundary_probe() -> dict[str, Any]:
    memory = LongTermMemory()
    legacy_payload = memory.store_episode(
        cycle=31,
        observation=baseline_observation(),
        prediction=baseline_prediction(),
        errors=baseline_errors(),
        action="hide",
        outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.5},
        body_state=baseline_body_state(),
    )
    legacy_payload["custom_flag"] = "keep-me"
    legacy_payload["custom_nested"] = {"note": "preserved"}
    store = MemoryStore.from_legacy_episodes([legacy_payload])
    entry = store.entries[0]
    entry.anchor_slots["action"] = "hide_revised"
    entry.anchor_slots["outcome"] = "mutated_outcome"
    entry.novelty = 0.99
    entry.salience = 0.77
    entry.support_count = 5
    restored_payload = store.to_legacy_episodes()[0]
    return {
        "timestamp_preserved": restored_payload["timestamp"] == legacy_payload["timestamp"],
        "action_reflected": restored_payload["action"],
        "outcome_reflected": restored_payload["predicted_outcome"],
        "prediction_error_reflected": restored_payload["prediction_error"],
        "total_surprise_reflected": restored_payload["total_surprise"],
        "support_count_reflected": restored_payload["support_count"],
        "unknown_fields_preserved": {
            "custom_flag": restored_payload.get("custom_flag"),
            "custom_nested": restored_payload.get("custom_nested"),
        },
    }


def _legacy_bridge_integration_probe() -> dict[str, Any]:
    memory = LongTermMemory(surprise_threshold=0.2, sleep_minimum_support=2, max_active_age=1)
    memory.ensure_memory_store()
    payload = memory.store_episode(
        cycle=31,
        observation=baseline_observation(),
        prediction=baseline_prediction(),
        errors=baseline_errors(),
        action="hide",
        outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.5},
        body_state=baseline_body_state(),
    )
    merged = memory.maybe_store_episode(
        cycle=32,
        observation=baseline_observation(),
        prediction=baseline_prediction(),
        errors=baseline_errors(),
        action="hide",
        outcome={"energy_delta": -0.1, "stress_delta": 0.2, "free_energy_drop": -0.5},
        body_state=baseline_body_state(),
    )
    memory.store_episode(
        cycle=33,
        observation=baseline_observation(),
        prediction=baseline_prediction(),
        errors=baseline_errors(),
        action="forage",
        outcome={"energy_delta": -0.05, "stress_delta": 0.1, "free_energy_drop": -0.4},
        body_state=baseline_body_state(),
    )
    memory.compress_episodes(current_cycle=40)
    restored = LongTermMemory.from_dict(memory.to_dict())
    replay_batch = restored.replay_during_sleep(rng=random.Random(0), limit=1)
    return {
        "episodes_after_store": len(memory.episodes),
        "store_entries_after_store": len(memory.memory_store.entries) if memory.memory_store is not None else 0,
        "merge_support_delta": merged.support_delta,
        "restored_store_entries": len(restored.memory_store.entries) if restored.memory_store is not None else 0,
        "replay_batch_size": len(replay_batch),
        "payload_episode_id": payload.get("episode_id"),
    }


def _store_transitions_boundary_probe() -> dict[str, Any]:
    store = MemoryStore()
    state = _identity_state()
    identity_entry = _with_decay_baseline(encode_memory(_identity_event(), state, SalienceConfig()), 12)
    identity_entry.retrieval_count = 1
    identity_entry.last_accessed = 20
    store.add(identity_entry)
    semantic_entry = _with_decay_baseline(encode_memory(_semantic_event(), state, SalienceConfig()), 16)
    semantic_entry.store_level = StoreLevel.MID
    semantic_entry.retrieval_count = 3
    semantic_entry.salience = 0.88
    semantic_entry.last_accessed = 26
    store.add(semantic_entry)
    noise_entry = encode_memory(_noise_event(), state, SalienceConfig())
    store.add(noise_entry)
    promoted_identity = store.get(identity_entry.id)
    promoted_long = store.get(semantic_entry.id)
    retained_noise = store.get(noise_entry.id)
    return {
        "identity_store_level": promoted_identity.store_level.value if promoted_identity is not None else None,
        "long_store_level": promoted_long.store_level.value if promoted_long is not None else None,
        "noise_store_level": retained_noise.store_level.value if retained_noise is not None else None,
        "identity_transition": dict(((promoted_identity.compression_metadata or {}).get("m45_internal", {}) if promoted_identity is not None else {}).get("last_promotion", {})),
        "long_transition": dict(((promoted_long.compression_metadata or {}).get("m45_internal", {}) if promoted_long is not None else {}).get("last_promotion", {})),
    }


def _store_transitions_integration_probe() -> dict[str, Any]:
    audit_fields = (
        "identity_link_strength",
        "identity_link_active",
        "self_relevance_multiplier",
        "base_short_to_mid_score",
        "boosted_short_to_mid_score",
        "score_cap_applied",
    )

    def _audit_excerpt(audit: dict[str, Any]) -> dict[str, Any]:
        return {field: audit.get(field) for field in audit_fields}

    store = MemoryStore()
    identity_entry = _with_decay_baseline(encode_memory(_identity_event(), _identity_state(), SalienceConfig()), 12)
    identity_entry.retrieval_count = 1
    store.add(identity_entry)
    noise_entry = encode_memory(_noise_event(), _identity_state(), SalienceConfig())
    store.add(noise_entry)
    linked_entry = _with_decay_baseline(encode_memory(_identity_event(), _identity_state(), SalienceConfig()), 12)
    linked_entry.retrieval_count = 1
    null_entry = _with_decay_baseline(encode_memory(_identity_event(), {}, SalienceConfig()), 12)
    null_entry.retrieval_count = 1
    linked_store = MemoryStore()
    null_store = MemoryStore()
    linked_store.add(linked_entry)
    null_store.add(null_entry)
    linked_promoted = linked_store.get(linked_entry.id)
    null_promoted = null_store.get(null_entry.id)
    linked_audit = dict(dict((linked_promoted.compression_metadata or {})).get("m47_promotion_audit", {})) if linked_promoted is not None else {}
    null_audit = dict(dict((null_promoted.compression_metadata or {})).get("m47_promotion_audit", {})) if null_promoted is not None else {}
    neutral_population = []
    for index in range(20):
        neutral_entry = encode_memory(_noise_event(), {}, SalienceConfig())
        neutral_entry.id = f"neutral-{index}"
        neutral_store = MemoryStore()
        neutral_store.add(neutral_entry)
        neutral_promoted = neutral_store.get(neutral_entry.id)
        neutral_population.append(
            1.0
            if neutral_promoted is not None and neutral_promoted.store_level is not StoreLevel.SHORT
            else 0.0
        )
    return {
        "identity_store_level": store.get(identity_entry.id).store_level.value if store.get(identity_entry.id) is not None else None,
        "noise_store_level": store.get(noise_entry.id).store_level.value if store.get(noise_entry.id) is not None else None,
        "identity_linked_store_level": linked_promoted.store_level.value if linked_promoted is not None else None,
        "identity_null_store_level": null_promoted.store_level.value if null_promoted is not None else None,
        "identity_linked_audit": _audit_excerpt(linked_audit),
        "identity_null_audit": _audit_excerpt(null_audit),
        "identity_linked_score": float(linked_audit.get("boosted_short_to_mid_score", linked_audit.get("short_to_mid_score", 0.0))),
        "identity_null_score": float(null_audit.get("boosted_short_to_mid_score", null_audit.get("short_to_mid_score", 0.0))),
        "identity_score_delta": float(linked_audit.get("boosted_short_to_mid_score", linked_audit.get("short_to_mid_score", 0.0)))
        - float(null_audit.get("boosted_short_to_mid_score", null_audit.get("short_to_mid_score", 0.0))),
        "neutral_promotion_rate": sum(neutral_population) / max(1, len(neutral_population)),
    }


BOUNDARY_PROBES: dict[str, Callable[[], dict[str, Any]]] = {
    "data_model_boundary": _data_model_boundary_probe,
    "salience_boundary": _salience_boundary_probe,
    "encoding_boundary": _encoding_boundary_probe,
    "decay_boundary": _decay_boundary_probe,
    "legacy_bridge_boundary": _legacy_bridge_boundary_probe,
    "store_transitions_boundary": _store_transitions_boundary_probe,
}

INTEGRATION_PROBES: dict[str, Callable[[], dict[str, Any]]] = {
    "data_model_integration": _data_model_integration_probe,
    "salience_integration": _salience_integration_probe,
    "encoding_integration": _encoding_integration_probe,
    "decay_integration": _decay_integration_probe,
    "legacy_bridge_integration": _legacy_bridge_integration_probe,
    "store_transitions_integration": _store_transitions_integration_probe,
}


def _run_probe_set(
    specs: list[dict[str, str]],
    registry: dict[str, Callable[[], dict[str, Any]]],
    *,
    channel: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        probe_id = spec["id"]
        observed = registry[probe_id]()
        results[probe_id] = _probe_result(
            gate=spec["gate"],
            probe_id=probe_id,
            channel=channel,
            observed=observed,
        )
    return results


def _ablation_evidence() -> dict[str, Any]:
    base_entry = encode_memory(_identity_event(), _identity_state(), SalienceConfig())
    ablated_entry = encode_memory(
        _identity_event(),
        _identity_state(),
        SalienceConfig(
            relevance_weights={"goal": 0.25, "threat": 0.25, "self": 0.0, "social": 0.25, "reward": 0.25}
        ),
    )
    return {
        "baseline_salience": base_entry.salience,
        "ablated_salience": ablated_entry.salience,
        "identity_relevance_drop": base_entry.salience - ablated_entry.salience,
    }


def _failure_injection_evidence() -> dict[str, Any]:
    cases: list[dict[str, str]] = []
    try:
        encode_memory({"memory_class": "procedural", "content": "Broken procedure"}, _identity_state(), SalienceConfig())
    except ValueError as exc:
        cases.append({"case": "missing_procedure_steps", "message": str(exc)})
    try:
        MemoryEntry(
            content="bad anchors",
            anchor_strengths={key: "weak" for key in ("time", "place", "agents", "action", "outcome")},
            support_count=1,
        )
    except ValueError as exc:
        cases.append({"case": "all_weak_episodic_anchors", "message": str(exc)})
    return {"cases": cases}


def run_probe_catalog(
    catalog: dict[str, object] | None = None,
    *,
    regression_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_catalog = build_probe_catalog() if catalog is None else catalog
    return {
        "probe_catalog": active_catalog,
        "boundary_probes": _run_probe_set(list(active_catalog["boundary"]), BOUNDARY_PROBES, channel="boundary"),
        "integration_probes": _run_probe_set(list(active_catalog["integration"]), INTEGRATION_PROBES, channel="integration"),
        "regression_summary": dict(regression_summary or {}),
        "ablation": _ablation_evidence(),
        "failure_injection": _failure_injection_evidence(),
    }


def build_m45_acceptance_payload(
    *,
    regression_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return run_probe_catalog(build_probe_catalog(), regression_summary=regression_summary)
