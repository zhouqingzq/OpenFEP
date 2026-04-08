from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory import LongTermMemory
from .memory_consolidation import (
    ConflictType,
    ReconstructionConfig,
    compress_episodic_cluster_to_semantic_skeleton,
    maybe_reconstruct,
    reconsolidate,
    validate_inference,
)
from .memory_model import AnchorStrength, MemoryClass, MemoryEntry, SourceType, StoreLevel
from .memory_retrieval import RecallArtifact, RetrievalQuery, compete_candidates
from .memory_store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
REPORTS_DIR = ROOT / "reports"

M46_REACCEPTANCE_EVIDENCE_PATH = REPORTS_DIR / "m46_reacceptance_evidence.json"
M46_REACCEPTANCE_SUMMARY_PATH = REPORTS_DIR / "m46_reacceptance_summary.md"

GATE_RETRIEVAL = "retrieval_multi_cue"
GATE_COMPETITION = "candidate_competition"
GATE_RECONSTRUCTION = "reconstruction_mechanism"
GATE_RECONSOLIDATION = "reconsolidation"
GATE_CONSOLIDATION = "offline_consolidation_pipeline"
GATE_INFERENCE = "inference_validation_gate"
GATE_LEGACY = "legacy_integration"
GATE_HONESTY = "report_honesty"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_RUN = "NOT_RUN"
FORMAL_CONCLUSION_NOT_ISSUED = "NOT_ISSUED"

SOURCE_KIND_REAL_API_RUN = "real_api_run"
SOURCE_KIND_REGRESSION_RUN = "regression_run"
SOURCE_KIND_SELF_AUDIT = "self_audit"

VALID_SOURCE_KINDS = {
    SOURCE_KIND_REAL_API_RUN,
    SOURCE_KIND_REGRESSION_RUN,
    SOURCE_KIND_SELF_AUDIT,
}

NOT_RUN_SCENARIOS = {"legacy_regression_prereq"}
REQUIRED_INTEGRATION_SCENARIOS = {
    "consolidation_validation_linkage",
    "inference_consolidation_validation_linkage",
}

GATE_ORDER = (
    GATE_RETRIEVAL,
    GATE_COMPETITION,
    GATE_RECONSTRUCTION,
    GATE_RECONSOLIDATION,
    GATE_CONSOLIDATION,
    GATE_INFERENCE,
    GATE_LEGACY,
    GATE_HONESTY,
)

GATE_CODES = {
    GATE_RETRIEVAL: "G1",
    GATE_COMPETITION: "G2",
    GATE_RECONSTRUCTION: "G3",
    GATE_RECONSOLIDATION: "G4",
    GATE_CONSOLIDATION: "G5",
    GATE_INFERENCE: "G6",
    GATE_LEGACY: "G7",
    GATE_HONESTY: "G8",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _discover_regression_targets() -> list[str]:
    targets: list[str] = []
    for prefix in ("m41", "m42", "m43", "m44", "m45"):
        for path in sorted(TESTS_DIR.glob(f"test_{prefix}*.py")):
            if path.suffix == ".py":
                targets.append(path.relative_to(ROOT).as_posix())
    return targets


REGRESSION_TARGETS = _discover_regression_targets()
PYTEST_SUMMARY_LINE_RE = re.compile(
    r"^\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,2}(?:,\s*\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,2})*\s+in\s+\d+(?:\.\d+)?s(?:\s+\([^)]+\))?$"
)


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validation_trace_from_snapshot(entry_snapshot: dict[str, object]) -> dict[str, float | str | bool]:
    metadata = _as_dict(entry_snapshot.get("compression_metadata"))
    replay_persistence = _clamp01(float(metadata.get("replay_persistence", min(1.0, float(entry_snapshot.get("retrieval_count", 0)) / 4.0))))
    support_score = _clamp01(min(1.0, float(entry_snapshot.get("support_count", 0)) / 5.0))
    cross_context_consistency = _clamp01(
        float(metadata.get("cross_context_consistency", min(1.0, len({str(item) for item in _as_list(entry_snapshot.get("context_tags"))}) / 3.0)))
    )
    predictive_gain = _clamp01(float(metadata.get("predictive_gain", min(1.0, float(entry_snapshot.get("relevance", 0.0)) + 0.1))))
    contradiction_penalty = _clamp01(
        float(metadata.get("contradiction_penalty", min(1.0, float(entry_snapshot.get("counterevidence_count", 0)) / 4.0)))
    )
    score = _clamp01(
        (0.25 * replay_persistence)
        + (0.30 * support_score)
        + (0.20 * cross_context_consistency)
        + (0.25 * predictive_gain)
        - (0.35 * contradiction_penalty)
    )
    threshold = 0.55
    if contradiction_penalty >= 0.75:
        expected_status = "contradicted"
    elif score >= threshold:
        expected_status = "validated"
    elif score >= 0.40:
        expected_status = "partially_supported"
    else:
        expected_status = "unvalidated"
    return {
        "replay_persistence": round(replay_persistence, 6),
        "support_score": round(support_score, 6),
        "cross_context_consistency": round(cross_context_consistency, 6),
        "predictive_gain": round(predictive_gain, 6),
        "contradiction_penalty": round(contradiction_penalty, 6),
        "score": round(score, 6),
        "threshold": threshold,
        "expected_validation_status": expected_status,
    }


def _reconsolidation_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, float | int]:
    return {
        "accessibility": round(float(after.get("accessibility", 0.0)) - float(before.get("accessibility", 0.0)), 6),
        "trace_strength": round(float(after.get("trace_strength", 0.0)) - float(before.get("trace_strength", 0.0)), 6),
        "retrieval_count": int(after.get("retrieval_count", 0)) - int(before.get("retrieval_count", 0)),
        "abstractness": round(float(after.get("abstractness", 0.0)) - float(before.get("abstractness", 0.0)), 6),
        "last_accessed": int(after.get("last_accessed", 0)) - int(before.get("last_accessed", 0)),
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
    source_confidence: float = 0.9,
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
    salience: float = 0.5,
    trace_strength: float = 0.5,
    relevance: float = 0.3,
    relevance_self: float = 0.2,
    relevance_threat: float = 0.2,
    relevance_goal: float = 0.3,
    relevance_social: float = 0.2,
    relevance_reward: float = 0.2,
    arousal: float = 0.3,
    novelty: float = 0.3,
    encoding_attention: float = 0.4,
) -> MemoryEntry:
    kwargs: dict[str, object] = {
        "id": entry_id,
        "content": content,
        "memory_class": memory_class,
        "store_level": store_level,
        "source_type": source_type,
        "created_at": created_at,
        "last_accessed": last_accessed,
        "valence": valence,
        "arousal": arousal,
        "encoding_attention": encoding_attention,
        "novelty": novelty,
        "relevance_goal": relevance_goal,
        "relevance_threat": relevance_threat,
        "relevance_self": relevance_self,
        "relevance_social": relevance_social,
        "relevance_reward": relevance_reward,
        "relevance": relevance,
        "salience": salience,
        "trace_strength": trace_strength,
        "accessibility": accessibility,
        "abstractness": abstractness,
        "source_confidence": source_confidence,
        "reality_confidence": reality_confidence,
        "semantic_tags": semantic_tags,
        "context_tags": context_tags,
        "anchor_slots": anchor_slots or {
            "action": "mentor_checkin",
            "outcome": "commitment_kept",
            "agents": "lin",
        },
        "anchor_strengths": anchor_strengths,
        "mood_context": mood_context,
        "retrieval_count": retrieval_count,
        "support_count": support_count,
        "counterevidence_count": counterevidence_count,
        "compression_metadata": compression_metadata,
        "is_dormant": is_dormant,
    }
    if memory_class is MemoryClass.PROCEDURAL:
        steps = procedure_steps or ["scan gauges", "vent pressure", "log readings"]
        kwargs["procedure_steps"] = steps
        kwargs["step_confidence"] = [0.9 for _ in steps]
        kwargs["execution_contexts"] = execution_contexts or ["reactor_room"]
    return MemoryEntry(**kwargs)


def _slug(value: str) -> str:
    sanitized = "".join(character.lower() if character.isalnum() else "_" for character in value)
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "unknown"


def _make_source_api_call_id(
    *,
    gate: str,
    scenario_id: str,
    api: str,
    source_seed: int | None,
) -> str:
    seed_label = f"seed_{source_seed}" if source_seed is not None else "seed_none"
    return ".".join((_slug(gate), _slug(scenario_id), _slug(api), seed_label))


def _record(
    *,
    gate: str,
    scenario_id: str,
    api: str,
    input_summary: dict[str, object],
    observed: dict[str, object],
    criteria_checks: dict[str, bool],
    notes: list[str] | None = None,
    status: str | None = None,
    source_kind: str = SOURCE_KIND_REAL_API_RUN,
    source_api_call_id: str | None = None,
    source_input_set_id: str | None = None,
    source_seed: int | None = None,
) -> dict[str, object]:
    if source_kind not in VALID_SOURCE_KINDS:
        raise ValueError(f"Unsupported source_kind: {source_kind}")
    record_status = status
    if record_status is None:
        record_status = STATUS_PASS if criteria_checks and all(criteria_checks.values()) else STATUS_FAIL
    return {
        "gate": gate,
        "scenario_id": scenario_id,
        "api": api,
        "input_summary": input_summary,
        "observed": observed,
        "criteria_checks": criteria_checks,
        "status": record_status,
        "notes": list(notes or []),
        "source_kind": source_kind,
        "source_api_call_id": source_api_call_id
        or _make_source_api_call_id(
            gate=gate,
            scenario_id=scenario_id,
            api=api,
            source_seed=source_seed,
        ),
        "source_input_set_id": source_input_set_id or scenario_id,
        "source_seed": source_seed,
    }


def build_m46_evidence_records(*, include_regressions: bool = False) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    records.extend(_build_retrieval_records())
    records.extend(_build_competition_records())
    records.extend(_build_reconstruction_records())
    records.extend(_build_reconsolidation_records())
    records.extend(_build_consolidation_records())
    records.extend(_build_inference_records())
    records.extend(_build_legacy_records(include_regressions=include_regressions))
    return records


def _gate_summary(gate: str, records: list[dict[str, object]]) -> dict[str, object]:
    statuses = [str(record["status"]) for record in records]
    if any(status == STATUS_FAIL for status in statuses):
        status = STATUS_FAIL
    elif any(status == STATUS_NOT_RUN for status in statuses):
        status = STATUS_NOT_RUN
    else:
        status = STATUS_PASS
    return {
        "gate": gate,
        "status": status,
        "scenario_ids": [str(record["scenario_id"]) for record in records],
        "counts": {
            "total": len(records),
            "passed": sum(1 for status_item in statuses if status_item == STATUS_PASS),
            "failed": sum(1 for status_item in statuses if status_item == STATUS_FAIL),
            "not_run": sum(1 for status_item in statuses if status_item == STATUS_NOT_RUN),
        },
        "notes": [
            note
            for record in records
            for note in record.get("notes", [])
            if isinstance(note, str) and note
        ],
    }


def _serialize_candidates(result: Any) -> list[dict[str, object]]:
    return [candidate.to_dict() for candidate in result.candidates]


def _serialize_recall(recall: RecallArtifact | None) -> dict[str, object] | None:
    return recall.to_dict() if recall is not None else None


def _entry_snapshot(entry: MemoryEntry) -> dict[str, object]:
    return entry.to_dict()


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


def _build_retrieval_records() -> list[dict[str, object]]:
    store = _retrieval_store()
    raw_contents = [entry.content for entry in store.entries]
    scenarios = [
        {
            "scenario_id": "retrieval_tag_primary",
            "query": RetrievalQuery(
                semantic_tags=["mentor", "promise"],
                context_tags=["lab"],
                content_keywords=["lin", "promise"],
                state_vector=[0.91, 0.22],
                reference_cycle=50,
            ),
            "current_mood": "reflective",
            "k": 4,
        },
        {
            "scenario_id": "retrieval_context_rich",
            "query": RetrievalQuery(
                semantic_tags=["plan"],
                context_tags=["lab", "weekly", "team"],
                state_vector=[0.88, 0.24],
                reference_cycle=50,
            ),
            "current_mood": "reflective",
            "k": 3,
        },
        {
            "scenario_id": "retrieval_negative_mood",
            "query": RetrievalQuery(
                semantic_tags=["promise"],
                context_tags=["storm"],
                reference_cycle=50,
            ),
            "current_mood": "anxious",
            "k": 3,
        },
        {
            "scenario_id": "retrieval_accessibility_pressure",
            "query": RetrievalQuery(
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab"],
                reference_cycle=50,
            ),
            "current_mood": "reflective",
            "k": 4,
        },
        {
            "scenario_id": "retrieval_procedural_outline",
            "query": RetrievalQuery(
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance"],
                reference_cycle=50,
                target_memory_class=MemoryClass.PROCEDURAL,
            ),
            "current_mood": "calm",
            "k": 1,
        },
    ]
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        result = store.retrieve(
            scenario["query"],
            current_mood=str(scenario["current_mood"]),
            k=int(scenario["k"]),
        )
        recall = result.recall_hypothesis
        candidate_ids = [candidate.entry_id for candidate in result.candidates]
        observed = {
            "candidates": _serialize_candidates(result),
            "recall_hypothesis": _serialize_recall(recall),
            "recall_confidence": result.recall_confidence,
            "source_trace": list(result.source_trace),
            "reconstruction_trace": dict(result.reconstruction_trace),
            "candidate_ids": candidate_ids,
        }
        criteria_checks: dict[str, bool]
        notes: list[str] = []
        if scenario["scenario_id"] == "retrieval_tag_primary":
            score_keys = set(result.candidates[0].score_breakdown) if result.candidates else set()
            criteria_checks = {
                "top_candidate_is_tag_primary": bool(result.candidates) and result.candidates[0].entry_id == "tag-primary",
                "score_breakdown_complete": score_keys
                == {"tag_overlap", "context_overlap", "mood_match", "accessibility", "recency"},
                "dormant_excluded": "dormant" not in candidate_ids,
                "recall_artifact_is_independent_type": isinstance(recall, RecallArtifact),
                "recall_has_primary_and_auxiliary_trace": bool(recall and recall.primary_entry_id == "tag-primary" and recall.auxiliary_entry_ids),
                "recall_content_not_raw_entry": bool(recall and recall.content not in raw_contents),
            }
            notes.append("Primary retrieval scenario captures ranking, source trace, and non-alias RecallArtifact behavior.")
        elif scenario["scenario_id"] == "retrieval_context_rich":
            context_overlap = result.candidates[0].score_breakdown.get("context_overlap", 0.0) if result.candidates else 0.0
            criteria_checks = {
                "top_candidate_is_context_rich": bool(result.candidates) and result.candidates[0].entry_id == "context-rich",
                "context_overlap_contributes": context_overlap > 0.0,
                "recall_hypothesis_present": isinstance(recall, RecallArtifact),
            }
        elif scenario["scenario_id"] == "retrieval_negative_mood":
            mood_match = result.candidates[0].score_breakdown.get("mood_match", 0.0) if result.candidates else 0.0
            criteria_checks = {
                "negative_memory_ranked_first": bool(result.candidates) and result.candidates[0].entry_id == "negative-mood",
                "mood_match_contributes": mood_match > 0.0,
                "recall_hypothesis_present": isinstance(recall, RecallArtifact),
            }
        elif scenario["scenario_id"] == "retrieval_accessibility_pressure":
            low_access_rank = candidate_ids.index("low-access") if "low-access" in candidate_ids else -1
            criteria_checks = {
                "low_access_not_top": bool(result.candidates) and result.candidates[0].entry_id != "low-access",
                "low_access_ranked_below_accessible_match": low_access_rank > 0,
                "recall_hypothesis_present": isinstance(recall, RecallArtifact),
            }
        else:
            criteria_checks = {
                "procedural_candidate_selected": bool(result.candidates) and result.candidates[0].entry_id == "procedure",
                "procedure_step_outline_present": bool(recall and recall.procedure_step_outline),
                "procedural_recall_not_raw_content": bool(recall and recall.content not in raw_contents),
            }
            notes.append("Procedural retrieval must expose a step outline rather than only narrative text.")
        records.append(
            _record(
                gate=GATE_RETRIEVAL,
                scenario_id=str(scenario["scenario_id"]),
                api="MemoryStore.retrieve",
                input_summary={
                    "query": scenario["query"].__dict__,
                    "current_mood": scenario["current_mood"],
                    "k": scenario["k"],
                },
                observed=observed,
                criteria_checks=criteria_checks,
                notes=notes,
                source_input_set_id=str(scenario["scenario_id"]),
            )
        )
    return records


def _build_competition_records() -> list[dict[str, object]]:
    store = MemoryStore(
        entries=[
            _entry(entry_id="dominant", content="dominant", semantic_tags=["mentor", "promise"], context_tags=["lab"], accessibility=0.9),
            _entry(entry_id="runner-up", content="runner", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4),
            _entry(entry_id="close-a", content="close-a", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.8),
            _entry(entry_id="close-b", content="close-b", semantic_tags=["mentor", "care"], context_tags=["lab"], accessibility=0.78),
        ]
    )
    scenarios = [
        {
            "scenario_id": "competition_dominant_margin",
            "query": RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=10),
            "k": 2,
        },
        {
            "scenario_id": "competition_close_margin",
            "query": RetrievalQuery(semantic_tags=["mentor", "care"], context_tags=["lab"], reference_cycle=10),
            "k": 3,
        },
    ]
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        result = store.retrieve(scenario["query"], k=int(scenario["k"]))
        competition = compete_candidates(result.candidates)
        observed = {
            "candidates": _serialize_candidates(result),
            "competition": competition.to_dict(),
        }
        if scenario["scenario_id"] == "competition_dominant_margin":
            criteria_checks = {
                "confidence_high": competition.confidence == "high",
                "interference_risk_false": competition.interference_risk is False,
                "competitors_empty": not competition.competitors,
                "dominance_margin_above_threshold": competition.dominance_margin > 0.15,
            }
        else:
            criteria_checks = {
                "confidence_low": competition.confidence == "low",
                "interference_risk_true": competition.interference_risk is True,
                "competitors_present": bool(competition.competitors),
                "competing_interpretations_present": bool(competition.competing_interpretations),
            }
        records.append(
            _record(
                gate=GATE_COMPETITION,
                scenario_id=str(scenario["scenario_id"]),
                api="compete_candidates",
                input_summary={"query": scenario["query"].__dict__, "k": scenario["k"]},
                observed=observed,
                criteria_checks=criteria_checks,
            )
        )
    return records


def _reconstruction_store() -> MemoryStore:
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
    return MemoryStore(entries=[base, semantic, low_conf, donor, procedural, donor_procedure])


def _build_reconstruction_records() -> list[dict[str, object]]:
    store = _reconstruction_store()
    config = ReconstructionConfig(current_cycle=20, current_state=_state())
    records: list[dict[str, object]] = []
    scenarios = [
        ("reconstruction_trigger_a", "base", "abstract_short_content"),
        ("reconstruction_trigger_b", "semantic", "semantic_abstractness"),
        ("reconstruction_trigger_c", "low-conf", "low_reality_after_retrieval"),
        ("reconstruction_procedural_protection", "procedural", "abstract_short_content"),
    ]
    for scenario_id, entry_id, expected_reason in scenarios:
        target = store.get(entry_id)
        assert target is not None
        before = _entry_snapshot(target)
        result = maybe_reconstruct(target, store.entries, store, config)
        after = _entry_snapshot(result.entry)
        observed = {
            "before": before,
            "after": after,
            "reconstruction_result": result.to_dict(),
        }
        criteria_checks: dict[str, bool]
        notes: list[str] = []
        if scenario_id == "reconstruction_trigger_a":
            criteria_checks = {
                "triggered": result.triggered,
                "trigger_reason_matches": result.trigger_reason == expected_reason,
                "borrow_limit_respected": len(result.borrowed_source_ids) <= 2,
                "source_type_reconstruction": result.entry.source_type is SourceType.RECONSTRUCTION,
                "reality_confidence_decreased": result.entry.reality_confidence < target.reality_confidence,
                "version_incremented": result.entry.version > target.version,
                "content_hash_changed": result.entry.content_hash != target.content_hash,
                "locked_strong_preserved": result.entry.anchor_slots["agents"] == "lin" and result.entry.anchor_slots["action"] == "mentor_checkin",
                "weak_anchor_filled": result.entry.anchor_slots["place"] == "community_lab",
                "trace_records_sources_and_fields": bool(result.reconstruction_trace.get("borrowed_source_ids")) and bool(result.reconstruction_trace.get("protected_fields")),
            }
        elif scenario_id == "reconstruction_trigger_b":
            criteria_checks = {
                "triggered": result.triggered,
                "trigger_reason_matches": result.trigger_reason == expected_reason,
                "borrow_limit_respected": len(result.borrowed_source_ids) <= 2,
                "semantic_has_more_reconstruction_freedom": "agents" not in result.protected_fields,
                "source_type_reconstruction": result.entry.source_type is SourceType.RECONSTRUCTION,
            }
        elif scenario_id == "reconstruction_trigger_c":
            criteria_checks = {
                "triggered": result.triggered,
                "trigger_reason_matches": result.trigger_reason == expected_reason,
                "borrow_limit_respected": len(result.borrowed_source_ids) <= 2,
                "source_type_reconstruction": result.entry.source_type is SourceType.RECONSTRUCTION,
                "reality_confidence_decreased": result.entry.reality_confidence < target.reality_confidence,
            }
        else:
            criteria_checks = {
                "triggered": result.triggered,
                "procedural_steps_preserved": result.entry.procedure_steps == target.procedure_steps,
                "supporting_context_borrowed": "night_shift" in result.entry.execution_contexts,
                "procedure_not_exposed_as_rewritten_anchor": "procedure_steps" not in result.reconstructed_fields,
            }
            notes.append("Procedural reconstruction may enrich contexts but must not rewrite core steps without evidence.")
        records.append(
            _record(
                gate=GATE_RECONSTRUCTION,
                scenario_id=scenario_id,
                api="maybe_reconstruct",
                input_summary={"entry_id": entry_id, "current_cycle": config.current_cycle},
                observed=observed,
                criteria_checks=criteria_checks,
                notes=notes,
            )
        )

    invalid_observed: dict[str, object]
    invalid_checks: dict[str, bool]
    try:
        MemoryEntry(
            id="invalid-anchors",
            content="Should fail",
            memory_class=MemoryClass.EPISODIC,
            store_level=StoreLevel.SHORT,
            source_type=SourceType.EXPERIENCE,
            semantic_tags=["mentor"],
            context_tags=["lab"],
            anchor_strengths={key: AnchorStrength.WEAK.value for key in ("time", "place", "agents", "action", "outcome")},
        )
        invalid_observed = {"raised": False}
        invalid_checks = {"invalid_all_weak_anchor_config_rejected": False}
    except ValueError as exc:
        invalid_observed = {"raised": True, "error": str(exc)}
        invalid_checks = {"invalid_all_weak_anchor_config_rejected": True}
    records.append(
        _record(
            gate=GATE_RECONSTRUCTION,
            scenario_id="reconstruction_invalid_anchor_rejection",
            api="MemoryEntry.__post_init__",
            input_summary={"memory_class": MemoryClass.EPISODIC.value, "anchor_strengths": "all weak"},
            observed=invalid_observed,
            criteria_checks=invalid_checks,
            notes=["Acceptance anti-degeneration check: episodic anchors cannot all be downgraded to weak."],
        )
    )
    return records


def _build_reconsolidation_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def add_record(
        *,
        scenario_id: str,
        entry: MemoryEntry,
        report: Any,
        before: dict[str, object],
        extra_observed: dict[str, object] | None = None,
        criteria_checks: dict[str, bool],
        notes: list[str] | None = None,
    ) -> None:
        after = _entry_snapshot(entry)
        observed = {
            "before": before,
            "after": after,
            "numeric_delta": _reconsolidation_delta(before, after),
            "report": report.to_dict(),
        }
        if extra_observed:
            observed.update(extra_observed)
        records.append(
            _record(
                gate=GATE_RECONSOLIDATION,
                scenario_id=scenario_id,
                api="reconsolidate",
                input_summary={"entry_id": entry.id},
                observed=observed,
                criteria_checks=criteria_checks,
                notes=notes,
            )
        )

    reinforce = _entry(entry_id="reinforce", content="stable", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2)
    reinforce_before = _entry_snapshot(reinforce)
    reinforce_report = reconsolidate(reinforce, None, None, current_cycle=30)
    add_record(
        scenario_id="reconsolidation_reinforcement_only",
        entry=reinforce,
        report=reinforce_report,
        before=reinforce_before,
        criteria_checks={
            "update_type_reinforcement_only": reinforce_report.update_type == "reinforcement_only",
            "accessibility_increased": float(reinforce.accessibility) > float(reinforce_before["accessibility"]),
            "trace_strength_increased": float(reinforce.trace_strength) > float(reinforce_before["trace_strength"]),
            "retrieval_count_incremented": int(reinforce.retrieval_count) == int(reinforce_before["retrieval_count"]) + 1,
            "abstractness_incremented": float(reinforce.abstractness) > float(reinforce_before["abstractness"]),
            "last_accessed_updated": int(reinforce.last_accessed) >= 30 and int(reinforce.last_accessed) > int(reinforce_before["last_accessed"]),
            "version_unchanged": reinforce_report.version_changed is False,
        },
    )

    rebind = _entry(entry_id="rebind", content="rebind", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.2, mood_context="reflective")
    rebind_before = _entry_snapshot(rebind)
    rebind_report = reconsolidate(rebind, "anxious", ["storm"], current_cycle=30)
    add_record(
        scenario_id="reconsolidation_contextual_rebinding",
        entry=rebind,
        report=rebind_report,
        before=rebind_before,
        criteria_checks={
            "update_type_contextual_rebinding": rebind_report.update_type == "contextual_rebinding",
            "mood_context_rebound": rebind.mood_context == "anxious",
            "context_tags_rebound": "storm" in rebind.context_tags,
            "version_unchanged": rebind_report.version_changed is False,
        },
    )

    reconstruction_store = MemoryStore(
        entries=[
            _entry(entry_id="reconstruct", content="thin", semantic_tags=["mentor"], context_tags=["lab"], accessibility=0.4, abstractness=0.8, retrieval_count=2),
            _entry(entry_id="donor", content="donor", semantic_tags=["mentor", "care"], context_tags=["lab"], anchor_slots={"time": None, "place": "community_lab", "agents": "lin", "action": "mentor_checkin", "outcome": "commitment_kept"}),
        ]
    )
    reconstruct = reconstruction_store.get("reconstruct")
    assert reconstruct is not None
    reconstruct_before = _entry_snapshot(reconstruct)
    reconstruct_report = reconsolidate(reconstruct, "reflective", ["lab"], store=reconstruction_store, current_cycle=30, current_state=_state())
    add_record(
        scenario_id="reconsolidation_structural_reconstruction",
        entry=reconstruct,
        report=reconstruct_report,
        before=reconstruct_before,
        criteria_checks={
            "update_type_structural_reconstruction": reconstruct_report.update_type == "structural_reconstruction",
            "fields_reconstructed_present": bool(reconstruct_report.fields_reconstructed),
            "version_changed": reconstruct_report.version_changed is True,
            "source_type_reconstruction": reconstruct.source_type is SourceType.RECONSTRUCTION,
            "reality_confidence_shifted": float(reconstruct.reality_confidence) != float(reconstruct_before["reality_confidence"]),
        },
    )

    conflict_store = _retrieval_store()
    conflict_artifact = conflict_store.retrieve(
        RetrievalQuery(semantic_tags=["mentor", "promise"], context_tags=["lab"], reference_cycle=50),
        current_mood="reflective",
        k=3,
    ).recall_hypothesis
    assert conflict_artifact is not None
    for scenario_id, conflict_type in (
        ("reconsolidation_factual_conflict", ConflictType.FACTUAL),
        ("reconsolidation_source_conflict", ConflictType.SOURCE),
        ("reconsolidation_interpretive_conflict", ConflictType.INTERPRETIVE),
    ):
        conflict_entry = _entry(
            entry_id=scenario_id,
            content="conflict",
            semantic_tags=["mentor"],
            context_tags=["lab"],
            accessibility=0.4,
            abstractness=0.2,
            source_confidence=0.9,
            reality_confidence=0.85,
        )
        conflict_before = _entry_snapshot(conflict_entry)
        conflict_report = reconsolidate(
            conflict_entry,
            "reflective",
            ["lab"],
            current_cycle=30,
            recall_artifact=conflict_artifact,
            conflict_type=conflict_type,
        )
        delta = conflict_report.confidence_delta
        criteria_checks = {
            "update_type_conflict_marking": conflict_report.update_type == "conflict_marking",
            "conflict_flag_recorded": conflict_type.value in conflict_report.conflict_flags,
            "competing_interpretation_preserved": bool(conflict_entry.competing_interpretations),
        }
        if conflict_type is ConflictType.FACTUAL:
            criteria_checks.update(
                {
                    "reality_confidence_hit_primary": float(delta["reality_confidence"]) < float(delta["source_confidence"]),
                    "counterevidence_incremented": int(conflict_entry.counterevidence_count) == int(conflict_before["counterevidence_count"]) + 1,
                }
            )
        elif conflict_type is ConflictType.SOURCE:
            criteria_checks.update(
                {
                    "source_confidence_hit_primary": float(delta["source_confidence"]) < float(delta["reality_confidence"]),
                    "counterevidence_not_incremented": int(conflict_entry.counterevidence_count) == int(conflict_before["counterevidence_count"]),
                }
            )
        else:
            criteria_checks.update(
                {
                    "confidence_not_directly_reduced": float(delta["source_confidence"]) == 0.0 and float(delta["reality_confidence"]) == 0.0,
                    "counterevidence_not_incremented": int(conflict_entry.counterevidence_count) == int(conflict_before["counterevidence_count"]),
                }
            )
        add_record(
            scenario_id=scenario_id,
            entry=conflict_entry,
            report=conflict_report,
            before=conflict_before,
            criteria_checks=criteria_checks,
        )

    procedural_store = MemoryStore(
        entries=[
            _entry(
                entry_id="proc-recall",
                content="Reactor procedure summary.",
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance"],
                memory_class=MemoryClass.PROCEDURAL,
                abstractness=0.8,
                retrieval_count=2,
                execution_contexts=["reactor_room"],
            ),
            _entry(
                entry_id="proc-donor",
                content="Secondary reactor procedure support.",
                semantic_tags=["reactor", "procedure"],
                context_tags=["maintenance", "night"],
                memory_class=MemoryClass.PROCEDURAL,
                execution_contexts=["night_shift"],
            ),
        ]
    )
    procedural_entry = procedural_store.get("proc-recall")
    assert procedural_entry is not None
    procedural_before = _entry_snapshot(procedural_entry)
    procedural_report = reconsolidate(
        procedural_entry,
        "calm",
        ["maintenance"],
        store=procedural_store,
        current_cycle=30,
        current_state=_state(),
    )
    add_record(
        scenario_id="reconsolidation_procedural_core_protection",
        entry=procedural_entry,
        report=procedural_report,
        before=procedural_before,
        criteria_checks={
            "procedure_steps_preserved": procedural_entry.procedure_steps == list(procedural_before["procedure_steps"]),
            "update_type_captured": procedural_report.update_type in {
                "contextual_rebinding",
                "structural_reconstruction",
            },
            "procedure_steps_not_listed_as_rewritten": "procedure_steps" not in procedural_report.fields_reconstructed,
            "execution_context_can_expand": "night_shift" in procedural_entry.execution_contexts,
        },
        notes=["Procedural memories may gain context but core action sequence remains protected."],
    )
    return records


def _build_consolidation_entries() -> list[MemoryEntry]:
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
            salience=0.72,
            trace_strength=0.55,
            abstractness=0.25,
            relevance_self=0.7 if index == 1 else 0.2,
            relevance=0.6,
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
        salience=0.05,
        trace_strength=0.01,
    )
    dormant_long = _entry(
        entry_id="dormant-long",
        content="old long-term trace",
        semantic_tags=["mentor", "history"],
        context_tags=["archive"],
        store_level=StoreLevel.LONG,
        accessibility=0.02,
        created_at=1,
        last_accessed=1,
        salience=0.2,
        trace_strength=0.02,
    )
    entries.extend([cleanup_entry, dormant_long])
    return entries


def _build_consolidation_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    source_entries = [
        _entry(
            entry_id=f"cluster-{index}",
            content=f"cluster {index}",
            semantic_tags=["mentor", "promise", "care"],
            context_tags=["lab", "weekly"],
            support_count=2,
            retrieval_count=1,
            abstractness=0.2 + (index * 0.05),
            salience=0.65,
        )
        for index in range(3)
    ]
    skeleton = compress_episodic_cluster_to_semantic_skeleton(source_entries)
    skeleton_metadata = dict(skeleton.compression_metadata or {})
    records.append(
        _record(
            gate=GATE_CONSOLIDATION,
            scenario_id="consolidation_semantic_skeleton_lineage",
            api="compress_episodic_cluster_to_semantic_skeleton",
            input_summary={"source_entry_ids": [entry.id for entry in source_entries]},
            observed={
                "source_entries": [entry.to_dict() for entry in source_entries],
                "skeleton": skeleton.to_dict(),
            },
            criteria_checks={
                "support_entry_ids_preserved": skeleton_metadata.get("support_entry_ids") == [entry.id for entry in source_entries],
                "discarded_detail_types_present": bool(skeleton_metadata.get("discarded_detail_types")),
                "stable_structure_present": bool(skeleton_metadata.get("stable_structure")),
                "lineage_type_present": bool(skeleton_metadata.get("lineage_type")),
                "abstractness_exceeds_sources": skeleton.abstractness > max(entry.abstractness for entry in source_entries),
            },
        )
    )

    store = MemoryStore(entries=_build_consolidation_entries())
    before_ids = [entry.id for entry in store.entries]
    report = store.run_consolidation_cycle(current_cycle=40, rng=random.Random(0), current_state=_state())
    extracted_entries = [store.get(entry_id) for entry_id in report.extracted_patterns]
    extracted_non_null = [entry for entry in extracted_entries if entry is not None]
    semantic_entries = [entry for entry in extracted_non_null if entry.memory_class is MemoryClass.SEMANTIC]
    inferred_entries = [entry for entry in extracted_non_null if entry.memory_class is MemoryClass.INFERRED]
    retained_source_ids = [entry.id for entry in store.entries if entry.id in before_ids and entry.id.startswith("ep-")]
    source_absorbed = {
        entry.id: dict(entry.compression_metadata or {}).get("absorbed_by")
        for entry in store.entries
        if entry.id.startswith("ep-")
    }
    records.append(
        _record(
            gate=GATE_CONSOLIDATION,
            scenario_id="consolidation_full_cycle",
            api="MemoryStore.run_consolidation_cycle",
            input_summary={"current_cycle": 40, "rng_seed": 0},
            observed={
                "report": report.to_dict(),
                "entries_after": [entry.to_dict() for entry in store.entries],
                "extracted_entries": [entry.to_dict() for entry in extracted_non_null],
                "retained_source_ids": retained_source_ids,
                "source_absorbed": source_absorbed,
            },
            criteria_checks={
                "upgrade_phase_executed": bool(report.upgrade.promoted_ids),
                "pattern_extraction_executed": bool(report.extracted_patterns),
                "replay_phase_executed": bool(report.replay_created_ids),
                "cleanup_phase_executed": bool(report.cleanup.deleted_ids or report.cleanup.dormant_ids or report.cleanup.absorbed_ids),
                "semantic_skeleton_created": bool(semantic_entries),
                "inferred_pattern_created": bool(inferred_entries),
                "source_episodics_retained_after_skeleton": len(retained_source_ids) >= 5,
                "sources_marked_absorbed_not_deleted": all(source_absorbed.values()),
                "report_has_stage_counts": set(report.to_dict()) == {
                    "upgrade",
                    "extracted_patterns",
                    "replay_created_ids",
                    "validated_inference_ids",
                    "cleanup",
                },
            },
            notes=["This scenario verifies the four-stage offline cycle with retention of episodic support sources."],
            source_seed=0,
        )
    )

    validation_linkage_store = MemoryStore(
        entries=[
            _entry(
                entry_id=f"validation-source-{index}",
                content=f"validation source {index}",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab", "weekly", "community"],
                support_count=5,
                retrieval_count=4,
                relevance=0.95,
                salience=0.95,
                trace_strength=0.8,
                accessibility=0.7,
                store_level=StoreLevel.MID,
            )
            for index in range(3)
        ]
    )
    validation_report = validation_linkage_store.run_consolidation_cycle(
        current_cycle=55,
        rng=random.Random(0),
        current_state=_state(),
    )
    validated_entries = [
        validation_linkage_store.get(entry_id)
        for entry_id in validation_report.validated_inference_ids
    ]
    validated_non_null = [entry for entry in validated_entries if entry is not None]
    records.append(
        _record(
            gate=GATE_CONSOLIDATION,
            scenario_id="consolidation_validation_linkage",
            api="MemoryStore.run_consolidation_cycle",
            input_summary={"current_cycle": 55, "rng_seed": 0, "input_set": "validation_linkage_store"},
            observed={
                "report": validation_report.to_dict(),
                "validated_entries": [entry.to_dict() for entry in validated_non_null],
            },
            criteria_checks={
                "validated_inference_ids_present": bool(validation_report.validated_inference_ids),
                "validated_entries_exist": len(validated_non_null) == len(validation_report.validated_inference_ids),
                "validated_ids_subset_replay_created": set(validation_report.validated_inference_ids).issubset(
                    set(validation_report.replay_created_ids)
                ),
                "validated_entries_promoted_to_long": bool(validated_non_null)
                and all(entry.store_level is StoreLevel.LONG for entry in validated_non_null),
                "validated_entries_marked_validated": bool(validated_non_null)
                and all(
                    dict(entry.compression_metadata or {}).get("validation_status") == "validated"
                    for entry in validated_non_null
                ),
            },
            notes=["Integration scenario: consolidation must surface validated inference ids that correspond to LONG entries."],
            source_seed=0,
        )
    )
    return records


def _build_inference_records() -> list[dict[str, object]]:
    validated = _entry(
        entry_id="validated",
        content="validated pattern",
        semantic_tags=["mentor", "pattern"],
        context_tags=["lab", "weekly", "community"],
        memory_class=MemoryClass.INFERRED,
        store_level=StoreLevel.MID,
        source_type=SourceType.INFERENCE,
        support_count=5,
        retrieval_count=4,
        reality_confidence=0.4,
        compression_metadata={"predictive_gain": 0.8, "cross_context_consistency": 0.9},
    )
    validated_before = _entry_snapshot(validated)
    validated_result = validate_inference(validated)
    validated_trace = _validation_trace_from_snapshot(validated_before)

    unvalidated = _entry(
        entry_id="unvalidated",
        content="weak hypothesis",
        semantic_tags=["mentor", "pattern"],
        context_tags=["lab"],
        memory_class=MemoryClass.INFERRED,
        store_level=StoreLevel.MID,
        source_type=SourceType.INFERENCE,
        support_count=1,
        retrieval_count=0,
        counterevidence_count=2,
        compression_metadata={"predictive_gain": 0.1, "cross_context_consistency": 0.2},
    )
    unvalidated_before = _entry_snapshot(unvalidated)
    unvalidated_result = validate_inference(unvalidated)
    unvalidated_trace = _validation_trace_from_snapshot(unvalidated_before)
    donor_store = MemoryStore(entries=[validated, unvalidated, _entry(entry_id="base", content="base", semantic_tags=["mentor"], context_tags=["lab"])])
    retrieval = donor_store.retrieve(
        RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=10),
        k=3,
    )
    recall = retrieval.recall_hypothesis
    candidate_ids = [candidate.entry_id for candidate in retrieval.candidates]
    linkage_store = MemoryStore(
        entries=[
            _entry(
                entry_id=f"inference-source-{index}",
                content=f"inference source {index}",
                semantic_tags=["mentor", "promise", "care"],
                context_tags=["lab", "weekly", "community"],
                support_count=5,
                retrieval_count=4,
                relevance=0.95,
                salience=0.95,
                trace_strength=0.8,
                accessibility=0.7,
                store_level=StoreLevel.MID,
            )
            for index in range(3)
        ]
    )
    linkage_report = linkage_store.run_consolidation_cycle(
        current_cycle=55,
        rng=random.Random(0),
        current_state=_state(),
    )
    linkage_entries = [
        linkage_store.get(entry_id)
        for entry_id in linkage_report.validated_inference_ids
    ]
    linkage_non_null = [entry for entry in linkage_entries if entry is not None]
    return [
        _record(
            gate=GATE_INFERENCE,
            scenario_id="inference_validated_upgrade",
            api="validate_inference",
            input_summary={"entry_id": "validated"},
            observed={
                "before": validated_before,
                "after": _entry_snapshot(validated),
                "validation": validated_result.to_dict(),
                "traceability": validated_trace,
            },
            criteria_checks={
                "score_exceeds_threshold": validated_result.score >= validated_result.threshold,
                "validation_passed": validated_result.passed is True,
                "status_validated": validated_result.validation_status == "validated",
                "upgraded_to_long": validated.store_level is StoreLevel.LONG,
                "reality_confidence_promoted": validated.reality_confidence >= 0.68,
                "traceability_score_matches_result": validated_trace["score"] == validated_result.score,
            },
        ),
        _record(
            gate=GATE_INFERENCE,
            scenario_id="inference_unvalidated_donor_blocked",
            api="validate_inference + MemoryStore.retrieve",
            input_summary={"entry_id": "unvalidated", "retrieval_query": RetrievalQuery(semantic_tags=["mentor", "pattern"], context_tags=["lab"], reference_cycle=10).__dict__},
            observed={
                "before": unvalidated_before,
                "after": _entry_snapshot(unvalidated),
                "validation": unvalidated_result.to_dict(),
                "traceability": unvalidated_trace,
                "retrieval_candidates": _serialize_candidates(retrieval),
                "recall_hypothesis": _serialize_recall(recall),
                "candidate_ids": candidate_ids,
            },
            criteria_checks={
                "score_below_threshold": unvalidated_result.score < unvalidated_result.threshold,
                "validation_blocked": unvalidated_result.passed is False,
                "status_unvalidated_or_contradicted": unvalidated_result.validation_status in {"unvalidated", "contradicted"},
                "not_upgraded_to_long": unvalidated.store_level is StoreLevel.MID,
                "still_visible_as_candidate": "unvalidated" in candidate_ids,
                "blocked_from_factual_donation": bool(recall and "unvalidated" not in recall.auxiliary_entry_ids),
                "traceability_score_matches_result": unvalidated_trace["score"] == unvalidated_result.score,
            },
            notes=["Unvalidated inferred memories may surface as low-confidence candidates but not as factual detail donors."],
        ),
        _record(
            gate=GATE_INFERENCE,
            scenario_id="inference_consolidation_validation_linkage",
            api="MemoryStore.run_consolidation_cycle + validate_inference",
            input_summary={"current_cycle": 55, "rng_seed": 0, "input_set": "inference_linkage_store"},
            observed={
                "report": linkage_report.to_dict(),
                "validated_entries": [
                    {
                        "entry": entry.to_dict(),
                        "traceability": _validation_trace_from_snapshot(entry.to_dict()),
                    }
                    for entry in linkage_non_null
                ],
            },
            criteria_checks={
                "validated_inference_ids_present": bool(linkage_report.validated_inference_ids),
                "validated_entries_exist": len(linkage_non_null) == len(linkage_report.validated_inference_ids),
                "validated_entries_long": bool(linkage_non_null)
                and all(entry.store_level is StoreLevel.LONG for entry in linkage_non_null),
                "traceability_inputs_present": bool(linkage_non_null)
                and all(
                    set(item["traceability"])
                    == {
                        "replay_persistence",
                        "support_score",
                        "cross_context_consistency",
                        "predictive_gain",
                        "contradiction_penalty",
                        "score",
                        "threshold",
                        "expected_validation_status",
                    }
                    for item in [
                        {"traceability": _validation_trace_from_snapshot(entry.to_dict())}
                        for entry in linkage_non_null
                    ]
                ),
                "metadata_scores_match_traceability": bool(linkage_non_null)
                and all(
                    round(float(dict(entry.compression_metadata or {}).get("inference_write_score", -1.0)), 6)
                    == _validation_trace_from_snapshot(entry.to_dict())["score"]
                    for entry in linkage_non_null
                ),
            },
            notes=["Integration scenario: validated replay entries must expose traceable score inputs and true LONG promotion."],
            source_seed=0,
        ),
    ]


def _bridge_store() -> MemoryStore:
    return MemoryStore(entries=_build_consolidation_entries()[:5])


def _run_regressions() -> dict[str, object]:
    if not REGRESSION_TARGETS:
        return {
            "executed": True,
            "command": [sys.executable, "-m", "pytest", "-q"],
            "files": [],
            "returncode": 0,
            "passed": True,
            "duration_seconds": 0.0,
            "stdout_tail": [],
            "summary_line": "0 files",
        }
    command = [sys.executable, "-m", "pytest", *REGRESSION_TARGETS, "-q"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_seconds = round(time.perf_counter() - started, 6)
    output_lines = "\n".join(part for part in [completed.stdout, completed.stderr] if part).splitlines()
    return {
        "executed": True,
        "command": command,
        "files": list(REGRESSION_TARGETS),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": duration_seconds,
        "stdout_tail": output_lines[-5:],
        "summary_line": output_lines[-1] if output_lines else "",
    }


def _build_legacy_records(*, include_regressions: bool) -> list[dict[str, object]]:
    bridge_store = _bridge_store()
    memory = LongTermMemory()
    memory.episodes = bridge_store.to_legacy_episodes()
    memory.ensure_memory_store()
    replay_batch = memory.replay_during_sleep(rng=random.Random(0), limit=2)
    bridge_record = _record(
        gate=GATE_LEGACY,
        scenario_id="legacy_bridge_replay_batch",
        api="LongTermMemory.replay_during_sleep",
        input_summary={"rng_seed": 0, "limit": 2},
        observed={
            "replay_batch_size": len(replay_batch),
            "store_cycle_callable": hasattr(memory.memory_store, "run_consolidation_cycle") if memory.memory_store is not None else False,
            "entries_match_after_bridge": len(memory.episodes) == len(memory.memory_store.entries) if memory.memory_store is not None else False,
        },
        criteria_checks={
            "replay_batch_non_empty": len(replay_batch) >= 1,
            "bridge_exposes_store_cycle": bool(memory.memory_store is not None and hasattr(memory.memory_store, "run_consolidation_cycle")),
            "bridge_preserves_entry_count": bool(memory.memory_store is not None and len(memory.episodes) == len(memory.memory_store.entries)),
        },
        source_seed=0,
    )
    report = memory.run_memory_consolidation_cycle(
        current_cycle=60,
        rng=random.Random(0),
        current_state=_state(),
    )
    cycle_record = _record(
        gate=GATE_LEGACY,
        scenario_id="legacy_bridge_consolidation_cycle",
        api="LongTermMemory.run_memory_consolidation_cycle",
        input_summary={"current_cycle": 60, "rng_seed": 0},
        observed={
            "consolidation_report": report.to_dict(),
            "episodes_after": len(memory.episodes),
            "store_entries_after": len(memory.memory_store.entries) if memory.memory_store is not None else 0,
        },
        criteria_checks={
            "bridge_cycle_returns_report": bool(report.to_dict()),
            "episodes_resynced_after_cycle": bool(memory.memory_store is not None and len(memory.episodes) == len(memory.memory_store.entries)),
        },
        source_seed=0,
    )
    if include_regressions:
        regression_summary = _run_regressions()
        regression_record = _record(
            gate=GATE_LEGACY,
            scenario_id="legacy_regression_prereq",
            api="pytest",
            input_summary={"targets": list(REGRESSION_TARGETS)},
            observed=regression_summary,
            criteria_checks={
                "regressions_executed": regression_summary.get("executed") is True,
                "regressions_passed": regression_summary.get("passed") is True,
                "target_list_complete": regression_summary.get("files") == REGRESSION_TARGETS,
            },
            source_kind=SOURCE_KIND_REGRESSION_RUN,
        )
    else:
        regression_record = _record(
            gate=GATE_LEGACY,
            scenario_id="legacy_regression_prereq",
            api="pytest",
            input_summary={"targets": list(REGRESSION_TARGETS)},
            observed={
                "reason": "Skipped by request: rebuild independent M4.6 evidence without running M4.1-M4.5 regressions.",
                "expected_targets": list(REGRESSION_TARGETS),
            },
            criteria_checks={"explicitly_skipped": True},
            notes=["Regression prerequisite intentionally deferred; formal acceptance conclusion must remain unissued."],
            status=STATUS_NOT_RUN,
            source_kind=SOURCE_KIND_REGRESSION_RUN,
        )
    return [bridge_record, cycle_record, regression_record]


def _expected_record_status(record: dict[str, object]) -> str:
    if str(record.get("scenario_id")) in NOT_RUN_SCENARIOS and dict(record.get("criteria_checks", {})).get("explicitly_skipped") is True:
        return STATUS_NOT_RUN
    criteria_checks = dict(record.get("criteria_checks", {}))
    return STATUS_PASS if criteria_checks and all(criteria_checks.values()) else STATUS_FAIL


def _expected_source_kind(record: dict[str, object]) -> str:
    if str(record.get("gate")) == GATE_HONESTY:
        return SOURCE_KIND_SELF_AUDIT
    if str(record.get("api")) == "pytest":
        return SOURCE_KIND_REGRESSION_RUN
    return SOURCE_KIND_REAL_API_RUN


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _retrieval_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    candidates = [item for item in _as_list(observed.get("candidates")) if isinstance(item, dict)]
    candidate_ids = [str(item) for item in _as_list(observed.get("candidate_ids"))]
    recall = observed.get("recall_hypothesis")
    recall_payload = recall if isinstance(recall, dict) else None
    checks = {
        "candidate_ids_match_candidates": candidate_ids == [str(candidate.get("entry_id")) for candidate in candidates],
        "recall_source_trace_matches_observed": recall_payload is None
        or [str(item) for item in _as_list(recall_payload.get("source_trace"))]
        == [str(item) for item in _as_list(observed.get("source_trace"))],
    }
    scenario_id = str(record.get("scenario_id"))
    if scenario_id == "retrieval_tag_primary":
        checks.update(
            {
                "tag_primary_still_ranked_first": bool(candidate_ids) and candidate_ids[0] == "tag-primary",
                "tag_primary_recall_matches_observed": bool(recall_payload)
                and str(recall_payload.get("primary_entry_id")) == "tag-primary",
            }
        )
    return checks


def _competition_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    candidates = [item for item in _as_list(observed.get("candidates")) if isinstance(item, dict)]
    competition = _as_dict(observed.get("competition"))
    top_candidate_id = str(candidates[0].get("entry_id")) if candidates else None
    competitor_ids = [str(item) for item in _as_list(competition.get("competitor_ids"))]
    checks = {
        "competition_primary_matches_top_candidate": top_candidate_id is not None
        and str(competition.get("primary_id")) == top_candidate_id,
        "competition_competitors_come_from_candidates": set(competitor_ids).issubset(
            {str(candidate.get("entry_id")) for candidate in candidates[1:]}
        ),
        "competition_margin_non_negative": float(competition.get("dominance_margin", -1.0)) >= 0.0,
    }
    if str(record.get("scenario_id")) == "competition_close_margin":
        checks["low_confidence_requires_competitors"] = competition.get("confidence") != "low" or bool(competitor_ids)
    return checks


def _reconstruction_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    if str(record.get("scenario_id")) == "reconstruction_invalid_anchor_rejection":
        return {
            "invalid_anchor_error_present": observed.get("raised") is True and bool(observed.get("error")),
        }
    before = _as_dict(observed.get("before"))
    after = _as_dict(observed.get("after"))
    reconstruction = _as_dict(observed.get("reconstruction_result"))
    trace = _as_dict(reconstruction.get("reconstruction_trace"))
    return {
        "reconstruction_entry_matches_after": str(reconstruction.get("entry_id")) == str(after.get("id")),
        "reconstruction_trace_borrowed_ids_align": _as_list(reconstruction.get("borrowed_source_ids"))
        == _as_list(trace.get("borrowed_source_ids")),
        "reconstruction_triggered_has_reason": reconstruction.get("triggered") is not True or bool(reconstruction.get("trigger_reason")),
        "reconstruction_after_version_not_lower": int(after.get("version", 0)) >= int(before.get("version", 0)),
    }


def _reconsolidation_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    before = _as_dict(observed.get("before"))
    after = _as_dict(observed.get("after"))
    report = _as_dict(observed.get("report"))
    checks = {
        "reconsolidation_report_matches_after": str(report.get("entry_id")) == str(after.get("id")),
        "reconsolidation_confidence_delta_keys_complete": set(_as_dict(report.get("confidence_delta"))) == {
            "source_confidence",
            "reality_confidence",
        },
        "reconsolidation_after_version_not_lower": int(after.get("version", 0)) >= int(before.get("version", 0)),
        "reconsolidation_numeric_delta_complete": set(_as_dict(observed.get("numeric_delta"))) == {
            "accessibility",
            "trace_strength",
            "retrieval_count",
            "abstractness",
            "last_accessed",
        },
    }
    if str(record.get("scenario_id")) == "reconsolidation_structural_reconstruction":
        checks["structural_reconstruction_changes_version"] = report.get("version_changed") is True and int(after.get("version", 0)) > int(before.get("version", 0))
    if str(record.get("scenario_id")) == "reconsolidation_reinforcement_only":
        numeric_delta = _as_dict(observed.get("numeric_delta"))
        checks["reinforcement_updates_last_accessed"] = int(numeric_delta.get("last_accessed", 0)) > 0
    return checks


def _consolidation_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    if str(record.get("scenario_id")) == "consolidation_semantic_skeleton_lineage":
        source_entries = [item for item in _as_list(observed.get("source_entries")) if isinstance(item, dict)]
        source_ids = [str(entry.get("id")) for entry in source_entries]
        skeleton = _as_dict(observed.get("skeleton"))
        metadata = _as_dict(skeleton.get("compression_metadata"))
        return {
            "skeleton_support_ids_match_sources": _as_list(metadata.get("support_entry_ids")) == source_ids,
            "skeleton_lineage_type_present": bool(metadata.get("lineage_type")),
        }
    if str(record.get("scenario_id")) == "consolidation_validation_linkage":
        report = _as_dict(observed.get("report"))
        validated_entries = [item for item in _as_list(observed.get("validated_entries")) if isinstance(item, dict)]
        return {
            "validated_entries_match_report_ids": [str(item.get("id")) for item in validated_entries]
            == [str(item) for item in _as_list(report.get("validated_inference_ids"))],
            "validated_entries_are_long": bool(validated_entries)
            and all(str(item.get("store_level")) == StoreLevel.LONG.value for item in validated_entries),
            "validated_entries_marked_validated": bool(validated_entries)
            and all(_as_dict(item.get("compression_metadata")).get("validation_status") == "validated" for item in validated_entries),
        }
    report = _as_dict(observed.get("report"))
    entries_after = [item for item in _as_list(observed.get("entries_after")) if isinstance(item, dict)]
    extracted_entries = [item for item in _as_list(observed.get("extracted_entries")) if isinstance(item, dict)]
    entry_ids_after = {str(entry.get("id")) for entry in entries_after}
    return {
        "consolidation_report_shape_complete": set(report) == {
            "upgrade",
            "extracted_patterns",
            "replay_created_ids",
            "validated_inference_ids",
            "cleanup",
        },
        "consolidation_extracted_entries_match_report": [str(entry.get("id")) for entry in extracted_entries]
        == [str(item) for item in _as_list(report.get("extracted_patterns"))],
        "retained_source_ids_exist_after_cycle": set(str(item) for item in _as_list(observed.get("retained_source_ids"))).issubset(entry_ids_after),
    }


def _inference_record_consistency(record: dict[str, object]) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    if str(record.get("scenario_id")) == "inference_consolidation_validation_linkage":
        report = _as_dict(observed.get("report"))
        validated_entries = [item for item in _as_list(observed.get("validated_entries")) if isinstance(item, dict)]
        return {
            "validated_entries_match_report_ids": [str(_as_dict(item.get("entry")).get("id")) for item in validated_entries]
            == [str(item) for item in _as_list(report.get("validated_inference_ids"))],
            "traceability_scores_match_metadata": bool(validated_entries)
            and all(
                round(float(_as_dict(_as_dict(item.get("entry")).get("compression_metadata")).get("inference_write_score", -1.0)), 6)
                == round(float(_as_dict(item.get("traceability")).get("score", -2.0)), 6)
                for item in validated_entries
            ),
            "validated_entries_are_long": bool(validated_entries)
            and all(str(_as_dict(item.get("entry")).get("store_level")) == StoreLevel.LONG.value for item in validated_entries),
        }
    after = _as_dict(observed.get("after"))
    validation = _as_dict(observed.get("validation"))
    traceability = _as_dict(observed.get("traceability"))
    checks = {
        "validation_entry_matches_after": str(validation.get("entry_id")) == str(after.get("id")),
        "validation_status_reflected_in_metadata": str(validation.get("validation_status"))
        == str(_as_dict(after.get("compression_metadata")).get("validation_status")),
        "traceability_score_matches_validation": round(float(traceability.get("score", -2.0)), 6)
        == round(float(validation.get("score", -1.0)), 6),
        "traceability_status_matches_validation": str(traceability.get("expected_validation_status")) == str(validation.get("validation_status")),
    }
    if str(record.get("scenario_id")) == "inference_unvalidated_donor_blocked":
        candidates = [item for item in _as_list(observed.get("retrieval_candidates")) if isinstance(item, dict)]
        candidate_ids = [str(item) for item in _as_list(observed.get("candidate_ids"))]
        recall = _as_dict(observed.get("recall_hypothesis"))
        checks.update(
            {
                "retrieval_candidate_ids_match_candidates": candidate_ids == [str(item.get("entry_id")) for item in candidates],
                "unvalidated_not_in_recall_auxiliary_ids": "unvalidated" not in {
                    str(item) for item in _as_list(recall.get("auxiliary_entry_ids"))
                },
            }
        )
    return checks


def _legacy_record_consistency(record: dict[str, object], *, include_regressions: bool) -> dict[str, bool]:
    observed = _as_dict(record.get("observed"))
    scenario_id = str(record.get("scenario_id"))
    if scenario_id == "legacy_regression_prereq":
        if include_regressions:
            command = observed.get("command")
            stdout_tail = [str(item) for item in _as_list(observed.get("stdout_tail"))]
            summary_line = str(observed.get("summary_line", ""))
            returncode = observed.get("returncode")
            passed = observed.get("passed")
            expected_command = [sys.executable, "-m", "pytest", *REGRESSION_TARGETS, "-q"]
            return {
                "regression_files_match_targets": observed.get("files") == REGRESSION_TARGETS,
                "regression_summary_has_execution_flag": observed.get("executed") is True,
                "regression_command_matches_expected": command == expected_command,
                "regression_duration_recorded": isinstance(observed.get("duration_seconds"), (int, float))
                and float(observed.get("duration_seconds")) >= 0.0,
                "regression_pass_flag_matches_returncode": isinstance(returncode, int)
                and isinstance(passed, bool)
                and ((returncode == 0) == passed),
                "regression_stdout_tail_present": bool(stdout_tail),
                "regression_summary_matches_stdout_tail": bool(stdout_tail) and summary_line == stdout_tail[-1],
                "regression_summary_looks_like_pytest_output": bool(summary_line)
                and bool(PYTEST_SUMMARY_LINE_RE.fullmatch(summary_line)),
            }
        return {
            "skip_reason_present": bool(observed.get("reason")),
            "skip_targets_match_expected": observed.get("expected_targets") == REGRESSION_TARGETS,
        }
    if scenario_id == "legacy_bridge_replay_batch":
        return {
            "legacy_bridge_reports_entry_match": observed.get("entries_match_after_bridge") is True,
            "legacy_bridge_reports_store_cycle": observed.get("store_cycle_callable") is True,
        }
    if scenario_id == "legacy_bridge_consolidation_cycle":
        return {
            "legacy_cycle_report_present": bool(_as_dict(observed.get("consolidation_report"))),
            "legacy_cycle_counts_are_non_negative": int(observed.get("episodes_after", -1)) >= 0
            and int(observed.get("store_entries_after", -1)) >= 0,
        }
    return {}


def _record_external_consistency_checks(
    record: dict[str, object],
    *,
    include_regressions: bool,
) -> dict[str, bool]:
    gate = str(record.get("gate"))
    if gate == GATE_RETRIEVAL:
        return _retrieval_record_consistency(record)
    if gate == GATE_COMPETITION:
        return _competition_record_consistency(record)
    if gate == GATE_RECONSTRUCTION:
        return _reconstruction_record_consistency(record)
    if gate == GATE_RECONSOLIDATION:
        return _reconsolidation_record_consistency(record)
    if gate == GATE_CONSOLIDATION:
        return _consolidation_record_consistency(record)
    if gate == GATE_INFERENCE:
        return _inference_record_consistency(record)
    if gate == GATE_LEGACY:
        return _legacy_record_consistency(record, include_regressions=include_regressions)
    return {}


def _gate_summaries_from_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        gate: _gate_summary(gate, [record for record in records if record["gate"] == gate])
        for gate in (
            GATE_RETRIEVAL,
            GATE_COMPETITION,
            GATE_RECONSTRUCTION,
            GATE_RECONSOLIDATION,
            GATE_CONSOLIDATION,
            GATE_INFERENCE,
            GATE_LEGACY,
        )
    }


def _build_honesty_record(
    records: list[dict[str, object]],
    *,
    include_regressions: bool,
) -> dict[str, object]:
    gate_names = {
        GATE_RETRIEVAL,
        GATE_COMPETITION,
        GATE_RECONSTRUCTION,
        GATE_RECONSOLIDATION,
        GATE_CONSOLIDATION,
        GATE_INFERENCE,
        GATE_LEGACY,
    }
    gate_summaries = _gate_summaries_from_records(records)
    gates_seen = {str(record["gate"]) for record in records}
    scenario_ids = {str(record["scenario_id"]) for record in records}
    empty_observed = [str(record["scenario_id"]) for record in records if not record.get("observed")]
    invalid_statuses = [
        str(record["scenario_id"])
        for record in records
        if str(record.get("status")) not in {STATUS_PASS, STATUS_FAIL, STATUS_NOT_RUN}
    ]
    not_run_ids = [str(record["scenario_id"]) for record in records if record.get("status") == STATUS_NOT_RUN]
    missing_provenance_fields: dict[str, list[str]] = {}
    mismatched_status_records: list[str] = []
    mismatched_source_kind_records: list[str] = []
    mismatched_source_api_call_id_records: list[str] = []
    external_check_failures: dict[str, list[str]] = {}
    source_api_call_ids: list[str] = []
    for record in records:
        scenario_id = str(record.get("scenario_id"))
        required_fields = [
            field_name
            for field_name in ("source_kind", "source_api_call_id", "source_input_set_id", "source_seed")
            if field_name not in record
        ]
        if required_fields:
            missing_provenance_fields[scenario_id] = required_fields
        if str(record.get("status")) != _expected_record_status(record):
            mismatched_status_records.append(scenario_id)
        if str(record.get("source_kind")) != _expected_source_kind(record):
            mismatched_source_kind_records.append(scenario_id)
        expected_call_id = _make_source_api_call_id(
            gate=str(record.get("gate")),
            scenario_id=scenario_id,
            api=str(record.get("api")),
            source_seed=int(record["source_seed"]) if isinstance(record.get("source_seed"), int) else None,
        )
        if str(record.get("source_api_call_id")) != expected_call_id:
            mismatched_source_api_call_id_records.append(scenario_id)
        if isinstance(record.get("source_api_call_id"), str):
            source_api_call_ids.append(str(record["source_api_call_id"]))
        consistency_checks = _record_external_consistency_checks(
            record,
            include_regressions=include_regressions,
        )
        failed_checks = [name for name, passed in consistency_checks.items() if not passed]
        if failed_checks:
            external_check_failures[scenario_id] = failed_checks
    duplicate_source_api_call_ids = sorted(
        {
            call_id
            for call_id in source_api_call_ids
            if source_api_call_ids.count(call_id) > 1
        }
    )
    return _record(
        gate=GATE_HONESTY,
        scenario_id="honesty_integrity_audit",
        api="m46_reacceptance.external_honesty_audit",
        input_summary={"include_regressions": include_regressions},
        observed={
            "gate_summaries": gate_summaries,
            "record_count": len(records),
            "gates_seen": sorted(gates_seen),
            "required_integration_scenarios": sorted(REQUIRED_INTEGRATION_SCENARIOS),
            "missing_integration_scenarios": sorted(REQUIRED_INTEGRATION_SCENARIOS - scenario_ids),
            "not_run_scenarios": not_run_ids,
            "empty_observed_scenarios": empty_observed,
            "invalid_status_scenarios": invalid_statuses,
            "missing_provenance_fields": missing_provenance_fields,
            "mismatched_status_records": mismatched_status_records,
            "mismatched_source_kind_records": mismatched_source_kind_records,
            "mismatched_source_api_call_id_records": mismatched_source_api_call_id_records,
            "duplicate_source_api_call_ids": duplicate_source_api_call_ids,
            "external_check_failures": external_check_failures,
        },
        criteria_checks={
            "all_required_gates_present": gates_seen == gate_names,
            "required_integration_scenarios_present": REQUIRED_INTEGRATION_SCENARIOS.issubset(scenario_ids),
            "no_empty_observed_payloads": not empty_observed,
            "record_statuses_valid": not invalid_statuses,
            "record_statuses_match_expected_truth": not mismatched_status_records,
            "only_explicit_scenarios_not_run": set(not_run_ids).issubset(NOT_RUN_SCENARIOS),
            "non_honesty_records_have_provenance": not missing_provenance_fields,
            "source_kinds_align_with_record_type": not mismatched_source_kind_records,
            "source_api_call_ids_unique": not duplicate_source_api_call_ids,
            "source_api_call_ids_align_with_records": not mismatched_source_api_call_id_records,
            "external_cross_checks_pass": not external_check_failures,
            "regression_skip_propagates_to_legacy_gate": gate_summaries[GATE_LEGACY]["status"]
            == (STATUS_PASS if include_regressions else STATUS_NOT_RUN),
        },
        notes=[
            "Honesty gate verifies externally auditable provenance and observed-result consistency for non-honesty records.",
            "This check does not issue formal acceptance; it only audits whether the rebuilt evidence can be independently verified.",
        ],
        source_kind=SOURCE_KIND_SELF_AUDIT,
        source_input_set_id="honesty_external_audit",
    )


def build_m46_reacceptance_report(*, include_regressions: bool = False) -> dict[str, object]:
    records = build_m46_evidence_records(include_regressions=include_regressions)

    gate_summaries = _gate_summaries_from_records(records)
    honesty_record = _build_honesty_record(records, include_regressions=include_regressions)
    records.append(honesty_record)
    gate_summaries[GATE_HONESTY] = _gate_summary(GATE_HONESTY, [honesty_record])

    gate_statuses = [summary["status"] for summary in gate_summaries.values()]
    if any(status == STATUS_FAIL for status in gate_statuses):
        evidence_rebuild_status = STATUS_FAIL
    elif any(status == STATUS_NOT_RUN for status in gate_statuses):
        evidence_rebuild_status = "INCOMPLETE"
    else:
        evidence_rebuild_status = STATUS_PASS

    return {
        "milestone_id": "M4.6",
        "mode": "independent_evidence_rebuild",
        "generated_at": _now_iso(),
        "formal_acceptance_conclusion": FORMAL_CONCLUSION_NOT_ISSUED,
        "evidence_rebuild_status": evidence_rebuild_status,
        "regression_policy": {
            "include_regressions": include_regressions,
            "regression_targets": list(REGRESSION_TARGETS),
        },
        "gate_summaries": gate_summaries,
        "evidence_records": records,
        "notes": [
            "This artifact rebuilds M4.6 evidence from real runtime APIs instead of reusing the prior self-attesting acceptance payload.",
            "This is an independent evidence rebuild, not a formal acceptance pass.",
            "Legacy M4.6 acceptance artifacts are historical only and must not replace this evidence chain.",
            "M4.1-M4.5 regression is intentionally not executed by default; the formal acceptance conclusion remains NOT_ISSUED.",
        ],
    }


def _gate_observation_summary(
    gate: str,
    summary: dict[str, object],
    records: list[dict[str, object]],
) -> str:
    if gate == GATE_RETRIEVAL:
        return "5 retrieval scenarios captured raw candidate rankings, score breakdowns, dormancy filtering, source traces, and procedural recall outlines."
    if gate == GATE_COMPETITION:
        return "2 competition runs captured dominant/high and close/low outcomes with interference metadata and competing interpretations."
    if gate == GATE_RECONSTRUCTION:
        return "A/B/C reconstruction triggers plus anchor-protection observations captured source_type, reality_confidence, version/content_hash, and reconstruction_trace."
    if gate == GATE_RECONSOLIDATION:
        return "7 reconsolidation runs captured reinforcement, rebinding, structural reconstruction, three conflict types, and procedural core-step protection."
    if gate == GATE_CONSOLIDATION:
        return "Offline consolidation evidence captured four-stage execution, extracted semantic/inferred entries, retained episodic supports, and stage report payloads."
    if gate == GATE_INFERENCE:
        return "Validated and blocked inferred cases captured write score, threshold, upgrade behavior, and donor restrictions during retrieval."
    if gate == GATE_LEGACY:
        executed = [record for record in records if record["status"] == STATUS_PASS]
        return (
            f"{len(executed)} bridge scenarios confirmed replay and consolidation-cycle calls return legal results through the legacy bridge."
        )
    return "Honesty audit captured per-gate evidence presence, tri-state status integrity, provenance fields, and cross-check consistency."


def _gate_gap_summary(gate: str, summary: dict[str, object], records: list[dict[str, object]]) -> str:
    if summary["status"] == STATUS_PASS:
        return "No gap within the executed scope."
    if gate == GATE_LEGACY and summary["status"] == STATUS_NOT_RUN:
        return "M4.1-M4.5 regression prerequisite was intentionally skipped, so the regression sub-item stays NOT_RUN and no formal PASS/BLOCK is issued."
    failing = [str(record["scenario_id"]) for record in records if record["status"] == STATUS_FAIL]
    not_run = [str(record["scenario_id"]) for record in records if record["status"] == STATUS_NOT_RUN]
    parts: list[str] = []
    if failing:
        parts.append(f"failed scenarios: {', '.join(failing)}")
    if not_run:
        parts.append(f"not-run scenarios: {', '.join(not_run)}")
    return "; ".join(parts) if parts else "See raw evidence records for missing execution details."


def _summary_lines(report: dict[str, object]) -> list[str]:
    gate_summaries = dict(report["gate_summaries"])
    records = list(report["evidence_records"])
    lines = [
        "# M4.6 Reacceptance Summary",
        "",
        f"Mode: `{report['mode']}`",
        f"Evidence Rebuild Status: `{report['evidence_rebuild_status']}`",
        f"Formal Acceptance Conclusion: `{report['formal_acceptance_conclusion']}`",
        "",
        "This is an independent evidence rebuild, not a formal acceptance pass.",
        "Legacy M4.6 acceptance artifacts are historical only and are not the primary evidence chain.",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name in GATE_ORDER:
        summary = gate_summaries[gate_name]
        counts = summary["counts"]
        gate_records = [record for record in records if record["gate"] == gate_name]
        gate_code = GATE_CODES[gate_name]
        observations = _gate_observation_summary(gate_name, summary, gate_records)
        gaps = _gate_gap_summary(gate_name, summary, gate_records)
        lines.append(
            f"- {gate_code} `{gate_name}`: `{summary['status']}` "
            f"(passed={counts['passed']}, failed={counts['failed']}, not_run={counts['not_run']}). "
            f"Observed: {observations} Gap: {gaps}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This run rebuilds independent evidence only; it does not issue a final M4.6 PASS/BLOCK decision.",
            "- This summary is not a formal acceptance pass and should not be read as one.",
            "- `legacy_integration` stays `NOT_RUN` when M4.1-M4.5 regression is intentionally skipped.",
        ]
    )
    return lines


def write_m46_reacceptance_artifacts(
    *,
    include_regressions: bool = False,
    reports_dir: Path | str | None = None,
) -> dict[str, str]:
    target_reports_dir = Path(reports_dir).resolve() if reports_dir is not None else REPORTS_DIR
    target_reports_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = target_reports_dir / M46_REACCEPTANCE_EVIDENCE_PATH.name
    summary_path = target_reports_dir / M46_REACCEPTANCE_SUMMARY_PATH.name

    report = build_m46_reacceptance_report(include_regressions=include_regressions)
    evidence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text("\n".join(_summary_lines(report)) + "\n", encoding="utf-8")
    return {"evidence": str(evidence_path), "summary": str(summary_path)}
