from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class ConflictSourceCategory(StrEnum):
    IDENTITY_ACTION = "identity_action"
    VALUE_HIERARCHY = "value_hierarchy"
    SURVIVAL_VS_EXPLORATION = "survival_vs_exploration"
    TRUST_VS_THREAT = "trust_vs_threat"
    SOCIAL_RUPTURE = "social_rupture"
    CONTINUITY_ANCHOR = "continuity_anchor"
    SELF_EXPECTATION_FALSIFICATION = "self_expectation_falsification"


class ConflictSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictPersistenceClass(StrEnum):
    EPISODIC = "episodic"
    RECURRING = "recurring"
    LONG_HORIZON = "long_horizon"
    IDENTITY_CRITICAL = "identity_critical"


class ReconciliationStatus(StrEnum):
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    PATCHED = "patched"
    PARTIALLY_RECONCILED = "partially_reconciled"
    REOPENED = "reopened"
    RECONCILED = "reconciled"
    ARCHIVED_UNRESOLVED = "archived_unresolved"
    ARCHIVED_RECONCILED = "archived_reconciled"


class ReconciliationOutcome(StrEnum):
    NONE = "none"
    LOCAL_PATCH = "local_patch"
    TEMPORARY_SUPPRESSION = "temporary_suppression"
    PARTIAL_REPAIR = "partial_repair"
    DEFERRED_CONFLICT = "deferred_conflict"
    UNRESOLVED_CHRONIC = "unresolved_chronic"
    DEEP_REPAIR = "deep_repair"


@dataclass(frozen=True)
class ConflictOrigin:
    signature: str
    source_category: str
    created_tick: int
    chapter_id: int | None = None
    description: str = ""
    source_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "source_category": self.source_category,
            "created_tick": int(self.created_tick),
            "chapter_id": self.chapter_id,
            "description": self.description,
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ConflictOrigin":
        if not payload:
            return cls(signature="", source_category=ConflictSourceCategory.IDENTITY_ACTION.value, created_tick=0)
        chapter_id = payload.get("chapter_id")
        return cls(
            signature=str(payload.get("signature", "")),
            source_category=str(payload.get("source_category", ConflictSourceCategory.IDENTITY_ACTION.value)),
            created_tick=int(payload.get("created_tick", 0)),
            chapter_id=int(chapter_id) if isinstance(chapter_id, (int, float)) else None,
            description=str(payload.get("description", "")),
            source_refs=tuple(str(item) for item in payload.get("source_refs", [])),
        )


@dataclass(frozen=True)
class ConflictPressure:
    tick: int
    chapter_id: int | None
    intensity: float
    source: str
    summary: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": int(self.tick),
            "chapter_id": self.chapter_id,
            "intensity": round(self.intensity, 6),
            "source": self.source,
            "summary": self.summary,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ConflictPressure":
        if not payload:
            return cls(tick=0, chapter_id=None, intensity=0.0, source="", summary="")
        chapter_id = payload.get("chapter_id")
        return cls(
            tick=int(payload.get("tick", 0)),
            chapter_id=int(chapter_id) if isinstance(chapter_id, (int, float)) else None,
            intensity=float(payload.get("intensity", 0.0)),
            source=str(payload.get("source", "")),
            summary=str(payload.get("summary", "")),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass
class RepairAttemptRecord:
    attempt_id: str
    tick: int
    policy: str
    local_success: bool
    classification: str
    target_action: str = ""
    repaired_action: str = ""
    pre_alignment: float = 0.0
    post_alignment: float = 0.0
    invalidated_tick: int | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "tick": int(self.tick),
            "policy": self.policy,
            "local_success": bool(self.local_success),
            "classification": self.classification,
            "target_action": self.target_action,
            "repaired_action": self.repaired_action,
            "pre_alignment": round(self.pre_alignment, 6),
            "post_alignment": round(self.post_alignment, 6),
            "invalidated_tick": self.invalidated_tick,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "RepairAttemptRecord":
        if not payload:
            return cls(attempt_id="", tick=0, policy="", local_success=False, classification="")
        invalidated_tick = payload.get("invalidated_tick")
        return cls(
            attempt_id=str(payload.get("attempt_id", "")),
            tick=int(payload.get("tick", 0)),
            policy=str(payload.get("policy", "")),
            local_success=bool(payload.get("local_success", False)),
            classification=str(payload.get("classification", "")),
            target_action=str(payload.get("target_action", "")),
            repaired_action=str(payload.get("repaired_action", "")),
            pre_alignment=float(payload.get("pre_alignment", 0.0)),
            post_alignment=float(payload.get("post_alignment", 0.0)),
            invalidated_tick=int(invalidated_tick) if isinstance(invalidated_tick, (int, float)) else None,
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass(frozen=True)
class ChapterBridge:
    chapter_id: int
    role: str
    tick: int
    summary: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter_id": int(self.chapter_id),
            "role": self.role,
            "tick": int(self.tick),
            "summary": self.summary,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ChapterBridge":
        if not payload:
            return cls(chapter_id=0, role="persistence", tick=0, summary="")
        return cls(
            chapter_id=int(payload.get("chapter_id", 0)),
            role=str(payload.get("role", "persistence")),
            tick=int(payload.get("tick", 0)),
            summary=str(payload.get("summary", "")),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass(frozen=True)
class NarrativeIntegrationRecord:
    tick: int
    chapter_id: int | None
    status: str
    summary: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": int(self.tick),
            "chapter_id": self.chapter_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NarrativeIntegrationRecord":
        if not payload:
            return cls(tick=0, chapter_id=None, status="", summary="")
        chapter_id = payload.get("chapter_id")
        return cls(
            tick=int(payload.get("tick", 0)),
            chapter_id=int(chapter_id) if isinstance(chapter_id, (int, float)) else None,
            status=str(payload.get("status", "")),
            summary=str(payload.get("summary", "")),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass
class ConflictThread:
    thread_id: str
    signature: str
    title: str
    created_tick: int
    latest_tick: int
    origin: ConflictOrigin
    linked_chapter_ids: list[int] = field(default_factory=list)
    linked_commitments: list[str] = field(default_factory=list)
    linked_values: list[str] = field(default_factory=list)
    linked_identity_elements: list[str] = field(default_factory=list)
    linked_social_entities: list[str] = field(default_factory=list)
    source_category: str = ConflictSourceCategory.IDENTITY_ACTION.value
    severity: str = ConflictSeverity.LOW.value
    recurrence_count: int = 0
    persistence_class: str = ConflictPersistenceClass.EPISODIC.value
    status: str = ReconciliationStatus.ACTIVE.value
    supporting_evidence: list[str] = field(default_factory=list)
    repair_attempt_history: list[RepairAttemptRecord] = field(default_factory=list)
    pressures: list[ConflictPressure] = field(default_factory=list)
    chapter_bridges: list[ChapterBridge] = field(default_factory=list)
    integration_records: list[NarrativeIntegrationRecord] = field(default_factory=list)
    verification_evidence_ids: list[str] = field(default_factory=list)
    stable_confirmations: int = 0
    failed_repairs: int = 0
    current_outcome: str = ReconciliationOutcome.NONE.value
    last_reopened_tick: int | None = None
    archived_tick: int | None = None
    last_status_reason: str = ""
    protected: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "signature": self.signature,
            "title": self.title,
            "created_tick": int(self.created_tick),
            "latest_tick": int(self.latest_tick),
            "origin": self.origin.to_dict(),
            "linked_chapter_ids": list(self.linked_chapter_ids),
            "linked_commitments": list(self.linked_commitments),
            "linked_values": list(self.linked_values),
            "linked_identity_elements": list(self.linked_identity_elements),
            "linked_social_entities": list(self.linked_social_entities),
            "source_category": self.source_category,
            "severity": self.severity,
            "recurrence_count": int(self.recurrence_count),
            "persistence_class": self.persistence_class,
            "status": self.status,
            "supporting_evidence": list(self.supporting_evidence),
            "repair_attempt_history": [item.to_dict() for item in self.repair_attempt_history],
            "pressures": [item.to_dict() for item in self.pressures],
            "chapter_bridges": [item.to_dict() for item in self.chapter_bridges],
            "integration_records": [item.to_dict() for item in self.integration_records],
            "verification_evidence_ids": list(self.verification_evidence_ids),
            "stable_confirmations": int(self.stable_confirmations),
            "failed_repairs": int(self.failed_repairs),
            "current_outcome": self.current_outcome,
            "last_reopened_tick": self.last_reopened_tick,
            "archived_tick": self.archived_tick,
            "last_status_reason": self.last_status_reason,
            "protected": bool(self.protected),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ConflictThread":
        if not payload:
            return cls(
                thread_id="",
                signature="",
                title="",
                created_tick=0,
                latest_tick=0,
                origin=ConflictOrigin.from_dict(None),
            )
        return cls(
            thread_id=str(payload.get("thread_id", "")),
            signature=str(payload.get("signature", "")),
            title=str(payload.get("title", "")),
            created_tick=int(payload.get("created_tick", 0)),
            latest_tick=int(payload.get("latest_tick", 0)),
            origin=ConflictOrigin.from_dict(payload.get("origin") if isinstance(payload.get("origin"), Mapping) else None),
            linked_chapter_ids=[int(item) for item in payload.get("linked_chapter_ids", [])],
            linked_commitments=[str(item) for item in payload.get("linked_commitments", [])],
            linked_values=[str(item) for item in payload.get("linked_values", [])],
            linked_identity_elements=[str(item) for item in payload.get("linked_identity_elements", [])],
            linked_social_entities=[str(item) for item in payload.get("linked_social_entities", [])],
            source_category=str(payload.get("source_category", ConflictSourceCategory.IDENTITY_ACTION.value)),
            severity=str(payload.get("severity", ConflictSeverity.LOW.value)),
            recurrence_count=int(payload.get("recurrence_count", 0)),
            persistence_class=str(payload.get("persistence_class", ConflictPersistenceClass.EPISODIC.value)),
            status=str(payload.get("status", ReconciliationStatus.ACTIVE.value)),
            supporting_evidence=[str(item) for item in payload.get("supporting_evidence", [])],
            repair_attempt_history=[
                RepairAttemptRecord.from_dict(item)
                for item in payload.get("repair_attempt_history", [])
                if isinstance(item, Mapping)
            ],
            pressures=[
                ConflictPressure.from_dict(item)
                for item in payload.get("pressures", [])
                if isinstance(item, Mapping)
            ],
            chapter_bridges=[
                ChapterBridge.from_dict(item)
                for item in payload.get("chapter_bridges", [])
                if isinstance(item, Mapping)
            ],
            integration_records=[
                NarrativeIntegrationRecord.from_dict(item)
                for item in payload.get("integration_records", [])
                if isinstance(item, Mapping)
            ],
            verification_evidence_ids=[str(item) for item in payload.get("verification_evidence_ids", [])],
            stable_confirmations=int(payload.get("stable_confirmations", 0)),
            failed_repairs=int(payload.get("failed_repairs", 0)),
            current_outcome=str(payload.get("current_outcome", ReconciliationOutcome.NONE.value)),
            last_reopened_tick=(
                int(payload.get("last_reopened_tick", 0))
                if isinstance(payload.get("last_reopened_tick"), (int, float))
                else None
            ),
            archived_tick=(
                int(payload.get("archived_tick", 0))
                if isinstance(payload.get("archived_tick"), (int, float))
                else None
            ),
            last_status_reason=str(payload.get("last_status_reason", "")),
            protected=bool(payload.get("protected", False)),
        )


@dataclass
class ReconciliationEngine:
    active_threads: list[ConflictThread] = field(default_factory=list)
    archived_threads: list[ConflictThread] = field(default_factory=list)
    recent_signature_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    last_tick: int = 0
    max_active_threads: int = 12
    archive_limit: int = 48

    def to_dict(self) -> dict[str, object]:
        return {
            "active_threads": [item.to_dict() for item in self.active_threads],
            "archived_threads": [item.to_dict() for item in self.archived_threads],
            "recent_signature_counts": {
                str(key): {str(k): int(v) for k, v in value.items()}
                for key, value in self.recent_signature_counts.items()
            },
            "last_tick": int(self.last_tick),
            "max_active_threads": int(self.max_active_threads),
            "archive_limit": int(self.archive_limit),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ReconciliationEngine":
        if not payload:
            return cls()
        recent_counts = payload.get("recent_signature_counts", {})
        return cls(
            active_threads=[
                ConflictThread.from_dict(item)
                for item in payload.get("active_threads", [])
                if isinstance(item, Mapping)
            ],
            archived_threads=[
                ConflictThread.from_dict(item)
                for item in payload.get("archived_threads", [])
                if isinstance(item, Mapping)
            ],
            recent_signature_counts=(
                {
                    str(key): {str(k): int(v) for k, v in dict(value).items()}
                    for key, value in dict(recent_counts).items()
                    if isinstance(value, Mapping)
                }
                if isinstance(recent_counts, Mapping)
                else {}
            ),
            last_tick=int(payload.get("last_tick", 0)),
            max_active_threads=max(2, int(payload.get("max_active_threads", 12))),
            archive_limit=max(8, int(payload.get("archive_limit", 48))),
        )

    def active_unresolved_threads(self) -> list[ConflictThread]:
        return [
            item
            for item in self.active_threads
            if item.status
            not in {
                ReconciliationStatus.RECONCILED.value,
                ReconciliationStatus.ARCHIVED_RECONCILED.value,
                ReconciliationStatus.ARCHIVED_UNRESOLVED.value,
            }
        ]

    def _thread_outcome_priority(self, outcome: str) -> int:
        ordering = {
            ReconciliationOutcome.DEEP_REPAIR.value: 0,
            ReconciliationOutcome.PARTIAL_REPAIR.value: 1,
            ReconciliationOutcome.LOCAL_PATCH.value: 2,
            ReconciliationOutcome.TEMPORARY_SUPPRESSION.value: 3,
            ReconciliationOutcome.DEFERRED_CONFLICT.value: 4,
            ReconciliationOutcome.UNRESOLVED_CHRONIC.value: 5,
            ReconciliationOutcome.NONE.value: 6,
        }
        return ordering.get(str(outcome), 7)

    def _dominant_thread_priority(self, thread: ConflictThread) -> tuple[object, ...]:
        chapters = len(set(int(item) for item in thread.linked_chapter_ids))
        bridge_count = len(thread.chapter_bridges)
        verification_count = len(thread.verification_evidence_ids)
        reconciled = thread.status == ReconciliationStatus.RECONCILED.value
        partially_reconciled = thread.status == ReconciliationStatus.PARTIALLY_RECONCILED.value
        status_priority = {
            ReconciliationStatus.RECONCILED.value: 0,
            ReconciliationStatus.PARTIALLY_RECONCILED.value: 1,
            ReconciliationStatus.PATCHED.value: 2,
            ReconciliationStatus.REOPENED.value: 3,
            ReconciliationStatus.ACTIVE.value: 4,
            ReconciliationStatus.SUPPRESSED.value: 5,
            ReconciliationStatus.ARCHIVED_RECONCILED.value: 6,
            ReconciliationStatus.ARCHIVED_UNRESOLVED.value: 7,
        }.get(thread.status, 8)
        return (
            status_priority,
            0 if reconciled and chapters >= 2 else 1,
            0 if partially_reconciled and chapters >= 2 else 1,
            0 if chapters >= 2 else 1,
            0 if bridge_count > 0 else 1,
            self._thread_outcome_priority(thread.current_outcome),
            -chapters,
            -bridge_count,
            -verification_count,
            -thread.stable_confirmations,
            -self._severity_weight(thread.severity),
            -thread.recurrence_count,
            thread.created_tick,
            thread.thread_id,
        )

    def dominant_thread(self) -> ConflictThread | None:
        candidates = [
            item
            for item in self.active_threads
            if item.status
            not in {
                ReconciliationStatus.ARCHIVED_RECONCILED.value,
                ReconciliationStatus.ARCHIVED_UNRESOLVED.value,
            }
        ]
        if not candidates:
            return None
        return sorted(candidates, key=self._dominant_thread_priority)[0]

    def _thread_summary(self, thread: ConflictThread | None) -> str:
        if thread is None:
            return "No long-horizon conflict thread is currently dominant."
        chapters = max(1, len(set(int(item) for item in thread.linked_chapter_ids)))
        status = str(thread.status).replace("_", " ")
        if (
            thread.status == ReconciliationStatus.RECONCILED.value
            and thread.current_outcome == ReconciliationOutcome.DEEP_REPAIR.value
        ):
            return f"This long-horizon conflict has been reconciled across {chapters} chapter(s)."
        if thread.status == ReconciliationStatus.PARTIALLY_RECONCILED.value:
            return f"This long-horizon conflict is partially reconciled across {chapters} chapter(s)."
        return f"This tension has persisted across {chapters} chapter(s) and is {status}."

    def explanation_payload(self) -> dict[str, object]:
        active_threads = sorted(self.active_threads, key=self._dominant_thread_priority)
        unresolved_threads = sorted(self.active_unresolved_threads(), key=self._dominant_thread_priority)
        dominant = self.dominant_thread()
        active = [item.to_dict() for item in active_threads]
        unresolved = [item.to_dict() for item in unresolved_threads]
        return {
            "summary": self._thread_summary(dominant),
            "dominant_thread": dominant.to_dict() if dominant is not None else {},
            "active_threads": active,
            "unresolved_threads": unresolved,
            "archived_threads": [item.to_dict() for item in self.archived_threads[-8:]],
            "counts": {
                "active": len(self.active_threads),
                "unresolved": len(unresolved),
                "archived": len(self.archived_threads),
            },
        }

    def action_bias(self, action: str) -> float:
        bias = 0.0
        for thread in self.active_unresolved_threads()[:4]:
            weight = min(0.14, 0.03 + 0.02 * self._severity_weight(thread.severity) + 0.01 * min(thread.recurrence_count, 4))
            if thread.source_category in {
                ConflictSourceCategory.IDENTITY_ACTION.value,
                ConflictSourceCategory.CONTINUITY_ANCHOR.value,
                ConflictSourceCategory.SELF_EXPECTATION_FALSIFICATION.value,
            }:
                if action in {"rest", "scan", "hide"}:
                    bias += weight
                elif action == "forage":
                    bias -= weight * 0.8
            if thread.source_category == ConflictSourceCategory.SURVIVAL_VS_EXPLORATION.value:
                if action in {"scan", "rest"}:
                    bias += weight
                elif action == "forage":
                    bias -= weight * 0.65
            if thread.source_category in {
                ConflictSourceCategory.SOCIAL_RUPTURE.value,
                ConflictSourceCategory.TRUST_VS_THREAT.value,
            }:
                if action == "scan":
                    bias += weight
                elif action == "seek_contact":
                    bias += weight * (0.40 if thread.status == ReconciliationStatus.PARTIALLY_RECONCILED.value else -0.35)
        return round(max(-0.32, min(0.32, bias)), 6)

    def workspace_focus(self) -> dict[str, float]:
        focus: dict[str, float] = {}
        for thread in self.active_unresolved_threads()[:4]:
            weight = round(min(0.95, 0.30 + 0.10 * self._severity_weight(thread.severity)), 6)
            channels = self._thread_channels(thread)
            for channel in channels:
                focus[channel] = max(focus.get(channel, 0.0), weight)
        return focus

    def memory_threshold_delta(self) -> float:
        if not self.active_unresolved_threads():
            return 0.0
        protected = any(item.protected for item in self.active_unresolved_threads())
        strongest = max(self._severity_weight(item.severity) for item in self.active_unresolved_threads())
        delta = -0.03 - strongest * 0.03 - (0.03 if protected else 0.0)
        return round(max(-0.16, min(0.0, delta)), 6)

    def maintenance_signal(self) -> dict[str, object]:
        tasks: list[str] = []
        recommended_action = ""
        priority_gain = 0.0
        suppressed_actions: list[str] = []
        for thread in self.active_unresolved_threads()[:4]:
            priority_gain += 0.06 + 0.02 * self._severity_weight(thread.severity)
            tasks.append(f"reconcile:{thread.thread_id}")
            if thread.source_category in {
                ConflictSourceCategory.SOCIAL_RUPTURE.value,
                ConflictSourceCategory.TRUST_VS_THREAT.value,
            }:
                recommended_action = recommended_action or "scan"
            else:
                recommended_action = recommended_action or "rest"
            if thread.protected:
                suppressed_actions.append("forage")
        return {
            "priority_gain": round(min(0.28, priority_gain), 6),
            "active_tasks": list(dict.fromkeys(tasks)),
            "recommended_action": recommended_action,
            "suppressed_actions": list(dict.fromkeys(suppressed_actions)),
        }

    def continuity_modifier(self) -> float:
        modifier = 0.0
        for thread in self.active_unresolved_threads():
            modifier -= 0.012 * self._severity_weight(thread.severity)
            if thread.protected:
                modifier -= 0.02
        for thread in self.active_threads:
            if thread.status == ReconciliationStatus.RECONCILED.value:
                modifier += 0.015 + min(0.02, thread.stable_confirmations * 0.004)
        return round(max(-0.12, min(0.08, modifier)), 6)

    def observe_runtime(
        self,
        *,
        tick: int,
        diagnostics,
        narrative,
        prediction_ledger,
        verification_loop,
        subject_state,
        continuity_score: float,
        slow_biases: Mapping[str, float] | None = None,
    ) -> dict[str, object]:
        chapter_id = getattr(getattr(narrative, "current_chapter", None), "chapter_id", None)
        candidates = self._collect_candidates(
            tick=tick,
            chapter_id=chapter_id,
            diagnostics=diagnostics,
            prediction_ledger=prediction_ledger,
            subject_state=subject_state,
            continuity_score=continuity_score,
            slow_biases=slow_biases or {},
        )
        matched_ids: list[str] = []
        created_ids: list[str] = []
        reopened_ids: list[str] = []
        updated_ids: list[str] = []
        for candidate in candidates:
            signature = str(candidate["signature"])
            self._note_signature(signature, tick=tick)
            thread = self._find_thread(signature)
            if thread is None:
                archived = self._reopen_archived(signature, tick=tick)
                if archived is not None:
                    thread = archived
                    reopened_ids.append(thread.thread_id)
                elif self._should_promote(candidate):
                    thread = self._create_thread(candidate)
                    self.active_threads.append(thread)
                    created_ids.append(thread.thread_id)
            if thread is None:
                continue
            self._apply_candidate(thread, candidate)
            matched_ids.append(thread.thread_id)
            updated_ids.append(thread.thread_id)
        self._attach_repair_attempts(tick=tick, diagnostics=diagnostics)
        self._attach_verification_evidence(tick=tick, verification_loop=verification_loop)
        self._advance_stability(
            tick=tick,
            matched_ids=set(matched_ids),
            chapter_id=chapter_id,
            continuity_score=continuity_score,
        )
        self._trim()
        self._write_back_to_narrative(
            narrative=narrative,
            tick=tick,
            reason="runtime_observation",
        )
        self.last_tick = tick
        return {
            "created_threads": created_ids,
            "reopened_threads": reopened_ids,
            "updated_threads": updated_ids,
            "active_threads": len(self.active_threads),
            "summary": self.explanation_payload()["summary"],
        }

    def sleep_review(
        self,
        *,
        tick: int,
        sleep_cycle_id: int,
        continuity_score: float,
        verification_loop,
        narrative=None,
    ) -> dict[str, object]:
        promoted: list[str] = []
        downgraded: list[str] = []
        archived: list[str] = []
        for thread in list(self.active_threads):
            if thread.status in {
                ReconciliationStatus.PATCHED.value,
                ReconciliationStatus.PARTIALLY_RECONCILED.value,
            } and continuity_score >= 0.78:
                thread.stable_confirmations += 1
                if thread.status == ReconciliationStatus.PATCHED.value and thread.stable_confirmations >= 1:
                    thread.status = ReconciliationStatus.PARTIALLY_RECONCILED.value
                    thread.current_outcome = ReconciliationOutcome.PARTIAL_REPAIR.value
                    promoted.append(thread.thread_id)
                if (
                    thread.status == ReconciliationStatus.PARTIALLY_RECONCILED.value
                    and thread.stable_confirmations >= 3
                    and len(thread.verification_evidence_ids) >= 2
                    and any(item.local_success for item in thread.repair_attempt_history)
                ):
                    thread.status = ReconciliationStatus.RECONCILED.value
                    thread.current_outcome = ReconciliationOutcome.DEEP_REPAIR.value
                    promoted.append(thread.thread_id)
            elif thread.status == ReconciliationStatus.SUPPRESSED.value and thread.recurrence_count >= 3:
                thread.current_outcome = ReconciliationOutcome.UNRESOLVED_CHRONIC.value
                downgraded.append(thread.thread_id)
            thread.integration_records.append(
                NarrativeIntegrationRecord(
                    tick=tick,
                    chapter_id=(thread.linked_chapter_ids[-1] if thread.linked_chapter_ids else None),
                    status=thread.status,
                    summary=f"sleep review {sleep_cycle_id} classified thread as {thread.status}",
                    evidence=tuple(thread.verification_evidence_ids[-2:]),
                )
            )
        self._attach_verification_evidence(tick=tick, verification_loop=verification_loop)
        for thread in list(self.active_threads):
            if thread.status == ReconciliationStatus.RECONCILED.value and not thread.protected and thread.stable_confirmations >= 4:
                thread.status = ReconciliationStatus.ARCHIVED_RECONCILED.value
                thread.archived_tick = tick
                self.archived_threads.append(thread)
                self.active_threads.remove(thread)
                archived.append(thread.thread_id)
            elif (
                thread.status in {ReconciliationStatus.SUPPRESSED.value, ReconciliationStatus.ACTIVE.value}
                and not thread.protected
                and tick - thread.latest_tick >= 10
            ):
                thread.status = ReconciliationStatus.ARCHIVED_UNRESOLVED.value
                thread.archived_tick = tick
                self.archived_threads.append(thread)
                self.active_threads.remove(thread)
                archived.append(thread.thread_id)
        self._trim()
        self._write_back_to_narrative(
            narrative=narrative,
            tick=tick,
            reason=f"sleep_review:{sleep_cycle_id}",
        )
        self.last_tick = tick
        return {
            "promoted_threads": promoted,
            "downgraded_threads": downgraded,
            "archived_threads": archived,
            "summary": self.explanation_payload()["summary"],
        }

    def _write_back_to_narrative(self, *, narrative, tick: int, reason: str) -> None:
        if narrative is None:
            return
        payload = self.explanation_payload()
        summary = str(payload.get("summary", ""))
        counts = dict(payload.get("counts", {}))
        unresolved = [dict(item) for item in payload.get("unresolved_threads", [])]
        dominant_payload = payload.get("dominant_thread", {})
        dominant = dict(dominant_payload) if isinstance(dominant_payload, Mapping) and dominant_payload else None
        chapter_id = None
        current_chapter = getattr(narrative, "current_chapter", None)
        if current_chapter is not None:
            chapter_id = getattr(current_chapter, "chapter_id", None)
            state_summary = getattr(current_chapter, "state_summary", None)
            if not isinstance(state_summary, dict):
                state_summary = {}
            state_summary["reconciliation"] = {
                "summary": summary,
                "counts": counts,
                "dominant_thread_id": str(dominant.get("thread_id", "")) if dominant else "",
                "dominant_status": str(dominant.get("status", "")) if dominant else "",
                "dominant_outcome": str(dominant.get("current_outcome", "")) if dominant else "",
                "linked_chapter_ids": list(dominant.get("linked_chapter_ids", [])) if dominant else [],
                "chapter_bridge_count": len(list(dominant.get("chapter_bridges", []))) if dominant else 0,
                "reason": reason,
                "tick": int(tick),
            }
            current_chapter.state_summary = state_summary
            key_events = list(getattr(current_chapter, "key_events", []))
            if summary and summary not in key_events:
                current_chapter.key_events = [*key_events[-4:], summary]

        transition_evidence = getattr(narrative, "chapter_transition_evidence", None)
        if not isinstance(transition_evidence, list):
            transition_evidence = []
        transition_entry = {
            "type": "reconciliation_writeback",
            "tick": int(tick),
            "reason": reason,
            "chapter_id": chapter_id,
            "summary": summary,
            "dominant_thread_id": str(dominant.get("thread_id", "")) if dominant else "",
            "dominant_status": str(dominant.get("status", "")) if dominant else "",
            "dominant_outcome": str(dominant.get("current_outcome", "")) if dominant else "",
            "linked_chapter_ids": list(dominant.get("linked_chapter_ids", [])) if dominant else [],
        }
        if not transition_evidence or transition_evidence[-1] != transition_entry:
            transition_evidence.append(transition_entry)
        narrative.chapter_transition_evidence = transition_evidence[-24:]

        contradiction_summary = getattr(narrative, "contradiction_summary", None)
        if not isinstance(contradiction_summary, dict):
            contradiction_summary = {}
        contradiction_summary["reconciliation"] = {
            "summary": summary,
            "counts": counts,
            "reason": reason,
            "tick": int(tick),
            "dominant_thread_id": str(dominant.get("thread_id", "")) if dominant else "",
            "dominant_status": str(dominant.get("status", "")) if dominant else "",
            "dominant_outcome": str(dominant.get("current_outcome", "")) if dominant else "",
            "linked_chapter_ids": list(dominant.get("linked_chapter_ids", [])) if dominant else [],
            "unresolved_thread_ids": [str(item.get("thread_id", "")) for item in unresolved[:4]],
        }
        narrative.contradiction_summary = contradiction_summary

        evidence_provenance = getattr(narrative, "evidence_provenance", None)
        if not isinstance(evidence_provenance, dict):
            evidence_provenance = {}
        for thread in self.active_threads[-8:]:
            evidence_provenance[f"reconciliation:{thread.thread_id}"] = {
                "thread_id": thread.thread_id,
                "signature": thread.signature,
                "status": thread.status,
                "current_outcome": thread.current_outcome,
                "linked_chapter_ids": list(thread.linked_chapter_ids),
                "verification_evidence_ids": list(thread.verification_evidence_ids[-4:]),
                "latest_tick": int(thread.latest_tick),
            }
        claim_revisions = self._write_back_to_claims(
            narrative=narrative,
            dominant=dominant,
            tick=tick,
        )
        if claim_revisions:
            evidence_provenance["reconciliation_claim_updates"] = {
                "tick": int(tick),
                "dominant_thread_id": str(dominant.get("thread_id", "")) if dominant else "",
                "updated_claim_ids": [str(item["claim_id"]) for item in claim_revisions],
                "contested_claim_ids": [
                    str(item["claim_id"]) for item in claim_revisions if bool(item.get("contested", False))
                ],
            }
        narrative.evidence_provenance = evidence_provenance

        significant_events = list(getattr(narrative, "significant_events", []))
        if summary and summary not in significant_events:
            narrative.significant_events = [*significant_events[-7:], summary]

        clause = self._reconciliation_clause(summary, dominant)
        narrative.autobiographical_summary = self._merge_reconciliation_clause(
            getattr(narrative, "autobiographical_summary", ""),
            clause,
        )
        narrative.core_summary = self._merge_reconciliation_clause(
            getattr(narrative, "core_summary", ""),
            clause,
        )
        self._recalibrate_commitments_from_claims(
            narrative=narrative,
            claim_revisions=claim_revisions,
            dominant=dominant,
            tick=tick,
        )
        narrative.last_updated_tick = max(int(getattr(narrative, "last_updated_tick", 0)), int(tick))
        narrative.version = int(getattr(narrative, "version", 0)) + 1

    def _write_back_to_claims(
        self,
        *,
        narrative,
        dominant: dict[str, object] | None,
        tick: int,
    ) -> list[dict[str, object]]:
        claims = getattr(narrative, "claims", None)
        if not isinstance(claims, list):
            return []
        target_claims = self._target_claims_for_reconciliation(
            narrative=narrative,
            dominant=dominant,
        )
        if not target_claims:
            return []
        thread_id = str(dominant.get("thread_id", "")) if dominant else ""
        status = str(dominant.get("status", "")) if dominant else ""
        outcome = str(dominant.get("current_outcome", "")) if dominant else ""
        linked_chapter_ids = [
            int(item) for item in dominant.get("linked_chapter_ids", []) if isinstance(item, int)
        ] if dominant else []
        evidence_ids = self._claim_reconciliation_evidence_ids(dominant)
        revisions: list[dict[str, object]] = []
        for claim in target_claims:
            if str(getattr(claim, "claim_type", "")) == "reconciliation":
                self._refresh_reconciliation_claim(claim=claim, dominant=dominant)
            claim.reconciliation_thread_id = thread_id
            claim.reconciliation_status = status
            claim.reconciliation_outcome = outcome
            claim.reconciliation_tick = int(tick)
            claim.reconciliation_source_chapter_ids = list(linked_chapter_ids)
            claim.reconciliation_evidence_ids = list(evidence_ids)
            claim.last_validated_at = max(int(getattr(claim, "last_validated_at", 0)), int(tick))
            claim.stale_since = None

            if status == ReconciliationStatus.RECONCILED.value:
                claim.supported_by = list(dict.fromkeys([*claim.supported_by, *evidence_ids]))[-16:]
                claim.support_count = max(int(claim.support_count), len(claim.supported_by))
                claim.support_score = round(max(float(claim.support_score), float(claim.support_count), 1.0), 4)
                if claim.contradict_count > 0:
                    claim.contradict_count = max(0, int(claim.contradict_count) - 1)
                claim.contradiction_score = round(max(0.0, float(claim.contradiction_score) * 0.5), 4)
                claim.reconciliation_contested = False
                claim.confidence = round(max(float(claim.confidence), 0.72), 4)
            elif status in {
                ReconciliationStatus.PARTIALLY_RECONCILED.value,
                ReconciliationStatus.REOPENED.value,
                ReconciliationStatus.ACTIVE.value,
                ReconciliationStatus.SUPPRESSED.value,
            }:
                claim.contradicted_by = list(dict.fromkeys([*claim.contradicted_by, *evidence_ids]))[-16:]
                claim.contradict_count = max(int(claim.contradict_count), len(claim.contradicted_by), 1)
                claim.contradiction_score = round(
                    max(float(claim.contradiction_score), float(claim.contradict_count), 0.5),
                    4,
                )
                claim.reconciliation_contested = True
                confidence_cap = 0.58 if status == ReconciliationStatus.PARTIALLY_RECONCILED.value else 0.45
                claim.confidence = round(min(float(claim.confidence), confidence_cap), 4)
            revisions.append(
                {
                    "claim_id": str(claim.claim_id),
                    "claim_key": str(claim.claim_key),
                    "contested": bool(getattr(claim, "reconciliation_contested", False)),
                }
            )

        contradiction_summary = getattr(narrative, "contradiction_summary", None)
        if isinstance(contradiction_summary, dict):
            contradiction_summary["reconciled_claim_ids"] = [
                str(item["claim_id"]) for item in revisions if not bool(item.get("contested", False))
            ]
            contradiction_summary["contested_claim_ids"] = [
                str(item["claim_id"]) for item in revisions if bool(item.get("contested", False))
            ]
            narrative.contradiction_summary = contradiction_summary
        return revisions

    def _refresh_reconciliation_claim(self, *, claim, dominant: dict[str, object] | None) -> None:
        if not dominant:
            return
        anchors = self._claim_anchor_tokens(dominant)
        claim_key = anchors[0] if anchors else "reconciliation"
        thread_id = str(dominant.get("thread_id", "unknown")).replace(":", "-")
        claim.claim_id = f"claim-reconciliation-{thread_id}"
        claim.claim_key = claim_key
        claim.text = (
            f"Reconciliation for {claim_key.replace('_', ' ')} remains "
            f"{str(dominant.get('status', 'active')).replace('_', ' ')} across chapters."
        )

    def _target_claims_for_reconciliation(self, *, narrative, dominant: dict[str, object] | None) -> list[object]:
        claims = getattr(narrative, "claims", None)
        if not isinstance(claims, list):
            return []
        if not claims:
            synthesized = self._synthesize_reconciliation_claim(dominant)
            if synthesized is None:
                return []
            narrative.claims = [synthesized]
            return narrative.claims
        if not dominant:
            return []
        anchors = set(self._claim_anchor_tokens(dominant))
        matched: list[object] = []
        explicit_thread_matches: list[object] = []
        for claim in claims:
            if self._claim_matches_reconciliation_thread(claim=claim, dominant=dominant):
                explicit_thread_matches.append(claim)
                continue
            claim_tokens = self._claim_tokens(claim)
            if anchors & claim_tokens:
                matched.append(claim)
        if matched:
            return matched[:4]
        if explicit_thread_matches:
            return explicit_thread_matches[:4]
        synthesized = self._synthesize_reconciliation_claim(dominant)
        if synthesized is None:
            return []
        claims.append(synthesized)
        return [synthesized]

    def _synthesize_reconciliation_claim(self, dominant: dict[str, object] | None):
        if not dominant:
            return None
        from .self_model import NarrativeClaim

        anchors = self._claim_anchor_tokens(dominant)
        if not anchors:
            return None
        claim_key = anchors[0]
        claim_text = (
            f"Reconciliation for {claim_key.replace('_', ' ')} remains "
            f"{str(dominant.get('status', 'active')).replace('_', ' ')} across chapters."
        )
        return NarrativeClaim(
            claim_id=f"claim-reconciliation-{str(dominant.get('thread_id', 'unknown')).replace(':', '-')}",
            claim_type="reconciliation",
            text=claim_text,
            claim_key=claim_key,
        )

    def _claim_anchor_tokens(self, dominant: dict[str, object] | None) -> list[str]:
        if not dominant:
            return []
        anchors = self._normalized_items(
            [
                *dominant.get("linked_commitments", []),
                *dominant.get("linked_identity_elements", []),
            ]
        )
        signature = str(dominant.get("signature", ""))
        if signature:
            anchors = tuple([*anchors, *self._normalized_items(signature.split(":"))])
        if "adaptive_exploration" in anchors:
            anchors = tuple([*anchors, "aggressive", "scan", "seek_contact"])
        if "core_survival" in anchors or "continuity" in anchors:
            anchors = tuple([*anchors, "survival_priority", "rest", "hide", "exploit_shelter"])
        return list(dict.fromkeys(item for item in anchors if item))

    def _claim_tokens(self, claim) -> set[str]:
        return set(
            self._normalized_items(
                [
                    getattr(claim, "claim_key", ""),
                    getattr(claim, "claim_type", ""),
                    *getattr(claim, "supported_by", []),
                    *getattr(claim, "contradicted_by", []),
                ]
            )
        )

    def _claim_matches_reconciliation_thread(self, *, claim, dominant: dict[str, object] | None) -> bool:
        if not dominant:
            return False
        dominant_thread_id = str(dominant.get("thread_id", ""))
        if dominant_thread_id and str(getattr(claim, "reconciliation_thread_id", "")) == dominant_thread_id:
            return True
        synthesized_claim_id = (
            f"claim-reconciliation-{dominant_thread_id.replace(':', '-')}" if dominant_thread_id else ""
        )
        return bool(
            synthesized_claim_id
            and str(getattr(claim, "claim_type", "")) == "reconciliation"
            and str(getattr(claim, "claim_id", "")) == synthesized_claim_id
        )

    def _claim_reconciliation_evidence_ids(self, dominant: dict[str, object] | None) -> list[str]:
        if not dominant:
            return []
        evidence = [
            *[str(item) for item in dominant.get("verification_evidence_ids", [])[-4:]],
            *[str(item) for item in dominant.get("supporting_evidence", [])[-4:]],
            f"thread:{str(dominant.get('thread_id', ''))}",
        ]
        return list(dict.fromkeys(item for item in evidence if item))

    def _recalibrate_commitments_from_claims(
        self,
        *,
        narrative,
        claim_revisions: list[dict[str, object]],
        dominant: dict[str, object] | None,
        tick: int,
    ) -> None:
        commitments = getattr(narrative, "commitments", None)
        claims = getattr(narrative, "claims", None)
        if not isinstance(commitments, list) or not isinstance(claims, list) or not claim_revisions:
            return
        revised_ids = {str(item["claim_id"]) for item in claim_revisions}
        revised_claims = [claim for claim in claims if str(getattr(claim, "claim_id", "")) in revised_ids]
        if not revised_claims:
            return
        dominant_status = str(dominant.get("status", "")) if dominant else ""
        dominant_thread_id = str(dominant.get("thread_id", "")) if dominant else ""
        evidence_ids = self._claim_reconciliation_evidence_ids(dominant)
        for commitment in commitments:
            source_claim_ids = {str(item) for item in getattr(commitment, "source_claim_ids", [])}
            relevant_claims = [
                claim for claim in revised_claims if str(getattr(claim, "claim_id", "")) in source_claim_ids
            ]
            if not relevant_claims:
                continue
            mean_confidence = sum(float(getattr(claim, "confidence", 0.0)) for claim in relevant_claims) / len(relevant_claims)
            any_contested = any(bool(getattr(claim, "reconciliation_contested", False)) for claim in relevant_claims)
            commitment.confidence = round(_clamp(mean_confidence, low=0.2, high=1.0), 4)
            if any_contested:
                commitment.priority = round(_clamp(float(commitment.priority) - 0.05, low=0.2, high=1.0), 4)
            else:
                commitment.priority = round(_clamp(float(commitment.priority) + 0.03, low=0.2, high=1.0), 4)
            commitment.last_reaffirmed_tick = int(tick)
            commitment.evidence_ids = list(dict.fromkeys([*getattr(commitment, "evidence_ids", []), *evidence_ids]))[-12:]
            if dominant_thread_id and dominant_thread_id not in commitment.evidence_ids:
                commitment.evidence_ids.append(dominant_thread_id)
                commitment.evidence_ids = commitment.evidence_ids[-12:]
            if dominant_status in {
                ReconciliationStatus.ACTIVE.value,
                ReconciliationStatus.REOPENED.value,
                ReconciliationStatus.SUPPRESSED.value,
            } and any_contested:
                commitment.last_violated_tick = int(tick)

    def _reconciliation_clause(self, summary: str, dominant: dict[str, object] | None) -> str:
        if dominant is None:
            return "Reconciliation: no long-horizon conflict thread is currently dominant."
        chapters = len(set(int(item) for item in dominant.get("linked_chapter_ids", []) if isinstance(item, int)))
        status = str(dominant.get("status", "")).replace("_", " ")
        title = str(dominant.get("title", "long-horizon conflict")).strip() or "long-horizon conflict"
        if summary:
            return f"Reconciliation: {summary}"
        return (
            f"Reconciliation: {title} spans {max(1, chapters)} chapter(s) "
            f"and is currently {status}."
        )

    def _merge_reconciliation_clause(self, text: str, clause: str) -> str:
        base = str(text or "").strip()
        marker = "Reconciliation:"
        if marker in base:
            base = base.split(marker, 1)[0].strip()
        if not base:
            return clause
        return f"{base} {clause}"

    def _collect_candidates(
        self,
        *,
        tick: int,
        chapter_id: int | None,
        diagnostics,
        prediction_ledger,
        subject_state,
        continuity_score: float,
        slow_biases: Mapping[str, float],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        if diagnostics is not None:
            violated = [str(item) for item in getattr(diagnostics, "violated_commitments", [])]
            relevant = [str(item) for item in getattr(diagnostics, "relevant_commitments", [])]
            social_alerts = [str(item) for item in getattr(diagnostics, "social_alerts", [])]
            identity_tension = float(getattr(diagnostics, "identity_tension", 0.0))
            inconsistency = float(getattr(diagnostics, "self_inconsistency_error", 0.0))
            if identity_tension >= 0.10 or inconsistency >= 0.10 or social_alerts:
                category = self._category_for_diagnostics(diagnostics, social_alerts=social_alerts)
                anchors = violated or relevant or social_alerts or [str(getattr(diagnostics, "conflict_type", "identity"))]
                signature = self._signature(category, anchors)
                candidates.append(
                    {
                        "signature": signature,
                        "title": str(getattr(diagnostics, "conflict_type", "long conflict")).replace("_", " "),
                        "source_category": category,
                        "intensity": max(identity_tension, inconsistency, 0.18),
                        "chapter_id": chapter_id,
                        "tick": tick,
                        "supporting_evidence": [
                            *violated[:3],
                            *social_alerts[:2],
                            str(getattr(diagnostics, "severity_level", "")),
                        ],
                        "linked_commitments": violated or relevant,
                        "linked_identity_elements": list(dict.fromkeys(violated or relevant)),
                        "linked_social_entities": social_alerts[:2],
                        "protected": bool(getattr(subject_state, "status_flags", {}).get("continuity_fragile", False))
                        or identity_tension >= 0.40,
                        "summary": f"diagnostics:{getattr(diagnostics, 'conflict_type', 'identity')}",
                    }
                )
        if prediction_ledger is not None:
            for discrepancy in prediction_ledger.top_discrepancies(limit=4):
                if not (
                    getattr(discrepancy, "chronic", False)
                    or getattr(discrepancy, "identity_relevant", False)
                    or getattr(discrepancy, "subject_critical", False)
                    or int(getattr(discrepancy, "recurrence_count", 0)) >= 2
                ):
                    continue
                category = self._category_for_discrepancy(str(getattr(discrepancy, "discrepancy_type", "")))
                anchors = list(getattr(discrepancy, "linked_commitments", ())) or [str(getattr(discrepancy, "discrepancy_type", ""))]
                candidates.append(
                    {
                        "signature": self._signature(category, anchors),
                        "title": str(getattr(discrepancy, "label", "ledger conflict")),
                        "source_category": category,
                        "intensity": min(1.0, float(getattr(discrepancy, "severity", 0.0)) + (0.12 if getattr(discrepancy, "chronic", False) else 0.0)),
                        "chapter_id": chapter_id,
                        "tick": tick,
                        "supporting_evidence": [str(item) for item in getattr(discrepancy, "supporting_evidence", ())][:3],
                        "linked_commitments": [str(item) for item in getattr(discrepancy, "linked_commitments", ())],
                        "linked_identity_elements": [str(item) for item in getattr(discrepancy, "linked_predictions", ())][:2],
                        "linked_social_entities": [],
                        "protected": bool(getattr(discrepancy, "identity_relevant", False) or getattr(discrepancy, "subject_critical", False)),
                        "summary": f"ledger:{getattr(discrepancy, 'discrepancy_type', '')}",
                    }
                )
        if continuity_score < 0.78:
            candidates.append(
                {
                    "signature": self._signature(
                        ConflictSourceCategory.CONTINUITY_ANCHOR.value,
                        list(getattr(subject_state, "continuity_anchors", ()))[:3] or ["continuity"],
                    ),
                    "title": "continuity anchor contradiction",
                    "source_category": ConflictSourceCategory.CONTINUITY_ANCHOR.value,
                    "intensity": 1.0 - continuity_score,
                    "chapter_id": chapter_id,
                    "tick": tick,
                    "supporting_evidence": list(getattr(subject_state, "continuity_anchors", ()))[:3] or ["continuity_score_drop"],
                    "linked_commitments": [],
                    "linked_identity_elements": list(getattr(subject_state, "continuity_anchors", ()))[:3],
                    "linked_social_entities": [],
                    "protected": True,
                    "summary": "continuity:fragile",
                }
            )
        if float(slow_biases.get("continuity_resilience", 0.5)) < 0.38 and candidates:
            for candidate in candidates:
                candidate["intensity"] = min(1.0, float(candidate["intensity"]) + 0.06)
                candidate["supporting_evidence"] = [*candidate["supporting_evidence"], "slow_resilience_drop"]
        return candidates

    def _should_promote(self, candidate: Mapping[str, object]) -> bool:
        counts = self.recent_signature_counts.get(str(candidate["signature"]), {})
        observed = int(counts.get("count", 0))
        intensity = float(candidate.get("intensity", 0.0))
        protected = bool(candidate.get("protected", False))
        return observed >= 2 or intensity >= 0.65 or protected

    def _create_thread(self, candidate: Mapping[str, object]) -> ConflictThread:
        signature = str(candidate["signature"])
        tick = int(candidate["tick"])
        chapter_id = candidate.get("chapter_id")
        origin = ConflictOrigin(
            signature=signature,
            source_category=str(candidate.get("source_category", ConflictSourceCategory.IDENTITY_ACTION.value)),
            created_tick=tick,
            chapter_id=int(chapter_id) if isinstance(chapter_id, int) else None,
            description=str(candidate.get("summary", candidate.get("title", ""))),
            source_refs=tuple(str(item) for item in candidate.get("supporting_evidence", [])),
        )
        thread = ConflictThread(
            thread_id=f"conflict:{signature}:{tick}",
            signature=signature,
            title=str(candidate.get("title", signature)),
            created_tick=tick,
            latest_tick=tick,
            origin=origin,
            linked_chapter_ids=([int(chapter_id)] if isinstance(chapter_id, int) else []),
            linked_commitments=[str(item) for item in candidate.get("linked_commitments", [])],
            linked_identity_elements=[str(item) for item in candidate.get("linked_identity_elements", [])],
            linked_social_entities=[str(item) for item in candidate.get("linked_social_entities", [])],
            source_category=str(candidate.get("source_category", ConflictSourceCategory.IDENTITY_ACTION.value)),
            severity=self._severity_name(float(candidate.get("intensity", 0.0))),
            recurrence_count=0,
            persistence_class=ConflictPersistenceClass.EPISODIC.value,
            status=ReconciliationStatus.ACTIVE.value,
            supporting_evidence=[str(item) for item in candidate.get("supporting_evidence", [])][:8],
            protected=bool(candidate.get("protected", False)),
        )
        thread.integration_records.append(
            NarrativeIntegrationRecord(
                tick=tick,
                chapter_id=(int(chapter_id) if isinstance(chapter_id, int) else None),
                status=thread.status,
                summary=f"promoted long-horizon conflict from {candidate.get('summary', 'runtime evidence')}",
                evidence=tuple(thread.supporting_evidence[:3]),
            )
        )
        return thread

    def _apply_candidate(self, thread: ConflictThread, candidate: Mapping[str, object]) -> None:
        tick = int(candidate.get("tick", thread.latest_tick))
        chapter_id = candidate.get("chapter_id")
        thread.latest_tick = tick
        thread.recurrence_count += 1
        thread.severity = self._severity_name(
            max(float(candidate.get("intensity", 0.0)), self._severity_value(thread.severity))
        )
        if isinstance(chapter_id, int) and chapter_id not in thread.linked_chapter_ids:
            role = "origin" if not thread.linked_chapter_ids else "persistence"
            thread.linked_chapter_ids.append(chapter_id)
            thread.chapter_bridges.append(
                ChapterBridge(
                    chapter_id=chapter_id,
                    role=role,
                    tick=tick,
                    summary=f"conflict persisted into chapter {chapter_id}",
                    evidence=tuple(str(item) for item in candidate.get("supporting_evidence", [])[:2]),
                )
            )
        for attr in ("linked_commitments", "linked_values", "linked_identity_elements", "linked_social_entities"):
            current = getattr(thread, attr)
            values = [str(item) for item in candidate.get(attr, [])]
            setattr(thread, attr, list(dict.fromkeys([*current, *values]))[:8])
        thread.supporting_evidence = list(dict.fromkeys([*thread.supporting_evidence, *[str(item) for item in candidate.get("supporting_evidence", [])]]))[:16]
        thread.pressures.append(
            ConflictPressure(
                tick=tick,
                chapter_id=int(chapter_id) if isinstance(chapter_id, int) else None,
                intensity=float(candidate.get("intensity", 0.0)),
                source=str(candidate.get("summary", "runtime")),
                summary=str(candidate.get("title", thread.title)),
                evidence=tuple(str(item) for item in candidate.get("supporting_evidence", [])[:3]),
            )
        )
        thread.pressures = thread.pressures[-16:]
        thread.persistence_class = self._persistence_class(thread)
        if thread.status in {
            ReconciliationStatus.RECONCILED.value,
            ReconciliationStatus.ARCHIVED_RECONCILED.value,
        }:
            thread.status = ReconciliationStatus.REOPENED.value
            thread.current_outcome = ReconciliationOutcome.UNRESOLVED_CHRONIC.value
            thread.last_reopened_tick = tick
            for attempt in thread.repair_attempt_history[-2:]:
                if attempt.invalidated_tick is None:
                    attempt.__dict__["invalidated_tick"] = tick
        elif thread.status == ReconciliationStatus.PATCHED.value and thread.recurrence_count >= 2:
            thread.status = ReconciliationStatus.REOPENED.value
            thread.current_outcome = ReconciliationOutcome.UNRESOLVED_CHRONIC.value
            thread.last_reopened_tick = tick
        elif thread.status not in {
            ReconciliationStatus.REOPENED.value,
            ReconciliationStatus.PARTIALLY_RECONCILED.value,
        }:
            thread.status = ReconciliationStatus.ACTIVE.value
        thread.stable_confirmations = 0
        thread.last_status_reason = str(candidate.get("summary", "reactivated"))

    def _advance_stability(
        self,
        *,
        tick: int,
        matched_ids: set[str],
        chapter_id: int | None,
        continuity_score: float,
    ) -> None:
        for thread in self.active_threads:
            if thread.thread_id in matched_ids:
                continue
            if thread.status in {
                ReconciliationStatus.PATCHED.value,
                ReconciliationStatus.PARTIALLY_RECONCILED.value,
                ReconciliationStatus.REOPENED.value,
            } and continuity_score >= 0.80:
                thread.stable_confirmations += 1
            if thread.status == ReconciliationStatus.REOPENED.value and continuity_score >= 0.76:
                thread.status = ReconciliationStatus.PARTIALLY_RECONCILED.value
                thread.current_outcome = ReconciliationOutcome.PARTIAL_REPAIR.value
            if chapter_id is not None and chapter_id not in thread.linked_chapter_ids and tick - thread.latest_tick <= 2:
                thread.linked_chapter_ids.append(chapter_id)

    def _attach_repair_attempts(self, *, tick: int, diagnostics) -> None:
        repair_result = getattr(diagnostics, "repair_result", {}) if diagnostics is not None else {}
        repair_policy = str(getattr(diagnostics, "repair_policy", "")) if diagnostics is not None else ""
        if not repair_policy and not repair_result:
            return
        thread = self._match_thread_for_repair_attempt(diagnostics=diagnostics)
        if thread is None:
            return
        success = bool(repair_result.get("success", False))
        post_alignment = float(
            repair_result.get(
                "post_alignment",
                getattr(diagnostics, "commitment_compatibility_score", 0.5),
            )
        )
        pre_alignment = float(repair_result.get("pre_alignment", max(0.0, post_alignment - 0.2)))
        classification = (
            ReconciliationOutcome.LOCAL_PATCH.value
            if success and post_alignment < 0.85
            else ReconciliationOutcome.PARTIAL_REPAIR.value
            if success
            else ReconciliationOutcome.TEMPORARY_SUPPRESSION.value
        )
        thread.repair_attempt_history.append(
            RepairAttemptRecord(
                attempt_id=f"{thread.thread_id}:repair:{tick}",
                tick=tick,
                policy=repair_policy or str(repair_result.get("policy", "")),
                local_success=success,
                classification=classification,
                target_action=str(repair_result.get("target_action", "")),
                repaired_action=str(repair_result.get("repaired_action", "")),
                pre_alignment=pre_alignment,
                post_alignment=post_alignment,
                evidence=tuple([str(item) for item in getattr(diagnostics, "violated_commitments", [])][:3]),
            )
        )
        thread.repair_attempt_history = thread.repair_attempt_history[-12:]
        if success:
            thread.status = (
                ReconciliationStatus.PATCHED.value
                if classification == ReconciliationOutcome.LOCAL_PATCH.value
                else ReconciliationStatus.PARTIALLY_RECONCILED.value
            )
            thread.current_outcome = classification
        else:
            thread.failed_repairs += 1
            thread.status = ReconciliationStatus.SUPPRESSED.value
            thread.current_outcome = ReconciliationOutcome.TEMPORARY_SUPPRESSION.value

    def _match_thread_for_repair_attempt(self, *, diagnostics) -> ConflictThread | None:
        explicit_thread_id = str(getattr(diagnostics, "repair_thread_id", "") or "")
        if explicit_thread_id:
            for thread in self.active_threads:
                if thread.thread_id == explicit_thread_id:
                    return thread
            return None

        explicit_signature = str(getattr(diagnostics, "repair_signature", "") or "")
        violated = self._normalized_items(getattr(diagnostics, "violated_commitments", ()))
        relevant = self._normalized_items(getattr(diagnostics, "relevant_commitments", ()))
        repair_targets = self._normalized_items(
            [*violated, *relevant, *getattr(diagnostics, "social_alerts", ())]
        )
        category = self._category_for_diagnostics(
            diagnostics,
            social_alerts=list(getattr(diagnostics, "social_alerts", [])),
        )
        derived_signatures: set[str] = set()
        if explicit_signature:
            derived_signatures.add(explicit_signature)
        if violated:
            derived_signatures.add(self._signature(category, list(violated)))
        if relevant:
            derived_signatures.add(self._signature(category, list(relevant)))

        ranked: list[tuple[int, int, ConflictThread]] = []
        for thread in self.active_unresolved_threads():
            score = 0
            primary_match = False
            anchor_hits = 0
            if explicit_signature and thread.signature == explicit_signature:
                score += 120
                primary_match = True
                anchor_hits += 2
            if thread.signature in derived_signatures:
                score += 95
                primary_match = True
                anchor_hits += 2

            thread_commitments = set(self._normalized_items(thread.linked_commitments))
            commitment_overlap = thread_commitments & set(violated or relevant)
            if commitment_overlap:
                score += 50 if thread_commitments.issuperset(violated or relevant) else 32
                primary_match = True
                anchor_hits += len(commitment_overlap)

            thread_identity = set(self._normalized_items(thread.linked_identity_elements))
            identity_overlap = thread_identity & set(repair_targets)
            if identity_overlap:
                score += 36 if thread_identity.issuperset(repair_targets) else 24
                primary_match = True
                anchor_hits += len(identity_overlap)

            if thread.source_category == category:
                score += 8
                anchor_hits += 1

            if primary_match and score > 0:
                ranked.append((score, anchor_hits, thread))

        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].created_tick, item[2].thread_id))
        if ranked[0][1] < 2 and not explicit_signature and not explicit_thread_id:
            return None
        if len(ranked) > 1 and (
            ranked[0][0] == ranked[1][0]
            or (ranked[0][0] - ranked[1][0] < 12 and ranked[0][1] <= ranked[1][1] + 1)
        ):
            return None
        return ranked[0][2]

    def _attach_verification_evidence(self, *, tick: int, verification_loop) -> None:
        recent_outcomes = [
            item
            for item in getattr(verification_loop, "archived_targets", [])[-6:]
            if int(getattr(item, "outcome_tick", 0)) >= max(0, tick - 2)
        ]
        for outcome in recent_outcomes:
            outcome_name = str(getattr(outcome, "outcome", ""))
            if outcome_name not in {"confirmed", "partially_supported"}:
                continue
            thread = self._match_thread_for_verification_outcome(outcome)
            if thread is None:
                continue
            evidence_id = f"{getattr(outcome, 'target_id', '')}:{getattr(outcome, 'outcome_tick', 0)}"
            if evidence_id not in thread.verification_evidence_ids:
                thread.verification_evidence_ids.append(evidence_id)
                thread.verification_evidence_ids = thread.verification_evidence_ids[-12:]

    def _match_thread_for_verification_outcome(self, outcome) -> ConflictThread | None:
        explicit_thread_id = str(getattr(outcome, "thread_id", "") or "")
        if explicit_thread_id:
            for thread in self.active_threads:
                if thread.thread_id == explicit_thread_id:
                    return thread
            return None

        explicit_signature = str(getattr(outcome, "signature", "") or "")
        linked_commitments = self._normalized_items(getattr(outcome, "linked_commitments", ()))
        linked_identity_anchors = self._normalized_items(
            getattr(outcome, "linked_identity_anchors", ())
        )
        linked_discrepancy_id = str(getattr(outcome, "linked_discrepancy_id", "") or "")
        prediction_id = str(getattr(outcome, "prediction_id", "") or "")
        target_channels = self._normalized_items(getattr(outcome, "target_channels", ()))
        prediction_type = str(getattr(outcome, "prediction_type", "") or "")
        derived_signatures = self._verification_outcome_signatures(
            explicit_signature=explicit_signature,
            linked_commitments=linked_commitments,
            linked_identity_anchors=linked_identity_anchors,
            target_channels=target_channels,
            prediction_type=prediction_type,
        )

        ranked: list[tuple[int, int, ConflictThread]] = []
        for thread in self.active_threads:
            score = 0
            primary_match = False
            anchor_hits = 0
            if explicit_signature and thread.signature == explicit_signature:
                score += 120
                primary_match = True
                anchor_hits += 2
            if thread.signature in derived_signatures:
                score += 90
                primary_match = True
                anchor_hits += 2

            thread_commitments = set(self._normalized_items(thread.linked_commitments))
            commitment_overlap = thread_commitments & set(linked_commitments)
            if commitment_overlap:
                score += 45 if thread_commitments.issuperset(linked_commitments) else 30
                primary_match = True
                anchor_hits += len(commitment_overlap)

            thread_identity = set(self._normalized_items(thread.linked_identity_elements))
            identity_overlap = thread_identity & set(linked_identity_anchors)
            if identity_overlap:
                score += 40 if thread_identity.issuperset(linked_identity_anchors) else 28
                primary_match = True
                anchor_hits += len(identity_overlap)

            if linked_discrepancy_id and linked_discrepancy_id in thread.supporting_evidence:
                score += 35
                primary_match = True
                anchor_hits += 1
            if prediction_id and any(prediction_id in item for item in thread.supporting_evidence):
                score += 12
                anchor_hits += 1
            if target_channels and self._verification_channels_align(thread, target_channels):
                score += 8
                anchor_hits += 1

            if primary_match and score > 0:
                ranked.append((score, anchor_hits, thread))

        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].created_tick, item[2].thread_id))
        if ranked[0][1] < 2 and not explicit_signature and not explicit_thread_id:
            return None
        if len(ranked) > 1 and (
            ranked[0][0] == ranked[1][0]
            or (ranked[0][0] - ranked[1][0] < 12 and ranked[0][1] <= ranked[1][1] + 1)
        ):
            return None
        return ranked[0][2]

    def _verification_outcome_signatures(
        self,
        *,
        explicit_signature: str,
        linked_commitments: tuple[str, ...],
        linked_identity_anchors: tuple[str, ...],
        target_channels: tuple[str, ...],
        prediction_type: str,
    ) -> set[str]:
        signatures: set[str] = set()
        if explicit_signature:
            signatures.add(explicit_signature)
        categories = self._verification_outcome_categories(
            linked_commitments=linked_commitments,
            linked_identity_anchors=linked_identity_anchors,
            target_channels=target_channels,
            prediction_type=prediction_type,
        )
        for category in categories:
            if linked_commitments:
                signatures.add(self._signature(category, list(linked_commitments)))
            if linked_identity_anchors:
                signatures.add(self._signature(category, list(linked_identity_anchors)))
        return signatures

    def _verification_outcome_categories(
        self,
        *,
        linked_commitments: tuple[str, ...],
        linked_identity_anchors: tuple[str, ...],
        target_channels: tuple[str, ...],
        prediction_type: str,
    ) -> tuple[str, ...]:
        categories: list[str] = []
        channel_set = set(target_channels)
        if linked_identity_anchors or "continuity" in channel_set or "continuity" in prediction_type:
            categories.append(ConflictSourceCategory.CONTINUITY_ANCHOR.value)
        if "social" in channel_set or "social" in prediction_type:
            categories.append(ConflictSourceCategory.SOCIAL_RUPTURE.value)
        if "danger" in channel_set:
            categories.append(ConflictSourceCategory.TRUST_VS_THREAT.value)
        if linked_commitments:
            categories.extend(
                [
                    ConflictSourceCategory.IDENTITY_ACTION.value,
                    ConflictSourceCategory.SELF_EXPECTATION_FALSIFICATION.value,
                ]
            )
        return tuple(dict.fromkeys(categories))

    def _verification_channels_align(
        self,
        thread: ConflictThread,
        target_channels: tuple[str, ...],
    ) -> bool:
        return bool(set(self._thread_channels(thread)) & set(target_channels))

    def _normalized_items(self, values) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in values or ():
            text = str(item).strip().lower().replace(" ", "_")
            if text:
                normalized.append(text)
        return tuple(dict.fromkeys(normalized))

    def _trim(self) -> None:
        self.active_threads.sort(
            key=lambda item: (
                not item.protected,
                item.status == ReconciliationStatus.RECONCILED.value,
                self._severity_weight(item.severity),
                item.recurrence_count,
                -item.latest_tick,
            )
        )
        if len(self.active_threads) > self.max_active_threads:
            keep = self.active_threads[: self.max_active_threads]
            for item in self.active_threads[self.max_active_threads :]:
                item.status = (
                    ReconciliationStatus.ARCHIVED_UNRESOLVED.value
                    if item.status != ReconciliationStatus.RECONCILED.value
                    else ReconciliationStatus.ARCHIVED_RECONCILED.value
                )
                item.archived_tick = self.last_tick
                self.archived_threads.append(item)
            self.active_threads = keep
        self.archived_threads = self.archived_threads[-self.archive_limit :]

    def _find_thread(self, signature: str) -> ConflictThread | None:
        for thread in self.active_threads:
            if thread.signature == signature:
                return thread
        return None

    def _reopen_archived(self, signature: str, *, tick: int) -> ConflictThread | None:
        for index, thread in enumerate(self.archived_threads):
            if thread.signature != signature:
                continue
            reopened = self.archived_threads.pop(index)
            reopened.status = ReconciliationStatus.REOPENED.value
            reopened.current_outcome = ReconciliationOutcome.UNRESOLVED_CHRONIC.value
            reopened.last_reopened_tick = tick
            reopened.archived_tick = None
            self.active_threads.append(reopened)
            return reopened
        return None

    def _note_signature(self, signature: str, *, tick: int) -> None:
        counts = dict(self.recent_signature_counts.get(signature, {}))
        last_tick = int(counts.get("last_tick", 0))
        count = int(counts.get("count", 0))
        count = count + 1 if tick - last_tick <= 6 else 1
        self.recent_signature_counts[signature] = {"count": count, "last_tick": tick}

    def _thread_channels(self, thread: ConflictThread) -> tuple[str, ...]:
        if thread.source_category in {
            ConflictSourceCategory.IDENTITY_ACTION.value,
            ConflictSourceCategory.CONTINUITY_ANCHOR.value,
            ConflictSourceCategory.SELF_EXPECTATION_FALSIFICATION.value,
        }:
            return ("continuity", "conflict")
        if thread.source_category in {
            ConflictSourceCategory.SOCIAL_RUPTURE.value,
            ConflictSourceCategory.TRUST_VS_THREAT.value,
        }:
            return ("social", "conflict")
        if thread.source_category == ConflictSourceCategory.SURVIVAL_VS_EXPLORATION.value:
            return ("danger", "novelty")
        return ("conflict",)

    def _signature(self, category: str, anchors: list[str]) -> str:
        normalized = [item.strip().lower().replace(" ", "_") for item in anchors if item]
        return f"{category}:{'|'.join(sorted(set(normalized))[:4]) or 'generic'}"

    def _category_for_diagnostics(self, diagnostics, *, social_alerts: list[str]) -> str:
        conflict_type = str(getattr(diagnostics, "conflict_type", ""))
        if social_alerts or conflict_type == "social_contradiction":
            return ConflictSourceCategory.SOCIAL_RUPTURE.value
        if conflict_type == "adaptation_vs_betrayal":
            return ConflictSourceCategory.SURVIVAL_VS_EXPLORATION.value
        if conflict_type == "stress_drift":
            return ConflictSourceCategory.VALUE_HIERARCHY.value
        if conflict_type == "temporary_deviation":
            return ConflictSourceCategory.IDENTITY_ACTION.value
        if "continuity" in conflict_type:
            return ConflictSourceCategory.CONTINUITY_ANCHOR.value
        return ConflictSourceCategory.SELF_EXPECTATION_FALSIFICATION.value

    def _category_for_discrepancy(self, discrepancy_type: str) -> str:
        if "social" in discrepancy_type:
            return ConflictSourceCategory.SOCIAL_RUPTURE.value
        if "continuity" in discrepancy_type:
            return ConflictSourceCategory.CONTINUITY_ANCHOR.value
        if "identity" in discrepancy_type:
            return ConflictSourceCategory.IDENTITY_ACTION.value
        if "danger" in discrepancy_type:
            return ConflictSourceCategory.TRUST_VS_THREAT.value
        return ConflictSourceCategory.SELF_EXPECTATION_FALSIFICATION.value

    def _severity_name(self, intensity: float) -> str:
        if intensity >= 0.85:
            return ConflictSeverity.CRITICAL.value
        if intensity >= 0.60:
            return ConflictSeverity.HIGH.value
        if intensity >= 0.30:
            return ConflictSeverity.MEDIUM.value
        return ConflictSeverity.LOW.value

    def _severity_value(self, severity: str) -> float:
        return {
            ConflictSeverity.LOW.value: 0.20,
            ConflictSeverity.MEDIUM.value: 0.45,
            ConflictSeverity.HIGH.value: 0.70,
            ConflictSeverity.CRITICAL.value: 0.92,
        }.get(severity, 0.20)

    def _severity_weight(self, severity: str) -> int:
        return {
            ConflictSeverity.LOW.value: 1,
            ConflictSeverity.MEDIUM.value: 2,
            ConflictSeverity.HIGH.value: 3,
            ConflictSeverity.CRITICAL.value: 4,
        }.get(severity, 1)

    def _persistence_class(self, thread: ConflictThread) -> str:
        if thread.protected:
            return ConflictPersistenceClass.IDENTITY_CRITICAL.value
        if len(set(thread.linked_chapter_ids)) >= 2 or thread.recurrence_count >= 4:
            return ConflictPersistenceClass.LONG_HORIZON.value
        if thread.recurrence_count >= 2:
            return ConflictPersistenceClass.RECURRING.value
        return ConflictPersistenceClass.EPISODIC.value
