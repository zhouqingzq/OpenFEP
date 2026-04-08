from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .memory_decay import (
    DecayReport,
    LONG_DORMANT_ACCESS_THRESHOLD,
    LONG_DORMANT_TRACE_THRESHOLD,
    SHORT_CLEANUP_TRACE_THRESHOLD,
    access_decay_rate,
    decay_accessibility,
    decay_accessibility_for_level,
    decay_trace_strength,
    decay_trace_strength_for_level,
    trace_decay_rate,
)
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _internal_metadata(entry: MemoryEntry) -> dict[str, object]:
    if entry.compression_metadata is None:
        entry.compression_metadata = {}
    internal = entry.compression_metadata.get("m45_internal")
    if not isinstance(internal, dict):
        internal = {}
        entry.compression_metadata["m45_internal"] = internal
    return internal


def _effective_cycle(entry: MemoryEntry, baseline_cycle: int) -> int:
    return max(entry.created_at, entry.last_accessed, baseline_cycle)


def _ensure_decay_baseline(entry: MemoryEntry) -> tuple[dict[str, object], int, float, float]:
    internal = _internal_metadata(entry)
    baseline_cycle = _coerce_int(
        internal.get("last_decay_cycle", max(entry.created_at, entry.last_accessed)),
        max(entry.created_at, entry.last_accessed),
    )
    base_trace = _coerce_float(internal.get("decay_base_trace_strength", entry.trace_strength), entry.trace_strength)
    base_access = _coerce_float(
        internal.get("decay_base_accessibility", entry.accessibility),
        entry.accessibility,
    )
    internal["last_decay_cycle"] = baseline_cycle
    internal["decay_base_trace_strength"] = base_trace
    internal["decay_base_accessibility"] = base_access
    return internal, baseline_cycle, base_trace, base_access


def _refresh_decay_baseline(entry: MemoryEntry, *, cycle: int) -> dict[str, object]:
    internal = _internal_metadata(entry)
    internal["last_decay_cycle"] = cycle
    internal["decay_base_trace_strength"] = entry.trace_strength
    internal["decay_base_accessibility"] = entry.accessibility
    return internal


LEGACY_MAPPED_KEYS = {
    "episode_id",
    "timestamp",
    "cycle",
    "state_vector",
    "state_snapshot",
    "action_taken",
    "action",
    "outcome_state",
    "outcome",
    "predicted_outcome",
    "prediction_error",
    "risk",
    "value_score",
    "total_surprise",
    "weighted_surprise",
    "embedding",
    "value_label",
    "preferred_probability",
    "preference_log_value",
    "observation",
    "prediction",
    "errors",
    "body_state",
    "support_count",
    "support",
    "last_seen_cycle",
    "identity_critical",
    "continuity_tags",
    "gating_reasons",
    "lifecycle_stage",
    "episode_family",
    "content",
}


def _legacy_unmapped(entry: MemoryEntry) -> dict[str, object]:
    if not isinstance(entry.compression_metadata, dict):
        return {}
    payload = entry.compression_metadata.get("legacy_unmapped")
    if not isinstance(payload, dict):
        return {}
    return deepcopy(payload)


def _legacy_template(entry: MemoryEntry) -> dict[str, object]:
    if not isinstance(entry.compression_metadata, dict):
        return {}
    payload = entry.compression_metadata.get("legacy_template")
    if not isinstance(payload, dict):
        return {}
    return deepcopy(payload)


def _legacy_episode_id(payload: dict[str, object], index: int = 0) -> str:
    timestamp = _coerce_int(payload.get("timestamp", payload.get("cycle", 0)))
    return str(payload.get("episode_id", f"legacy-{timestamp}-{index}"))


def _legacy_payload_matches(entry: MemoryEntry, payload: dict[str, object]) -> bool:
    return _legacy_template(entry) == deepcopy(payload)


def _legacy_entry_from_payload(payload: dict[str, object], index: int = 0) -> MemoryEntry:
    timestamp = _coerce_int(payload.get("timestamp", payload.get("cycle", 0)))
    action = str(payload.get("action", "unknown"))
    outcome = str(
        payload.get(
            "predicted_outcome",
            payload.get("value_label", _outcome_summary(payload.get("outcome", "neutral"))),
        )
    )
    content = str(
        payload.get(
            "content",
            f"Legacy episode at cycle {timestamp}: {action} -> {outcome}",
        )
    )
    total_surprise = _coerce_float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0)))
    support_count = _coerce_int(payload.get("support_count", payload.get("support", 1)), 1)
    store_level = StoreLevel.SHORT
    if bool(payload.get("identity_critical", False)) or total_surprise >= 0.9:
        store_level = StoreLevel.LONG
    elif support_count >= 2 or total_surprise >= 0.5:
        store_level = StoreLevel.MID
    metadata = {
        "legacy_template": deepcopy(payload),
        "legacy_unmapped": {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in LEGACY_MAPPED_KEYS
        },
        "m45_internal": {"last_decay_cycle": timestamp},
    }
    return MemoryEntry(
        id=_legacy_episode_id(payload, index=index),
        content=content,
        memory_class=MemoryClass.EPISODIC,
        store_level=store_level,
        source_type=SourceType.EXPERIENCE,
        created_at=timestamp,
        last_accessed=_coerce_int(payload.get("last_seen_cycle", timestamp)),
        valence=_coerce_float(payload.get("value_score", 0.0)),
        arousal=_clamp(total_surprise),
        encoding_attention=_clamp(total_surprise),
        novelty=_clamp(_coerce_float(payload.get("prediction_error", 0.0))),
        relevance_goal=_clamp(abs(_coerce_float(payload.get("value_relevance", 0.0)))),
        relevance_threat=_clamp(_coerce_float(payload.get("threat_significance", 0.0))),
        relevance_self=1.0 if bool(payload.get("identity_critical", False)) else 0.0,
        relevance_social=_clamp(0.6 if str(payload.get("episode_family", "")) == "social_signal" else 0.0),
        relevance_reward=_clamp(abs(_coerce_float(payload.get("value_score", 0.0)))),
        relevance=_clamp(
            max(
                _coerce_float(payload.get("value_relevance", 0.0)),
                _coerce_float(payload.get("threat_significance", 0.0)),
                1.0 if bool(payload.get("identity_critical", False)) else 0.0,
            )
        ),
        salience=_clamp(total_surprise),
        trace_strength=max(0.05, _clamp(total_surprise)),
        accessibility=max(0.05, _clamp(total_surprise)),
        abstractness=0.25,
        source_confidence=0.9,
        reality_confidence=0.85,
        semantic_tags=sorted(
            {
                action,
                outcome,
                *[str(item) for item in payload.get("continuity_tags", []) if str(item)],
                str(payload.get("episode_family", "")),
            }
            - {""}
        ),
        context_tags=sorted(
            {
                *[str(item) for item in payload.get("gating_reasons", []) if str(item)],
                str(payload.get("lifecycle_stage", "")),
            }
            - {""}
        ),
        anchor_slots={
            "time": str(timestamp),
            "place": None,
            "agents": None,
            "action": action,
            "outcome": outcome,
        },
        anchor_strengths={
            "time": "weak",
            "place": "weak",
            "agents": "strong",
            "action": "strong",
            "outcome": "strong",
        },
        mood_context=str(payload.get("predicted_outcome", "")),
        retrieval_count=0,
        support_count=max(1, support_count),
        counterevidence_count=0,
        competing_interpretations=None,
        compression_metadata=metadata,
        derived_from=[],
        is_dormant=bool(payload.get("lifecycle_stage") == "archived_summary"),
    )


def _outcome_summary(value: Any) -> str:
    if isinstance(value, dict):
        summary = value.get("summary")
        if summary is not None:
            return str(summary)
    return str(value or "")


def _update_legacy_outcome_payload(value: object, outcome_text: str) -> object:
    if isinstance(value, dict):
        updated = deepcopy(value)
        updated["summary"] = outcome_text
        return updated
    if value is None:
        return {"summary": outcome_text}
    return outcome_text


def _update_legacy_action_payload(value: object, action: str) -> object:
    if isinstance(value, dict):
        updated = deepcopy(value)
        updated["name"] = action
        return updated
    if value is None:
        return action
    return action


def _has_source_conflict(entry: MemoryEntry) -> bool:
    if not isinstance(entry.compression_metadata, dict):
        return False
    metadata = entry.compression_metadata
    if metadata.get("source_conflict") or metadata.get("provenance_conflict"):
        return True
    conflict_type = str(metadata.get("conflict_type", "")).lower()
    if conflict_type == "source":
        return True
    conflicts = metadata.get("source_conflicts")
    return isinstance(conflicts, list) and bool(conflicts)


def _has_reality_conflict(entry: MemoryEntry) -> bool:
    if entry.counterevidence_count > 0 or bool(entry.competing_interpretations):
        return True
    if not isinstance(entry.compression_metadata, dict):
        return False
    metadata = entry.compression_metadata
    if metadata.get("factual_conflict") or metadata.get("counterevidence_present"):
        return True
    conflict_type = str(metadata.get("conflict_type", "")).lower()
    if conflict_type == "factual":
        return True
    conflicts = metadata.get("reality_conflicts")
    return isinstance(conflicts, list) and bool(conflicts)


@dataclass
class MemoryStore:
    entries: list[MemoryEntry] = field(default_factory=list)
    short_to_mid_salience_threshold: float = 0.55
    short_to_mid_retrieval_threshold: int = 2
    mid_to_long_salience_threshold: float = 0.80
    mid_to_long_retrieval_threshold: int = 3
    identity_priority_threshold: float = 0.75
    identity_long_threshold: float = 0.85
    last_cleanup_report: dict[str, object] = field(default_factory=dict)

    def _find_index(self, entry_id: str) -> int | None:
        for index, entry in enumerate(self.entries):
            if entry.id == entry_id:
                return index
        return None

    def _promote_store_level(self, entry: MemoryEntry) -> None:
        internal, baseline_cycle, base_trace, base_access = _ensure_decay_baseline(entry)
        effective_cycle = _effective_cycle(entry, baseline_cycle)
        old_level = entry.store_level
        new_level = old_level
        reasons: list[str] = []

        if new_level is StoreLevel.SHORT and (
            (
                entry.salience >= self.short_to_mid_salience_threshold
                and entry.retrieval_count >= self.short_to_mid_retrieval_threshold
            )
            or entry.relevance_self >= self.identity_priority_threshold
        ):
            new_level = StoreLevel.MID
            if entry.relevance_self >= self.identity_priority_threshold:
                reasons.append("identity_priority")
            else:
                reasons.append("salience_retrieval")

        if new_level is StoreLevel.MID and (
            (
                entry.salience >= self.mid_to_long_salience_threshold
                and entry.retrieval_count >= self.mid_to_long_retrieval_threshold
            )
            or (entry.relevance_self >= self.identity_long_threshold and entry.retrieval_count >= 2)
        ):
            if old_level is StoreLevel.SHORT and new_level is StoreLevel.MID:
                reasons.append("promotion_chain")
            new_level = StoreLevel.LONG
            if entry.relevance_self >= self.identity_long_threshold and entry.retrieval_count >= 2:
                reasons.append("identity_long_priority")
            else:
                reasons.append("salience_retrieval")

        if new_level is old_level:
            return

        elapsed = max(0, effective_cycle - baseline_cycle)
        old_trace = decay_trace_strength_for_level(
            base_trace,
            old_level,
            elapsed,
            memory_class=entry.memory_class,
        )
        old_access = decay_accessibility_for_level(base_access, old_level, elapsed)
        new_trace = decay_trace_strength_for_level(
            base_trace,
            new_level,
            elapsed,
            memory_class=entry.memory_class,
        )
        new_access = decay_accessibility_for_level(base_access, new_level, elapsed)

        entry.store_level = new_level
        entry.trace_strength = new_trace
        entry.accessibility = new_access
        internal = _refresh_decay_baseline(entry, cycle=effective_cycle)
        promotion_record = {
            "reason": "+".join(reasons) if reasons else "promotion",
            "effective_cycle": effective_cycle,
            "elapsed_since_baseline": elapsed,
            "old_level": old_level.value,
            "new_level": new_level.value,
            "trace_before": old_trace,
            "trace_after": new_trace,
            "accessibility_before": old_access,
            "accessibility_after": new_access,
            "trace_rate_before": trace_decay_rate(old_level, memory_class=entry.memory_class),
            "trace_rate_after": trace_decay_rate(new_level, memory_class=entry.memory_class),
            "access_rate_before": access_decay_rate(old_level),
            "access_rate_after": access_decay_rate(new_level),
        }
        history = internal.get("promotion_history")
        if not isinstance(history, list):
            history = []
            internal["promotion_history"] = history
        history.append(promotion_record)
        internal["last_promotion"] = promotion_record

    def add(self, entry: MemoryEntry) -> str:
        entry.sync_content_hash()
        self._promote_store_level(entry)
        _, baseline_cycle, _, _ = _ensure_decay_baseline(entry)
        effective_cycle = _effective_cycle(entry, baseline_cycle)
        _refresh_decay_baseline(entry, cycle=effective_cycle)
        existing_index = self._find_index(entry.id)
        if existing_index is None:
            self.entries.append(entry)
        else:
            self.entries[existing_index] = entry
        return entry.id

    def upsert_legacy_episode(self, payload: dict[str, object], *, index: int = 0) -> str:
        entry_id = _legacy_episode_id(payload, index=index)
        existing_index = self._find_index(entry_id)
        if existing_index is not None and _legacy_payload_matches(self.entries[existing_index], payload):
            return self.entries[existing_index].id
        entry = _legacy_entry_from_payload(payload, index=index)
        if existing_index is None:
            self.entries.append(entry)
        else:
            self.entries[existing_index] = entry
        return entry.id

    def remove_legacy_episode(self, episode_id: str) -> bool:
        existing_index = self._find_index(str(episode_id))
        if existing_index is None:
            return False
        self.entries.pop(existing_index)
        return True

    def replace_legacy_group(self, episodes: list[dict[str, object]]) -> dict[str, list[str]]:
        existing_by_id = {entry.id: entry for entry in self.entries}
        reused_ids: list[str] = []
        upserted_ids: list[str] = []
        new_entries: list[MemoryEntry] = []
        for index, payload in enumerate(episodes):
            entry_id = _legacy_episode_id(payload, index=index)
            existing = existing_by_id.get(entry_id)
            if existing is not None and _legacy_payload_matches(existing, payload):
                reused_ids.append(entry_id)
                new_entries.append(existing)
                continue
            upserted_ids.append(entry_id)
            new_entries.append(_legacy_entry_from_payload(payload, index=index))
        removed_ids = [entry.id for entry in self.entries if entry.id not in {item.id for item in new_entries}]
        self.entries = new_entries
        return {
            "reused_ids": reused_ids,
            "upserted_ids": upserted_ids,
            "removed_ids": removed_ids,
        }

    def get(self, entry_id: str) -> MemoryEntry | None:
        index = self._find_index(entry_id)
        return None if index is None else self.entries[index]

    def query_by_tags(self, tags: list[str], k: int = 5) -> list[MemoryEntry]:
        normalized = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        scored: list[tuple[float, float, MemoryEntry]] = []
        for entry in self.entries:
            entry_tags = {tag.lower() for tag in [*entry.semantic_tags, *entry.context_tags]}
            overlap = len(normalized & entry_tags)
            if not overlap and normalized:
                continue
            score = float(overlap) + (0.25 * entry.accessibility) + (0.15 * entry.salience)
            scored.append((score, entry.trace_strength, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:k]]

    def retrieve(
        self,
        query,
        *,
        current_mood: str | None = None,
        k: int = 5,
    ):
        from .memory_retrieval import retrieve

        return retrieve(query, self, current_mood=current_mood, k=k)

    def reconsolidate_entry(
        self,
        entry_id: str,
        *,
        current_mood: str | None = None,
        current_context_tags: list[str] | None = None,
        current_cycle: int | None = None,
        current_state: dict[str, object] | None = None,
        recall_artifact=None,
        conflict_type=None,
    ):
        from .memory_consolidation import reconsolidate

        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        return reconsolidate(
            entry,
            current_mood,
            current_context_tags,
            store=self,
            current_cycle=current_cycle,
            current_state=current_state,
            recall_artifact=recall_artifact,
            conflict_type=conflict_type,
        )

    def run_consolidation_cycle(
        self,
        current_cycle: int,
        rng: random.Random,
        current_state: dict[str, object] | None = None,
    ):
        from .memory_consolidation import run_consolidation_cycle

        return run_consolidation_cycle(
            self,
            current_cycle=current_cycle,
            rng=rng,
            current_state=current_state,
        )

    def mark_dormant(self, entry_id: str) -> None:
        entry = self.get(entry_id)
        if entry is not None:
            entry.is_dormant = True

    def cleanup_short(self, threshold: float = SHORT_CLEANUP_TRACE_THRESHOLD) -> int:
        deleted: list[str] = []
        retained_low_value: list[str] = []
        keep: list[MemoryEntry] = []
        for entry in self.entries:
            if entry.store_level is StoreLevel.SHORT and entry.trace_strength <= threshold:
                deleted.append(entry.id)
                continue
            if entry.trace_strength <= threshold:
                retained_low_value.append(entry.id)
            keep.append(entry)
        self.entries = keep
        self.last_cleanup_report = {
            "threshold": threshold,
            "deleted_short_residue": deleted,
            "retained_non_short_low_trace": retained_low_value,
        }
        return len(deleted)

    def _should_abstract_entry(self, entry: MemoryEntry, elapsed: int) -> bool:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return False
        return bool(
            elapsed >= 8
            and (
                entry.retrieval_count <= 1
                or entry.support_count <= 1
                or entry.abstractness >= 0.60
                or entry.is_dormant
            )
        )

    def _source_confidence_drift(self, entry: MemoryEntry, elapsed: int) -> float:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return 0.0
        pressure = 0.0
        if entry.source_type is SourceType.RECONSTRUCTION:
            pressure += 0.0009 * elapsed
        if entry.support_count <= 1:
            pressure += 0.0005 * elapsed
        if _has_source_conflict(entry):
            pressure += 0.0012 * elapsed
        return min(0.18, pressure)

    def _reality_confidence_drift(self, entry: MemoryEntry, elapsed: int) -> float:
        if entry.store_level not in {StoreLevel.MID, StoreLevel.LONG}:
            return 0.0
        pressure = 0.0
        if entry.counterevidence_count > 0:
            pressure += min(0.12, 0.01 * entry.counterevidence_count) + (0.001 * elapsed)
        if bool(entry.competing_interpretations):
            pressure += 0.0008 * elapsed
        if _has_reality_conflict(entry):
            pressure += 0.0012 * elapsed
        if entry.abstractness >= 0.75 and entry.retrieval_count == 0:
            pressure += 0.0005 * elapsed
        return min(0.22, pressure)

    def apply_decay(self, current_cycle: int) -> DecayReport:
        report = DecayReport(current_cycle=current_cycle)
        for entry in self.entries:
            report.processed_entries += 1
            internal, baseline_cycle, base_trace, base_access = _ensure_decay_baseline(entry)
            elapsed = max(0, current_cycle - baseline_cycle)
            if elapsed <= 0:
                continue
            entry.trace_strength = decay_trace_strength_for_level(
                base_trace,
                entry.store_level,
                elapsed,
                memory_class=entry.memory_class,
            )
            entry.accessibility = decay_accessibility_for_level(base_access, entry.store_level, elapsed)
            if self._should_abstract_entry(entry, elapsed):
                new_abstractness = _clamp(entry.abstractness + min(0.16, 0.002 * elapsed))
                if new_abstractness > entry.abstractness:
                    entry.abstractness = new_abstractness
                    report.abstracted_entries.append(entry.id)
            source_before = entry.source_confidence
            reality_before = entry.reality_confidence
            source_drift = self._source_confidence_drift(entry, elapsed)
            reality_drift = self._reality_confidence_drift(entry, elapsed)
            if source_drift > 0.0:
                entry.source_confidence = _clamp(entry.source_confidence - source_drift)
            if reality_drift > 0.0:
                entry.reality_confidence = _clamp(entry.reality_confidence - reality_drift)
            if entry.source_confidence != source_before:
                report.source_confidence_drifted.append(entry.id)
            if entry.reality_confidence != reality_before:
                report.reality_confidence_drifted.append(entry.id)
            if entry.source_confidence != source_before or entry.reality_confidence != reality_before:
                report.confidence_drifted.append(entry.id)
            if (
                entry.store_level is StoreLevel.LONG
                and entry.trace_strength <= LONG_DORMANT_TRACE_THRESHOLD
                and entry.accessibility <= LONG_DORMANT_ACCESS_THRESHOLD
            ):
                entry.is_dormant = True
                report.dormant_marked.append(entry.id)
            _refresh_decay_baseline(entry, cycle=current_cycle)
        before_cleanup = {entry.id for entry in self.entries}
        self.cleanup_short()
        after_cleanup = {entry.id for entry in self.entries}
        report.deleted_short_residue = sorted(before_cleanup - after_cleanup)
        return report

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "short_to_mid_salience_threshold": self.short_to_mid_salience_threshold,
            "short_to_mid_retrieval_threshold": self.short_to_mid_retrieval_threshold,
            "mid_to_long_salience_threshold": self.mid_to_long_salience_threshold,
            "mid_to_long_retrieval_threshold": self.mid_to_long_retrieval_threshold,
            "identity_priority_threshold": self.identity_priority_threshold,
            "identity_long_threshold": self.identity_long_threshold,
            "last_cleanup_report": dict(self.last_cleanup_report),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MemoryStore:
        return cls(
            entries=[
                MemoryEntry.from_dict(item)
                for item in payload.get("entries", [])
                if isinstance(item, dict)
            ],
            short_to_mid_salience_threshold=_coerce_float(
                payload.get("short_to_mid_salience_threshold"), 0.55
            ),
            short_to_mid_retrieval_threshold=_coerce_int(
                payload.get("short_to_mid_retrieval_threshold"), 2
            ),
            mid_to_long_salience_threshold=_coerce_float(
                payload.get("mid_to_long_salience_threshold"), 0.80
            ),
            mid_to_long_retrieval_threshold=_coerce_int(
                payload.get("mid_to_long_retrieval_threshold"), 3
            ),
            identity_priority_threshold=_coerce_float(
                payload.get("identity_priority_threshold"), 0.75
            ),
            identity_long_threshold=_coerce_float(
                payload.get("identity_long_threshold"), 0.85
            ),
            last_cleanup_report=dict(payload.get("last_cleanup_report", {}))
            if isinstance(payload.get("last_cleanup_report"), dict)
            else {},
        )

    @classmethod
    def from_legacy_episodes(cls, episodes: list[dict[str, object]]) -> MemoryStore:
        store = cls()
        store.replace_legacy_group(episodes)
        return store

    def to_legacy_episodes(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for entry in self.entries:
            template = _legacy_template(entry)
            extras = _legacy_unmapped(entry)
            action = entry.anchor_slots.get("action") or str(
                template.get("action", template.get("action_taken", "unknown"))
            )
            outcome_text = entry.anchor_slots.get("outcome") or str(
                template.get("predicted_outcome", template.get("value_label", entry.content))
            )
            base_payload = {
                "episode_id": entry.id,
                "timestamp": entry.created_at,
                "cycle": entry.created_at,
                "state_vector": deepcopy(template.get("state_vector"))
                if isinstance(template.get("state_vector"), dict)
                else {},
                "state_snapshot": deepcopy(template.get("state_snapshot"))
                if isinstance(template.get("state_snapshot"), dict)
                else {},
                "action_taken": _update_legacy_action_payload(template.get("action_taken"), action),
                "action": action,
                "outcome_state": _update_legacy_outcome_payload(template.get("outcome_state"), outcome_text),
                "outcome": _update_legacy_outcome_payload(template.get("outcome"), outcome_text),
                "predicted_outcome": outcome_text,
                "prediction_error": entry.novelty,
                "risk": _coerce_float(template.get("risk", 0.0)),
                "value_score": entry.valence,
                "total_surprise": entry.salience,
                "weighted_surprise": entry.salience,
                "embedding": deepcopy(template.get("embedding"))
                if isinstance(template.get("embedding"), list)
                else [],
                "value_label": outcome_text,
                "preferred_probability": _coerce_float(template.get("preferred_probability", 0.0)),
                "preference_log_value": _coerce_float(template.get("preference_log_value", 0.0)),
                "observation": deepcopy(template.get("observation"))
                if isinstance(template.get("observation"), dict)
                else {},
                "prediction": deepcopy(template.get("prediction"))
                if isinstance(template.get("prediction"), dict)
                else {},
                "errors": deepcopy(template.get("errors"))
                if isinstance(template.get("errors"), dict)
                else {},
                "body_state": deepcopy(template.get("body_state"))
                if isinstance(template.get("body_state"), dict)
                else {},
                "support_count": entry.support_count,
                "support": entry.support_count,
                "last_seen_cycle": entry.last_accessed,
                "identity_critical": entry.relevance_self >= self.identity_priority_threshold,
                "continuity_tags": list(entry.semantic_tags),
                "gating_reasons": list(entry.context_tags),
                "lifecycle_stage": template.get("lifecycle_stage", "validated_episode"),
                "episode_family": template.get("episode_family", ""),
                "content": entry.content,
            }
            for key, value in extras.items():
                if key in LEGACY_MAPPED_KEYS:
                    continue
                base_payload[key] = deepcopy(value)
            payloads.append(base_payload)
        return payloads
