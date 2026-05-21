"""M14.3 unified proactive target selection.

This module is deliberately structural. It does not generate persona text and
does not infer subjective desire; it chooses whether an existing durable signal
is traceable enough to become a proactive proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_boredom import boredom_band, normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import _bounded_float, _mapping, _string_list, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_memory_efe import normalize_memory_efe_state
from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state

VAGUE_NEXT_CHECKS = frozenset({"", "later", "regular", "someday", "soon", "next", "follow_up", "check_later"})
TRACEABLE_NEXT_CHECKS = frozenset({"next_user_turn", "next_turn", "after_next_user_message"})


@dataclass(frozen=True)
class ProactiveTarget:
    trigger: str
    traceable_expectation_id: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    proposed_topic: str = ""
    ordinary_language_intent: str = ""
    source_kind: str = ""
    urgency_band: str = "medium"
    risk_band: str = "low"
    selection_reason_codes: list[str] = field(default_factory=list)


def _clean_id(raw: Any) -> str:
    return str(raw or "").strip()[:120]


def _status_open(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "open") or "open").strip().lower() in {"open", "pending", "active", ""}


def _content_summary(row: Mapping[str, Any]) -> str:
    for key in ("content_summary", "summary", "title", "content", "text"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value[:240]
    return ""


def _evidence_refs(row: Mapping[str, Any]) -> list[str]:
    refs = _string_list(row.get("evidence_refs"), limit=8)
    refs.extend(_string_list(row.get("bound_memory_ids"), limit=8))
    row_id = _clean_id(row.get("id") or row.get("expectation_id"))
    if row_id and (refs or str(row.get("source_kind", "")) == "scheduled_outreach"):
        refs.append(row_id)
    return list(dict.fromkeys(refs))[:8]


def next_check_is_vague(value: Any) -> bool:
    return str(value or "").strip().casefold() in VAGUE_NEXT_CHECKS


def is_traceable_open_item(row: Mapping[str, Any]) -> bool:
    if not _status_open(row):
        return False
    next_check = str(row.get("next_check", row.get("next_step", "")) or "").strip().casefold()
    scheduled_id = _clean_id(row.get("scheduled_intent_id") or row.get("intent_id"))
    due_at = row.get("due_at_epoch") or row.get("due_at")
    if scheduled_id and due_at:
        return True
    if next_check not in TRACEABLE_NEXT_CHECKS:
        return False
    return bool(_evidence_refs(row) and _content_summary(row))


def build_traceable_proactive_intent(expectation: Mapping[str, Any]) -> str:
    summary = _content_summary(expectation)
    if summary:
        return f"Follow up on the unresolved expectation: {summary[:180]}"
    trace_id = _clean_id(expectation.get("expectation_id") or expectation.get("id"))
    return f"Follow up on traceable expectation {trace_id}" if trace_id else ""


def _target_from_memory_efe(memory_efe_evaluation: Any | None, m13_state: Mapping[str, Any]) -> ProactiveTarget | None:
    if memory_efe_evaluation is None:
        memory_efe = normalize_memory_efe_state(normalize_m13_drive_state(m13_state).get("memory_efe"))
        should = bool(memory_efe.get("should_outreach"))
        trace_id = _clean_id(memory_efe.get("traceable_expectation_id"))
        refs = _string_list(memory_efe.get("evidence_refs"), limit=8)
        reason_codes = _string_list(memory_efe.get("reason_codes"), limit=8)
        intent = ""
        eligible = memory_efe.get("eligible_for_efe", []) or []
    else:
        should = bool(getattr(memory_efe_evaluation, "should_outreach", False))
        trace_id = _clean_id(getattr(memory_efe_evaluation, "traceable_expectation_id", ""))
        refs = _string_list(getattr(memory_efe_evaluation, "evidence_refs", []), limit=8)
        reason_codes = _string_list(getattr(memory_efe_evaluation, "reason_codes", []), limit=8)
        eligible = [getattr(row, "to_dict", lambda: row)() for row in getattr(memory_efe_evaluation, "eligible_for_efe", []) or []]
        intent = ""
    if not should or not trace_id or not refs:
        return None
    for row in eligible:
        if isinstance(row, Mapping) and _clean_id(row.get("expectation_id")) == trace_id:
            intent = build_traceable_proactive_intent(row)
            topic = _content_summary(row) or trace_id
            source_kind = str(row.get("source_kind", "pending_expectation") or "pending_expectation")
            break
    else:
        topic = trace_id
        source_kind = "pending_expectation"
    return ProactiveTarget(
        trigger="memory_efe_outreach",
        traceable_expectation_id=trace_id,
        evidence_refs=refs,
        proposed_topic=topic[:120],
        ordinary_language_intent=intent or f"Follow up on traceable expectation {trace_id}",
        source_kind=source_kind,
        urgency_band="medium",
        risk_band="low",
        selection_reason_codes=list(dict.fromkeys(["memory_efe_should_outreach", *reason_codes]))[:8],
    )


def _target_from_correction(m13_state: Mapping[str, Any]) -> ProactiveTarget | None:
    reward = normalize_affective_reward_proxy_state(normalize_m13_drive_state(m13_state).get("affective_reward_proxy"))
    if _bounded_float(reward.get("opponent_strength")) < 0.35:
        return None
    for row in reward.get("pending_settlements", []) or []:
        if isinstance(row, Mapping) and bool(row.get("prior_safety_repair")):
            pid = _clean_id(row.get("pending_id"))
            return ProactiveTarget(
                trigger="correction_followup",
                traceable_expectation_id=pid,
                evidence_refs=[pid] if pid else [],
                proposed_topic=str(row.get("prior_topic_fingerprint", "repair_thread"))[:120],
                ordinary_language_intent="Offer a concise clarification after the prior repair pressure.",
                source_kind="prior_safety_repair",
                urgency_band="medium",
                risk_band="medium",
                selection_reason_codes=["prior_safety_repair_settlement"],
            )
    return None


def _target_from_boredom(m13_state: Mapping[str, Any]) -> ProactiveTarget | None:
    boredom = normalize_boredom_state(normalize_m13_drive_state(m13_state).get("boredom"))
    level = _bounded_float(boredom.get("boredom_level"))
    target = str(boredom.get("last_exploration_target", "") or "").strip()
    refs = _string_list(boredom.get("recent_plan_terms"), limit=4)
    if boredom_band(level) not in {"medium", "high"} or level < 0.35 or not target or not refs:
        return None
    return ProactiveTarget(
        trigger="boredom_exploration_target",
        evidence_refs=refs,
        proposed_topic=target[:120],
        ordinary_language_intent=f"Offer a small fresh angle on: {target[:140]}",
        source_kind="boredom_exploration",
        urgency_band="low",
        risk_band="low",
        selection_reason_codes=["boredom_band_with_evidence_refs"],
    )


def _target_from_scheduled_signal(structural_signals: Mapping[str, Any]) -> ProactiveTarget | None:
    for row in structural_signals.get("queued_outreach", []) or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status", "pending") or "pending") != "pending":
            continue
        if str(row.get("trigger", "")) != "scheduled_outreach":
            continue
        intent_id = _clean_id(row.get("source_intent_id"))
        refs = _string_list(row.get("trigger_evidence_refs") or row.get("evidence_refs"), limit=8)
        if not intent_id:
            continue
        return ProactiveTarget(
            trigger="scheduled_outreach",
            traceable_expectation_id=intent_id,
            evidence_refs=refs or [intent_id],
            proposed_topic=str(row.get("proposed_topic", "scheduled outreach") or "scheduled outreach")[:120],
            ordinary_language_intent=str(row.get("ordinary_language_intent", "") or "")[:240],
            source_kind="scheduled_intent",
            urgency_band="high",
            risk_band="low",
            selection_reason_codes=["locked_scheduled_outbox_pending"],
        )
    return None


def select_proactive_target(
    state: Mapping[str, Any],
    m13_state: Mapping[str, Any],
    *,
    memory_efe_evaluation: Any | None = None,
    structural_signals: Mapping[str, Any] | None = None,
) -> ProactiveTarget | None:
    sig = _mapping(structural_signals)
    scheduled = _target_from_scheduled_signal(sig)
    if scheduled is not None:
        return scheduled
    mem = _target_from_memory_efe(memory_efe_evaluation, m13_state)
    if mem is not None:
        return mem
    if memory_efe_evaluation is None:
        memory_efe = normalize_memory_efe_state(normalize_m13_drive_state(m13_state).get("memory_efe"))
        if memory_efe.get("traceable_expectation_id") or memory_efe.get("eligible_for_efe"):
            return None
    else:
        if getattr(memory_efe_evaluation, "traceable_expectation_id", "") or getattr(
            memory_efe_evaluation, "eligible_for_efe", []
        ):
            return None
    correction = _target_from_correction(m13_state)
    if correction is not None:
        return correction
    if mem is None:
        return _target_from_boredom(m13_state)
    return None
