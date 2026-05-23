"""M14.3 unified proactive target selection.

This module is deliberately structural. It does not generate persona text and
does not infer subjective desire; it chooses whether an existing durable signal
is traceable enough to become a proactive proposal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_boredom import boredom_band, normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import _bounded_float, _mapping, _string_list, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_memory_efe import normalize_memory_efe_state
from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state

VAGUE_NEXT_CHECKS = frozenset({"", "later", "regular", "someday", "soon", "next", "follow_up", "check_later"})
TRACEABLE_NEXT_CHECKS = frozenset({"next_user_turn", "next_turn", "after_next_user_message"})
TRACEABLE_DELIVERY_TRIGGERS = frozenset(
    {
        "memory_efe_outreach",
        "scheduled_outreach",
        "correction_followup",
        "relationship_reconnect_pull",
        "affective_path_stale_proactive",
    }
)
GENERIC_INTENT_TOKENS = frozenset(
    {
        "follow",
        "up",
        "unresolved",
        "expectation",
        "traceable",
        "pending",
        "open",
        "item",
        "the",
        "on",
        "a",
        "an",
    }
)


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


def traceable_proactive_reply_grounded(
    reply: str,
    *,
    ordinary_language_intent: str,
    trigger: str,
) -> bool:
    """Deterministic post-generation check for traceable outreach triggers."""
    if trigger not in TRACEABLE_DELIVERY_TRIGGERS:
        return True
    text = str(reply or "").strip()
    if not text:
        return False
    if trigger == "scheduled_outreach":
        return len(text) >= 8
    intent = str(ordinary_language_intent or "").strip()
    prefix = "Follow up on the unresolved expectation: "
    if intent.startswith(prefix):
        summary = intent[len(prefix) :].strip()
        if len(summary) >= 6:
            tokens = [
                token
                for token in re.findall(r"[\w一-龥]{2,}", summary)
                if token.casefold() not in GENERIC_INTENT_TOKENS
            ]
            if not tokens:
                return len(text) >= 8
            return any(token.casefold() in text.casefold() for token in tokens if len(token) >= 2)
    intent_tokens = [
        token
        for token in re.findall(r"[\w一-龥]{2,}", intent)
        if len(token) >= 2 and token.casefold() not in GENERIC_INTENT_TOKENS
    ][:8]
    if intent_tokens:
        return any(token.casefold() in text.casefold() for token in intent_tokens)
    return len(text) >= 8


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
    if not should or not refs:
        return None
    if not trace_id:
        for row in eligible:
            if isinstance(row, Mapping):
                trace_id = _clean_id(row.get("expectation_id") or row.get("id"))
                if trace_id:
                    break
    for row in eligible:
        if trace_id and isinstance(row, Mapping) and _clean_id(row.get("expectation_id") or row.get("id")) == trace_id:
            intent = build_traceable_proactive_intent(row)
            topic = _content_summary(row) or trace_id
            source_kind = str(row.get("source_kind", "pending_expectation") or "pending_expectation")
            break
    else:
        topic = trace_id or refs[0]
        source_kind = "memory_efe_bound_memory"
    return ProactiveTarget(
        trigger="memory_efe_outreach",
        traceable_expectation_id=trace_id,
        evidence_refs=refs,
        proposed_topic=topic[:120],
        ordinary_language_intent=intent
        or (
            f"Follow up on traceable expectation {trace_id}"
            if trace_id
            else f"Reconnect around the memory dynamics tension bound to recalled evidence {refs[0]}"
        ),
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


def _current_user_id(state: Mapping[str, Any]) -> str:
    temporal = _mapping(state.get("temporal_state"))
    share_trace = _mapping(temporal.get("last_share_trace"))
    user_id = _clean_id(share_trace.get("user_id"))
    if user_id:
        return user_id
    rel_store = _mapping(_mapping(state.get("relationship_value_memories")).get("by_user"))
    if len(rel_store) == 1:
        return _clean_id(next(iter(rel_store.keys())))
    return ""


def _relationship_value_rows(state: Mapping[str, Any], user_id: str) -> list[dict[str, Any]]:
    rel_store = _mapping(_mapping(state.get("relationship_value_memories")).get("by_user"))
    raw_rows = rel_store.get(user_id, [])
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        confidence = _bounded_float(item.get("confidence"), default=0.0)
        priority = str(item.get("priority", "medium") or "medium").strip().lower()
        summary = str(item.get("summary", "") or "").strip()
        if confidence < 0.6 or priority not in {"high", "medium"} or not summary:
            continue
        rows.append(
            {
                "id": _clean_id(item.get("id")),
                "summary": summary[:180],
                "prediction_constraint": str(item.get("prediction_constraint", "") or "").strip()[:240],
                "priority": priority,
                "confidence": confidence,
            }
        )
    rows.sort(key=lambda row: (row["priority"] == "high", row["confidence"]), reverse=True)
    return rows[:4]


def _top_traction_action_for_user(m13_state: Mapping[str, Any], user_id: str) -> tuple[str, float]:
    traction = _mapping(normalize_m13_drive_state(m13_state).get("traction_by_action"))
    suffix = f"|{user_id}" if user_id else ""
    best_action = ""
    best_value = 0.0
    for key, value in traction.items():
        key_text = str(key)
        if suffix and not key_text.endswith(suffix):
            continue
        action = key_text.split("|", 1)[0].strip()
        score = _bounded_float(value)
        if action and score > best_value:
            best_action = action
            best_value = score
    return best_action, best_value


def _target_from_relationship_pull(state: Mapping[str, Any], m13_state: Mapping[str, Any]) -> ProactiveTarget | None:
    normalized = normalize_m13_drive_state(m13_state)
    user_id = _current_user_id(state)
    rel_rows = _relationship_value_rows(state, user_id) if user_id else []
    relation_map = _mapping(normalized.get("relation_path_precision"))
    relation_precision = _bounded_float(relation_map.get(user_id)) if user_id else 0.0
    relationship_context = bool(rel_rows) or relation_precision >= 0.35
    if not relationship_context:
        return None

    reward = normalize_affective_reward_proxy_state(normalized.get("affective_reward_proxy"))
    pending = [row for row in reward.get("pending_settlements", []) or [] if isinstance(row, Mapping)]
    path_feels_stale = bool(reward.get("path_feels_stale_proxy"))
    unsettled_reward = bool(pending) or (
        _bounded_float(reward.get("last_relief_proxy")) >= 0.25
        and _bounded_float(reward.get("last_net_reward_proxy")) < 0.35
    )
    top_action, _top_pull = _top_traction_action_for_user(normalized, user_id)
    relationship_action = top_action in {"empathize", "ask_question", "clarify", "self_disclose"}
    if not (relationship_action or path_feels_stale or unsettled_reward):
        return None

    refs = [row["id"] for row in rel_rows if row.get("id")]
    refs.extend(
        _clean_id(row.get("pending_id") or row.get("settlement_id"))
        for row in pending[:4]
        if _clean_id(row.get("pending_id") or row.get("settlement_id"))
    )
    if not refs and user_id and relation_precision >= 0.35:
        refs.append(f"relation_path_precision:{user_id}")
    refs = list(dict.fromkeys(refs))[:8]
    if not refs:
        return None

    summary = rel_rows[0]["summary"] if rel_rows else "stable relationship path"
    topic = summary[:120]
    reason_codes: list[str] = []
    if rel_rows:
        reason_codes.append("active_relationship_value_memory")
    if relation_precision >= 0.35:
        reason_codes.append("relation_path_precision_high")
    if relationship_action:
        reason_codes.append("relationship_action_top")
    if path_feels_stale:
        reason_codes.append("path_feels_stale_proxy")
    if unsettled_reward:
        reason_codes.append("unsettled_affective_path")

    trigger = "relationship_reconnect_pull" if relationship_action or rel_rows else "affective_path_stale_proactive"
    source_kind = "relationship_reconnect_pull" if trigger == "relationship_reconnect_pull" else "affective_path_stale"
    return ProactiveTarget(
        trigger=trigger,
        evidence_refs=refs,
        proposed_topic=topic,
        ordinary_language_intent=(
            "Reconnect around the active relationship value context: "
            f"{summary[:150]}. Offer one short concrete continuation without pressure."
        ),
        source_kind=source_kind,
        urgency_band="low",
        risk_band="low",
        selection_reason_codes=list(dict.fromkeys(reason_codes))[:8],
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
    relationship = _target_from_relationship_pull(state, m13_state)
    if relationship is not None:
        return relationship
    if mem is None:
        return _target_from_boredom(m13_state)
    return None
