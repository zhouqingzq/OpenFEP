"""MVP-local M13.3 bounded UI initiative: proposals, suppression, audit events.

Engineering policy only; not subjective agency or background autonomy.
"""

from __future__ import annotations

import copy
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_boredom import boredom_band, normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state

PROACTIVE_SOURCE = "m13_initiative_policy"
PROACTIVE_SURROGATE_USER_TEXT = (
    "[system proactive tick: generate only if the pending proposal remains useful, safe, "
    "and concrete. One short message; no demand for response; no claim of subjective need.]"
)

DEFAULT_MAX_PROACTIVE_PER_SESSION = 1
DEFAULT_IDLE_THRESHOLD_SECONDS = 120
DEFAULT_COOLDOWN_TURNS = 2
PROPOSAL_TTL_SECONDS = 300
MAX_EVIDENCE_REFS = 8

SUPPRESSION_REASONS: frozenset[str] = frozenset(
    {
        "disabled",
        "not_opted_in",
        "cooldown_active",
        "idle_time_too_short",
        "no_high_value_target",
        "safety_risk",
        "insufficient_evidence",
        "session_limit_reached",
        "user_active",
        "implicit_idle_disabled",
        "proposal_expired",
        "proposal_not_found",
    }
)

_ALLOWED_TRIGGERS: frozenset[str] = frozenset(
    {
        "manual_continue",
        "open_item_next_check",
        "boredom_exploration_target",
        "user_continue_later",
        "correction_followup",
        "explicit_remind_request",
    }
)

_BLOCKED_TRIGGER_PATTERNS = re.compile(
    r"(?i)(lonely|loneliness|寂寞|孤独|好想你|想你|jealous|依赖你|guilt|punish|"
    r"无聊死了|needed to talk|got bored waiting|attention capture)",
)
_EXPLICIT_REMIND_PATTERNS = re.compile(
    r"(?i)(提醒|follow[\s-]?up|记得|别忘|keep thinking|继续想|稍后提醒|帮我记)",
)
_USER_CONTINUE_LATER_PATTERNS = re.compile(
    r"(?i)(稍后|待会|回头|continue later|明天再|下次再|later继续)",
)
_FORBIDDEN_VISIBLE_PROACTIVE_PATTERNS = re.compile(
    r"(?i)(got bored|needed to talk|lonely|loneliness|寂寞|孤独|好想你|"
    r"依赖你|jealous|addicted|成瘾|punish you|内疚你)",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()[:240]]
    if isinstance(value, list):
        return [str(item).strip()[:240] for item in value[:limit] if str(item).strip()]
    return []


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_initiative_state() -> dict[str, Any]:
    return {
        "enabled": False,
        "user_opt_in": False,
        "implicit_idle_delivery": False,
        "manual_continue_button": True,
        "idle_threshold_seconds": DEFAULT_IDLE_THRESHOLD_SECONDS,
        "cooldown_turns": DEFAULT_COOLDOWN_TURNS,
        "cooldown_until_timestamp": 0,
        "max_proactive_per_session": DEFAULT_MAX_PROACTIVE_PER_SESSION,
        "proactive_count_this_session": 0,
        "last_proactive_turn_at": 0,
        "last_proactive_turn_index": -1,
        "pending_proactive_proposal": {},
        "last_suppression_reason": "",
        "engineering_proxy_label": "mvp_local_m13_initiative",
    }


def normalize_initiative_state(raw: Any) -> dict[str, Any]:
    base = default_initiative_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["enabled"] = bool(merged.get("enabled"))
    merged["user_opt_in"] = bool(merged.get("user_opt_in"))
    merged["implicit_idle_delivery"] = bool(merged.get("implicit_idle_delivery"))
    merged["manual_continue_button"] = bool(merged.get("manual_continue_button", True))
    merged["idle_threshold_seconds"] = max(
        30, int(merged.get("idle_threshold_seconds", DEFAULT_IDLE_THRESHOLD_SECONDS) or DEFAULT_IDLE_THRESHOLD_SECONDS)
    )
    merged["cooldown_turns"] = max(0, int(merged.get("cooldown_turns", DEFAULT_COOLDOWN_TURNS) or DEFAULT_COOLDOWN_TURNS))
    merged["cooldown_until_timestamp"] = int(merged.get("cooldown_until_timestamp", 0) or 0)
    merged["max_proactive_per_session"] = max(
        1, int(merged.get("max_proactive_per_session", DEFAULT_MAX_PROACTIVE_PER_SESSION) or DEFAULT_MAX_PROACTIVE_PER_SESSION)
    )
    merged["proactive_count_this_session"] = max(0, int(merged.get("proactive_count_this_session", 0) or 0))
    merged["last_proactive_turn_at"] = int(merged.get("last_proactive_turn_at", 0) or 0)
    merged["last_proactive_turn_index"] = int(merged.get("last_proactive_turn_index", -1) or -1)
    pending = merged.get("pending_proactive_proposal")
    merged["pending_proactive_proposal"] = dict(pending) if isinstance(pending, Mapping) else {}
    merged["last_suppression_reason"] = str(merged.get("last_suppression_reason", "") or "")[:64]
    return merged


def merge_initiative_into_m13_state(m13_state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_m13_drive_state(m13_state)
    state["initiative"] = normalize_initiative_state(state.get("initiative"))
    return state


@dataclass
class ProactiveTurnProposal:
    proposal_id: str
    created_at: int
    source: str
    trigger: str
    trigger_evidence_refs: list[str]
    urgency_band: str
    expected_user_value_band: str
    risk_band: str
    proposed_action: str
    proposed_topic: str
    ordinary_language_intent: str
    expires_at: int
    cooldown_cost: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "created_at": self.created_at,
            "source": self.source,
            "trigger": self.trigger,
            "trigger_evidence_refs": list(self.trigger_evidence_refs[:MAX_EVIDENCE_REFS]),
            "urgency_band": self.urgency_band,
            "expected_user_value_band": self.expected_user_value_band,
            "risk_band": self.risk_band,
            "proposed_action": self.proposed_action,
            "proposed_topic": self.proposed_topic,
            "ordinary_language_intent": self.ordinary_language_intent,
            "expires_at": self.expires_at,
            "cooldown_cost": self.cooldown_cost,
            "engineering_proxy_label": "mvp_local_m13_initiative",
        }


@dataclass
class ProactiveInitiativeCheckResult:
    proposal: ProactiveTurnProposal | None
    suppression_reason: str
    events: list[dict[str, Any]] = field(default_factory=list)
    state_fields_read: list[str] = field(default_factory=list)


def proactive_visible_text_is_safe(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if _FORBIDDEN_VISIBLE_PROACTIVE_PATTERNS.search(cleaned):
        return False
    return True


def _recent_user_text(state: Mapping[str, Any]) -> str:
    temporal = _mapping(state.get("temporal_state"))
    parts = [
        str(temporal.get("last_user_text", "") or ""),
    ]
    for row in state.get("short_term_memory", []) or []:
        if isinstance(row, Mapping) and str(row.get("role", "")).lower() in {"user", "interlocutor"}:
            parts.append(str(row.get("content", row.get("text", "")) or ""))
    return " ".join(parts)[:480]


def _safety_risk_from_state(state: Mapping[str, Any], m13_state: Mapping[str, Any]) -> bool:
    reward = normalize_affective_reward_proxy_state(
        normalize_m13_drive_state(m13_state).get("affective_reward_proxy")
    )
    return _bounded_float(reward.get("opponent_strength")) >= 0.5


def _open_item_target(state: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    for item in state.get("open_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "open")).strip().lower()
        if status not in {"open", "pending", "active", ""}:
            continue
        title = str(item.get("title", item.get("summary", "")) or "").strip()
        next_check = str(item.get("next_check", item.get("next_step", title)) or "").strip()
        if not next_check:
            continue
        item_id = str(item.get("id", "") or "").strip()
        topic = title[:120] or next_check[:120]
        intent = f"Follow up on the open item: {next_check[:140]}"
        refs = [item_id] if item_id else []
        return ("open_item_next_check", topic, intent, refs)
    return None


def _boredom_target(m13_state: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    boredom = normalize_boredom_state(normalize_m13_drive_state(m13_state).get("boredom"))
    level = _bounded_float(boredom.get("boredom_level"))
    band = boredom_band(level)
    target = str(boredom.get("last_exploration_target", "") or "").strip()
    if band not in {"medium", "high"} or not target:
        return None
    if level < 0.35:
        return None
    intent = f"Offer a small fresh angle on: {target[:140]}"
    return ("boredom_exploration_target", target[:120], intent, _string_list(boredom.get("recent_plan_terms"), limit=4))


def _continue_later_target(state: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    recent = _recent_user_text(state)
    if _USER_CONTINUE_LATER_PATTERNS.search(recent):
        intent = "Check whether the user is ready to continue the paused thread."
        return ("user_continue_later", "paused_thread", intent, [])
    return None


def _explicit_remind_target(state: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    recent = _recent_user_text(state)
    if _EXPLICIT_REMIND_PATTERNS.search(recent):
        intent = "Provide the concise follow-up the user asked to be reminded about."
        return ("explicit_remind_request", "requested_followup", intent, [])
    return None


def _correction_followup_target(m13_state: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    reward = normalize_affective_reward_proxy_state(
        normalize_m13_drive_state(m13_state).get("affective_reward_proxy")
    )
    if _bounded_float(reward.get("opponent_strength")) < 0.35:
        return None
    pending = reward.get("pending_settlements", []) or []
    for row in pending:
        if not isinstance(row, Mapping):
            continue
        if bool(row.get("prior_safety_repair")):
            topic = str(row.get("prior_topic_fingerprint", "repair_thread"))[:120]
            intent = "Offer a concise clarification after the prior repair pressure."
            pid = str(row.get("pending_id", ""))
            return ("correction_followup", topic, intent, [pid] if pid else [])
    return None


def _pick_high_value_target(
    state: Mapping[str, Any],
    m13_state: Mapping[str, Any],
    *,
    manual_continue: bool,
) -> tuple[str, str, str, list[str]] | None:
    if manual_continue:
        for finder in (
            _open_item_target,
            lambda s: _boredom_target(m13_state),
            _continue_later_target,
            _explicit_remind_target,
            lambda s: _correction_followup_target(m13_state),
        ):
            found = finder(state)
            if found:
                return found
        return ("manual_continue", "current_thread", "Continue the current thread with one useful next step.", [])
    for finder in (
        _open_item_target,
        lambda s: _boredom_target(m13_state),
        _continue_later_target,
        _explicit_remind_target,
        lambda s: _correction_followup_target(m13_state),
    ):
        found = finder(state)
        if found:
            return found
    return None


def _build_proposal(
    *,
    trigger: str,
    proposed_topic: str,
    ordinary_language_intent: str,
    evidence_refs: list[str],
    now: int,
    initiative: Mapping[str, Any],
    urgency_band: str = "medium",
    expected_user_value_band: str = "medium",
    risk_band: str = "low",
    proposed_action: str = "answer",
) -> ProactiveTurnProposal:
    if trigger not in _ALLOWED_TRIGGERS:
        trigger = "manual_continue"
    return ProactiveTurnProposal(
        proposal_id=_new_id("m13_prop"),
        created_at=now,
        source=PROACTIVE_SOURCE,
        trigger=trigger,
        trigger_evidence_refs=_string_list(evidence_refs, limit=MAX_EVIDENCE_REFS),
        urgency_band=urgency_band,
        expected_user_value_band=expected_user_value_band,
        risk_band=risk_band,
        proposed_action=proposed_action,
        proposed_topic=proposed_topic[:120],
        ordinary_language_intent=ordinary_language_intent[:240],
        expires_at=now + PROPOSAL_TTL_SECONDS,
        cooldown_cost=int(initiative.get("cooldown_turns", DEFAULT_COOLDOWN_TURNS) or DEFAULT_COOLDOWN_TURNS),
    )


def evaluate_proactive_initiative(
    state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    idle_seconds: float = 0.0,
    manual_continue: bool = False,
    user_typing: bool = False,
    implicit_idle_request: bool = False,
) -> tuple[dict[str, Any], ProactiveInitiativeCheckResult]:
    """Deterministic cheap policy; returns updated state and check result."""
    m13_state = merge_initiative_into_m13_state(state.get("m13_drive_state", {}))
    initiative = normalize_initiative_state(m13_state.get("initiative"))
    fields_read = [
        "initiative.enabled",
        "initiative.user_opt_in",
        "open_items",
        "temporal_state",
        "m13_drive_state.boredom",
        "m13_drive_state.affective_reward_proxy",
    ]
    events: list[dict[str, Any]] = [
        {
            "type": "M13ProactiveCheckEvent",
            "turn_index": turn_index,
            "at": now,
            "user_opt_in": initiative.get("user_opt_in"),
            "enabled": initiative.get("enabled"),
            "idle_seconds": round(float(idle_seconds), 3),
            "manual_continue": manual_continue,
            "implicit_idle_request": implicit_idle_request,
            "engineering_proxy_label": "mvp_local_m13_initiative",
        }
    ]

    def suppress(reason: str) -> ProactiveInitiativeCheckResult:
        initiative["last_suppression_reason"] = reason
        initiative["pending_proactive_proposal"] = {}
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        events.append(
            {
                "type": "M13ProactiveSuppressionEvent",
                "turn_index": turn_index,
                "reason": reason,
                "user_opt_in": initiative.get("user_opt_in"),
                "proactive_count_this_session": initiative.get("proactive_count_this_session"),
                "engineering_proxy_label": "mvp_local_m13_initiative",
            }
        )
        return ProactiveInitiativeCheckResult(
            proposal=None,
            suppression_reason=reason,
            events=events,
            state_fields_read=fields_read,
        )

    if not initiative.get("user_opt_in"):
        return state, suppress("not_opted_in")
    if not initiative.get("enabled"):
        return state, suppress("disabled")
    if user_typing:
        return state, suppress("user_active")
    if int(initiative.get("proactive_count_this_session", 0) or 0) >= int(
        initiative.get("max_proactive_per_session", DEFAULT_MAX_PROACTIVE_PER_SESSION) or 1
    ):
        return state, suppress("session_limit_reached")
    cooldown_until = int(initiative.get("cooldown_until_timestamp", 0) or 0)
    if cooldown_until > now:
        return state, suppress("cooldown_active")
    last_turn = int(initiative.get("last_proactive_turn_index", -1) or -1)
    cooldown_turns = int(initiative.get("cooldown_turns", DEFAULT_COOLDOWN_TURNS) or DEFAULT_COOLDOWN_TURNS)
    if last_turn >= 0 and turn_index - last_turn <= cooldown_turns:
        return state, suppress("cooldown_active")

    if implicit_idle_request:
        if not initiative.get("implicit_idle_delivery"):
            return state, suppress("implicit_idle_disabled")
        threshold = float(initiative.get("idle_threshold_seconds", DEFAULT_IDLE_THRESHOLD_SECONDS))
        if idle_seconds < threshold:
            return state, suppress("idle_time_too_short")

    if not manual_continue and not implicit_idle_request:
        return state, suppress("implicit_idle_disabled")

    recent = _recent_user_text(state)
    if _BLOCKED_TRIGGER_PATTERNS.search(recent):
        return state, suppress("safety_risk")
    if _safety_risk_from_state(state, m13_state):
        return state, suppress("safety_risk")

    target = _pick_high_value_target(state, m13_state, manual_continue=manual_continue)
    if target is None:
        return state, suppress("no_high_value_target")

    trigger, topic, intent, refs = target
    proposal = _build_proposal(
        trigger=trigger,
        proposed_topic=topic,
        ordinary_language_intent=intent,
        evidence_refs=refs,
        now=now,
        initiative=initiative,
        urgency_band="high" if manual_continue else "medium",
    )
    initiative["pending_proactive_proposal"] = proposal.to_dict()
    initiative["last_suppression_reason"] = ""
    m13_state["initiative"] = initiative
    state["m13_drive_state"] = m13_state
    events.append(
        {
            "type": "M13ProactiveProposalEvent",
            "turn_index": turn_index,
            "proposal_id": proposal.proposal_id,
            "trigger": proposal.trigger,
            "urgency_band": proposal.urgency_band,
            "risk_band": proposal.risk_band,
            "ordinary_language_intent": proposal.ordinary_language_intent,
            "engineering_proxy_label": "mvp_local_m13_initiative",
        }
    )
    return state, ProactiveInitiativeCheckResult(
        proposal=proposal,
        suppression_reason="",
        events=events,
        state_fields_read=fields_read,
    )


def proposal_from_initiative_state(initiative: Mapping[str, Any], *, now: int) -> ProactiveTurnProposal | None:
    pending = _mapping(initiative.get("pending_proactive_proposal"))
    if not pending:
        return None
    expires = int(pending.get("expires_at", 0) or 0)
    if expires and expires < now:
        return None
    trigger = str(pending.get("trigger", "manual_continue"))
    return ProactiveTurnProposal(
        proposal_id=str(pending.get("proposal_id", "")),
        created_at=int(pending.get("created_at", now) or now),
        source=str(pending.get("source", PROACTIVE_SOURCE)),
        trigger=trigger,
        trigger_evidence_refs=_string_list(pending.get("trigger_evidence_refs"), limit=MAX_EVIDENCE_REFS),
        urgency_band=str(pending.get("urgency_band", "medium")),
        expected_user_value_band=str(pending.get("expected_user_value_band", "medium")),
        risk_band=str(pending.get("risk_band", "low")),
        proposed_action=str(pending.get("proposed_action", "answer")),
        proposed_topic=str(pending.get("proposed_topic", "")),
        ordinary_language_intent=str(pending.get("ordinary_language_intent", "")),
        expires_at=expires,
        cooldown_cost=int(pending.get("cooldown_cost", DEFAULT_COOLDOWN_TURNS) or DEFAULT_COOLDOWN_TURNS),
    )


def mark_proactive_turn_consumed(
    m13_state: dict[str, Any],
    *,
    now: int,
    turn_index: int,
    proposal: ProactiveTurnProposal,
) -> dict[str, Any]:
    state = merge_initiative_into_m13_state(m13_state)
    initiative = normalize_initiative_state(state.get("initiative"))
    initiative["proactive_count_this_session"] = int(initiative.get("proactive_count_this_session", 0) or 0) + 1
    initiative["last_proactive_turn_at"] = now
    initiative["last_proactive_turn_index"] = turn_index
    initiative["cooldown_until_timestamp"] = now + max(30, proposal.cooldown_cost * 45)
    initiative["pending_proactive_proposal"] = {}
    initiative["last_suppression_reason"] = ""
    state["initiative"] = initiative
    return state


def set_initiative_user_opt_in(m13_state: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    state = merge_initiative_into_m13_state(m13_state)
    initiative = normalize_initiative_state(state.get("initiative"))
    initiative["user_opt_in"] = bool(enabled)
    initiative["enabled"] = bool(enabled)
    if not enabled:
        initiative["pending_proactive_proposal"] = {}
        initiative["last_suppression_reason"] = "not_opted_in"
    state["initiative"] = initiative
    return state
