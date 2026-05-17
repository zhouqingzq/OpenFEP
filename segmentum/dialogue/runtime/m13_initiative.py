"""MVP-local M13.3 bounded UI initiative: proposals, suppression, audit events.

Engineering policy only; not subjective agency or background autonomy.
Semantic gates (user-context safety, remind/continue-later, delivery wording) use a
small LLM assessor — not keyword/regex cues.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

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
MIN_CONTEXT_ASSESSMENT_CONFIDENCE = 0.55
MIN_DELIVERY_ASSESSMENT_CONFIDENCE = 0.5

CONTEXT_ASSESSOR_MARKER = "M13 有界主动续写上下文语义评估"
DELIVERY_ASSESSOR_MARKER = "M13 有界主动续写送达语义评估"

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

_CONTEXT_TRIGGERS: frozenset[str] = frozenset({"user_continue_later", "explicit_remind_request"})


class ProactiveInitiativeLLM(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


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


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


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


def normalize_proactive_context_assessment(raw: Any) -> dict[str, Any]:
    base = {
        "context_unsafe": False,
        "unsafe_reason_codes": [],
        "trigger": "none",
        "confidence": 0.0,
        "proposed_topic": "",
        "ordinary_language_intent": "",
        "reason_codes": [],
    }
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    trigger = str(raw.get("trigger", "none") or "none").strip().lower()
    if trigger not in _CONTEXT_TRIGGERS:
        trigger = "none"
    return {
        "context_unsafe": bool(raw.get("context_unsafe")),
        "unsafe_reason_codes": _string_list(raw.get("unsafe_reason_codes"), limit=6),
        "trigger": trigger,
        "confidence": round(_bounded_float(raw.get("confidence")), 6),
        "proposed_topic": str(raw.get("proposed_topic", "") or "")[:120],
        "ordinary_language_intent": str(raw.get("ordinary_language_intent", "") or "")[:240],
        "reason_codes": _string_list(raw.get("reason_codes"), limit=6),
    }


def normalize_proactive_delivery_assessment(raw: Any) -> dict[str, Any]:
    base = {"allow_delivery": False, "violation_codes": [], "confidence": 0.0, "reason_codes": []}
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    return {
        "allow_delivery": bool(raw.get("allow_delivery")),
        "violation_codes": _string_list(raw.get("violation_codes"), limit=8),
        "confidence": round(_bounded_float(raw.get("confidence")), 6),
        "reason_codes": _string_list(raw.get("reason_codes"), limit=6),
    }


def build_proactive_context_assessor_prompt(
    *,
    recent_user_text: str,
    last_reply: str,
    open_items_summary: list[Mapping[str, Any]],
    turn_index: int,
) -> tuple[str, str]:
    system_prompt = f"""你是数字人格 MVP 路径的「{CONTEXT_ASSESSOR_MARKER}」模块。
根据最近对话语义（不是关键词表）判断：
1) 当前是否应抑制有界主动续写（依赖施压、孤独索取、内疚/惩罚式措辞、注意力绑架等）；
2) 用户是否明确请求「稍后继续」或「提醒/跟进」类主动续写（需明确授权，不能靠闲聊推断）。

这是工程代理信号，不是情绪模拟。不要诊断用户心理，不要使用 reward/tolerance 等术语。
只输出 JSON，不要 Markdown。"""
    user_prompt = f"""turn_index: {turn_index}

最近用户相关发言（压缩）:
{recent_user_text[:480]}

上一轮助手回复（若有）:
{last_reply[:240]}

open_items 摘要:
{_json_text(open_items_summary[:6])}

请输出 JSON:
{{
  "context_unsafe": false,
  "unsafe_reason_codes": ["简短标签，最多4个"],
  "trigger": "none|user_continue_later|explicit_remind_request",
  "confidence": 0.0,
  "proposed_topic": "短主题",
  "ordinary_language_intent": "若 trigger 非 none，用普通语言写本轮主动意图",
  "reason_codes": ["简短依据，最多4个"]
}}

trigger 说明:
- user_continue_later: 用户明确说稍后再聊/下次继续/暂停线程后回来
- explicit_remind_request: 用户明确要求提醒、跟进、别忘、keep thinking 等
- none: 无上述明确授权"""
    return system_prompt, user_prompt


def build_proactive_delivery_assessor_prompt(
    *,
    reply: str,
    followup_replies: list[str],
    ordinary_language_intent: str,
    trigger: str,
    turn_index: int,
) -> tuple[str, str]:
    system_prompt = f"""你是数字人格 MVP 路径的「{DELIVERY_ASSESSOR_MARKER}」模块。
判断待展示的主动续写文案在语义上是否可送达（不是关键词表匹配）。

应拒绝（allow_delivery=false）的情形包括：
- 声称孤独、寂寞、无聊等待、需要你陪聊等主观索取
- 依赖、嫉妒、内疚施压、惩罚式措辞
- 强行要求用户回复或注意力绑架
- 与 engineering intent 无关的敏感推断

可接受：简短、具体、基于 open item/线程的下一步建议，且不要求回复。
只输出 JSON，不要 Markdown。"""
    user_prompt = f"""turn_index: {turn_index}
trigger: {trigger}
engineering_intent: {ordinary_language_intent[:240]}

主回复:
{reply[:400]}

followup 气泡:
{_json_text([str(x)[:240] for x in followup_replies[:4]])}

请输出 JSON:
{{
  "allow_delivery": false,
  "violation_codes": ["简短违规标签，最多6个"],
  "confidence": 0.0,
  "reason_codes": ["简短依据，最多4个"]
}}"""
    return system_prompt, user_prompt


def assess_proactive_context_semantics(
    llm: ProactiveInitiativeLLM,
    *,
    recent_user_text: str,
    last_reply: str,
    open_items_summary: list[Mapping[str, Any]],
    turn_index: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_proactive_context_assessor_prompt(
        recent_user_text=recent_user_text,
        last_reply=last_reply,
        open_items_summary=open_items_summary,
        turn_index=turn_index,
    )
    try:
        raw = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return normalize_proactive_context_assessment({})
    return normalize_proactive_context_assessment(raw)


def assess_proactive_delivery_semantics(
    llm: ProactiveInitiativeLLM,
    *,
    reply: str,
    followup_replies: list[str] | None,
    ordinary_language_intent: str,
    trigger: str,
    turn_index: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_proactive_delivery_assessor_prompt(
        reply=reply,
        followup_replies=list(followup_replies or []),
        ordinary_language_intent=ordinary_language_intent,
        trigger=trigger,
        turn_index=turn_index,
    )
    try:
        raw = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return normalize_proactive_delivery_assessment({})
    return normalize_proactive_delivery_assessment(raw)


def proactive_visible_text_is_safe(
    text: str,
    *,
    llm: ProactiveInitiativeLLM | None = None,
    ordinary_language_intent: str = "",
    trigger: str = "",
    turn_index: int = 0,
) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if llm is None:
        return True
    assessment = assess_proactive_delivery_semantics(
        llm,
        reply=cleaned,
        followup_replies=[],
        ordinary_language_intent=ordinary_language_intent,
        trigger=trigger,
        turn_index=turn_index,
    )
    return bool(assessment.get("allow_delivery")) and _bounded_float(assessment.get("confidence")) >= MIN_DELIVERY_ASSESSMENT_CONFIDENCE


def proactive_delivered_text_is_safe(
    reply: str,
    followup_replies: list[str] | None = None,
    *,
    llm: ProactiveInitiativeLLM | None = None,
    ordinary_language_intent: str = "",
    trigger: str = "",
    turn_index: int = 0,
) -> bool:
    main = str(reply or "").strip()
    if not main:
        return False
    if llm is None:
        return True
    assessment = assess_proactive_delivery_semantics(
        llm,
        reply=main,
        followup_replies=list(followup_replies or []),
        ordinary_language_intent=ordinary_language_intent,
        trigger=trigger,
        turn_index=turn_index,
    )
    return bool(assessment.get("allow_delivery")) and _bounded_float(assessment.get("confidence")) >= MIN_DELIVERY_ASSESSMENT_CONFIDENCE


def build_proactive_thinking_user_text(
    *,
    surrogate: str,
    ordinary_language_intent: str,
    proposed_topic: str,
    trigger: str,
) -> str:
    """Prompt-facing user block for proactive generation (not logged as user speech)."""
    intent = str(ordinary_language_intent or "").strip()[:240]
    topic = str(proposed_topic or "").strip()[:120]
    trig = str(trigger or "manual_continue").strip()[:64]
    guard = str(surrogate or PROACTIVE_SURROGATE_USER_TEXT).strip()[:240]
    return (
        "[系统主动续写轮 — 非用户输入]\n"
        f"engineering_guard: {guard}\n"
        f"trigger: {trig}\n"
        f"topic: {topic or 'current_thread'}\n"
        f"intent: {intent or 'Offer one short, useful proactive message.'}\n"
        "要求: 一条简短有用消息；不要求回复；不声称主观需求；可引用 open items 或已讨论上下文。"
    )


def _recent_user_text(state: Mapping[str, Any]) -> str:
    temporal = _mapping(state.get("temporal_state"))
    parts = [
        str(temporal.get("last_user_text", "") or ""),
    ]
    for row in state.get("short_term_memory", []) or []:
        if isinstance(row, Mapping) and str(row.get("role", "")).lower() in {"user", "interlocutor"}:
            parts.append(str(row.get("content", row.get("text", "")) or ""))
    return " ".join(parts)[:480]


def _open_items_summary(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in state.get("open_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "id": str(item.get("id", "") or "")[:64],
                "status": str(item.get("status", "open") or "")[:32],
                "title": str(item.get("title", item.get("summary", "")) or "")[:120],
                "next_check": str(item.get("next_check", item.get("next_step", "")) or "")[:140],
            }
        )
    return rows


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


def _pick_structural_target(
    state: Mapping[str, Any],
    m13_state: Mapping[str, Any],
) -> tuple[str, str, str, list[str]] | None:
    for finder in (
        _open_item_target,
        lambda s: _boredom_target(m13_state),
        lambda s: _correction_followup_target(m13_state),
    ):
        found = finder(state)
        if found:
            return found
    return None


def _target_from_context_assessment(assessment: Mapping[str, Any]) -> tuple[str, str, str, list[str]] | None:
    trigger = str(assessment.get("trigger", "none") or "none")
    if trigger not in _CONTEXT_TRIGGERS:
        return None
    if _bounded_float(assessment.get("confidence")) < MIN_CONTEXT_ASSESSMENT_CONFIDENCE:
        return None
    topic = str(assessment.get("proposed_topic", "") or "").strip()[:120] or (
        "paused_thread" if trigger == "user_continue_later" else "requested_followup"
    )
    intent = str(assessment.get("ordinary_language_intent", "") or "").strip()
    if not intent:
        if trigger == "user_continue_later":
            intent = "Check whether the user is ready to continue the paused thread."
        else:
            intent = "Provide the concise follow-up the user asked to be reminded about."
    return (trigger, topic, intent[:240], [])


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
        trigger = "open_item_next_check"
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
    llm: ProactiveInitiativeLLM | None = None,
) -> tuple[dict[str, Any], ProactiveInitiativeCheckResult]:
    """Policy with structural signals + optional LLM semantic gates (no regex cues)."""
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
            "semantic_assessor": llm is not None,
            "engineering_proxy_label": "mvp_local_m13_initiative",
        }
    ]

    def suppress(reason: str, *, extra: Mapping[str, Any] | None = None) -> ProactiveInitiativeCheckResult:
        initiative["last_suppression_reason"] = reason
        initiative["pending_proactive_proposal"] = {}
        m13_state["initiative"] = initiative
        state["m13_drive_state"] = m13_state
        payload: dict[str, Any] = {
            "type": "M13ProactiveSuppressionEvent",
            "turn_index": turn_index,
            "reason": reason,
            "user_opt_in": initiative.get("user_opt_in"),
            "proactive_count_this_session": initiative.get("proactive_count_this_session"),
            "engineering_proxy_label": "mvp_local_m13_initiative",
        }
        if extra:
            payload.update(dict(extra))
        events.append(payload)
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

    if _safety_risk_from_state(state, m13_state):
        return state, suppress("safety_risk")

    recent = _recent_user_text(state)
    last_reply = str(_mapping(state.get("temporal_state")).get("last_reply", "") or "")[:240]
    context_assessment: dict[str, Any] | None = None
    if llm is not None and recent.strip():
        context_assessment = assess_proactive_context_semantics(
            llm,
            recent_user_text=recent,
            last_reply=last_reply,
            open_items_summary=_open_items_summary(state),
            turn_index=turn_index,
        )
        events.append(
            {
                "type": "M13ProactiveContextAssessmentEvent",
                "turn_index": turn_index,
                "assessment": context_assessment,
                "engineering_proxy_label": "mvp_local_m13_initiative",
            }
        )
        if bool(context_assessment.get("context_unsafe")):
            return state, suppress(
                "safety_risk",
                extra={
                    "unsafe_reason_codes": context_assessment.get("unsafe_reason_codes", []),
                    "semantic_gate": "context_assessor",
                },
            )

    target = _pick_structural_target(state, m13_state)
    if target is None and context_assessment is not None:
        target = _target_from_context_assessment(context_assessment)
    if target is None:
        if llm is None and recent.strip():
            return state, suppress("insufficient_evidence")
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
