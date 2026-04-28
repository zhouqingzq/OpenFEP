"""M5.3 response generation: deterministic rule surface and LLM protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .seed_utils import pick_index
from .types import TranscriptUtterance
from .utils import clamp as _clamp01


class ResponseGenerator(Protocol):
    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history: Sequence[TranscriptUtterance],
        *,
        master_seed: int,
        turn_index: int,
    ) -> str: ...


_RULE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ask_question": ("能再说具体一点吗？", "你指的是哪一部分？", "我想确认一下你的意思。"),
    "introduce_topic": ("换个角度，我们聊聊这个。", "我想提一个新的切入口。", "要不先谈谈这件事？"),
    "share_opinion": ("我的想法是这样。", "坦白说，我倾向于这样看。", "从我的角度，这件事可以这么处理。"),
    "elaborate": ("我补充几句。", "展开来说。", "换句话说。"),
    "agree": ("我同意。", "是的，我也这么想。", "有道理，我跟你的看法一致。"),
    "empathize": ("我能理解你的感受。", "这一定不容易。", "听起来你压力很大，我在听。"),
    "joke": ("轻松一下，别太紧张。", "开个玩笑缓和一下。", "哈哈，我们深呼吸一下。"),
    "disagree": ("这一点我不太同意。", "我有不同看法。", "我需要谨慎地反驳一下。"),
    "deflect": ("我们先不深入这个。", "换个说法。", "也许可以稍后再谈这个。"),
    "minimal_response": ("嗯。", "知道了。", "收到。"),
    "disengage": ("我需要先到这里。", "今天先聊到这吧。", "抱歉，我得停下来。"),
}

_ACTION_FALLBACKS: dict[str, str] = {
    "ask_question": "我想先确认",
    "introduce_topic": "换个角度",
    "share_opinion": "我倾向于",
    "elaborate": "我补充",
    "agree": "我认同",
    "empathize": "我理解",
    "joke": "轻松一点",
    "disagree": "我有一点不同看法",
    "deflect": "先放一放",
    "minimal_response": "嗯",
    "disengage": "先到这里",
}


def _policy_lift_metadata(personality_state: Mapping[str, object], action: str) -> dict[str, object]:
    policies = personality_state.get("preferred_policies")
    if not isinstance(policies, Mapping):
        return {
            "applied": False,
            "strategy_confidence": 0.0,
            "policy_evidence_count": 0,
            "policy_evidence_weight": 0.0,
            "policy_source": "missing",
            "policy_context_bucket": "",
            "conditional_policy_frequency": 0.0,
            "conditional_policy_strategy_frequency": 0.0,
            "conditional_policy_support": 0.0,
            "conditional_policy_top_strategy": "",
            "policy_action_selection_lift_applied": False,
        }
    try:
        confidence = _clamp01(float(policies.get("strategy_confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        evidence_count = int(policies.get("policy_evidence_count", 0) or 0)
    except (TypeError, ValueError):
        evidence_count = 0
    distribution = policies.get("action_distribution", {})
    action_frequency = 0.0
    if isinstance(distribution, Mapping):
        try:
            action_frequency = float(distribution.get(action, 0.0))
        except (TypeError, ValueError):
            action_frequency = 0.0
    raw_preferences = policies.get("learned_preferences", [])
    learned_preferences = {str(item) for item in raw_preferences} if isinstance(raw_preferences, list) else set()
    applied = bool(
        evidence_count >= 3
        and confidence >= 0.20
        and (action in learned_preferences or action_frequency >= 0.12)
    )
    selection_context = personality_state.get("policy_action_selection_context", {})
    if not isinstance(selection_context, Mapping):
        selection_context = {}
    return {
        "applied": applied,
        "strategy_confidence": round(float(confidence), 6),
        "policy_evidence_count": int(evidence_count),
        "policy_evidence_weight": round(float(action_frequency), 6),
        "policy_source": "evidence_bearing_distribution" if evidence_count > 0 else "surface_only_or_missing",
        "policy_context_bucket": str(selection_context.get("policy_context_bucket", "")),
        "conditional_policy_frequency": selection_context.get("conditional_policy_frequency", 0.0),
        "conditional_policy_strategy_frequency": selection_context.get(
            "conditional_policy_strategy_frequency",
            0.0,
        ),
        "conditional_policy_support": selection_context.get("conditional_policy_support", 0.0),
        "conditional_policy_top_strategy": str(selection_context.get("conditional_policy_top_strategy", "")),
        "policy_action_selection_lift_applied": bool(
            selection_context.get("policy_action_selection_lift_applied", False)
        ),
    }


def _policy_detail_priority_bucket(bucket: str) -> bool:
    base_bucket = str(bucket).split("|partner:", 1)[0]
    return base_bucket in {
        "ctx:partner_low_info_ack",
        "ctx:partner_short_confirmation",
        "ctx:partner_short_other",
        "ctx:partner_statement_topicish",
    }


def _bucket_personality(personality_state: Mapping[str, object]) -> str:
    parts: list[str] = []
    traits = personality_state.get("slow_traits")
    if isinstance(traits, Mapping):
        for key in sorted(traits.keys()):
            try:
                parts.append(f"{key}:{round(float(traits[key]), 2)}")
            except (TypeError, ValueError):
                parts.append(f"{key}:?")
    return "|".join(parts) if parts else "default"


# ── Rhetorical move thresholds ─────────────────────────────────────────────
# These thresholds govern how personality traits map to conversational style.
# Calibrated to produce visibly different surface behavior for agents whose
# slow traits differ by >= 0.25 on at least two dimensions (M5.4 surface
# ablation gate).  Tuning them shifts the gentleness/suspiciousness/curiosity
# boundary; they are NOT validated against human conversational data.

_RHETORICAL_GUARDED_CAUTION_FLOOR = 0.50   # caution >= this → guarded possible
_RHETORICAL_GUARDED_TRUST_CEILING = 0.50   # trust  <= this → guarded possible
_RHETORICAL_WARM_SOCIAL_FLOOR = 0.42       # social >= this → warm possible
_RHETORICAL_WARM_TRUST_FLOOR = 0.50        # trust  >= this → warm possible
_RHETORICAL_EXPLORE_EXPLORATION_FLOOR = 0.56  # exploration >= this (for share_opinion)
_RHETORICAL_HIGH_CAUTION = 0.62            # definitely guarded
_RHETORICAL_LOW_TRUST = 0.38               # definitely guarded
_RHETORICAL_HIGH_EXPLORATION = 0.62        # definitely exploratory
_RHETORICAL_HIGH_SOCIAL = 0.62             # definitely warm
_RHETORICAL_HIGH_TRUST = 0.62              # definitely warm


def _rhetorical_move(personality_state: Mapping[str, object], action: str) -> str:
    """Map (slow_traits, action) → conversational stance label.

    Four stances:
    - ``"guarded_short"`` — defensive, brief, boundary-protecting
    - ``"exploratory_questioning"`` — curious, probing, topic-introducing
    - ``"warm_supportive"`` — agreeable, empathetic, elaborating
    - ``"direct_advisory"`` — neutral, factual (fallback)
    """
    traits = personality_state.get("slow_traits")
    if not isinstance(traits, Mapping):
        return "direct_advisory"
    try:
        social = float(traits.get("social_approach", 0.5))
    except (TypeError, ValueError):
        social = 0.5
    try:
        caution = float(traits.get("caution_bias", 0.5))
    except (TypeError, ValueError):
        caution = 0.5
    try:
        trust = float(traits.get("trust_stance", 0.5))
    except (TypeError, ValueError):
        trust = 0.5
    try:
        exploration = float(traits.get("exploration_posture", 0.5))
    except (TypeError, ValueError):
        exploration = 0.5
    if action in {"deflect", "minimal_response", "disengage", "disagree"} and (
        caution >= _RHETORICAL_GUARDED_CAUTION_FLOOR or trust <= _RHETORICAL_GUARDED_TRUST_CEILING
    ):
        return "guarded_short"
    if action in {"ask_question", "introduce_topic"} or (
        action == "share_opinion" and exploration >= _RHETORICAL_EXPLORE_EXPLORATION_FLOOR
    ):
        return "exploratory_questioning"
    if action in {"agree", "empathize", "elaborate"} and (
        social >= _RHETORICAL_WARM_SOCIAL_FLOOR or trust >= _RHETORICAL_WARM_TRUST_FLOOR
    ):
        return "warm_supportive"
    if caution >= _RHETORICAL_HIGH_CAUTION or trust <= _RHETORICAL_LOW_TRUST:
        return "guarded_short"
    if exploration >= _RHETORICAL_HIGH_EXPLORATION:
        return "exploratory_questioning"
    if social >= _RHETORICAL_HIGH_SOCIAL or trust >= _RHETORICAL_HIGH_TRUST:
        return "warm_supportive"
    return "direct_advisory"


def _latest_interlocutor_text(conversation_history: Sequence[TranscriptUtterance]) -> str:
    for item in reversed(conversation_history):
        role = str(item.get("role", ""))
        if role == "interlocutor":
            text = str(item.get("text", "")).strip()
            if text:
                return text
    return ""


def _quote_focus(text: str) -> str:
    trimmed = text.strip().replace("\n", " ")
    if not trimmed:
        return ""
    compact = " ".join(trimmed.split())
    return compact[:18]


def _list_from_profile(profile: Mapping[str, object], key: str) -> list[str]:
    raw = profile.get(key, [])
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _context_matches(value: str, context_text: str) -> bool:
    value = value.strip().lower()
    context_text = context_text.strip().lower()
    if not value or not context_text:
        return False
    if len(value) >= 3 and value in context_text:
        return True
    tokens = [tok for tok in value.replace("_", " ").split() if len(tok) >= 3]
    return bool(tokens and any(tok in context_text for tok in tokens))


def _profile_confidence(profile: Mapping[str, object], *, anchor_match: bool) -> tuple[float, str]:
    reasons: list[str] = []
    try:
        reply_count = int(profile.get("reply_count", 0) or 0)
    except (TypeError, ValueError):
        reply_count = 0
    try:
        ultra_short = float(profile.get("ultra_short_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        ultra_short = 0.0
    confidence = 1.0
    if reply_count < 4:
        confidence *= 0.45
        reasons.append("low_reply_count")
    elif reply_count < 8:
        confidence *= 0.75
        reasons.append("limited_reply_count")
    if ultra_short >= 0.70:
        confidence *= 0.45
        reasons.append("ultra_short_high")
    elif ultra_short >= 0.45:
        confidence *= 0.75
        reasons.append("ultra_short_moderate")
    if not anchor_match:
        confidence *= 0.80
        reasons.append("anchor_mismatch")
    return round(max(0.0, min(1.0, confidence)), 6), ",".join(reasons)


def _choose(items: list[str], *, seed: int, label: str, turn_index: int, action: str, fallback: str = "") -> str:
    if not items:
        return fallback
    idx = pick_index(seed, "surface-profile", label, turn_index, action, modulo=len(items))
    return items[idx]


def _profile_action_phrase(profile: Mapping[str, object], action: str, *, seed: int, turn_index: int) -> str:
    raw = profile.get("action_phrases", {})
    if not isinstance(raw, Mapping):
        return ""
    candidates = raw.get(action)
    if not isinstance(candidates, list) or not candidates:
        for alt in ("elaborate", "empathize", "agree", "ask_question"):
            candidates = raw.get(alt)
            if isinstance(candidates, list) and candidates:
                break
    if not isinstance(candidates, list):
        return ""
    return _choose([str(item) for item in candidates], seed=seed, label="action", turn_index=turn_index, action=action)


def _generic_focused_reply(action: str, base: str, focus: str) -> str:
    if focus and action in {"ask_question", "elaborate", "agree", "disagree", "empathize"}:
        if action in {"ask_question", "disagree"}:
            return f"关于“{focus}”，{base}"
        if action == "empathize":
            return f"你提到“{focus}”，{base}"
        return f"你说“{focus}”，{base}"
    return base


def _set_expression_sources(diagnostics: dict[str, object] | None, sources: list[str]) -> None:
    if diagnostics is None:
        return
    cleaned = [source for source in sources if source]
    diagnostics["profile_expression_sources"] = cleaned
    diagnostics["profile_expression_source"] = ",".join(cleaned) if cleaned else "generic"


def _matched_anchor_candidates(
    profile: Mapping[str, object],
    *,
    partner_uid: str,
    context_text: str,
) -> tuple[list[str], bool, bool, str]:
    candidate_groups: list[tuple[str, list[str], bool]] = []
    partner_context_raw = profile.get("partner_context_tokens", {})
    if isinstance(partner_context_raw, Mapping):
        partner_context = partner_context_raw.get(partner_uid, [])
        if isinstance(partner_context, list) and partner_context:
            candidate_groups.append(
                (
                    "partner_context",
                    [str(item).strip() for item in partner_context if str(item).strip()],
                    True,
                )
            )
    context_tokens = _list_from_profile(profile, "context_top_tokens")
    if context_tokens:
        candidate_groups.append(("context_global", context_tokens, False))
    partner_tokens_raw = profile.get("partner_tokens", {})
    if isinstance(partner_tokens_raw, Mapping):
        partner_tokens = partner_tokens_raw.get(partner_uid, [])
        if isinstance(partner_tokens, list) and partner_tokens:
            candidate_groups.append(
                (
                    "reply_partner_legacy",
                    [str(item).strip() for item in partner_tokens if str(item).strip()],
                    True,
                )
            )
    top_tokens = _list_from_profile(profile, "top_tokens")
    if top_tokens:
        candidate_groups.append(("reply_global_legacy", top_tokens, False))
    for source, tokens, partner_specific in candidate_groups:
        matched = [token for token in tokens[:12] if _context_matches(token, context_text)]
        if matched:
            return matched, True, bool(partner_specific), source
    return [], False, False, "none"


def _profile_reply(
    *,
    action: str,
    base: str,
    focus: str,
    profile: Mapping[str, object],
    partner_uid: str,
    context_text: str,
    rhetorical_move: str,
    master_seed: int,
    turn_index: int,
    policy_lift_applied: bool = False,
    policy_detail_applied: bool = False,
    diagnostics: dict[str, object] | None = None,
) -> str:
    source = str(profile.get("source", "profile"))
    population_state_only = source == "population_average" or source.startswith("population:")
    matched_tokens, anchor_match, partner_anchor_used, topic_anchor_source = _matched_anchor_candidates(
        profile,
        partner_uid=partner_uid,
        context_text=context_text,
    )
    confidence, degraded_reason = _profile_confidence(profile, anchor_match=anchor_match)
    phrase = (
        _profile_action_phrase(profile, action, seed=master_seed, turn_index=turn_index)
        if confidence >= 0.80 and (anchor_match or confidence >= 0.90)
        else ""
    )
    opening = _choose(
        _list_from_profile(profile, "opening_phrases"),
        seed=master_seed,
        label="opening",
        turn_index=turn_index,
        action=action,
    )
    connector = _choose(
        _list_from_profile(profile, "connector_phrases"),
        seed=master_seed,
        label="connector",
        turn_index=turn_index,
        action=action,
        fallback=_ACTION_FALLBACKS.get(action, "我补充"),
    )
    anchor = _choose(
        matched_tokens,
        seed=master_seed,
        label="anchor",
        turn_index=turn_index,
        action=action,
    )
    try:
        median_reply_chars = int(profile.get("median_reply_chars", 0) or 0)
    except (TypeError, ValueError):
        median_reply_chars = 0
    length_bucket = "short" if median_reply_chars <= 12 else "medium" if median_reply_chars <= 40 else "long"
    if diagnostics is not None:
        diagnostics.update(
            {
                "surface_source": source,
                "population_surface_state_only": population_state_only,
                "profile_phrase_used": bool(phrase),
                "profile_phrase": phrase,
                "profile_opening_used": bool(opening),
                "topic_anchor_used": bool(anchor),
                "topic_anchor": anchor,
                "topic_anchor_source": topic_anchor_source if anchor else "none",
                "partner_anchor_used": partner_anchor_used,
                "profile_confidence": confidence,
                "profile_degraded_reason": degraded_reason,
                "profile_anchor_match": anchor_match,
                "profile_length_bucket": length_bucket,
                "policy_lift_applied": bool(policy_lift_applied),
                "policy_action_selection_lift_applied": bool(policy_detail_applied),
                "surface_shortcut_suppressed": bool(population_state_only),
            }
        )

    try:
        ultra_short = float(profile.get("ultra_short_ratio", 0.0))
    except (TypeError, ValueError):
        ultra_short = 0.0
    raw_ultra_short = ultra_short
    policy_context_bucket = ""
    force_policy_detail = False
    if diagnostics is not None:
        policy_context_bucket = str(diagnostics.get("policy_context_bucket", ""))
        force_policy_detail = bool(
            policy_detail_applied and _policy_detail_priority_bucket(policy_context_bucket)
        )
    if policy_lift_applied and raw_ultra_short >= 0.45 and diagnostics is not None:
        diagnostics["surface_shortcut_suppressed"] = True
    if population_state_only:
        ultra_short = 0.0

    target_context_surface = not (
        population_state_only
        or source.startswith("wrong_user")
    )
    generic_focus = focus if target_context_surface else ""
    expression_available = bool(phrase or connector or opening or anchor)
    if confidence < 0.75 and not (force_policy_detail and generic_focus):
        _set_expression_sources(diagnostics, ["generic_focus" if generic_focus else "generic"])
        reply = _generic_focused_reply(action, base, generic_focus)
        if population_state_only and len(reply.strip()) <= 12:
            if diagnostics is not None:
                diagnostics["surface_shortcut_suppressed"] = True
            reply = reply.rstrip("。.!?") + "，我会按当前信息继续判断。"
        return reply
    if not expression_available and not (force_policy_detail and generic_focus):
        _set_expression_sources(diagnostics, ["generic_focus" if generic_focus else "generic"])
        reply = _generic_focused_reply(action, base, generic_focus)
        if population_state_only and len(reply.strip()) <= 12:
            if diagnostics is not None:
                diagnostics["surface_shortcut_suppressed"] = True
            reply = reply.rstrip("。.!?") + "，我会按当前信息继续判断。"
        return reply

    if ultra_short >= 0.45 and action in {"minimal_response", "agree", "deflect"} and not policy_lift_applied:
        if diagnostics is not None:
            diagnostics["rhetorical_move"] = rhetorical_move
        selected_source = (
            "action_phrase"
            if phrase
            else "connector"
            if connector
            else "opening"
            if opening
            else "generic"
        )
        _set_expression_sources(diagnostics, [selected_source])
        return phrase or connector or opening or base
    if ultra_short >= 0.45 and action in {"minimal_response", "agree", "deflect"} and policy_lift_applied:
        if diagnostics is not None:
            diagnostics["surface_shortcut_suppressed"] = True
    if ultra_short >= 0.60 and phrase and not policy_lift_applied:
        if diagnostics is not None:
            diagnostics["rhetorical_move"] = rhetorical_move
        _set_expression_sources(diagnostics, ["action_phrase"])
        return phrase

    bits: list[str] = []
    expression_sources: list[str] = []
    if (
        force_policy_detail
        and target_context_surface
        and focus
        and action in {"agree", "minimal_response", "deflect", "disengage"}
    ):
        bits.append(f'关于“{focus}”，我先按当前线索稳住回应。')
        expression_sources.append("policy_detail")
    if target_context_surface and focus and action in {
        "ask_question",
        "elaborate",
        "agree",
        "disagree",
        "empathize",
        "share_opinion",
    }:
        bits.append(f"关于“{focus}”")
        expression_sources.append("focus")
    if phrase and confidence >= 0.80:
        bits.append(phrase)
        expression_sources.append("action_phrase")
    elif connector and confidence >= 0.75:
        bits.append(connector)
        expression_sources.append("connector")
    elif opening and confidence >= 0.80:
        bits.append(opening)
        expression_sources.append("opening")
    if anchor and confidence >= 0.90 and anchor not in " ".join(bits):
        bits.append(f"我会把{anchor}也放进判断里")
        expression_sources.append("anchor")
    if policy_detail_applied and target_context_surface and focus and action in {
        "ask_question",
        "introduce_topic",
        "share_opinion",
        "agree",
        "empathize",
        "elaborate",
        "joke",
    }:
        if action in {"ask_question", "introduce_topic", "share_opinion"}:
            bits.append(f"我会围绕“{focus}”再确认具体线索")
        else:
            bits.append(f"我会把“{focus}”的细节接住并继续确认")
        expression_sources.append("policy_detail")
    if (
        policy_detail_applied
        and target_context_surface
        and focus
        and action in {"deflect", "minimal_response", "disengage"}
    ):
        bits.append(f"关于“{focus}”，我先保持简短边界")
        expression_sources.append("policy_detail")
    if not bits:
        bits.append(base)
        expression_sources.append("generic")
    reply = "，".join(bits)
    if not reply.endswith(("。", "！", "？", ".", "!", "?")):
        reply += "。"
    if population_state_only and len(reply.strip()) <= 12:
        if diagnostics is not None:
            diagnostics["surface_shortcut_suppressed"] = True
        reply = reply.rstrip("。.!?") + "，我会按当前信息继续判断。"
    if policy_lift_applied and len(reply.strip()) <= 12 and focus:
        if diagnostics is not None:
            diagnostics["surface_shortcut_suppressed"] = True
        reply = f"关于“{focus}”，{reply.rstrip('。.!?')}，我会再补充一点。"
    _set_expression_sources(diagnostics, expression_sources)
    return reply


# ── LLM prompt helpers ──────────────────────────────────────────────────────

_ACTION_INSTRUCTIONS: dict[str, str] = {
    "ask_question": "自然地提出一个问题，引导对方继续说下去",
    "introduce_topic": "换个新话题聊聊",
    "share_opinion": "分享你的真实看法或观点",
    "elaborate": "补充更多细节，展开说说",
    "agree": "表示同意或认同对方的观点",
    "empathize": "表达理解和共情，给对方情感上的支持",
    "joke": "用轻松幽默的方式回应，活跃气氛",
    "disagree": "委婉地表达不同意见",
    "deflect": "巧妙地避开这个话题，不深入讨论",
    "minimal_response": "简单回应一下就好，不用展开",
    "disengage": "礼貌自然地结束这段对话",
}

_TRAIT_LABELS: dict[str, tuple[str, str]] = {
    "caution_bias": ("大大咧咧、不拘小节", "谨慎小心、思虑周全"),
    "threat_sensitivity": ("心态平和、不容易感到威胁", "警觉敏锐、对环境变化很敏感"),
    "trust_stance": ("多疑戒备、不容易相信别人", "信任开放、愿意相信他人"),
    "exploration_posture": ("安于现状、喜欢熟悉的事物", "充满好奇、喜欢探索新事物"),
    "social_approach": ("内向安静、喜欢独处", "外向健谈、喜欢社交"),
}


def _describe_trait(value: float, low_label: str, high_label: str) -> str:
    if value <= 0.25:
        return f"非常{low_label}"
    elif value <= 0.40:
        return f"比较{low_label}"
    elif value <= 0.60:
        return "中性"
    elif value <= 0.75:
        return f"比较{high_label}"
    else:
        return f"非常{high_label}"


def _emotional_label(value: float) -> str:
    if value >= 0.65:
        return "积极愉快"
    elif value >= 0.55:
        return "偏正面"
    elif value >= 0.45:
        return "中性平稳"
    elif value >= 0.30:
        return "偏负面"
    else:
        return "低落沮丧"


def _conflict_label(value: float) -> str:
    if value >= 0.70:
        return "剑拔弩张，火药味很重"
    elif value >= 0.50:
        return "有明显分歧和张力"
    elif value >= 0.30:
        return "有一些小摩擦"
    else:
        return "气氛平和"


def _format_history(history: Sequence[TranscriptUtterance], max_turns: int = 8) -> str:
    if not history:
        return "（这是对话的开始）"
    recent = history[-(max_turns * 2):]  # pairs of utterances
    lines: list[str] = []
    for u in recent:
        role = str(u.get("role", ""))
        text = str(u.get("text", "")).strip()
        if not text:
            continue
        label = "对方" if role == "interlocutor" else "我"
        lines.append(f"{label}：{text}")
    return "\n".join(lines) if lines else "（这是对话的开始）"


def _build_system_prompt(
    action: str,
    personality_state: dict[str, object],
    emotional_tone: float,
    conflict_tension: float,
) -> str:
    traits = personality_state.get("slow_traits")
    trait_lines: list[str] = []
    if isinstance(traits, Mapping):
        for key in ("caution_bias", "threat_sensitivity", "trust_stance",
                     "exploration_posture", "social_approach"):
            labels = _TRAIT_LABELS.get(key, ("低", "高"))
            try:
                v = float(traits.get(key, 0.5))
            except (TypeError, ValueError):
                v = 0.5
            desc = _describe_trait(v, *labels)
            trait_lines.append(f"- {labels[0]} ↔ {labels[1]}：{desc}")

    action_instruction = _ACTION_INSTRUCTIONS.get(action, "自然地回应对方")
    emotional_desc = _emotional_label(emotional_tone)
    conflict_desc = _conflict_label(conflict_tension)

    # Narrative priors add color
    priors = personality_state.get("narrative_priors")
    priors_text = ""
    if isinstance(priors, Mapping):
        try:
            trust_prior = float(priors.get("trust_prior", 0.0))
        except (TypeError, ValueError):
            trust_prior = 0.0
        if trust_prior >= 0.50:
            priors_text = "\n你过往的经历让你对人际关系总体持正面态度。"
        elif trust_prior <= -0.50:
            priors_text = "\n你过往的经历让你对人际关系总体持谨慎态度。"

    return f"""你是一个正在和朋友微信聊天的人。你的聊天风格自然、口语化，像真人一样。

## 你的性格特质
{chr(10).join(trait_lines) if trait_lines else '- 性格中性平和'}
{priors_text}
## 当前对话
你需要：{action_instruction}
对方情绪：{emotional_desc}
对话气氛：{conflict_desc}

## 回复要求
- 用中文回复，自然口语化，像真人微信聊天
- 1-4句话，不要太长
- 不要说"作为一个人工智能"或类似的话，你就是你
- 直接回复内容本身，不要加引号或前缀标签"""


# ── LLMGenerator ────────────────────────────────────────────────────────────

class LLMGenerator:
    """Structured LLM surface backed by OpenRouter (deepseek/deepseek-v4-flash).

    Personality state (slow_traits + narrative_priors) is injected via a
    structured system prompt.  Action and observation channels constrain the
    response direction.  Conversation history provides dialogue coherence.

    Errors are recorded in ``last_diagnostics`` and returned as a visible
    error string — there is no silent fallback to rule-based generation.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-v4-flash",
        temperature: float = 0.7,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.last_diagnostics: dict[str, object] = {}

    # ── config loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_openrouter_config() -> dict:
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "secrets" / "openrouter.json"
        )
        if not config_path.exists():
            return {}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    # ── generate ────────────────────────────────────────────────────────

    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history: Sequence[TranscriptUtterance],
        *,
        master_seed: int,
        turn_index: int,
    ) -> str:
        del master_seed, turn_index  # reserved for future caching

        cfg = self._load_openrouter_config()
        api_key = cfg.get("api_key")
        base_url = str(cfg.get("base_url", "https://openrouter.ai/api/v1"))

        if not api_key:
            self.last_diagnostics = {
                "llm_error": "missing_api_key",
                "llm_error_detail": "secrets/openrouter.json missing or invalid",
            }
            return "[LLM 错误：未配置 API key，请检查 secrets/openrouter.json]"

        observation = dialogue_context.get("observation")
        if isinstance(observation, Mapping):
            emotional_tone = float(observation.get("emotional_tone", 0.5))
            conflict_tension = float(observation.get("conflict_tension", 0.0))
        else:
            emotional_tone = 0.5
            conflict_tension = 0.0

        current_turn = str(dialogue_context.get("current_turn", "")).strip()
        history_text = _format_history(conversation_history)

        system_prompt = _build_system_prompt(
            action, personality_state, emotional_tone, conflict_tension
        )

        user_content = f"对话历史：\n{history_text}\n\n对方刚说：{current_turn}\n\n请回复："

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            import requests
        except ImportError:
            self.last_diagnostics = {
                "llm_error": "missing_requests",
                "llm_error_detail": "requests library not installed",
            }
            return "[LLM 错误：缺少 requests 库，无法调用 API]"

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            elapsed = round(time.monotonic() - start, 3)
        except requests.Timeout:
            self.last_diagnostics = {
                "llm_error": "timeout",
                "llm_error_detail": f"API timeout after {self.timeout_seconds}s",
                "llm_latency_ms": int(self.timeout_seconds * 1000),
            }
            return f"[LLM 超时：{self.timeout_seconds}秒内未收到回复]"
        except requests.ConnectionError as exc:
            self.last_diagnostics = {
                "llm_error": "connection_error",
                "llm_error_detail": str(exc),
            }
            return "[LLM 错误：无法连接到 OpenRouter API]"
        except Exception as exc:
            self.last_diagnostics = {
                "llm_error": "request_failed",
                "llm_error_detail": f"{type(exc).__name__}: {exc}",
            }
            return f"[LLM 错误：请求失败 — {type(exc).__name__}]"

        if response.status_code != 200:
            self.last_diagnostics = {
                "llm_error": f"http_{response.status_code}",
                "llm_error_detail": response.text[:500],
                "llm_latency_ms": int(elapsed * 1000),
            }
            return f"[LLM 错误：API 返回 HTTP {response.status_code}]"

        try:
            body = response.json()
        except ValueError:
            self.last_diagnostics = {
                "llm_error": "invalid_json",
                "llm_error_detail": response.text[:500],
                "llm_latency_ms": int(elapsed * 1000),
            }
            return "[LLM 错误：API 返回非 JSON 响应]"

        try:
            reply = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            self.last_diagnostics = {
                "llm_error": "unexpected_response_shape",
                "llm_error_detail": str(exc),
                "llm_raw_body": body,
            }
            return "[LLM 错误：API 返回格式异常]"

        # Remove common model artifacts
        for prefix in ("回复：", "我：", "我说：", "答："):
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

        self.last_diagnostics = {
            "llm_model": self.model,
            "llm_latency_ms": int(elapsed * 1000),
            "llm_tokens_prompt": body.get("usage", {}).get("prompt_tokens", 0),
            "llm_tokens_completion": body.get("usage", {}).get("completion_tokens", 0),
            "llm_tokens_total": body.get("usage", {}).get("total_tokens", 0),
        }
        return reply


class RuleBasedGenerator:
    """Deterministic template surface with optional train-only profile anchors."""

    def __init__(self) -> None:
        self.last_diagnostics: dict[str, object] = {}

    def generate(
        self,
        action: str,
        dialogue_context: dict[str, object],
        personality_state: dict[str, object],
        conversation_history: Sequence[TranscriptUtterance],
        *,
        master_seed: int,
        turn_index: int,
    ) -> str:
        observation = dialogue_context.get("observation")
        if isinstance(observation, Mapping):
            conflict = _clamp01(float(observation.get("conflict_tension", 0.0)))
            emotional = _clamp01(float(observation.get("emotional_tone", 0.5)))
        else:
            conflict = 0.0
            emotional = 0.5
        current_turn = str(dialogue_context.get("current_turn", "")).strip()
        prior_turn = _latest_interlocutor_text(conversation_history)
        focus = _quote_focus(current_turn or prior_turn)
        templates = _RULE_TEMPLATES.get(action, ("我在。", "请继续。", "我在听。"))
        bucket = _bucket_personality(personality_state)
        style = _rhetorical_move(personality_state, action)
        policy_meta = _policy_lift_metadata(personality_state, action)
        policy_lift_applied = bool(policy_meta.get("applied", False))
        policy_detail_applied = bool(policy_meta.get("policy_action_selection_lift_applied", False))
        if policy_lift_applied and action in {"ask_question", "introduce_topic", "share_opinion"}:
            style = "exploratory_questioning"
        elif policy_lift_applied and action in {"agree", "empathize", "elaborate", "joke"}:
            style = "warm_supportive"
        idx = pick_index(
            master_seed,
            "surface",
            turn_index,
            action,
            bucket,
            style,
            round(conflict, 2),
            round(emotional, 2),
            focus,
            modulo=len(templates),
        )
        base = templates[idx]
        generation_diagnostics: dict[str, object] = {
            "template_id": f"{action}:{idx}",
            "surface_source": "generic",
            "profile_phrase_used": False,
            "profile_phrase": "",
            "profile_opening_used": False,
            "topic_anchor_used": False,
            "topic_anchor": "",
            "topic_anchor_source": "none",
            "partner_anchor_used": False,
            "profile_confidence": 0.0,
            "profile_degraded_reason": "",
            "profile_anchor_match": False,
            "profile_length_bucket": "none",
            "profile_expression_sources": [],
            "profile_expression_source": "generic",
            "rhetorical_move": style,
            "policy_lift_applied": policy_lift_applied,
            "policy_evidence_weight": policy_meta.get("policy_evidence_weight", 0.0),
            "policy_strategy_confidence": policy_meta.get("strategy_confidence", 0.0),
            "policy_context_bucket": policy_meta.get("policy_context_bucket", ""),
            "conditional_policy_frequency": policy_meta.get("conditional_policy_frequency", 0.0),
            "conditional_policy_strategy_frequency": policy_meta.get(
                "conditional_policy_strategy_frequency",
                0.0,
            ),
            "conditional_policy_support": policy_meta.get("conditional_policy_support", 0.0),
            "conditional_policy_top_strategy": policy_meta.get("conditional_policy_top_strategy", ""),
            "policy_action_selection_lift_applied": policy_detail_applied,
            "calibration_policy_source": policy_meta.get("policy_source", "missing"),
            "surface_shortcut_suppressed": False,
        }
        if style == "warm_supportive" and action in {"ask_question", "empathize", "agree", "elaborate"}:
            base = "我在认真听你说，" + base
        elif style == "guarded_short" and action in {"deflect", "minimal_response", "disengage", "disagree"}:
            base = "我先保守一点，" + base

        profile = personality_state.get("surface_profile")
        if isinstance(profile, Mapping) and int(profile.get("reply_count", 0) or 0) > 0:
            reply = _profile_reply(
                action=action,
                base=base,
                focus=focus,
                profile=profile,
                partner_uid=str(dialogue_context.get("partner_uid", "")),
                context_text=current_turn or prior_turn,
                rhetorical_move=style,
                master_seed=master_seed,
                turn_index=turn_index,
                policy_lift_applied=policy_lift_applied,
                policy_detail_applied=policy_detail_applied,
                diagnostics=generation_diagnostics,
            )
            self.last_diagnostics = generation_diagnostics
            return reply

        self.last_diagnostics = generation_diagnostics
        if focus and action in {"ask_question", "elaborate", "agree", "disagree", "empathize"}:
            if action in {"ask_question", "disagree"}:
                return f"关于“{focus}”，{base}"
            if action == "empathize":
                return f"你提到“{focus}”，{base}"
            return f"你说“{focus}”，{base}"

        if conflict >= 0.70 and action in {"agree", "joke"}:
            return "我先不激化冲突，" + base
        return base
