"""M5.3 response generation: deterministic rule surface and LLM protocol."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .seed_utils import pick_index
from .types import TranscriptUtterance


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _rhetorical_move(personality_state: Mapping[str, object], action: str) -> str:
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
    if caution >= 0.62 or trust <= 0.38:
        return "guarded_short"
    if exploration >= 0.62 or action in {"ask_question", "introduce_topic"}:
        return "exploratory_questioning"
    if social >= 0.62 or trust >= 0.62:
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


def _matched_anchor_candidates(
    profile: Mapping[str, object],
    *,
    partner_uid: str,
    context_text: str,
) -> tuple[list[str], bool, bool]:
    tokens = _list_from_profile(profile, "top_tokens")
    partner_anchor_available = False
    partner_tokens_raw = profile.get("partner_tokens", {})
    if isinstance(partner_tokens_raw, Mapping):
        partner_tokens = partner_tokens_raw.get(partner_uid, [])
        if isinstance(partner_tokens, list) and partner_tokens:
            partner_anchor_available = True
            tokens = [str(item).strip() for item in partner_tokens if str(item).strip()] + tokens
    matched = [token for token in tokens[:12] if _context_matches(token, context_text)]
    return matched, bool(matched), bool(partner_anchor_available and matched)


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
    diagnostics: dict[str, object] | None = None,
) -> str:
    matched_tokens, anchor_match, partner_anchor_used = _matched_anchor_candidates(
        profile,
        partner_uid=partner_uid,
        context_text=context_text,
    )
    confidence, degraded_reason = _profile_confidence(profile, anchor_match=anchor_match)
    phrase = (
        _profile_action_phrase(profile, action, seed=master_seed, turn_index=turn_index)
        if confidence >= 0.88
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
                "surface_source": str(profile.get("source", "profile")),
                "profile_phrase_used": bool(phrase),
                "profile_phrase": phrase,
                "profile_opening_used": bool(opening),
                "topic_anchor_used": bool(anchor),
                "topic_anchor": anchor,
                "partner_anchor_used": partner_anchor_used,
                "profile_confidence": confidence,
                "profile_degraded_reason": degraded_reason,
                "profile_anchor_match": anchor_match,
                "profile_length_bucket": length_bucket,
            }
        )

    try:
        ultra_short = float(profile.get("ultra_short_ratio", 0.0))
    except (TypeError, ValueError):
        ultra_short = 0.0

    source = str(profile.get("source", "profile"))
    target_context_surface = not (
        source == "population_average"
        or source.startswith("population:")
        or source.startswith("wrong_user")
    )
    generic_focus = focus if target_context_surface else ""
    expression_available = bool(phrase or connector or opening or anchor)
    if confidence < 0.75:
        return _generic_focused_reply(action, base, generic_focus)
    if not expression_available:
        return _generic_focused_reply(action, base, generic_focus)

    if ultra_short >= 0.45 and action in {"minimal_response", "agree", "deflect"}:
        if diagnostics is not None:
            diagnostics["rhetorical_move"] = rhetorical_move
        return phrase or connector or opening or base
    if ultra_short >= 0.60 and phrase:
        if diagnostics is not None:
            diagnostics["rhetorical_move"] = rhetorical_move
        return phrase

    bits: list[str] = []
    if target_context_surface and focus and action in {
        "ask_question",
        "elaborate",
        "agree",
        "disagree",
        "empathize",
        "share_opinion",
    }:
        bits.append(f"关于“{focus}”")
    if phrase and confidence >= 0.85:
        bits.append(phrase)
    elif connector and confidence >= 0.75:
        bits.append(connector)
    elif opening and confidence >= 0.80:
        bits.append(opening)
    if anchor and confidence >= 0.90 and anchor not in " ".join(bits):
        bits.append(f"我会把{anchor}也放进判断里")
    if not bits:
        bits.append(base)
    reply = "，".join(bits)
    if not reply.endswith(("。", "！", "？", ".", "!", "?")):
        reply += "。"
    return reply


class LLMGenerator:
    """Deferred M5.6: structured LLM surface; CI and M5.3 acceptance use RuleBasedGenerator only."""

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
        raise NotImplementedError(
            "LLMGenerator requires M5.6 runtime wiring; use RuleBasedGenerator for M5.3."
        )


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
            "partner_anchor_used": False,
            "profile_confidence": 0.0,
            "profile_degraded_reason": "",
            "profile_anchor_match": False,
            "profile_length_bucket": "none",
            "rhetorical_move": style,
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
