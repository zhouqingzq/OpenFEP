"""M5.3 response generation: rule-based (deterministic) and protocol for LLM."""

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


# Deterministic per-action templates (Chinese).
_RULE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ask_question": ("能再说具体一点吗？", "你指的是哪一部分？", "我想确认一下你的意思？"),
    "introduce_topic": ("换个角度，我们聊聊这个吧。", "我想提一个新话题。", "要不谈谈这件事？"),
    "share_opinion": ("我的想法是……", "坦白说，我倾向于这样看。", "从我的角度，这件事……"),
    "elaborate": ("我补充几句。", "展开来说……", "换句话说……"),
    "agree": ("我同意。", "是的，我也这么想。", "有道理，我跟你的看法一致。"),
    "empathize": ("我能理解你的感受。", "这一定不容易。", "听起来你压力很大，我在听。"),
    "joke": ("轻松一下，别太紧张。", "开个玩笑缓和气氛。", "哈哈，我们深呼吸一下。"),
    "disagree": ("这一点我不太同意。", "我有不同看法。", "请原谅，我必须反驳一下。"),
    "deflect": ("我们先不深入这个吧。", "换个说法……", "也许可以稍后再谈这个。"),
    "minimal_response": ("嗯。", "知道了。", "收到。"),
    "disengage": ("我需要先到这里。", "今天先聊到这吧。", "抱歉，我得停下来。"),
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


def _style_tag(personality_state: Mapping[str, object]) -> str:
    traits = personality_state.get("slow_traits")
    if not isinstance(traits, Mapping):
        return "neutral"
    try:
        social = float(traits.get("social_approach", 0.5))
    except (TypeError, ValueError):
        social = 0.5
    try:
        caution = float(traits.get("caution_bias", 0.5))
    except (TypeError, ValueError):
        caution = 0.5
    if social >= 0.62:
        return "warm"
    if caution >= 0.62:
        return "guarded"
    return "neutral"


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
    """Deterministic template surface: index from derive(master, surface, turn, action, …)."""

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
        templates = _RULE_TEMPLATES.get(
            action,
            ("我在。", "请继续。", "我在听。"),
        )
        bucket = _bucket_personality(personality_state)
        style = _style_tag(personality_state)
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
        if style == "warm" and action in {"ask_question", "empathize", "agree", "elaborate"}:
            base = "我在认真听你说，" + base
        elif style == "guarded" and action in {"deflect", "minimal_response", "disengage", "disagree"}:
            base = "我先保守一点，" + base

        if focus and action in {"ask_question", "elaborate", "agree", "disagree", "empathize"}:
            if action == "ask_question":
                return f"关于“{focus}”，{base}"
            if action == "disagree":
                return f"关于“{focus}”，{base}"
            if action == "empathize":
                return f"你提到“{focus}”，{base}"
            return f"你说“{focus}”，{base}"

        if conflict >= 0.70 and action in {"agree", "joke"}:
            return "我先不激化冲突，" + base
        return base
