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


# Minimal K-templates per action (Chinese); expanded in tests / demo as needed
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
        del dialogue_context, conversation_history
        templates = _RULE_TEMPLATES.get(
            action,
            ("我在。", "请继续。", "我在听。"),
        )
        bucket = _bucket_personality(personality_state)
        idx = pick_index(
            master_seed,
            "surface",
            turn_index,
            action,
            bucket,
            modulo=len(templates),
        )
        return templates[idx]
