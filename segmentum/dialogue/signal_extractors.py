from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SignalExtractor(Protocol):
    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float: ...


class SemanticContentExtractor:
    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del partner_uid, session_context
        current = set(_tokenize(current_turn))
        if not current:
            return 0.0
        if not conversation_history:
            return 1.0
        history = set()
        for turn in conversation_history[-12:]:
            history.update(_tokenize(turn))
        overlap = _jaccard(current, history)
        return round(_clamp(1.0 - overlap), 6)


class TopicNoveltyExtractor:
    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del partner_uid, session_context
        current = set(_tokenize(current_turn))
        if not current:
            return 0.0
        history = set()
        for turn in conversation_history[-20:]:
            history.update(_tokenize(turn))
        if not history:
            return 1.0
        unseen_ratio = len(current - history) / max(1.0, len(current))
        return round(_clamp(unseen_ratio), 6)


class EmotionalToneExtractor:
    _POS = frozenset({"好", "开心", "谢谢", "喜欢", "赞", "great", "good", "love"})
    _NEG = frozenset({"烦", "糟", "讨厌", "生气", "难过", "bad", "hate", "angry"})

    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del conversation_history, partner_uid, session_context
        tokens = _tokenize(current_turn)
        if not tokens:
            return 0.5
        pos = sum(1 for token in tokens if token in self._POS)
        neg = sum(1 for token in tokens if token in self._NEG)
        raw = 0.5 + (pos - neg) / max(2.0, len(tokens))
        return round(_clamp(raw), 6)


class ConflictTensionExtractor:
    _NEGATION = frozenset({"不", "不是", "没", "别", "不要", "no", "not"})
    _CHALLENGE = frozenset({"凭什么", "为什么", "胡说", "错了", "离谱"})

    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del conversation_history, partner_uid, session_context
        tokens = _tokenize(current_turn)
        punctuation_pressure = min(1.0, (current_turn.count("?") + current_turn.count("？") + current_turn.count("!")) / 4.0)
        negation = sum(1 for token in tokens if token in self._NEGATION) / max(1.0, len(tokens))
        challenge = sum(1 for phrase in self._CHALLENGE if phrase in current_turn) / max(1.0, len(self._CHALLENGE))
        score = (0.45 * punctuation_pressure) + (0.35 * negation) + (0.20 * challenge)
        return round(_clamp(score), 6)


@dataclass(slots=True)
class _RelState:
    value: float = 0.10
    turns: int = 0


class RelationshipDepthExtractor:
    def __init__(self, *, update_rate: float = 0.03, max_step: float = 0.04) -> None:
        self.update_rate = update_rate
        self.max_step = max_step
        self._state: dict[int, _RelState] = {}

    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del session_context
        state = self._state.get(partner_uid, _RelState())
        length_signal = min(1.0, len(_tokenize(current_turn)) / 30.0)
        history_depth = min(1.0, len(conversation_history) / 40.0)
        target = _clamp(0.08 + (0.45 * history_depth) + (0.20 * length_signal))
        delta = max(-self.max_step, min(self.max_step, target - state.value))
        state.value = _clamp(state.value + (delta * self.update_rate / max(self.update_rate, 1e-6)))
        state.turns += 1
        self._state[partner_uid] = state
        return round(state.value, 6)


@dataclass(slots=True)
class _HiddenIntentState:
    estimate: float = 0.10
    suspicious_evidence: float = 0.0
    cooperative_evidence: float = 0.0


class HiddenIntentExtractor:
    def __init__(
        self,
        *,
        alpha: float = 0.03,
        decay: float = 0.985,
        max_step: float = 0.035,
    ) -> None:
        self.alpha = alpha
        self.decay = decay
        self.max_step = max_step
        self._state: dict[int, _HiddenIntentState] = {}

    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del conversation_history
        state = self._state.get(partner_uid, _HiddenIntentState())
        text = current_turn.strip()
        question_ratio = _clamp((text.count("?") + text.count("？")) / 2.0)
        imperative_ratio = _clamp(sum(1 for w in ("必须", "立刻", "马上", "快点") if w in text) / 2.0)
        cooperative_ratio = _clamp(sum(1 for w in ("谢谢", "请", "我们", "一起", "抱歉") if w in text) / 3.0)
        self_disclosure_ratio = _clamp(sum(1 for w in ("我觉得", "我今天", "我担心", "我希望") if w in text) / 2.0)

        suspicious_signal = _clamp(
            (0.45 * question_ratio) + (0.40 * imperative_ratio) + (0.15 * (1.0 - self_disclosure_ratio))
        )
        cooperative_signal = _clamp(
            (0.45 * cooperative_ratio) + (0.25 * (1.0 - imperative_ratio)) + (0.30 * self_disclosure_ratio)
        )
        state.suspicious_evidence = (state.suspicious_evidence * self.decay) + suspicious_signal
        state.cooperative_evidence = (state.cooperative_evidence * self.decay) + cooperative_signal

        total = state.suspicious_evidence + state.cooperative_evidence + 1e-6
        balance = state.suspicious_evidence / total
        target = 0.07 + (0.16 * balance)
        conflict_trend = float(session_context.get("conflict_trend", 0.0)) if isinstance(session_context, dict) else 0.0
        target = _clamp(target + (0.03 * _clamp(conflict_trend, -1.0, 1.0)))

        raw_delta = target - state.estimate
        capped_delta = max(-self.max_step, min(self.max_step, raw_delta))
        state.estimate = _clamp(state.estimate + (self.alpha * (capped_delta / max(self.alpha, 1e-9))))
        self._state[partner_uid] = state
        return round(state.estimate, 6)

