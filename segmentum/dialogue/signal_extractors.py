from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Protocol

from .utils import clamp as _clamp


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
    _POS = frozenset({
        "好", "赞", "棒", "妙",
        "开心", "谢谢", "喜欢", "不错", "有意思", "有趣", "放松",
        "舒服", "快乐", "高兴", "期待", "幸运",
        "great", "good", "love", "nice", "wonderful", "happy",
    })
    _NEG = frozenset({
        "烦", "糟", "累", "苦",
        "讨厌", "生气", "难过", "担心", "焦虑", "压力", "失望",
        "批评", "烦躁", "疲惫", "不舒服", "受不了", "不开心",
        "无聊", "害怕", "后悔",
        "bad", "hate", "angry", "sad", "terrible", "awful",
    })

    def extract(
        self,
        current_turn: str,
        conversation_history: list[str],
        partner_uid: int,
        session_context: dict[str, object],
    ) -> float:
        del conversation_history, partner_uid, session_context
        raw_text = current_turn.lower()
        tokens = _tokenize(current_turn)
        if not tokens:
            return 0.5
        # Single-char keywords via token matching (backward compatible)
        pos = sum(1 for token in tokens if token in self._POS)
        neg = sum(1 for token in tokens if token in self._NEG)
        # Multi-char keywords via raw substring matching (fixes dead code)
        pos += sum(1 for w in self._POS if len(w) > 1 and w in raw_text)
        neg += sum(1 for w in self._NEG if len(w) > 1 and w in raw_text)
        raw = 0.5 + (pos - neg) / max(2.0, len(tokens))
        return round(_clamp(raw), 6)


class ConflictTensionExtractor:
    _NEGATION = frozenset({
        "不", "没", "别",
        "不是", "不要", "不行", "不可能", "不对", "不好",
        "no", "not", "never",
    })
    _CHALLENGE = frozenset({
        "凭什么", "为什么", "胡说", "错了", "离谱",
        "不同意", "不适合", "不专业", "效率低", "懒人",
        "误会", "挑剔", "借口",
    })

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
        # Single-char negation via token, multi-char via raw substring
        negation_token = sum(1 for token in tokens if token in self._NEGATION)
        negation_substr = sum(1 for w in self._NEGATION if len(w) > 1 and w in current_turn)
        negation = (negation_token + negation_substr) / max(1.0, len(tokens))
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
        imperative_ratio = _clamp(sum(1 for w in (
            "必须", "立刻", "马上", "快点", "应该", "一定要", "赶紧",
        ) if w in text) / 3.0)
        cooperative_ratio = _clamp(sum(1 for w in (
            "谢谢", "请", "我们", "一起", "抱歉", "理解", "商量", "帮忙", "方便",
        ) if w in text) / 4.0)
        self_disclosure_ratio = _clamp(sum(1 for w in (
            "我觉得", "我今天", "我担心", "我希望", "我感觉", "我想", "我最近",
            "我之前", "我试着",
        ) if w in text) / 3.0)

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


# ── Structured LLM channel extractor ────────────────────────────────────────

_CHANNEL_EXTRACTION_SYSTEM = """\
Analyze the conversation turn and output a JSON object with exactly 6 scores.
Each score is a float between 0.0 and 1.0.  Output ONLY the JSON object.

Channel definitions:
- emotional_tone: emotional valence (0=very negative/distressed, 0.5=neutral, 1=very positive/happy)
- conflict_tension: disagreement or confrontation level (0=none, 0.5=moderate friction, 1=intense)
- hidden_intent: likelihood of unstated/covert motives behind the words (0=none, 1=obviously hiding something)
- semantic_content: how much substantive/meaningful content (0=empty/ritual, 1=rich/detailed)
- topic_novelty: how new or surprising the topic is relative to everyday chat (0=mundane, 1=completely new)
- relationship_depth: intimacy/trust/closeness signaled (0=surface/transactional, 1=deep/vulnerable)

Scoring guidelines:
- emotional_tone: "我太开心了"→0.85+, "我很难过"→0.15-, "今天天气还行"→0.5, sarcasm should lower the score
- conflict_tension: polite disagreement→0.3, direct challenge→0.6+, insult→0.8+
- hidden_intent: probing questions with unclear motive→0.6+, genuine sharing→0.1-
- semantic_content: "嗯"→0.05, detailed story→0.8+
- topic_novelty: routine greeting→0.1, completely new subject→0.7+
- relationship_depth: "吃了吗"→0.1, sharing a secret→0.7+"""


class LLMChannelExtractor:
    """Extract all 6 dialogue channels in one structured LLM call.

    Uses a short system prompt to produce a JSON dict of channel scores.
    On any failure (API error, parse error, timeout) returns an empty dict
    so the caller can fall back to rule-based extraction.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-v4-flash",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.last_diagnostics: dict[str, object] = {}

    @staticmethod
    def _load_config() -> dict:
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

    def extract_all(
        self,
        current_turn: str,
        conversation_history: list[str],
    ) -> dict[str, float]:
        """Return {channel: value} for all 6 channels, or {} on failure."""
        cfg = self._load_config()
        api_key = cfg.get("api_key")
        base_url = str(cfg.get("base_url", "https://openrouter.ai/api/v1"))

        if not api_key:
            self.last_diagnostics = {"llm_extraction_error": "missing_api_key"}
            return {}

        # Build a minimal context snippet from recent history
        recent = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        history_block = "\n".join(f"- {t}" for t in recent) if recent else "(no prior turns)"

        user_content = (
            f"Recent conversation:\n{history_block}\n\n"
            f'Current message: "{current_turn}"\n\n'
            f"JSON:"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _CHANNEL_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,  # deterministic for channel extraction
            "max_tokens": 2048,  # headroom: DeepSeek v4 Flash reasoning tokens consume budget
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            import requests
        except ImportError:
            self.last_diagnostics = {"llm_extraction_error": "missing_requests"}
            return {}

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            elapsed = round(time.monotonic() - start, 3)
        except Exception as exc:
            self.last_diagnostics = {
                "llm_extraction_error": f"{type(exc).__name__}: {exc}",
            }
            return {}

        if response.status_code != 200:
            self.last_diagnostics = {
                "llm_extraction_error": f"http_{response.status_code}",
                "llm_extraction_detail": response.text[:300],
            }
            return {}

        try:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            if raw is None:
                self.last_diagnostics = {
                    "llm_extraction_error": "null_content",
                    "llm_extraction_finish_reason": str(
                        body.get("choices", [{}])[0].get("finish_reason", "unknown")
                    ),
                }
                return {}
            raw = raw.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.last_diagnostics = {
                "llm_extraction_error": f"response_parse: {exc}",
            }
            return {}

        # Extract JSON from the response (may be wrapped in ```json fences)
        if "```" in raw:
            # Extract content between first ``` and last ```
            blocks = raw.split("```")
            # Find the block after "json" marker if present
            json_str = ""
            for i, block in enumerate(blocks):
                if block.strip().startswith("json"):
                    if i + 1 < len(blocks):
                        json_str = blocks[i + 1].strip()
                        break
            if not json_str and len(blocks) >= 2:
                json_str = blocks[1].strip()
            raw = json_str or raw

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage: look for a JSON object in the text
            import re as _re
            m = _re.search(r'\{[^{}]*\}', raw.replace("\n", " "))
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    self.last_diagnostics = {
                        "llm_extraction_error": "json_parse_failed",
                        "llm_extraction_raw": raw[:500],
                    }
                    return {}
            else:
                self.last_diagnostics = {
                    "llm_extraction_error": "json_parse_failed",
                    "llm_extraction_raw": raw[:500],
                }
                return {}

        if not isinstance(parsed, dict):
            self.last_diagnostics = {"llm_extraction_error": "output_not_a_dict"}
            return {}

        expected_channels = {
            "emotional_tone", "conflict_tension", "hidden_intent",
            "semantic_content", "topic_novelty", "relationship_depth",
        }
        channels: dict[str, float] = {}
        for ch in expected_channels:
            if ch not in parsed:
                self.last_diagnostics = {
                    "llm_extraction_error": f"missing_channel_{ch}",
                }
                return {}
            try:
                val = float(parsed[ch])
            except (TypeError, ValueError):
                self.last_diagnostics = {
                    "llm_extraction_error": f"non_numeric_{ch}",
                }
                return {}
            channels[ch] = round(_clamp(val), 6)

        usage = body.get("usage", {})
        self.last_diagnostics = {
            "llm_extraction_model": self.model,
            "llm_extraction_latency_ms": int(elapsed * 1000),
            "llm_extraction_tokens": usage.get("total_tokens", 0),
        }
        return channels

