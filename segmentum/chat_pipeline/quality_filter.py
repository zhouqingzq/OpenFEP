from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re

from .parser import ChatMessage
from .session_builder import ConversationSession

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")

DEFAULT_SPAM_PATTERNS = [
    r"加微信",
    r"点击链接",
    r"返利",
    r"兼职刷单",
    r"扫码",
]
ULTRA_SHORT_TOKENS = {
    "嗯",
    "哈",
    "哈哈",
    "哦",
    "啊",
    "好",
    "ok",
    "OK",
    "收到",
    "在吗",
}


@dataclass
class FilterResult:
    message: ChatMessage
    kept: bool
    tags: list[str]
    redacted_body: str


class QualityFilter:
    def __init__(
        self,
        *,
        spam_url_threshold: int = 2,
        spam_template_patterns: list[str] | None = None,
        normalize_chinese: str | None = "simplified",
    ) -> None:
        self.spam_url_threshold = spam_url_threshold
        self.spam_template_patterns = spam_template_patterns or list(DEFAULT_SPAM_PATTERNS)
        self.normalize_chinese = normalize_chinese
        self._spam_regexes = [re.compile(p, re.IGNORECASE) for p in self.spam_template_patterns]
        self._opencc_converter = None
        if normalize_chinese in {"simplified", "traditional"}:
            try:
                from opencc import OpenCC  # type: ignore

                self._opencc_converter = OpenCC("t2s" if normalize_chinese == "simplified" else "s2t")
            except Exception:
                self._opencc_converter = None

    def _normalize(self, body: str) -> tuple[str, bool]:
        if self._opencc_converter is None:
            return body, False
        converted = self._opencc_converter.convert(body)
        return converted, converted != body

    def filter_message(self, msg: ChatMessage) -> FilterResult:
        tags: list[str] = []
        body = msg.body
        url_count = len(URL_RE.findall(body))
        is_template_spam = any(regex.search(body) for regex in self._spam_regexes)
        is_spam = url_count >= self.spam_url_threshold or is_template_spam
        if is_spam:
            tags.append("spam")

        short_candidate = body.strip()
        if len(short_candidate) <= 2 and short_candidate in ULTRA_SHORT_TOKENS:
            tags.append("ultra_short")

        redacted = URL_RE.sub("[URL]", body)
        redacted = EMAIL_RE.sub("[EMAIL]", redacted)
        redacted = PHONE_RE.sub("[PHONE]", redacted)
        if redacted != body:
            tags.append("pii_redacted")

        normalized, changed = self._normalize(redacted)
        if changed:
            tags.append("chinese_normalized")

        return FilterResult(
            message=msg,
            kept=not is_spam,
            tags=tags,
            redacted_body=normalized,
        )

    def filter_session(self, session: ConversationSession) -> ConversationSession:
        filtered_turns: list[ChatMessage] = []
        tag_counts: Counter[str] = Counter()
        for turn in session.turns:
            result = self.filter_message(turn)
            tag_counts.update(result.tags)
            if not result.kept:
                continue
            filtered_turns.append(replace(turn, body=result.redacted_body))

        metadata = dict(session.metadata)
        metadata["filter_tag_counts"] = dict(sorted(tag_counts.items()))
        metadata["valid_turn_count"] = len(filtered_turns)
        metadata["dropped"] = len(filtered_turns) < 2
        if filtered_turns:
            metadata["turn_count"] = len(filtered_turns)
            metadata["duration_seconds"] = int(
                (filtered_turns[-1].timestamp - filtered_turns[0].timestamp).total_seconds()
            )
            count_a = sum(1 for t in filtered_turns if t.sender_uid == session.uid_a)
            count_b = sum(1 for t in filtered_turns if t.sender_uid == session.uid_b)
            metadata["message_count_uid_a"] = count_a
            metadata["message_count_uid_b"] = count_b
            metadata["message_ratio_uid_a_to_uid_b"] = float(count_a) / float(count_b) if count_b else float(count_a)
        return ConversationSession(
            session_id=session.session_id,
            uid_a=session.uid_a,
            uid_b=session.uid_b,
            start_time=filtered_turns[0].timestamp if filtered_turns else session.start_time,
            end_time=filtered_turns[-1].timestamp if filtered_turns else session.end_time,
            turns=filtered_turns,
            metadata=metadata,
        )
