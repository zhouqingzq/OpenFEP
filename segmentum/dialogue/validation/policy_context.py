from __future__ import annotations

import re
from typing import Collection


_PUNCT_RE = re.compile(r"[\s.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\[\](){}'\"\-_/\\]+")

_QUESTION_MARKERS = (
    "?",
    "？",
    "why",
    "what",
    "which",
    "how",
    "when",
    "吗",
    "呢",
    "么",
)
_PAYMENT_MARKERS = (
    "payment",
    "pay",
    "paid",
    "account",
    "invoice",
    "bill",
    "transfer",
    "bank",
    "汇款",
    "转账",
    "账户",
    "账号",
    "付款",
    "打款",
)
_TASK_MARKERS = (
    "schedule",
    "meeting",
    "coordinate",
    "arrange",
    "plan",
    "deadline",
    "follow up",
    "sync",
    "排期",
    "安排",
    "同步",
    "跟进",
    "确认流程",
    "处理一下",
)
_AFFILIATIVE_MARKERS = (
    "lol",
    "haha",
    "233",
    "thanks",
    "thank you",
    "感谢",
    "謝謝",
    "开心",
    "[",
)
_CONFLICT_MARKERS = (
    "stop",
    "leave me",
    "dont",
    "don't",
    "no more",
    "别",
    "不要",
    "算了",
    "闭嘴",
    "烦",
)
_TEMPORAL_MARKERS = (
    "today",
    "tonight",
    "tomorrow",
    "tmr",
    "later",
    "soon",
    "next week",
    "next month",
    "before",
    "after",
    "wait",
    "today",
    "now",
    "今天",
    "今晚",
    "明天",
    "后天",
    "等下",
    "等会",
    "待会",
    "一会",
    "晚点",
    "稍后",
    "下周",
    "下个月",
)
_SHORT_CONFIRMATION_MARKERS = (
    "ok",
    "okay",
    "sure",
    "yes",
    "yep",
    "got it",
    "kk",
    "fine",
    "好",
    "好的",
    "行",
    "可以",
    "收到",
    "知道了",
    "明白",
    "嗯",
)
_TOPICISH_MARKERS = (
    "about",
    "regarding",
    "topic",
    "project",
    "plan",
    "idea",
    "details",
    "situation",
    "issue",
    "question",
    "关于",
    "这个",
    "那个",
    "项目",
    "计划",
    "想法",
    "细节",
    "情况",
    "问题",
    "内容",
)


def _compact(text: str) -> str:
    return _PUNCT_RE.sub("", str(text).strip().lower())


def _has_any(text: str, compact: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text).strip().lower()
    return any(
        marker
        and (
            marker.lower() in lowered
            or marker.lower().replace(" ", "") in compact
        )
        for marker in markers
    )


def dialogue_policy_context_bucket(text: str) -> str:
    """Train/decision-safe bucket built only from the partner turn."""
    normalized = str(text).strip()
    if not normalized:
        return "ctx:empty"
    compact = _compact(normalized)
    if _has_any(normalized, compact, _QUESTION_MARKERS):
        return "ctx:partner_question"
    if _has_any(normalized, compact, _PAYMENT_MARKERS):
        return "ctx:payment_or_account"
    if _has_any(normalized, compact, _TASK_MARKERS):
        return "ctx:task_coordination"
    if _has_any(normalized, compact, _AFFILIATIVE_MARKERS):
        return "ctx:affiliative"
    if _has_any(normalized, compact, _CONFLICT_MARKERS):
        return "ctx:conflict_or_boundary"
    if len(compact) >= 48:
        return "ctx:partner_long"
    if len(compact) <= 4:
        if _has_any(normalized, compact, _TEMPORAL_MARKERS):
            return "ctx:partner_low_info_temporal"
        if _has_any(normalized, compact, _SHORT_CONFIRMATION_MARKERS):
            return "ctx:partner_low_info_ack"
        return "ctx:partner_low_info_ack" if len(compact) <= 2 else "ctx:partner_short_other"
    if len(compact) <= 8:
        if _has_any(normalized, compact, _TEMPORAL_MARKERS):
            return "ctx:partner_short_temporal"
        if _has_any(normalized, compact, _SHORT_CONFIRMATION_MARKERS):
            return "ctx:partner_short_confirmation"
        return "ctx:partner_short_other"
    if _has_any(normalized, compact, _TEMPORAL_MARKERS):
        return "ctx:partner_statement_temporal"
    if _has_any(normalized, compact, _TOPICISH_MARKERS):
        return "ctx:partner_statement_topicish"
    return "ctx:partner_statement_other"


def dialogue_partner_policy_context_bucket(text: str, partner_uid: object) -> str:
    """Optional finer bucket keyed by current partner id plus safe text bucket."""
    base_bucket = dialogue_policy_context_bucket(text)
    partner_key = str(partner_uid).strip()
    if not partner_key:
        return base_bucket
    return f"{base_bucket}|partner:{partner_key}"


def dialogue_policy_context_candidates(text: str, partner_uid: object | None = None) -> tuple[str, ...]:
    """Ordered policy-context lookup candidates: partner-conditioned first, then global."""
    base_bucket = dialogue_policy_context_bucket(text)
    partner_key = str(partner_uid).strip()
    if not partner_key:
        return (base_bucket,)
    return (f"{base_bucket}|partner:{partner_key}", base_bucket)


def resolve_dialogue_policy_context_bucket(
    text: str,
    partner_uid: object | None = None,
    *,
    available_buckets: Collection[object] | None = None,
) -> str:
    """Choose the preferred bucket, falling back to the global safe bucket when unsupported."""
    candidates = dialogue_policy_context_candidates(text, partner_uid)
    if available_buckets is None:
        return candidates[0]
    available = {str(item) for item in available_buckets}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return candidates[-1]
