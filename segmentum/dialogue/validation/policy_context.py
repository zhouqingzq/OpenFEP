from __future__ import annotations

import re


_PUNCT_RE = re.compile(r"[\s.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\[\](){}'\"\-_/\\]+")

_QUESTION_MARKERS = (
    "?",
    "锛?",
    "鍚?",
    "鍝?",
    "浠€涔?",
    "鎬庝箞",
    "涓轰粈涔?",
    "鑳戒笉鑳?",
    "鍙笉鍙互",
    "鏄惁",
    "濡備綍",
    "閿?",
)
_PAYMENT_MARKERS = (
    "浠樻",
    "鏀粯",
    "杞处",
    "鏀舵",
    "閲戦",
    "璐﹀彿",
    "璐︽埛",
    "閾惰鍗?",
    "浜岀淮鐮?",
    "鍙戠エ",
    "璁㈠崟",
)
_TASK_MARKERS = (
    "澶勭悊",
    "瀹夋帓",
    "纭",
    "鑱旂郴",
    "甯綘",
    "鎴戝厛",
    "鎼炲畾",
    "鏌ヤ竴涓?",
    "浠诲姟",
    "涓嬩竴姝?",
    "杈涜嫤",
    "鏀跺埌",
)
_AFFILIATIVE_MARKERS = (
    "鍝堝搱",
    "鍛靛懙",
    "lol",
    "haha",
    "233",
    "馃",
    "[",
)
_CONFLICT_MARKERS = (
    "涓嶈",
    "涓嶅悓鎰?",
    "涓嶈鍚?",
    "涓嶈兘",
    "鍒户缁?",
    "鐢熸皵",
    "闅惧彈",
)


def _compact(text: str) -> str:
    return _PUNCT_RE.sub("", str(text).strip().lower())


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker and marker in text for marker in markers)


def dialogue_policy_context_bucket(text: str) -> str:
    """Train/decision-safe bucket built only from the partner turn."""
    normalized = str(text).strip()
    if not normalized:
        return "ctx:empty"
    compact = _compact(normalized)
    if len(compact) <= 2:
        return "ctx:partner_low_info"
    if _has_any(normalized, _QUESTION_MARKERS):
        return "ctx:partner_question"
    if _has_any(normalized, _PAYMENT_MARKERS):
        return "ctx:payment_or_account"
    if _has_any(normalized, _TASK_MARKERS):
        return "ctx:task_coordination"
    if _has_any(normalized.lower(), _AFFILIATIVE_MARKERS):
        return "ctx:affiliative"
    if _has_any(normalized, _CONFLICT_MARKERS):
        return "ctx:conflict_or_boundary"
    if len(compact) <= 8:
        return "ctx:partner_short"
    if len(compact) >= 48:
        return "ctx:partner_long"
    return "ctx:partner_statement"


def dialogue_partner_policy_context_bucket(text: str, partner_uid: object) -> str:
    """Optional finer bucket keyed by current partner id plus safe text bucket."""
    base_bucket = dialogue_policy_context_bucket(text)
    partner_key = str(partner_uid).strip()
    if not partner_key:
        return base_bucket
    return f"{base_bucket}|partner:{partner_key}"
