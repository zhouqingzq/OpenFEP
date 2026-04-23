from __future__ import annotations

import re


EXPLOIT_REPLY_FUNCTIONS = frozenset(
    {
        "transactional_ack",
        "task_coordination",
        "payment_or_account_info",
        "affiliative_humor",
        "emoji_affiliation",
        "substantive_response",
    }
)
EXPLORE_REPLY_FUNCTIONS = frozenset({"information_request"})
ESCAPE_REPLY_FUNCTIONS = frozenset(
    {
        "refusal_boundary",
        "defer_or_stop",
        "low_info_other",
        "empty",
    }
)


_COOPERATIVE_ACKS = frozenset(
    {
        "ok",
        "okay",
        "sure",
        "fine",
        "noted",
        "got it",
        "gotit",
        "yes",
        "yep",
        "收到",
        "知道",
        "知道了",
        "好",
        "好的",
        "嗯",
        "嗯嗯",
        "可以",
        "行",
        "对",
        "谢谢",
        "感谢",
        # Common mojibake variants present in existing fixtures/artifacts.
        "鏀跺埌",
        "鐭ラ亾",
        "濂",
        "濂界殑",
        "鍡",
        "鍙互",
        "璎濊瑵",
        "璋㈣阿",
        "鎰熻瑵",
        "鎰熻阿",
    }
)

_QUESTION_MARKERS = (
    "?",
    "？",
    "吗",
    "哪",
    "什么",
    "怎么",
    "为什么",
    "能不能",
    "可不可以",
    "是否",
    "如何",
    "锛?",
    "鍚?",
    "鍝?",
    "鎬庝箞",
    "鎬庨杭",
)
_PAYMENT_MARKERS = (
    "付款",
    "支付",
    "转账",
    "收款",
    "金额",
    "账号",
    "账户",
    "银行卡",
    "二维码",
    "发票",
    "订单",
    "尾款",
    "定金",
    "璐﹀彿",
    "閼?",
    "閽?",
    "鍖?",
    "姹?",
)
_TASK_MARKERS = (
    "处理",
    "安排",
    "确认",
    "联系",
    "帮你",
    "我先",
    "搞定",
    "查一下",
    "任务",
    "下一步",
    "辛苦",
    "收到",
    "有需要",
    "澶勭悊",
    "铏曠悊",
    "纭",
    "鑱旂郴",
    "鑱怠",
    "甯綘",
    "骞綘",
    "鎴戝厛",
    "鎼跺埌",
    "鎶㈠埌",
    "浠诲姟",
    "杈涜嫤",
)
_HUMOR_MARKERS = (
    "哈哈",
    "呵呵",
    "笑死",
    "开玩笑",
    "lol",
    "haha",
    "hh",
    "233",
    "鍝堝搱",
    "鍛靛懙",
)
_EMOJI_MARKERS = (
    "😀",
    "😄",
    "😆",
    "😂",
    "🤣",
    "🙂",
    "😉",
    "[哈哈]",
    "[爱心]",
    "[抱拳]",
    "[OK]",
    "[鍝堝搱]",
    "[鎰涘績]",
    "[鐖卞績]",
)
_REFUSAL_MARKERS = (
    "不建议",
    "不同意",
    "不认同",
    "不能",
    "不太",
    "先别",
    "别继续",
    "先放",
    "暂时不",
    "到这里",
    "先停",
    "不用了",
    "算了",
    "涓嶅缓璁",
    "涓嶅缓璀",
    "涓嶅お",
    "涓嶈兘",
    "鍏堝垾",
    "鍏堝埆",
    "鍏堟斁",
    "鏆傛椂涓",
    "鍒拌繖閲",
    "鍏堝仠",
)
_PUNCT_RE = re.compile(r"[\s.,!?;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a\[\](){}'\"\-_/\\]+")


def compact_reply(text: str) -> str:
    return _PUNCT_RE.sub("", str(text).strip().lower())


def is_cooperative_ack(text: str) -> bool:
    lowered = " ".join(str(text).strip().lower().split())
    compact = compact_reply(text)
    return lowered in _COOPERATIVE_ACKS or compact in _COOPERATIVE_ACKS


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker and marker in text for marker in markers)


def classify_reply_function(text: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        return "empty"
    lowered = normalized.lower()
    compact = compact_reply(normalized)
    if _has_any(normalized, _QUESTION_MARKERS):
        return "information_request"
    if _has_any(normalized, _REFUSAL_MARKERS):
        return "defer_or_stop" if any(m in normalized for m in ("到这里", "先停", "鍒拌繖閲", "鍏堝仠")) else "refusal_boundary"
    if is_cooperative_ack(normalized):
        return "transactional_ack"
    if _has_any(normalized, _PAYMENT_MARKERS):
        return "payment_or_account_info"
    if _has_any(normalized, _TASK_MARKERS):
        return "task_coordination"
    if _has_any(lowered, _HUMOR_MARKERS):
        return "affiliative_humor"
    if _has_any(normalized, _EMOJI_MARKERS):
        return "emoji_affiliation"
    if len(compact) <= 2 or (len(compact) <= 8 and compact.isdigit()):
        return "low_info_other"
    return "substantive_response"


def reply_function_strategy(reply_function: str) -> str:
    if reply_function in EXPLORE_REPLY_FUNCTIONS:
        return "explore"
    if reply_function in EXPLOIT_REPLY_FUNCTIONS:
        return "exploit"
    return "escape"


def representative_action_for_reply_function(reply_function: str) -> str:
    if reply_function == "information_request":
        return "ask_question"
    if reply_function in {"transactional_ack", "emoji_affiliation"}:
        return "agree"
    if reply_function == "affiliative_humor":
        return "joke"
    if reply_function in {"task_coordination", "payment_or_account_info", "substantive_response"}:
        return "elaborate"
    if reply_function == "defer_or_stop":
        return "disengage"
    if reply_function == "refusal_boundary":
        return "deflect"
    return "minimal_response"


def behavioral_policy_weight_for_reply_function(reply_function: str) -> float:
    if reply_function == "empty":
        return 0.0
    if reply_function == "low_info_other":
        return 0.55
    if reply_function == "transactional_ack":
        return 0.80
    if reply_function in {
        "task_coordination",
        "payment_or_account_info",
        "affiliative_humor",
        "emoji_affiliation",
        "information_request",
        "refusal_boundary",
        "defer_or_stop",
    }:
        return 1.0
    return 0.85
