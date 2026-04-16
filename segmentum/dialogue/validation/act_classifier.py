from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from ..actions import DIALOGUE_ACTION_STRATEGY_MAP


@dataclass(slots=True)
class ActionPrediction:
    label_11: str
    label_3: str
    confidence: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ask_question": ("吗", "?", "？", "为什么", "怎么", "如何", "哪", "是否", "能不能", "可不可以"),
    "introduce_topic": ("换个话题", "聊聊", "说到", "顺便", "另外", "我们谈谈", "新话题"),
    "share_opinion": ("我觉得", "我认为", "我的看法", "在我看来", "我倾向于"),
    "elaborate": ("具体说", "展开", "补充", "进一步", "详细", "换句话说"),
    "agree": ("同意", "没错", "是的", "对", "有道理", "赞同"),
    "empathize": ("理解", "辛苦", "不容易", "心疼", "抱抱", "感受"),
    "joke": ("哈哈", "hh", "233", "玩笑", "开个玩笑", "笑死"),
    "disagree": ("不同意", "不太对", "错", "反对", "不行", "并非"),
    "deflect": ("先不聊", "先不说", "稍后再说", "换个说法", "先这样"),
    "minimal_response": ("嗯", "哦", "好", "收到", "知道了", "行"),
    "disengage": ("先到这", "先这样", "我要走了", "不聊了", "改天聊", "先停"),
}

_ASCII_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ask_question": ("why", "what", "how", "can you", "could you", "would you", "explain", "compare"),
    "introduce_topic": ("new topic", "another topic", "switch topic", "talk about", "by the way"),
    "share_opinion": ("i think", "i believe", "my view", "in my view", "i would say", "i prefer"),
    "elaborate": ("add detail", "expand", "more detail", "supporting context", "unpack", "clarify"),
    "agree": ("agree", "exactly", "makes sense", "you are right", "yes"),
    "empathize": ("understand", "that sounds hard", "sorry", "that hurts", "i hear you"),
    "joke": ("haha", "lol", "funny", "joke", "hh", "233"),
    "disagree": ("disagree", "i object", "not right", "i do not buy", "no way"),
    "deflect": ("not now", "skip this", "let us move on", "another time", "park this"),
    "minimal_response": ("ok", "fine", "noted", "sure", "got it"),
    "disengage": ("bye", "goodbye", "talk later", "i have to go", "stop here"),
}

for _action_name, _extra_keywords in _ASCII_ACTION_KEYWORDS.items():
    _ACTION_KEYWORDS[_action_name] = _ACTION_KEYWORDS.get(_action_name, ()) + _extra_keywords


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _score_action(text: str, action: str) -> float:
    normalized = text.lower().strip()
    if not normalized:
        return 0.0
    hits = 0.0
    for kw in _ACTION_KEYWORDS.get(action, ()):
        if kw in normalized:
            hits += 1.0
    if action == "ask_question":
        if "?" in normalized or "？" in normalized:
            hits += 1.5
    if action == "joke":
        if normalized.count("哈") >= 2:
            hits += 1.0
    if action == "minimal_response" and len(_tokenize(normalized)) <= 2:
        hits += 1.0
    if action == "disengage" and any(token in normalized for token in ("拜拜", "回头", "先撤")):
        hits += 1.0
    if action == "agree" and any(token in normalized for token in ("确实", "没毛病", "赞")):
        hits += 1.0
    if action == "disagree" and any(token in normalized for token in ("不是", "别", "不该")):
        hits += 0.8
    return hits


def _fallback_action(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "minimal_response"
    tokens = _tokenize(normalized)
    if "?" in normalized or "？" in normalized:
        return "ask_question"
    if len(tokens) <= 2:
        return "minimal_response"
    if any(item in normalized for item in ("哈哈", "玩笑")):
        return "joke"
    if any(item in normalized for item in ("不同意", "不行", "错")):
        return "disagree"
    if any(item in normalized for item in ("理解", "辛苦", "不容易")):
        return "empathize"
    if any(item in normalized for item in ("先到这", "不聊了", "改天")):
        return "disengage"
    return "elaborate"


def _safe_strategy(label_11: str) -> str:
    return DIALOGUE_ACTION_STRATEGY_MAP.get(label_11, "explore")


class DialogueActClassifier:
    """Deterministic rule+keyword classifier for 11 actions and 3 strategies."""

    def predict(self, text: str, *, context: Mapping[str, object] | None = None) -> ActionPrediction:
        del context
        scores: dict[str, float] = {}
        for action in DIALOGUE_ACTION_STRATEGY_MAP:
            scores[action] = _score_action(text, action)
        best_action = max(scores, key=lambda name: (scores[name], name))
        best_score = float(scores[best_action])
        if best_score <= 0.0:
            best_action = _fallback_action(text)
            confidence = 0.40
        else:
            ranked = sorted(scores.values(), reverse=True)
            second = ranked[1] if len(ranked) > 1 else 0.0
            margin = max(0.0, best_score - second)
            confidence = min(0.99, 0.55 + (margin * 0.15) + (best_score * 0.05))
        return ActionPrediction(
            label_11=best_action,
            label_3=_safe_strategy(best_action),
            confidence=round(float(confidence), 6),
        )

    def predict_batch(self, texts: list[str]) -> list[ActionPrediction]:
        return [self.predict(item) for item in texts]


def _macro_f1(true_labels: list[str], pred_labels: list[str], labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall <= 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return sum(f1_values) / float(len(f1_values)) if f1_values else 0.0


def validate_act_classifier(
    samples: list[dict[str, str]],
    *,
    min_macro_f1_3class: float = 0.70,
) -> dict[str, object]:
    classifier = DialogueActClassifier()
    if not samples:
        return {
            "sample_count": 0,
            "macro_f1_3class": 0.0,
            "macro_f1_11class": 0.0,
            "min_macro_f1_3class": float(min_macro_f1_3class),
            "passed_3class_gate": False,
            "behavioral_hard_metric_enabled": False,
            "notes": "no samples provided",
        }
    true_11: list[str] = []
    true_3: list[str] = []
    pred_11: list[str] = []
    pred_3: list[str] = []
    for sample in samples:
        text = str(sample.get("text", ""))
        pred = classifier.predict(text)
        gold_11 = str(sample.get("label_11", "")).strip()
        if not gold_11:
            gold_11 = str(sample.get("label", "")).strip()
        if not gold_11:
            continue
        gold_3 = str(sample.get("label_3", "")).strip() or _safe_strategy(gold_11)
        true_11.append(gold_11)
        true_3.append(gold_3)
        pred_11.append(pred.label_11)
        pred_3.append(pred.label_3)

    labels_11 = sorted(set(true_11) | set(pred_11))
    labels_3 = sorted(set(true_3) | set(pred_3))
    macro_f1_3 = _macro_f1(true_3, pred_3, labels_3)
    macro_f1_11 = _macro_f1(true_11, pred_11, labels_11)
    passed = macro_f1_3 >= float(min_macro_f1_3class)
    return {
        "sample_count": int(len(true_11)),
        "macro_f1_3class": round(float(macro_f1_3), 6),
        "macro_f1_11class": round(float(macro_f1_11), 6),
        "min_macro_f1_3class": float(min_macro_f1_3class),
        "passed_3class_gate": bool(passed),
        "behavioral_hard_metric_enabled": bool(passed),
        "degradation_required": bool(not passed),
    }

