"""Fixed labeled samples for `validate_act_classifier` (gate on 3-class macro-F1).

Held-out style: mix of clear examples and a few harder lines so the gate is not trivially
passed by a handful of hand-picked strings alone.
"""

from __future__ import annotations

# Default eval set used by run_validation pipeline (>= 11 rows; includes edge-ish lines).
DEFAULT_CLASSIFIER_EVAL_SAMPLES: list[dict[str, str]] = [
    {"text": "你能具体说说吗？", "label_11": "ask_question", "label_3": "explore"},
    {"text": "这是不是意味着要延期？", "label_11": "ask_question", "label_3": "explore"},
    {"text": "我同意你的看法。", "label_11": "agree", "label_3": "exploit"},
    {"text": "嗯，有道理。", "label_11": "agree", "label_3": "exploit"},
    {"text": "我能理解你的感受。", "label_11": "empathize", "label_3": "exploit"},
    {"text": "哈哈这个太好笑了", "label_11": "joke", "label_3": "exploit"},
    {"text": "这一点我不同意。", "label_11": "disagree", "label_3": "escape"},
    {"text": "我不太认可这个结论。", "label_11": "disagree", "label_3": "escape"},
    {"text": "我们先不聊这个。", "label_11": "deflect", "label_3": "escape"},
    {"text": "嗯。", "label_11": "minimal_response", "label_3": "escape"},
    {"text": "我先到这里，改天聊。", "label_11": "disengage", "label_3": "escape"},
    {"text": "我补充一下细节。", "label_11": "elaborate", "label_3": "exploit"},
    {"text": "我想提一个新话题。", "label_11": "introduce_topic", "label_3": "explore"},
    {"text": "在我看来这个方案更稳。", "label_11": "share_opinion", "label_3": "explore"},
    {"text": "换个角度，预算和风险怎么平衡？", "label_11": "ask_question", "label_3": "explore"},
    {"text": "收到，我这边先记下来。", "label_11": "minimal_response", "label_3": "escape"},
]
