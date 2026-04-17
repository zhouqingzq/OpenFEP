"""Fixed labeled samples for M5.4 dialogue-act classifier gates.

The smoke set is kept for small unit checks.  The gate set is intentionally
larger and phrased independently from the classifier's hand-written examples;
the hard behavioral metric is enabled only when this gate passes.
"""

from __future__ import annotations


SMOKE_CLASSIFIER_EVAL_SAMPLES: list[dict[str, str]] = [
    {"text": "Can you explain this?", "label_11": "ask_question", "label_3": "explore"},
    {"text": "I will add concrete details now.", "label_11": "elaborate", "label_3": "exploit"},
    {"text": "ok", "label_11": "minimal_response", "label_3": "escape"},
]


CLASSIFIER_GATE_EVAL_SAMPLES: list[dict[str, str]] = [
    {"text": "Can you compare the two options?", "label_11": "ask_question", "label_3": "explore"},
    {"text": "What makes this riskier?", "label_11": "ask_question", "label_3": "explore"},
    {"text": "By the way, let us talk about a new topic.", "label_11": "introduce_topic", "label_3": "explore"},
    {"text": "I think this option is safer.", "label_11": "share_opinion", "label_3": "explore"},
    {"text": "In my view the budget risk matters most.", "label_11": "share_opinion", "label_3": "explore"},
    {"text": "Let me expand with more detail.", "label_11": "elaborate", "label_3": "exploit"},
    {"text": "Here is the supporting context in more detail.", "label_11": "elaborate", "label_3": "exploit"},
    {"text": "I agree, that makes sense.", "label_11": "agree", "label_3": "exploit"},
    {"text": "I understand; that sounds hard.", "label_11": "empathize", "label_3": "exploit"},
    {"text": "hh that is funnier than expected", "label_11": "joke", "label_3": "exploit"},
    {"text": "I disagree with that conclusion.", "label_11": "disagree", "label_3": "escape"},
    {"text": "Not now, let us move on.", "label_11": "deflect", "label_3": "escape"},
    {"text": "ok", "label_11": "minimal_response", "label_3": "escape"},
    {"text": "Talk later, I have to go.", "label_11": "disengage", "label_3": "escape"},
    {"text": "Skip this for now; another time.", "label_11": "deflect", "label_3": "escape"},
]


DEFAULT_CLASSIFIER_EVAL_SAMPLES = SMOKE_CLASSIFIER_EVAL_SAMPLES

# Formal M5.4 runs must provide independent human-labeled Chinese train/gate
# files via scripts/run_m54_validation.py.  Keeping these defaults empty makes
# missing labels fail formal acceptance instead of silently reusing toy samples.
DEFAULT_CLASSIFIER_TRAIN_SAMPLES: list[dict[str, str]] = []
DEFAULT_CLASSIFIER_GATE_SAMPLES: list[dict[str, str]] = []
