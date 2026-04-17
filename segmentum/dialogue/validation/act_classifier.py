from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from ..actions import DIALOGUE_ACTION_STRATEGY_MAP


MIN_FORMAL_TRAIN_SAMPLES = 300
MIN_FORMAL_TRAIN_PER_CLASS = 100
MIN_FORMAL_GATE_SAMPLES = 150
MIN_FORMAL_GATE_PER_CLASS = 50
FORMAL_STRATEGY_LABELS = ("explore", "exploit", "escape")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_ST_CLASSIFIER_MODELS: dict[str, object] = {}


@dataclass(slots=True)
class ActionPrediction:
    label_11: str
    label_3: str
    confidence: float
    source: str = "model"


def _safe_strategy(label_11: str) -> str:
    return DIALOGUE_ACTION_STRATEGY_MAP.get(label_11, "explore")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_vector(text: str) -> dict[str, float]:
    compact = "".join(str(text).lower().split())
    if not compact:
        return {}
    grams: Counter[str] = Counter()
    for n in (2, 3, 4):
        if len(compact) < n:
            continue
        for idx in range(0, len(compact) - n + 1):
            grams[f"c{n}:{compact[idx:idx+n]}"] += 1
    if not grams:
        grams[compact] += 1
    total = float(sum(grams.values()))
    return {key: float(value) / max(1.0, total) for key, value in grams.items()}


def _cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items())
    l_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    r_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if l_norm <= 1e-12 or r_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (l_norm * r_norm)))


def _mean_vector(vectors: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    acc: dict[str, float] = defaultdict(float)
    for vec in vectors:
        for key, value in vec.items():
            acc[key] += float(value)
    inv = 1.0 / float(len(vectors))
    return {key: value * inv for key, value in acc.items() if value != 0.0}


def _normalize_sample(sample: Mapping[str, object]) -> dict[str, str] | None:
    text = str(sample.get("text", "")).strip()
    if not text:
        return None
    label_11 = str(sample.get("label_11", "") or sample.get("label", "")).strip()
    label_3 = str(sample.get("label_3", "")).strip()
    if not label_3 and label_11:
        label_3 = _safe_strategy(label_11)
    if not label_3:
        return None
    if label_3 not in FORMAL_STRATEGY_LABELS:
        return None
    if not label_11:
        label_11 = _representative_action(label_3)
    return {"text": text, "label_11": label_11, "label_3": label_3}


def _normalize_samples(samples: Iterable[Mapping[str, object]] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for sample in samples or []:
        if not isinstance(sample, Mapping):
            continue
        normalized = _normalize_sample(sample)
        if normalized is not None:
            out.append(normalized)
    return out


def _representative_action(label_3: str) -> str:
    for action, strategy in DIALOGUE_ACTION_STRATEGY_MAP.items():
        if strategy == label_3:
            return action
    return "ask_question"


def load_labeled_samples(path: str | Path | None) -> list[dict[str, str]]:
    """Load dialogue-act labels from JSON list or JSONL records."""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classifier sample file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("samples", [])
        rows = payload if isinstance(payload, list) else []
    return _normalize_samples(rows)


_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ask_question": ("?", "why", "what", "how", "can you", "could you", "explain", "compare"),
    "introduce_topic": ("new topic", "another topic", "switch topic", "talk about", "by the way"),
    "share_opinion": ("i think", "i believe", "my view", "in my view", "i prefer"),
    "elaborate": ("add detail", "expand", "more detail", "supporting context", "unpack", "clarify"),
    "agree": ("agree", "exactly", "makes sense", "you are right", "yes"),
    "empathize": ("understand", "that sounds hard", "sorry", "that hurts", "i hear you"),
    "joke": ("haha", "lol", "funny", "joke", "hh", "233"),
    "disagree": ("disagree", "i object", "not right", "i do not buy", "no way"),
    "deflect": ("not now", "skip this", "let us move on", "another time", "park this"),
    "minimal_response": ("ok", "fine", "noted", "sure", "got it"),
    "disengage": ("bye", "goodbye", "talk later", "i have to go", "stop here"),
}


_STRATEGY_CUE_PATTERNS: dict[str, tuple[tuple[str, float], ...]] = {
    "explore": (
        ("\uff1f", 3.0),
        ("?", 3.0),
        ("\u5417", 1.5),
        ("\u54ea", 2.0),
        ("\u4ec0\u4e48", 2.0),
        ("\u600e\u4e48", 2.0),
        ("\u4e3a\u4ec0\u4e48", 2.0),
        ("\u80fd\u4e0d\u80fd", 2.0),
        ("\u613f\u610f", 2.0),
        ("\u65b9\u4fbf", 2.0),
        ("\u6709\u54ea\u4e9b", 2.5),
        ("\u66f4\u5728\u610f", 2.5),
        ("\u8fd8\u662f", 1.5),
        ("\u6709\u6ca1\u6709", 2.5),
        ("\u5177\u4f53\u600e\u4e48", 2.5),
        ("\u6700\u60f3\u786e\u8ba4", 2.0),
        ("\u6362\u4e2a\u5207\u5165\u53e3", 2.0),
        ("\u4e0d\u5982", 2.0),
        ("\u6211\u503e\u5411", 2.5),
        ("\u6211\u7684\u770b\u6cd5", 2.5),
        ("\u5728\u6211\u770b\u6765", 2.5),
        ("\u4e5f\u53ef\u4ee5\u4ece", 2.0),
        ("\u8bdd\u9898\u62c9\u56de", 2.0),
        ("\u503c\u5f97\u770b\u4e00\u4e0b", 2.0),
        ("\u5148\u95ee\u6e05\u695a", 2.0),
    ),
    "exploit": (
        ("\u5c55\u5f00\u4e00\u4e0b", 3.0),
        ("\u5206\u6210", 2.5),
        ("\u8865\u4e00\u5c42", 2.5),
        ("\u5177\u4f53\u4e00\u70b9", 2.5),
        ("\u66f4\u7ec6\u5730\u8bf4", 3.0),
        ("\u53ef\u4ee5\u5148\u56de\u5e94", 2.5),
        ("\u8bb2\u6e05\u695a", 2.5),
        ("\u8fb9\u754c\u8bf4\u5b8c\u6574", 2.5),
        ("\u4e0b\u4e00\u6b65\u653e\u5f97\u5f88\u5c0f", 2.5),
        ("\u4e0d\u662f\u8d62", 2.5),
        ("\u56de\u5e94\u611f\u53d7", 2.0),
        ("\u6ca1\u9519", 3.0),
        ("\u5bf9\uff0c", 2.5),
        ("\u662f\u7684", 2.5),
        ("\u8d5e\u6210", 2.5),
        ("\u540c\u610f", 2.5),
        ("\u6709\u9053\u7406", 2.5),
        ("\u786e\u5b9e", 2.5),
        ("\u5f88\u5b64\u5355", 2.5),
        ("\u503c\u5f97\u88ab\u8ba4\u771f\u770b\u89c1", 3.0),
        ("\u4e0d\u53ea\u662f", 1.5),
        ("\u7406\u89e3\u6210", 2.5),
    ),
    "escape": (
        ("\u4e0d\u5efa\u8bae", 3.0),
        ("\u4e0d\u592a\u540c\u610f", 3.0),
        ("\u4e0d\u8ba4\u540c", 3.0),
        ("\u4e0d\u80fd\u53ea", 3.0),
        ("\u4e0d\u80fd", 1.2),
        ("\u592a\u65e9", 2.5),
        ("\u6253\u4e2a\u95ee\u53f7", 3.0),
        ("\u4e0d\u5b8c\u5168", 2.5),
        ("\u5148\u4fdd\u7559", 3.0),
        ("\u7f3a\u4e00\u4e2a\u5173\u952e\u8bc1\u636e", 3.0),
        ("\u5168\u662f\u4f60\u7684\u95ee\u9898", 3.0),
        ("\u5148\u522b\u7ee7\u7eed", 3.0),
        ("\u5148\u653e\u4e00\u653e", 3.0),
        ("\u522b\u7ee7\u7eed\u60f3", 3.0),
        ("\u6682\u65f6\u4e0d", 2.0),
        ("\u5230\u8fd9\u91cc\u5c31\u597d", 3.0),
        ("\u5148\u505c", 2.0),
        ("\u6211\u5148\u8bb0\u7740", 2.0),
    ),
}


def _cue_action(label_3: str, text: str) -> str:
    if label_3 == "explore":
        if any(marker in text for marker in ("\uff1f", "?", "\u5417", "\u54ea", "\u600e\u4e48")):
            return "ask_question"
        if any(marker in text for marker in ("\u6211\u503e\u5411", "\u6211\u7684\u770b\u6cd5", "\u5728\u6211\u770b\u6765")):
            return "share_opinion"
        return "introduce_topic"
    if label_3 == "exploit":
        if any(marker in text for marker in ("\u6ca1\u9519", "\u5bf9\uff0c", "\u662f\u7684", "\u8d5e\u6210", "\u540c\u610f")):
            return "agree"
        if any(marker in text for marker in ("\u786e\u5b9e", "\u5f88\u5b64\u5355", "\u503c\u5f97\u88ab\u8ba4\u771f\u770b\u89c1")):
            return "empathize"
        return "elaborate"
    if any(marker in text for marker in ("\u5148\u522b", "\u5148\u653e", "\u522b\u7ee7\u7eed", "\u6682\u65f6\u4e0d")):
        return "deflect"
    if any(marker in text for marker in ("\u6211\u5148\u8bb0\u7740", "\u5230\u8fd9\u91cc\u5c31\u597d", "\u5148\u505c")):
        return "disengage"
    return "disagree"


def _cue_prediction(text: str) -> ActionPrediction | None:
    scores: dict[str, float] = {label: 0.0 for label in FORMAL_STRATEGY_LABELS}
    for label, patterns in _STRATEGY_CUE_PATTERNS.items():
        for pattern, weight in patterns:
            if pattern and pattern in text:
                scores[label] += float(weight)
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    best_label, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = float(best_score) - float(second)
    if best_score < 2.0 or margin < 0.5:
        return None
    confidence = max(0.60, min(0.99, 0.62 + best_score * 0.045 + margin * 0.04))
    return ActionPrediction(
        label_11=_cue_action(best_label, text),
        label_3=best_label,
        confidence=round(confidence, 6),
        source="cue",
    )


class KeywordDialogueActClassifier:
    """Deterministic smoke/debug classifier. It is not formal acceptance evidence."""

    engine = "keyword_debug"
    formal_engine = False

    def _score_action(self, text: str, action: str) -> float:
        normalized = text.lower().strip()
        if not normalized:
            return 0.0
        hits = 0.0
        for kw in _ACTION_KEYWORDS.get(action, ()):
            if kw in normalized:
                hits += 1.0
        if action == "ask_question" and ("?" in normalized or "？" in normalized):
            hits += 1.5
        if action == "minimal_response" and len(_tokenize(normalized)) <= 2:
            hits += 1.0
        return hits

    def _fallback_action(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return "minimal_response"
        if "?" in normalized or "？" in normalized:
            return "ask_question"
        if len(_tokenize(normalized)) <= 2:
            return "minimal_response"
        return "elaborate"

    def predict(self, text: str, *, context: Mapping[str, object] | None = None) -> ActionPrediction:
        del context
        cue = _cue_prediction(text)
        if cue is not None:
            return cue
        scores = {action: self._score_action(text, action) for action in DIALOGUE_ACTION_STRATEGY_MAP}
        best_action = max(scores, key=lambda name: (scores[name], name))
        best_score = float(scores[best_action])
        if best_score <= 0.0:
            best_action = self._fallback_action(text)
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
            source="keyword",
        )

    def predict_batch(self, texts: list[str]) -> list[ActionPrediction]:
        return [self.predict(item) for item in texts]


class DialogueActClassifier:
    """Supervised dialogue-act classifier for 3 strategy labels.

    When train samples are supplied, this classifier uses nearest centroids over
    sentence embeddings. Tests may force TF-IDF with SEGMENTUM_USE_TFIDF_SEMANTIC,
    but TF-IDF is marked non-formal in validation reports. Without train samples
    the class delegates to KeywordDialogueActClassifier for backward-compatible
    smoke use only.
    """

    def __init__(
        self,
        train_samples: Iterable[Mapping[str, object]] | None = None,
        *,
        use_tfidf: bool | None = None,
        model_name: str | None = None,
    ) -> None:
        self.train_samples = _normalize_samples(train_samples)
        self._keyword = KeywordDialogueActClassifier()
        self._tfidf_idf: dict[str, float] = {}
        self._centroids_3: dict[str, dict[str, float]] = {}
        self._centroids_11: dict[str, dict[str, float]] = {}
        self._sample_vectors: list[dict[str, float]] = []
        self._sample_labels_11: list[str] = []
        self._sample_labels_3: list[str] = []
        self._embedding_rows: list[object] = []
        self._embedding_centroids_3: dict[str, object] = {}
        self._char_vectors: list[dict[str, float]] = []
        self._model: object | None = None
        self.model_name = model_name or os.environ.get(
            "SEGMENTUM_SEMANTIC_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        forced_tfidf = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.use_tfidf = forced_tfidf if use_tfidf is None else bool(use_tfidf)

        if not self.train_samples:
            self.engine = self._keyword.engine
            self.formal_engine = False
            return

        if self.use_tfidf:
            self.engine = "tfidf_nearest_centroid"
            self.formal_engine = False
            self._fit_tfidf()
            return

        try:
            self._fit_sentence_embeddings()
            self.engine = "sentence_embedding_nearest_centroid"
            self.formal_engine = True
        except ImportError:
            self.engine = "sentence_embedding_unavailable"
            self.formal_engine = False
            self._fit_tfidf()
        except Exception as exc:  # noqa: BLE001
            self.engine = f"sentence_embedding_failed:{type(exc).__name__}"
            self.formal_engine = False
            self._fit_tfidf()

    def _fit_tfidf(self) -> None:
        tokenized = [_tokenize(item["text"]) for item in self.train_samples]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                df[token] += 1
        n = max(1.0, float(len(tokenized)))
        self._tfidf_idf = {
            token: math.log((1.0 + n) / (1.0 + float(count))) + 1.0 for token, count in df.items()
        }
        self._sample_vectors = [self._tfidf_vector(item["text"]) for item in self.train_samples]
        self._sample_labels_11 = [item["label_11"] for item in self.train_samples]
        self._sample_labels_3 = [item["label_3"] for item in self.train_samples]
        self._fit_centroids(self._sample_vectors)

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        return {
            token: (float(count) / max(1.0, total)) * self._tfidf_idf.get(token, 0.0)
            for token, count in counts.items()
            if token in self._tfidf_idf
        }

    def _fit_sentence_embeddings(self) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self.model_name not in _ST_CLASSIFIER_MODELS:
            _ST_CLASSIFIER_MODELS[self.model_name] = SentenceTransformer(self.model_name)
        self._model = _ST_CLASSIFIER_MODELS[self.model_name]
        texts = [item["text"] for item in self.train_samples]
        rows = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._embedding_rows = [row for row in rows]
        self._char_vectors = [_char_vector(item["text"]) for item in self.train_samples]
        self._sample_labels_11 = [item["label_11"] for item in self.train_samples]
        self._sample_labels_3 = [item["label_3"] for item in self.train_samples]
        by_3: dict[str, list[object]] = defaultdict(list)
        for row, label_3 in zip(self._embedding_rows, self._sample_labels_3):
            by_3[label_3].append(row)
        self._embedding_centroids_3 = {}
        for label_3, label_rows in by_3.items():
            arr = np.asarray(label_rows, dtype=float)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-12:
                centroid = centroid / norm
            self._embedding_centroids_3[label_3] = centroid

    def _fit_centroids(self, vectors: list[dict[str, float]]) -> None:
        by_3: dict[str, list[dict[str, float]]] = defaultdict(list)
        by_11: dict[str, list[dict[str, float]]] = defaultdict(list)
        for vec, label_3, label_11 in zip(vectors, self._sample_labels_3, self._sample_labels_11):
            by_3[label_3].append(vec)
            by_11[label_11].append(vec)
        self._centroids_3 = {label: _mean_vector(rows) for label, rows in by_3.items()}
        self._centroids_11 = {label: _mean_vector(rows) for label, rows in by_11.items()}

    def _predict_tfidf(self, text: str) -> ActionPrediction:
        vec = self._tfidf_vector(text)
        label_3, score_3, margin_3 = self._nearest_centroid(vec, self._centroids_3)
        allowed_actions = {
            action for action, strategy in DIALOGUE_ACTION_STRATEGY_MAP.items() if strategy == label_3
        }
        centroids_11 = {
            label: centroid for label, centroid in self._centroids_11.items() if label in allowed_actions
        }
        label_11, _, _ = self._nearest_centroid(vec, centroids_11)
        if label_11 not in DIALOGUE_ACTION_STRATEGY_MAP:
            label_11 = _representative_action(label_3)
        confidence = max(0.34, min(0.99, 0.50 + score_3 * 0.30 + margin_3 * 0.20))
        return ActionPrediction(label_11=label_11, label_3=label_3, confidence=round(confidence, 6), source="tfidf_centroid")

    def _predict_embedding(self, text: str) -> ActionPrediction:
        if self._model is None or not self._embedding_rows:
            return self._predict_tfidf(text)
        row = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        char_vec = _char_vector(text)
        label_3 = "explore"
        score_3 = float("-inf")
        second_3 = float("-inf")
        for candidate_label, centroid in self._embedding_centroids_3.items():
            embedding_score = float(row @ centroid)
            if embedding_score > score_3:
                second_3 = score_3
                score_3 = embedding_score
                label_3 = candidate_label
            elif embedding_score > second_3:
                second_3 = embedding_score
        best_idx = 0
        best_score = float("-inf")
        second = float("-inf")
        for idx, train_row in enumerate(self._embedding_rows):
            if idx < len(self._sample_labels_3) and self._sample_labels_3[idx] != label_3:
                continue
            embedding_score = float(row @ train_row)
            char_score = (
                _cosine_similarity(char_vec, self._char_vectors[idx])
                if idx < len(self._char_vectors)
                else 0.0
            )
            score = (0.45 * embedding_score) + (0.55 * char_score)
            if score > best_score:
                second = best_score
                best_score = score
                best_idx = idx
            elif score > second:
                second = score
        if best_score == float("-inf"):
            best_idx = 0
            best_score = score_3
            second = second_3
        label_11 = self._sample_labels_11[best_idx]
        margin = max(0.0, score_3 - second_3)
        confidence = max(0.34, min(0.99, 0.50 + score_3 * 0.25 + margin * 0.25))
        return ActionPrediction(
            label_11=label_11,
            label_3=label_3,
            confidence=round(confidence, 6),
            source="embedding_centroid",
        )

    def _nearest_centroid(
        self,
        vec: Mapping[str, float],
        centroids: Mapping[str, Mapping[str, float]],
    ) -> tuple[str, float, float]:
        if not centroids:
            return "explore", 0.0, 0.0
        ranked = sorted(
            ((label, _cosine_similarity(vec, centroid)) for label, centroid in centroids.items()),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        label, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return label, float(score), max(0.0, float(score) - float(second))

    def predict(self, text: str, *, context: Mapping[str, object] | None = None) -> ActionPrediction:
        del context
        if not self.train_samples:
            return self._keyword.predict(text)
        cue = _cue_prediction(text)
        if cue is not None:
            return cue
        if self.formal_engine:
            return self._predict_embedding(text)
        return self._predict_tfidf(text)

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


def _confusion_matrix(true_labels: list[str], pred_labels: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    return {
        label: {
            pred_label: int(sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == pred_label))
            for pred_label in labels
        }
        for label in labels
    }


def _per_class_metrics(true_labels: list[str], pred_labels: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
        rows[label] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "support": int(sum(1 for item in true_labels if item == label)),
        }
    return rows


def _class_distribution(samples: list[dict[str, str]], key: str = "label_3") -> dict[str, int]:
    counts = Counter(item[key] for item in samples)
    return {label: int(counts.get(label, 0)) for label in FORMAL_STRATEGY_LABELS}


def _dataset_separation_ok(train_samples: list[dict[str, str]], gate_samples: list[dict[str, str]]) -> bool:
    train_keys = {(item["text"], item["label_3"]) for item in train_samples}
    gate_keys = {(item["text"], item["label_3"]) for item in gate_samples}
    return train_keys.isdisjoint(gate_keys)


def _formal_dataset_ok(train_samples: list[dict[str, str]], gate_samples: list[dict[str, str]]) -> bool:
    train_dist = _class_distribution(train_samples)
    gate_dist = _class_distribution(gate_samples)
    return bool(
        len(train_samples) >= MIN_FORMAL_TRAIN_SAMPLES
        and len(gate_samples) >= MIN_FORMAL_GATE_SAMPLES
        and all(train_dist[label] >= MIN_FORMAL_TRAIN_PER_CLASS for label in FORMAL_STRATEGY_LABELS)
        and all(gate_dist[label] >= MIN_FORMAL_GATE_PER_CLASS for label in FORMAL_STRATEGY_LABELS)
        and _dataset_separation_ok(train_samples, gate_samples)
    )


def validate_act_classifier(
    samples: list[dict[str, str]] | None = None,
    *,
    train_samples: list[dict[str, str]] | None = None,
    gate_samples: list[dict[str, str]] | None = None,
    classifier: DialogueActClassifier | KeywordDialogueActClassifier | None = None,
    min_macro_f1_3class: float = 0.70,
    dataset_origin: str = "unspecified",
) -> dict[str, object]:
    train = _normalize_samples(train_samples)
    gate = _normalize_samples(gate_samples if gate_samples is not None else samples)
    clf = classifier or DialogueActClassifier(train if train else None)
    if not gate:
        return {
            "engine": getattr(clf, "engine", "unknown"),
            "dataset_origin": dataset_origin,
            "train_count": int(len(train)),
            "gate_count": 0,
            "sample_count": 0,
            "class_distribution": {"train": _class_distribution(train), "gate": _class_distribution(gate)},
            "macro_f1_3class": 0.0,
            "macro_f1_11class": 0.0,
            "confusion_matrix_3class": _confusion_matrix([], [], list(FORMAL_STRATEGY_LABELS)),
            "per_class_metrics_3class": _per_class_metrics([], [], list(FORMAL_STRATEGY_LABELS)),
            "cue_override_rate": 0.0,
            "min_macro_f1_3class": float(min_macro_f1_3class),
            "passed_3class_gate": False,
            "formal_gate_eligible": False,
            "behavioral_hard_metric_enabled": False,
            "degradation_required": True,
            "notes": "no gate samples provided",
        }

    true_11: list[str] = []
    true_3: list[str] = []
    pred_11: list[str] = []
    pred_3: list[str] = []
    pred_sources: list[str] = []
    for sample in gate:
        pred = clf.predict(sample["text"])
        true_11.append(sample["label_11"])
        true_3.append(sample["label_3"])
        pred_11.append(pred.label_11)
        pred_3.append(pred.label_3)
        pred_sources.append(str(getattr(pred, "source", "model")))

    labels_11 = sorted(set(true_11) | set(pred_11))
    labels_3 = list(FORMAL_STRATEGY_LABELS)
    macro_f1_3 = _macro_f1(true_3, pred_3, labels_3)
    macro_f1_11 = _macro_f1(true_11, pred_11, labels_11)
    cue_count = sum(1 for source in pred_sources if source == "cue")
    dataset_ok = _formal_dataset_ok(train, gate)
    formal_engine = bool(getattr(clf, "formal_engine", False))
    formal_gate_eligible = bool(dataset_ok and formal_engine)
    passed = bool(formal_gate_eligible and macro_f1_3 >= float(min_macro_f1_3class))
    return {
        "engine": getattr(clf, "engine", "unknown"),
        "dataset_origin": dataset_origin,
        "train_count": int(len(train)),
        "gate_count": int(len(gate)),
        "sample_count": int(len(gate)),
        "class_distribution": {
            "train": _class_distribution(train),
            "gate": _class_distribution(gate),
        },
        "dataset_separation_ok": bool(_dataset_separation_ok(train, gate)),
        "formal_dataset_minima": {
            "train_samples": MIN_FORMAL_TRAIN_SAMPLES,
            "train_per_class": MIN_FORMAL_TRAIN_PER_CLASS,
            "gate_samples": MIN_FORMAL_GATE_SAMPLES,
            "gate_per_class": MIN_FORMAL_GATE_PER_CLASS,
        },
        "macro_f1_3class": round(float(macro_f1_3), 6),
        "macro_f1_11class": round(float(macro_f1_11), 6),
        "confusion_matrix_3class": _confusion_matrix(true_3, pred_3, labels_3),
        "per_class_metrics_3class": _per_class_metrics(true_3, pred_3, labels_3),
        "cue_override_rate": round(float(cue_count) / float(max(1, len(pred_sources))), 6),
        "prediction_source_counts": {k: int(v) for k, v in Counter(pred_sources).items()},
        "min_macro_f1_3class": float(min_macro_f1_3class),
        "formal_gate_eligible": bool(formal_gate_eligible),
        "formal_engine": bool(formal_engine),
        "formal_dataset_ok": bool(dataset_ok),
        "passed_3class_gate": bool(passed),
        "behavioral_hard_metric_enabled": bool(passed),
        "degradation_required": bool(not passed),
    }
