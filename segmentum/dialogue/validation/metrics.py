from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import os
import re
from typing import Mapping

_ST_MODEL: object | None = None
_ST_MODEL_NAME: str | None = None
_ST_EMBEDDING_CACHE: dict[tuple[str, str], object] = {}

from ...personality_analyzer import PersonalityAnalyzer
from ..actions import DIALOGUE_ACTION_STRATEGY_MAP

_STRATEGY_LABELS = frozenset({"explore", "exploit", "escape"})


@dataclass(slots=True)
class SimilarityResult:
    metric_name: str
    value: float
    details: dict[str, object]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_COOPERATIVE_ACKS = frozenset(
    {
        "ok",
        "okay",
        "sure",
        "fine",
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
        "對",
        "对",
        "謝謝",
        "谢谢",
        "感謝",
        "感谢",
    }
)
_PUNCT = tuple("。！？,，.!?")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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


def _tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    tokenized = [_tokenize(text) for text in texts]
    df: Counter[str] = Counter()
    for tokens in tokenized:
        for token in set(tokens):
            df[token] += 1
    n = max(1.0, float(len(texts)))
    idf = {token: math.log((1.0 + n) / (1.0 + float(cnt))) + 1.0 for token, cnt in df.items()}
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        vec = {token: (float(cnt) / total) * idf.get(token, 0.0) for token, cnt in counts.items()}
        vectors.append(vec)
    return vectors


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)]


def _bleu4(generated: list[str], real: list[str]) -> float:
    if not generated or not real:
        return 0.0
    pairs = list(zip(generated, real))
    precisions: list[float] = []
    for n in range(1, 5):
        matched = 0
        total = 0
        for gen, ref in pairs:
            g_ngrams = _ngrams(_tokenize(gen), n)
            r_ngrams = Counter(_ngrams(_tokenize(ref), n))
            total += len(g_ngrams)
            for item in g_ngrams:
                if r_ngrams[item] > 0:
                    matched += 1
                    r_ngrams[item] -= 1
        precisions.append(float(matched) / float(total) if total > 0 else 0.0)
    if any(p <= 0.0 for p in precisions):
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / 4.0)
    gen_len = sum(len(_tokenize(gen)) for gen, _ in pairs)
    ref_len = sum(len(_tokenize(ref)) for _, ref in pairs)
    if gen_len <= 0:
        return 0.0
    if gen_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (float(ref_len) / float(gen_len)))
    return max(0.0, min(1.0, bp * geo))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, l_token in enumerate(left, start=1):
        for j, r_token in enumerate(right, start=1):
            if l_token == r_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l(generated: list[str], real: list[str]) -> float:
    pairs = list(zip(generated, real))
    if not pairs:
        return 0.0
    scores: list[float] = []
    for gen, ref in pairs:
        g = _tokenize(gen)
        r = _tokenize(ref)
        if not g or not r:
            scores.append(0.0)
            continue
        lcs = float(_lcs_length(g, r))
        precision = lcs / float(len(g))
        recall = lcs / float(len(r))
        if precision + recall <= 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / float(len(scores))


def _distribution_jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left.keys()) | set(right.keys()))
    if not keys:
        return 0.0
    p = [float(left.get(key, 0.0)) for key in keys]
    q = [float(right.get(key, 0.0)) for key in keys]
    p_total = sum(p)
    q_total = sum(q)
    if p_total <= 0.0 or q_total <= 0.0:
        return 0.0
    p = [value / p_total for value in p]
    q = [value / q_total for value in q]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for x, y in zip(a, b):
            if x <= 0.0:
                continue
            out += x * math.log(x / max(y, 1e-12))
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def surface_similarity(generated: list[str], real: list[str]) -> SimilarityResult:
    bleu = _bleu4(generated, real)
    rouge = _rouge_l(generated, real)
    value = (bleu + rouge) / 2.0
    return SimilarityResult(
        metric_name="surface_similarity",
        value=round(float(value), 6),
        details={"bleu4": round(float(bleu), 6), "rouge_l": round(float(rouge), 6)},
    )


def _load_sentence_transformer(model_name: str):
    """Lazy singleton for SentenceTransformer (multilingual sentence embeddings)."""
    global _ST_MODEL, _ST_MODEL_NAME
    if _ST_MODEL is not None and _ST_MODEL_NAME == model_name:
        return _ST_MODEL
    from sentence_transformers import SentenceTransformer

    _ST_MODEL = SentenceTransformer(model_name)
    _ST_MODEL_NAME = model_name
    return _ST_MODEL


def _semantic_similarity_tfidf_pairs(
    pairs: list[tuple[str, str]],
    *,
    fallback_reason: str | None = None,
) -> SimilarityResult:
    vectors = _tfidf_vectors([item for pair in pairs for item in pair])
    pair_scores: list[float] = []
    for idx in range(0, len(vectors), 2):
        pair_scores.append(_cosine_similarity(vectors[idx], vectors[idx + 1]))
    value = sum(pair_scores) / float(len(pair_scores))
    details: dict[str, object] = {
        "pair_count": int(len(pair_scores)),
        "method": "tfidf_cosine_fallback" if fallback_reason else "tfidf_cosine",
    }
    if fallback_reason:
        details["fallback_reason"] = fallback_reason
    return SimilarityResult(
        metric_name="semantic_similarity",
        value=round(float(max(0.0, min(1.0, value))), 6),
        details=details,
    )


def _semantic_similarity_sentence_embedding(pairs: list[tuple[str, str]]) -> SimilarityResult:
    model_name = os.environ.get(
        "SEGMENTUM_SEMANTIC_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    pair_scores, embedding_dim = _sentence_embedding_pair_scores(pairs, model_name=model_name)
    real_texts = [real for _, real in pairs]
    weights = semantic_pair_weights(real_texts)
    value = weighted_semantic_mean(pair_scores, weights)
    raw_mean = sum(pair_scores) / float(len(pair_scores))
    return SimilarityResult(
        metric_name="semantic_similarity",
        value=round(float(max(0.0, min(1.0, value))), 6),
        details={
            "pair_count": int(len(pair_scores)),
            "method": "sentence_embedding_cosine",
            "model": model_name,
            "embedding_dim": embedding_dim,
            "aggregation": "information_weighted_pair_mean",
            "raw_pair_mean": round(float(raw_mean), 6),
            "weighted_pair_mean": round(float(value), 6),
            "mean_pair_weight": round(float(sum(weights) / float(len(weights))) if weights else 0.0, 6),
            "info_bucket_counts": dict(Counter(semantic_pair_info_buckets(real_texts))),
        },
    )


def _sentence_embedding_pair_scores(
    pairs: list[tuple[str, str]],
    *,
    model_name: str,
) -> tuple[list[float], int]:
    import numpy as np

    model = _load_sentence_transformer(model_name)
    texts = [text for pair in pairs for text in pair]
    missing: list[str] = []
    seen_missing: set[str] = set()
    for text in texts:
        key = (model_name, text)
        if key in _ST_EMBEDDING_CACHE or text in seen_missing:
            continue
        missing.append(text)
        seen_missing.add(text)
    if missing:
        encoded = model.encode(
            missing,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for text, row in zip(missing, encoded):
            _ST_EMBEDDING_CACHE[(model_name, text)] = np.asarray(row, dtype=np.float64)
    arr = np.asarray([_ST_EMBEDDING_CACHE[(model_name, text)] for text in texts], dtype=np.float64)
    pair_scores: list[float] = []
    for i in range(0, len(arr), 2):
        pair_scores.append(float(max(0.0, min(1.0, np.dot(arr[i], arr[i + 1])))))
    embedding_dim = int(arr.shape[1]) if arr.size else 0
    return pair_scores, embedding_dim


def semantic_similarity(generated: list[str], real: list[str]) -> SimilarityResult:
    """Pairwise semantic similarity: multilingual sentence embeddings + cosine (preferred) or TF-IDF fallback."""
    pairs = list(zip(generated, real))
    if not pairs:
        return SimilarityResult("semantic_similarity", 0.0, {"pair_count": 0, "method": "none"})

    force_tfidf = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if force_tfidf:
        result = _semantic_similarity_tfidf_pairs(pairs)
        scores = semantic_pair_scores(generated, real)
        weights = semantic_pair_weights(real)
        weighted = weighted_semantic_mean(scores, weights)
        result.value = round(float(weighted), 6)
        result.details.update(
            {
                "aggregation": "information_weighted_pair_mean",
                "raw_pair_mean": round(float(sum(scores) / float(len(scores))) if scores else 0.0, 6),
                "weighted_pair_mean": round(float(weighted), 6),
                "mean_pair_weight": round(float(sum(weights) / float(len(weights))) if weights else 0.0, 6),
                "info_bucket_counts": dict(Counter(semantic_pair_info_buckets(real))),
            }
        )
        return result

    try:
        return _semantic_similarity_sentence_embedding(pairs)
    except ImportError:
        return _semantic_similarity_tfidf_pairs(
            pairs,
            fallback_reason="sentence_transformers_not_installed; pip install segmentum[validation]",
        )
    except Exception as exc:  # noqa: BLE001 — keep validation run alive; see details
        return _semantic_similarity_tfidf_pairs(pairs, fallback_reason=repr(exc))


def semantic_pair_scores(generated: list[str], real: list[str]) -> list[float]:
    """Return one semantic score per generated/real pair using the configured engine."""
    pairs = list(zip(generated, real))
    if not pairs:
        return []
    force_tfidf = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if force_tfidf:
        vectors = _tfidf_vectors([item for pair in pairs for item in pair])
        return [
            round(float(max(0.0, min(1.0, _cosine_similarity(vectors[idx], vectors[idx + 1])))), 6)
            for idx in range(0, len(vectors), 2)
        ]
    try:
        model_name = os.environ.get(
            "SEGMENTUM_SEMANTIC_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        scores, _ = _sentence_embedding_pair_scores(pairs, model_name=model_name)
        return [round(float(score), 6) for score in scores]
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - diagnostics should not abort validation
        pass
    vectors = _tfidf_vectors([item for pair in pairs for item in pair])
    return [
        round(float(max(0.0, min(1.0, _cosine_similarity(vectors[idx], vectors[idx + 1])))), 6)
        for idx in range(0, len(vectors), 2)
    ]


def semantic_pair_info_bucket(text: str) -> str:
    """Bucket real replies by information content for formal semantic aggregation."""
    compact = " ".join(str(text).strip().lower().split())
    if not compact:
        return "empty"
    stripped = re.sub(r"[\s.,!?;:，。！？；：~…]+", "", compact)
    tokens = _tokenize(compact)
    token_count = len(tokens)
    char_count = len(stripped)
    if stripped in _COOPERATIVE_ACKS or compact in _COOPERATIVE_ACKS:
        return "low_info_ack"
    if char_count <= 2 or token_count <= 1:
        return "ultra_low_info"
    if char_count <= 8 or token_count <= 2:
        return "low_info_short"
    if char_count <= 20 or token_count <= 5:
        return "medium_info"
    return "high_info"


def semantic_pair_weight_for_text(text: str) -> float:
    bucket = semantic_pair_info_bucket(text)
    if bucket == "empty":
        return 0.0
    if bucket in {"low_info_ack", "ultra_low_info"}:
        return 0.15
    if bucket == "low_info_short":
        return 0.35
    if bucket == "medium_info":
        return 0.70
    return 1.0


def semantic_pair_weights(real: list[str]) -> list[float]:
    return [round(float(semantic_pair_weight_for_text(text)), 6) for text in real]


def semantic_pair_info_buckets(real: list[str]) -> list[str]:
    return [semantic_pair_info_bucket(text) for text in real]


def weighted_semantic_mean(scores: list[float], weights: list[float]) -> float:
    total = min(len(scores), len(weights))
    if total <= 0:
        return 0.0
    pairs = [(float(scores[idx]), max(0.0, float(weights[idx]))) for idx in range(total)]
    denom = sum(weight for _, weight in pairs)
    if denom <= 1e-12:
        return sum(score for score, _ in pairs) / float(len(pairs))
    return sum(score * weight for score, weight in pairs) / denom


def stylistic_similarity(generated: list[str], real: list[str]) -> SimilarityResult:
    def _features(texts: list[str]) -> dict[str, float]:
        text = "\n".join(texts)
        tokens = _tokenize(text)
        lengths = [_tokenize(line) for line in texts]
        length_bins: Counter[str] = Counter()
        for item in lengths:
            bin_id = int(len(item) / 5)
            length_bins[f"len_bin_{bin_id}"] += 1
        punct: Counter[str] = Counter(char for char in text if char in _PUNCT)
        emoji_count = sum(1 for ch in text if ord(ch) > 0x1000 and ch not in _PUNCT)
        type_token = float(len(set(tokens))) / float(len(tokens)) if tokens else 0.0
        out: dict[str, float] = {}
        for key, value in length_bins.items():
            out[f"length:{key}"] = float(value)
        for key, value in punct.items():
            out[f"punct:{key}"] = float(value)
        out["emoji_freq"] = float(emoji_count)
        out["type_token"] = float(type_token) * max(1, len(tokens))
        return out

    g_feat = _features(generated)
    r_feat = _features(real)
    jsd = _distribution_jsd(g_feat, r_feat)
    similarity = max(0.0, min(1.0, 1.0 - (jsd / math.log(2.0))))
    return SimilarityResult(
        metric_name="stylistic_similarity",
        value=round(float(similarity), 6),
        details={"jsd": round(float(jsd), 6)},
    )


def personality_similarity(generated: list[str], real: list[str]) -> SimilarityResult:
    analyzer = PersonalityAnalyzer()
    generated_result = analyzer.analyze(generated)
    real_result = analyzer.analyze(real)
    keys = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    left = {key: float(generated_result.big_five.get(key, 0.0)) for key in keys}
    right = {key: float(real_result.big_five.get(key, 0.0)) for key in keys}
    cosine = _cosine_similarity(left, right)
    return SimilarityResult(
        metric_name="personality_similarity",
        value=round(float(max(0.0, min(1.0, cosine))), 6),
        details={"vector_keys": list(keys)},
    )


def behavioral_similarity(
    generated_actions: list[str],
    real_actions: list[str],
    *,
    granularity: str = "strategy",
) -> SimilarityResult:
    if granularity not in {"strategy", "action11"}:
        raise ValueError("granularity must be 'strategy' or 'action11'")
    if granularity == "strategy":
        if (
            generated_actions
            and real_actions
            and all(item in _STRATEGY_LABELS for item in generated_actions)
            and all(item in _STRATEGY_LABELS for item in real_actions)
        ):
            gen_labels = list(generated_actions)
            real_labels = list(real_actions)
        else:
            gen_labels = [DIALOGUE_ACTION_STRATEGY_MAP.get(item, "explore") for item in generated_actions]
            real_labels = [DIALOGUE_ACTION_STRATEGY_MAP.get(item, "explore") for item in real_actions]
    else:
        gen_labels = list(generated_actions)
        real_labels = list(real_actions)
    gen_dist = Counter(gen_labels)
    real_dist = Counter(real_labels)
    keys = sorted(set(gen_dist.keys()) | set(real_dist.keys()))
    gen_total = float(sum(gen_dist.values()))
    real_total = float(sum(real_dist.values()))
    if gen_total <= 0.0 or real_total <= 0.0:
        return SimilarityResult(
            metric_name=f"behavioral_similarity_{granularity}",
            value=0.0,
            details={"chi_square_distance": 0.0, "categories": keys},
        )
    chi2 = 0.0
    for key in keys:
        p = float(gen_dist.get(key, 0)) / gen_total
        q = float(real_dist.get(key, 0)) / real_total
        denom = max(q, 1e-9)
        chi2 += ((p - q) ** 2) / denom
    similarity = 1.0 / (1.0 + chi2)
    return SimilarityResult(
        metric_name=f"behavioral_similarity_{granularity}",
        value=round(float(max(0.0, min(1.0, similarity))), 6),
        details={"chi_square_distance": round(float(chi2), 6), "categories": keys},
    )


def balanced_behavioral_similarity(
    generated_actions: list[str],
    real_actions: list[str],
    *,
    granularity: str = "strategy",
) -> SimilarityResult:
    if granularity not in {"strategy", "action11"}:
        raise ValueError("granularity must be 'strategy' or 'action11'")
    if granularity == "strategy":
        if (
            generated_actions
            and real_actions
            and all(item in _STRATEGY_LABELS for item in generated_actions)
            and all(item in _STRATEGY_LABELS for item in real_actions)
        ):
            gen_labels = list(generated_actions)
            real_labels = list(real_actions)
        else:
            gen_labels = [DIALOGUE_ACTION_STRATEGY_MAP.get(item, "explore") for item in generated_actions]
            real_labels = [DIALOGUE_ACTION_STRATEGY_MAP.get(item, "explore") for item in real_actions]
        keys = sorted(_STRATEGY_LABELS)
    else:
        gen_labels = list(generated_actions)
        real_labels = list(real_actions)
        keys = sorted(set(gen_labels) | set(real_labels))
    gen_dist = Counter(gen_labels)
    real_dist = Counter(real_labels)
    gen_total = float(sum(gen_dist.values()))
    real_total = float(sum(real_dist.values()))
    if gen_total <= 0.0 or real_total <= 0.0 or not keys:
        return SimilarityResult(
            metric_name=f"balanced_behavioral_similarity_{granularity}",
            value=0.0,
            details={"balanced_l1_distance": 0.0, "categories": keys},
        )
    distances: list[float] = []
    for key in keys:
        p = float(gen_dist.get(key, 0)) / gen_total
        q = float(real_dist.get(key, 0)) / real_total
        distances.append(abs(p - q))
    balanced_distance = sum(distances) / float(len(distances))
    similarity = 1.0 - balanced_distance
    return SimilarityResult(
        metric_name=f"balanced_behavioral_similarity_{granularity}",
        value=round(float(max(0.0, min(1.0, similarity))), 6),
        details={"balanced_l1_distance": round(float(balanced_distance), 6), "categories": keys},
    )


def agent_state_similarity(
    agent_train: object,
    agent_full: object,
) -> SimilarityResult:
    def _state_vector(agent: object) -> dict[str, float]:
        if hasattr(agent, "to_dict"):
            payload = agent.to_dict()  # type: ignore[call-arg]
        elif isinstance(agent, Mapping):
            payload = dict(agent)
        else:
            payload = {}
        learner = payload.get("slow_variable_learner", {})
        state = learner.get("state", {}) if isinstance(learner, Mapping) else {}
        traits = state.get("traits", {}) if isinstance(state, Mapping) else {}
        self_model = payload.get("self_model", {})
        priors = self_model.get("narrative_priors", {}) if isinstance(self_model, Mapping) else {}
        out: dict[str, float] = {}
        if isinstance(traits, Mapping):
            for key, value in traits.items():
                try:
                    out[f"trait:{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        if isinstance(priors, Mapping):
            for key, value in priors.items():
                try:
                    out[f"prior:{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        return out

    left = _state_vector(agent_train)
    right = _state_vector(agent_full)
    cosine = _cosine_similarity(left, right)
    return SimilarityResult(
        metric_name="agent_state_similarity",
        value=round(float(max(0.0, min(1.0, cosine))), 6),
        details={"vector_length": int(len(set(left.keys()) | set(right.keys())))},
    )

