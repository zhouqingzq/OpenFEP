from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import math
import random
import re
from typing import Mapping


class SplitStrategy(str, Enum):
    RANDOM = "random"
    TEMPORAL = "temporal"
    PARTNER = "partner"
    TOPIC = "topic"


@dataclass(slots=True)
class DataSplit:
    strategy: SplitStrategy
    train_sessions: list[dict]
    holdout_sessions: list[dict]
    split_metadata: dict[str, object]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _session_start_ts(session: Mapping[str, object]) -> str:
    start = _safe_text(session.get("start_time"))
    if start:
        return start
    turns = session.get("turns", [])
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, Mapping):
                ts = _safe_text(turn.get("timestamp"))
                if ts:
                    return ts
    return ""


def _session_partner_uid(user_uid: int, session: Mapping[str, object]) -> int:
    uid_a = _parse_int(session.get("uid_a"), user_uid)
    uid_b = _parse_int(session.get("uid_b"), user_uid)
    if uid_a == user_uid:
        return uid_b
    return uid_a


def _collect_sessions(user_dataset: Mapping[str, object]) -> list[dict]:
    raw = user_dataset.get("sessions", [])
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _bounded_cut(total: int, train_ratio: float) -> tuple[int, int]:
    if total <= 1:
        return total, 0
    train_n = max(1, min(total - 1, int(round(total * train_ratio))))
    holdout_n = total - train_n
    return train_n, holdout_n


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _session_document(session: Mapping[str, object]) -> str:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return ""
    parts: list[str] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        body = _safe_text(turn.get("body"))
        if body:
            parts.append(body)
    return " ".join(parts)


def _build_tfidf(
    docs: list[str],
    train_indices: set[int],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    tokenized = [_tokenize(doc) for doc in docs]
    train_df: Counter[str] = Counter()
    train_count = 0
    for idx, tokens in enumerate(tokenized):
        if idx not in train_indices:
            continue
        train_count += 1
        unique = set(tokens)
        for token in unique:
            train_df[token] += 1
    idf: dict[str, float] = {}
    denom = max(1, train_count)
    for token, df in train_df.items():
        idf[token] = math.log((1.0 + denom) / (1.0 + float(df))) + 1.0
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        vec: dict[str, float] = {}
        for token, cnt in counts.items():
            if token not in idf:
                continue
            tf = float(cnt) / max(1.0, total)
            vec[token] = tf * idf[token]
        vectors.append(vec)
    return vectors, idf


def _cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = 0.0
    for key, value in left.items():
        dot += float(value) * float(right.get(key, 0.0))
    l_norm = math.sqrt(sum(float(v) * float(v) for v in left.values()))
    r_norm = math.sqrt(sum(float(v) * float(v) for v in right.values()))
    if l_norm <= 1e-12 or r_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (l_norm * r_norm)))


def _cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(0.0, 1.0 - _cosine_similarity(left, right))


def _mean_vector(indices: list[int], vectors: list[dict[str, float]]) -> dict[str, float]:
    if not indices:
        return {}
    acc: dict[str, float] = defaultdict(float)
    for idx in indices:
        for key, value in vectors[idx].items():
            acc[key] += float(value)
    inv = 1.0 / float(len(indices))
    return {key: value * inv for key, value in acc.items() if value != 0.0}


def _kmeans_fitted_on_train(
    vectors: list[dict[str, float]],
    k: int,
    *,
    seed: int,
    fit_indices: set[int],
    max_iter: int = 25,
) -> tuple[list[int], list[dict[str, float]]]:
    """KMeans-style clustering: centroids updated from *train* sessions only; all points assigned."""
    n = len(vectors)
    fit_list = sorted(fit_indices)
    if n == 0 or not fit_list:
        return [], []
    if k <= 1:
        return [0] * n, [_mean_vector(list(range(n)), vectors)]
    rng = random.Random(seed)
    init = rng.sample(fit_list, k=min(k, len(fit_list)))
    centroids = [dict(vectors[idx]) for idx in init]
    while len(centroids) < k:
        centroids.append({})
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for idx, vec in enumerate(vectors):
            best_c = 0
            best_s = float("-inf")
            for c_idx, center in enumerate(centroids):
                score = _cosine_similarity(vec, center)
                if score > best_s:
                    best_s = score
                    best_c = c_idx
            if labels[idx] != best_c:
                labels[idx] = best_c
                changed = True
        grouped: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(labels):
            grouped[cid].append(idx)
        for c_idx in range(k):
            train_members = [i for i in grouped.get(c_idx, []) if i in fit_indices]
            if train_members:
                centroids[c_idx] = _mean_vector(train_members, vectors)
        if not changed:
            break
    return labels, centroids


def _cluster_sizes(labels: list[int], indices: set[int] | None = None) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for idx, cid in enumerate(labels):
        if indices is not None and idx not in indices:
            continue
        out[int(cid)] += 1
    return dict(out)


def _merge_small_clusters(
    labels: list[int],
    centroids: list[dict[str, float]],
    *,
    min_size: int,
    vectors: list[dict[str, float]],
    fit_indices: set[int] | None = None,
) -> tuple[list[int], list[dict[str, float]], list[dict[str, object]]]:
    merge_steps: list[dict[str, object]] = []
    merged = list(labels)
    while True:
        sizes = _cluster_sizes(merged, indices=fit_indices)
        small = [cid for cid, size in sizes.items() if size < min_size]
        if not small:
            break
        active = sorted(sizes.keys())
        if len(active) <= 1:
            break
        source = min(small, key=lambda cid: (sizes[cid], cid))
        candidates = [cid for cid in active if cid != source]
        target = min(
            candidates,
            key=lambda cid: _cosine_distance(centroids[source], centroids[cid]),
        )
        merge_steps.append(
            {
                "source_cluster": int(source),
                "target_cluster": int(target),
                "source_size": int(sizes.get(source, 0)),
                "target_size": int(sizes.get(target, 0)),
            }
        )
        for idx, cid in enumerate(merged):
            if cid == source:
                merged[idx] = target
        members_by_cluster: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(merged):
            members_by_cluster[cid].append(idx)
        for cid, members in members_by_cluster.items():
            train_m = [i for i in members if fit_indices is None or i in fit_indices]
            if train_m:
                centroids[cid] = _mean_vector(train_m, vectors)
            elif members:
                centroids[cid] = _mean_vector(members, vectors)
    return merged, centroids, merge_steps


def _recompute_centroids(
    labels: list[int],
    vectors: list[dict[str, float]],
    fit_indices: set[int] | None = None,
) -> list[dict[str, float]]:
    members_by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        members_by_cluster[cid].append(idx)
    max_cid = max(members_by_cluster) if members_by_cluster else -1
    centroids: list[dict[str, float]] = [{} for _ in range(max_cid + 1)]
    for cid, members in members_by_cluster.items():
        train_m = [i for i in members if fit_indices is None or i in fit_indices]
        if train_m:
            centroids[cid] = _mean_vector(train_m, vectors)
        else:
            centroids[cid] = _mean_vector(members, vectors)
    return centroids


def _silhouette_score(
    labels: list[int],
    vectors: list[dict[str, float]],
    *,
    indices: set[int] | None = None,
) -> float:
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        if indices is not None and idx not in indices:
            continue
        clusters[cid].append(idx)
    if len(clusters) <= 1:
        return -1.0
    values: list[float] = []
    scored_indices = sorted(indices) if indices is not None else list(range(len(labels)))
    for idx in scored_indices:
        cid = labels[idx]
        same = clusters[cid]
        if len(same) <= 1:
            values.append(0.0)
            continue
        a = sum(_cosine_distance(vectors[idx], vectors[j]) for j in same if j != idx) / float(len(same) - 1)
        b = float("inf")
        for other_cid, members in clusters.items():
            if other_cid == cid:
                continue
            dist = sum(_cosine_distance(vectors[idx], vectors[j]) for j in members) / float(len(members))
            if dist < b:
                b = dist
        denom = max(a, b, 1e-9)
        values.append((b - a) / denom)
    if not values:
        return -1.0
    return sum(values) / float(len(values))


def _topic_not_applicable(
    sessions: list[dict],
    metadata: dict[str, object],
    reason: str,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    out = dict(metadata)
    out.update(
        {
            "topic_split_not_applicable": True,
            "reason": reason,
            "train_count": int(len(sessions)),
            "holdout_count": 0,
            "strict_topic_no_random_fallback": True,
        }
    )
    return sessions[:], [], out


def _distribution_jsd(left: Mapping[int, int], right: Mapping[int, int]) -> float:
    keys = sorted(set(left.keys()) | set(right.keys()))
    if not keys:
        return 0.0
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    p = [float(left.get(key, 0)) / left_total for key in keys]
    q = [float(right.get(key, 0)) / right_total for key in keys]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for x, y in zip(a, b):
            if x <= 0.0:
                continue
            total += x * math.log(x / max(y, 1e-12))
        return total

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _split_random(
    sessions: list[dict],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    order = list(range(len(sessions)))
    rng = random.Random(seed)
    rng.shuffle(order)
    train_n, holdout_n = _bounded_cut(len(sessions), train_ratio)
    holdout_ids = set(order[:holdout_n])
    train, holdout = [], []
    for idx, session in enumerate(sessions):
        if idx in holdout_ids:
            holdout.append(session)
        else:
            train.append(session)
    metadata = {
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "session_count": int(len(sessions)),
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
    }
    return train, holdout, metadata


def _split_temporal(
    sessions: list[dict],
    *,
    train_ratio: float,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    indexed = [(idx, _session_start_ts(session), session) for idx, session in enumerate(sessions)]
    indexed.sort(key=lambda item: (item[1], item[0]))
    train_n, _ = _bounded_cut(len(indexed), train_ratio)
    train_idx = {item[0] for item in indexed[:train_n]}
    train, holdout = [], []
    for idx, session in enumerate(sessions):
        if idx in train_idx:
            train.append(session)
        else:
            holdout.append(session)
    metadata = {
        "train_ratio": float(train_ratio),
        "session_count": int(len(sessions)),
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
        "ordered_by": "session_start_time",
    }
    return train, holdout, metadata


def _split_partner(
    user_uid: int,
    sessions: list[dict],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    by_partner: dict[int, list[int]] = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_partner[_session_partner_uid(user_uid, session)].append(idx)
    partners = sorted(by_partner.keys())
    rng = random.Random(seed)
    rng.shuffle(partners)
    _, holdout_target = _bounded_cut(len(sessions), train_ratio)
    holdout_idx: set[int] = set()
    holdout_partners: list[int] = []
    for partner in partners:
        if len(holdout_idx) >= holdout_target and holdout_partners:
            break
        holdout_partners.append(partner)
        holdout_idx.update(by_partner[partner])
    if len(holdout_idx) >= len(sessions):
        last_partner = holdout_partners.pop() if holdout_partners else None
        if last_partner is not None:
            for idx in by_partner[last_partner]:
                holdout_idx.discard(idx)
    if not holdout_idx and partners:
        holdout_partners = [partners[0]]
        holdout_idx.update(by_partner[partners[0]])
    train, holdout = [], []
    for idx, session in enumerate(sessions):
        if idx in holdout_idx:
            holdout.append(session)
        else:
            train.append(session)
    metadata = {
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "partner_count": int(len(partners)),
        "holdout_partners": [int(item) for item in sorted(set(holdout_partners))],
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
    }
    return train, holdout, metadata


def _topic_try_stratified_jsd(
    sessions: list[dict],
    *,
    labels: list[int],
    train_ratio: float,
    seed: int,
    jsd_threshold: float,
    max_retries: int,
) -> tuple[set[int], set[int], float, int] | None:
    """Pick holdout per cluster; return (train_idx, holdout_idx, jsd, attempt) or None."""
    n_sessions = len(sessions)
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        by_cluster[int(cid)].append(idx)

    for attempt in range(max_retries):
        holdout_idx: set[int] = set()
        for cid in sorted(by_cluster.keys()):
            members = list(by_cluster[cid])
            rng = random.Random(seed + 1000 + (attempt * 131) + cid)
            rng.shuffle(members)
            _, holdout_n = _bounded_cut(len(members), train_ratio)
            selected = members[:holdout_n]
            holdout_idx.update(selected)
        train_idx = set(range(n_sessions)) - holdout_idx
        if not train_idx or not holdout_idx:
            continue
        train_dist = Counter(labels[idx] for idx in train_idx)
        holdout_dist = Counter(labels[idx] for idx in holdout_idx)
        jsd = _distribution_jsd(train_dist, holdout_dist)
        if jsd <= jsd_threshold:
            return train_idx, holdout_idx, float(jsd), int(attempt)
    return None


def _split_topic(
    sessions: list[dict],
    *,
    train_ratio: float,
    seed: int,
    min_cluster_size: int = 5,
    max_k: int = 8,
    jsd_threshold: float = 0.10,
    max_retries: int = 8,
    max_train_fit_iterations: int = 24,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    """Topic split: TF-IDF + KMeans fit **only** on current train indices; iterate until
    stratified train indices match fit indices (strict no-leakage w.r.t. final train set).
    """
    n_sessions = len(sessions)
    metadata: dict[str, object] = {
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "strict_train_only_fit": True,
        "topic_kmeans_fitted_on_train_sessions_only": True,
        "min_cluster_size": int(min_cluster_size),
        "max_k": int(max_k),
        "max_train_fit_iterations": int(max_train_fit_iterations),
        "jsd_threshold": float(jsd_threshold),
        "strict_topic_no_random_fallback": True,
    }
    if n_sessions <= 1:
        return _topic_not_applicable(sessions, metadata, "insufficient_sessions")

    docs = [_session_document(item) for item in sessions]
    effective_k_max = min(int(max_k), n_sessions // max(1, int(min_cluster_size)))
    metadata["k_search_max_by_all_sessions"] = int(effective_k_max)
    if effective_k_max < 2:
        return _topic_not_applicable(sessions, metadata, "k_range_empty")

    fit_indices: set[int] | None = None
    seen_masks: set[frozenset[int]] = set()

    for fit_iter in range(max(1, int(max_train_fit_iterations))):
        if fit_indices is None:
            init_train, _, _ = _split_random(
                sessions, train_ratio=train_ratio, seed=seed + fit_iter * 7919
            )
            init_ids = {id(item) for item in init_train}
            fit_indices = {idx for idx, item in enumerate(sessions) if id(item) in init_ids}

        vectors, _ = _build_tfidf(docs, fit_indices)
        fit_k_max = min(int(max_k), len(fit_indices) // max(1, int(min_cluster_size)))
        metadata["k_search_max_by_train_sessions"] = int(fit_k_max)
        if fit_k_max < 2:
            return _topic_not_applicable(sessions, metadata, "train_k_range_empty")

        best: dict[str, object] | None = None
        for k in range(2, fit_k_max + 1):
            labels, centroids = _kmeans_fitted_on_train(
                vectors,
                k,
                seed=seed + k + fit_iter * 97,
                fit_indices=fit_indices,
            )
            labels, centroids, merge_steps = _merge_small_clusters(
                labels,
                list(centroids),
                min_size=min_cluster_size,
                vectors=vectors,
                fit_indices=fit_indices,
            )
            centroids = _recompute_centroids(labels, vectors, fit_indices=fit_indices)
            sizes = _cluster_sizes(labels, indices=fit_indices)
            if any(size < min_cluster_size for size in sizes.values()):
                continue
            score = _silhouette_score(labels, vectors, indices=fit_indices)
            candidate = {
                "k": int(k),
                "labels": labels,
                "centroids": centroids,
                "sizes": {str(cid): int(size) for cid, size in sorted(sizes.items())},
                "all_assigned_cluster_sizes": {
                    str(cid): int(size) for cid, size in sorted(_cluster_sizes(labels).items())
                },
                "merge_steps": merge_steps,
                "silhouette": float(score),
                "silhouette_scope": "train_fit_sessions_only",
            }
            if best is None:
                best = candidate
            else:
                prev = float(best["silhouette"])
                if score > prev + 1e-12 or (abs(score - prev) <= 1e-12 and int(k) < int(best["k"])):
                    best = candidate

        if best is None:
            return _topic_not_applicable(sessions, metadata, "no_valid_cluster_layout")

        labels = list(best["labels"])
        strat = _topic_try_stratified_jsd(
            sessions,
            labels=labels,
            train_ratio=train_ratio,
            seed=seed,
            jsd_threshold=jsd_threshold,
            max_retries=max_retries,
        )
        if strat is None:
            return _topic_not_applicable(sessions, metadata, "jsd_quality_gate_failed")

        train_idx, holdout_idx, jsd, attempt = strat
        train_list = [sessions[idx] for idx in sorted(train_idx)]
        holdout_list = [sessions[idx] for idx in sorted(holdout_idx)]

        meta_round: dict[str, object] = {
            "k_selected": int(best["k"]),
            "cluster_sizes": dict(best["sizes"]),
            "cluster_sizes_scope": "train_fit_sessions_only",
            "all_assigned_cluster_sizes": dict(best["all_assigned_cluster_sizes"]),
            "merge_steps": list(best["merge_steps"]),
            "silhouette": round(float(best["silhouette"]), 6),
            "silhouette_scope": str(best["silhouette_scope"]),
            "topic_jsd_train_holdout": round(float(jsd), 6),
            "jsd_threshold": float(jsd_threshold),
            "topic_split_retry_count": int(attempt),
            "topic_train_fit_iteration": int(fit_iter),
            "topic_train_fit_iterations": int(fit_iter + 1),
            "topic_strict_fit_converged": bool(train_idx == fit_indices),
            "train_count": int(len(train_list)),
            "holdout_count": int(len(holdout_list)),
            "topic_split_not_applicable": False,
            "topic_holdout_transform_only": True,
        }
        merged = {**metadata, **meta_round}

        if train_idx == fit_indices:
            return train_list, holdout_list, merged

        mask = frozenset(train_idx)
        if mask in seen_masks:
            return _topic_not_applicable(sessions, metadata, "topic_strict_fit_cycle_detected")
        seen_masks.add(mask)
        fit_indices = set(train_idx)

    return _topic_not_applicable(sessions, metadata, "topic_strict_fit_not_converged")


def split_user_data(
    user_dataset: dict,
    strategy: SplitStrategy,
    *,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> DataSplit:
    sessions = _collect_sessions(user_dataset)
    user_uid = _parse_int(user_dataset.get("uid"), 0)
    if not sessions:
        return DataSplit(
            strategy=strategy,
            train_sessions=[],
            holdout_sessions=[],
            split_metadata={
                "train_ratio": float(train_ratio),
                "seed": int(seed),
                "session_count": 0,
                "train_count": 0,
                "holdout_count": 0,
                "topic_split_not_applicable": bool(strategy == SplitStrategy.TOPIC),
            },
        )

    if strategy == SplitStrategy.RANDOM:
        train, holdout, meta = _split_random(sessions, train_ratio=train_ratio, seed=seed)
    elif strategy == SplitStrategy.TEMPORAL:
        train, holdout, meta = _split_temporal(sessions, train_ratio=train_ratio)
    elif strategy == SplitStrategy.PARTNER:
        train, holdout, meta = _split_partner(
            user_uid,
            sessions,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif strategy == SplitStrategy.TOPIC:
        train, holdout, meta = _split_topic(
            sessions,
            train_ratio=train_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported split strategy: {strategy}")

    metadata = dict(meta)
    metadata.setdefault("session_count", int(len(sessions)))
    metadata.setdefault("train_count", int(len(train)))
    metadata.setdefault("holdout_count", int(len(holdout)))
    metadata["strategy"] = str(strategy.value)
    return DataSplit(
        strategy=strategy,
        train_sessions=train,
        holdout_sessions=holdout,
        split_metadata=metadata,
    )

