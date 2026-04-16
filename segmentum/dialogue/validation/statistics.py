from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import mean
@dataclass(slots=True)
class ComparisonResult:
    metric_name: str
    personality_mean: float
    baseline_a_mean: float
    baseline_b_mean: float
    baseline_c_mean: float
    vs_a_pvalue: float
    vs_b_pvalue: float
    vs_c_pvalue: float
    vs_a_mean_diff: float
    vs_b_mean_diff: float
    vs_c_mean_diff: float
    vs_a_better: bool
    vs_b_better: bool
    vs_c_better: bool
    vs_a_significant: bool
    vs_b_significant: bool
    vs_c_significant: bool


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _one_sided_paired_t_greater(diffs: list[float]) -> float:
    """P(H1: mean > 0) for paired differences; normal approx."""
    n = len(diffs)
    if n <= 1:
        return 1.0
    mu = mean(diffs)
    if mu <= 1e-12:
        return 1.0
    var = sum((item - mu) ** 2 for item in diffs) / float(n - 1)
    if var <= 1e-12:
        return 0.0
    se = math.sqrt(var / float(n))
    t_stat = mu / max(se, 1e-12)
    p = 1.0 - _normal_cdf(t_stat)
    return max(0.0, min(1.0, p))


def _wilcoxon_pvalue_two_sided(diffs: list[float]) -> float:
    non_zero = [item for item in diffs if abs(item) > 1e-12]
    n = len(non_zero)
    if n == 0:
        return 1.0
    ranked = sorted(enumerate(non_zero), key=lambda item: abs(item[1]))
    ranks = [0] * n
    for rank, (idx, _) in enumerate(ranked, start=1):
        ranks[idx] = rank
    w_plus = sum(rank for rank, diff in zip(ranks, non_zero) if diff > 0.0)
    total = n * (n + 1) // 2
    w = min(w_plus, total - w_plus)
    distribution: Counter[int] = Counter({0: 1})
    for rank in range(1, n + 1):
        next_dist: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            next_dist[subtotal] += count
            next_dist[subtotal + rank] += count
        distribution = next_dist
    denom = float(2**n)
    low_tail = sum(count for subtotal, count in distribution.items() if subtotal <= w) / denom
    p = min(1.0, 2.0 * low_tail)
    return max(0.0, min(1.0, p))


def _wilcoxon_greater_p_scipy(diffs: list[float]) -> float:
    from scipy.stats import wilcoxon

    arr = [float(x) for x in diffs]
    if not arr:
        return 1.0
    result = wilcoxon(arr, alternative="greater", zero_method="wilcox", mode="auto")
    p = float(result.pvalue)
    return max(0.0, min(1.0, p))


def _wilcoxon_greater_p_fallback(diffs: list[float]) -> float:
    """Conservative fallback when scipy is unavailable: reject if mean diff <= 0; else crude upper bound."""
    if not diffs:
        return 1.0
    if mean(diffs) <= 1e-12:
        return 1.0
    p_two = _wilcoxon_pvalue_two_sided(diffs)
    return max(0.0, min(1.0, p_two / 2.0))


def _wilcoxon_greater_p(diffs: list[float]) -> float:
    try:
        return _wilcoxon_greater_p_scipy(diffs)
    except ImportError:
        return _wilcoxon_greater_p_fallback(diffs)
    except Exception:  # noqa: BLE001
        return _wilcoxon_greater_p_fallback(diffs)


def paired_comparison(
    personality_scores: list[float],
    baseline_scores: list[float],
    *,
    test: str = "wilcoxon",
    alpha: float = 0.05,
) -> tuple[float, bool, float, bool]:
    """Paired test H1: personality > baseline.

    Returns (p_value, significant_better, mean_diff, better) where significant_better means
    mean_diff > 0 and p_value < alpha under the one-sided test.
    """
    n = min(len(personality_scores), len(baseline_scores))
    if n == 0:
        return 1.0, False, 0.0, False
    diffs = [float(personality_scores[idx]) - float(baseline_scores[idx]) for idx in range(n)]
    mean_diff = float(mean(diffs))
    better = mean_diff > 1e-12

    if test == "wilcoxon":
        if not better:
            p_value = 1.0
        else:
            p_value = _wilcoxon_greater_p(diffs)
    elif test == "t_test":
        p_value = _one_sided_paired_t_greater(diffs)
    else:
        raise ValueError("test must be 'wilcoxon' or 't_test'")

    significant_better = bool(better and p_value < float(alpha))
    return round(float(p_value), 6), significant_better, mean_diff, better
