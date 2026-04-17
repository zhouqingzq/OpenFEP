from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean


@dataclass(slots=True)
class ComparisonResult:
    """Aggregate comparison row. Baseline fields are None for personality-only metrics (e.g. agent_state)."""

    metric_name: str
    personality_mean: float
    baseline_a_mean: float | None = None
    baseline_b_mean: float | None = None
    baseline_c_mean: float | None = None
    vs_a_pvalue: float | None = None
    vs_b_pvalue: float | None = None
    vs_c_pvalue: float | None = None
    vs_a_mean_diff: float | None = None
    vs_b_mean_diff: float | None = None
    vs_c_mean_diff: float | None = None
    vs_a_better: bool | None = None
    vs_b_better: bool | None = None
    vs_c_better: bool | None = None
    vs_a_significant: bool | None = None
    vs_b_significant: bool | None = None
    vs_c_significant: bool | None = None
    statistical_valid: bool = True
    statistical_error: str | None = None
    users_included: int | None = None
    users_skipped_no_metric: int | None = None
    interpretation_notes: str | None = None


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


def _wilcoxon_greater_p_scipy(diffs: list[float]) -> float:
    from scipy.stats import wilcoxon

    arr = [float(x) for x in diffs]
    if not arr:
        return 1.0
    result = wilcoxon(arr, alternative="greater", zero_method="wilcox", mode="auto")
    p = float(result.pvalue)
    return max(0.0, min(1.0, p))


def scipy_wilcoxon_available() -> bool:
    try:
        from scipy.stats import wilcoxon  # noqa: F401

        return True
    except ImportError:
        return False


def _wilcoxon_greater_p(diffs: list[float]) -> float:
    return _wilcoxon_greater_p_scipy(diffs)


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
            try:
                p_value = _wilcoxon_greater_p(diffs)
            except ImportError:
                p_value = 1.0
    elif test == "t_test":
        p_value = _one_sided_paired_t_greater(diffs)
    else:
        raise ValueError("test must be 'wilcoxon' or 't_test'")

    significant_better = bool(better and p_value < float(alpha))
    return round(float(p_value), 6), significant_better, mean_diff, better
