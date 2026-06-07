"""M18.7.1 Held-Out Calibration For LLM Confidence.

M18.7.1 owns the held-out calibration step for the
`addressee_hypothesis` and `reaction_attribution_hypothesis`
fields introduced by M18.7. The calibration harness reads
the M18.7 state surface, compares each prediction's
`confidence` against hand-annotated ground truth, and
writes the per-field calibration report to a new state
surface `state["m18_7_1_calibration"]`.

M18.7.1 is a pure analysis layer. It does NOT call the
LLM (the LLM call is the existing M18.7 conscious loop's
responsibility). It does NOT modify the M18.7 prompt
template, the M18.7 normalize contract, the M18.5 reply
policy, the M20.4 dispatcher / settler, or the M20.4
threshold constants. It does NOT auto-revise the
`M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` or
`M20_4_TIE_BREAKER_CONFIDENCE_MIN` constants — it
surfaces `candidate_*` values with a frozen caveat
attributing the decision to M20.4.

The pure functions (compute_reliability_bins,
compute_ece, compute_brier, compute_accuracy,
derive_drift_signals, recommend_thresholds) take
plain Python lists and return frozen dataclasses.
They do not import the runtime, the LLM, or any
I/O surface. The top-level
`run_m18_7_1_calibration_harness` is a thin loop
over `MVPDialogueRuntime.run_turn`; it reads
`state["m18_7_attribution_hypotheses"]` and feeds
the (prediction, ground_truth) pairs into the
pure calibrators.

Path A, M10, and `conversation_loop.py` are
explicitly out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# === Frozen v1 constants ==================================================


# M18.7.1 §1 — bounded state surface key. Single dict,
# not a rolling list (the calibration is a one-shot
# analysis per `record_m18_7_1_calibration` call).
M18_7_1_CALIBRATION_SURFACE_KEY: str = "m18_7_1_calibration"

# M18.7.1 §1 — 10 equal-width reliability bins on [0.0, 1.0].
# DECIDED 1: frozen at width 0.10 in v1.
M18_7_1_BIN_WIDTH: float = 0.10
M18_7_1_N_BINS: int = 10

# M18.7.1 §1 — frozen bin labels, lower-bound inclusive,
# upper-bound exclusive except the last bin (which is
# closed on both ends to catch confidence == 1.0).
M18_7_1_BIN_LABELS: tuple[str, ...] = (
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
)

# M18.7.1 §1 — engineering proxy label. Shared style with
# M18.7 / M20.4 `mvp_local_group_attribution`; M18.7.1
# appends the calibration scope to keep it distinct.
M18_7_1_ENGINEERING_PROXY_LABEL: str = (
    "mvp_local_group_attribution_calibration"
)

# M18.7.1 §1 — current v1 M20.4 thresholds. M18.7.1 reads
# these for the `current_*` fields in the
# threshold_recommendation. M18.7.1 NEVER mutates the
# underlying M20.4 module constants.
M18_7_1_ADMIT_MIN_CURRENT: float = 0.4
M18_7_1_TIE_BREAKER_MIN_CURRENT: float = 0.85

# M18.7.1 §1 — minimum `n_present` for emitting a non-
# `insufficient_data` drift signal or a non-`None`
# candidate threshold. DECIDED 9: frozen at 5.
M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL: int = 5

# M18.7.1 §1 — threshold-neighborhood half-width for
# the candidate threshold search. The candidate is the
# bin boundary closest to the current v1 threshold that
# has the smallest `|gap|` among populated bins within
# ± M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH of the
# current threshold.
M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH: float = 0.15

# M18.7.1 §1 — high-band lower bound for
# `overconfidence_at_high_band`.
M18_7_1_HIGH_BAND_LOWER_BOUND: float = 0.80

# M18.7.1 §1 — low-band upper bound for
# `underconfidence_at_low_band`.
M18_7_1_LOW_BAND_UPPER_BOUND: float = 0.40

# M18.7.1 §1 — gap threshold for over-/under-confidence
# drift signals. A bin's `|gap|` must exceed this to
# count as a miscalibration signal.
M18_7_1_DRIFT_GAP_THRESHOLD: float = 0.15

# M18.7.1 §1 — flat-curve `|gap|` upper bound. A
# populated bin with `|gap| < M18_7_1_FLAT_CURVE_GAP`
# contributes to the `flat_curve` signal.
M18_7_1_FLAT_CURVE_GAP: float = 0.10

# M18.7.1 §1 — minimum populated bins for the
# `flat_curve` signal to fire.
M18_7_1_FLAT_CURVE_MIN_POPULATED_BINS: int = 3

# M18.7.1 §1 — frozen drift-signal enum (DECIDED 8).
ALLOWED_M18_7_1_DRIFT_SIGNALS: frozenset[str] = frozenset({
    "overconfidence_at_high_band",
    "underconfidence_at_low_band",
    "bimodal",
    "flat_curve",
    "insufficient_data",
})

# M18.7.1 §1 — frozen assertion-kind enum (used by the
# fixture loader; matches the M18.7.1 fixture schema).
ALLOWED_M18_7_1_ASSERTION_KIND: frozenset[str] = frozenset({
    "addressee_only",
    "reaction_only",
    "both",
    "neither",
    "probe",
})

# M18.7.1 §1 — frozen confidence-band hint (design
# intent for the fixture; not parsed by the calibration
# harness).
ALLOWED_M18_7_1_CONFIDENCE_BAND: frozenset[str] = frozenset({
    "low",
    "medium",
    "high",
})

# M18.7.1 §1 — frozen `expected_confidence_band` bands
# used to test that the fixture covers all three
# confidence regions.
ALLOWED_M18_7_1_CONFIDENCE_BAND_VALUES: frozenset[str] = frozenset(
    {"low", "medium", "high"}
)

# M18.7.1 §1 — frozen caveat string. M20.4 acceptance
# may read this string verbatim.
M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT: str = (
    "decision belongs to M20.4; M18.7.1 only surfaces"
)

# M18.7.1 §1 — sentinel for "not decidable" ground
# truth. The calibration harness skips `"unknown"`
# ground truth without scoring the prediction.
M18_7_1_GT_UNKNOWN_SENTINEL: str = "unknown"

# M18.7.1 §1 — frozen ground truth keys per field.
M18_7_1_ADDRESSEE_GT_KEYS: frozenset[str] = frozenset({
    "addressed_to_assistant",
    "addressee_participant_id",
})
M18_7_1_REACTION_GT_KEYS: frozenset[str] = frozenset({
    "reaction_to_turn_id",
    "reaction_to_participant_id",
    "is_about_assistant_claim",
})


# === Frozen dataclasses ===================================================


@dataclass(frozen=True)
class BinStats:
    """A single reliability-diagram bin.

    `mean_confidence` and `accuracy` are 0.0 when
    `count == 0`. `gap` is `|accuracy - mean_confidence|`,
    0.0 when `count == 0`.
    """

    label: str
    count: int
    mean_confidence: float
    accuracy: float
    gap: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "count": int(self.count),
            "mean_confidence": round(float(self.mean_confidence), 6),
            "accuracy": round(float(self.accuracy), 6),
            "gap": round(float(self.gap), 6),
        }


@dataclass(frozen=True)
class CalibrationFieldReport:
    """Calibration report for a single M18.7 field.

    `n_total` is the total turns the field was scored on.
    `n_present` is the number of turns where the LLM
    returned a non-empty hypothesis. `n_unknown` is the
    number of turns where the ground truth was
    `"unknown"`. `n_correct` / `n_incorrect` are computed
    only over the `n_present` predictions.
    """

    n_total: int
    n_present: int
    n_unknown: int
    n_correct: int
    n_incorrect: int
    accuracy: float
    brier: float
    ece: float
    reliability_bins: list[BinStats]
    drift_signals: list[str]
    threshold_recommendation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_total": int(self.n_total),
            "n_present": int(self.n_present),
            "n_unknown": int(self.n_unknown),
            "n_correct": int(self.n_correct),
            "n_incorrect": int(self.n_incorrect),
            "accuracy": round(float(self.accuracy), 6),
            "brier": round(float(self.brier), 6),
            "ece": round(float(self.ece), 6),
            "reliability_bins": [b.to_dict() for b in self.reliability_bins],
            "drift_signals": list(self.drift_signals),
            "threshold_recommendation": {
                k: v
                for k, v in self.threshold_recommendation.items()
            },
        }


@dataclass(frozen=True)
class CalibrationHarnessReport:
    """Top-level harness report spanning both M18.7 fields.

    `fixture_name` is the path or label of the fixture
    that produced this report. `n_fixtures` is the
    number of fixture steps the runner replayed.
    """

    fixture_name: str
    n_fixtures: int
    addressee: CalibrationFieldReport
    reaction: CalibrationFieldReport
    drift_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_name": str(self.fixture_name),
            "n_fixtures": int(self.n_fixtures),
            "addressee": self.addressee.to_dict(),
            "reaction": self.reaction.to_dict(),
            "drift_signals": list(self.drift_signals),
        }


@dataclass(frozen=True)
class AddresseePrediction:
    """LLM's M18.7 `addressee_hypothesis` for one turn.

    `present` is False when the LLM returned `{}` (the
    M18.7 DECIDED 6 silent "no hypothesis" answer). When
    `present` is False, `addressed_to_assistant`,
    `participant_id`, and `confidence` are meaningless
    and may take their default values.
    """

    present: bool
    addressed_to_assistant: bool
    participant_id: str
    confidence: float


@dataclass(frozen=True)
class AddresseeGroundTruth:
    """Hand-annotated ground truth for `addressee_hypothesis`.

    `addressed_to_assistant` may be True / False / None
    (None is the `"unknown"` sentinel).
    `addressee_participant_id` is the expected
    participant_id, or `""` if M18.4 forbids, or None
    for `"unknown"`.
    """

    addressed_to_assistant: bool | None
    addressee_participant_id: str | None


@dataclass(frozen=True)
class ReactionPrediction:
    """LLM's M18.7 `reaction_attribution_hypothesis` for one turn.

    `present` is False when the LLM returned `{}`.
    """

    present: bool
    reaction_to_turn_id: str
    reaction_to_participant_id: str
    is_about_assistant_claim: bool
    confidence: float


@dataclass(frozen=True)
class ReactionGroundTruth:
    """Hand-annotated ground truth for
    `reaction_attribution_hypothesis`.

    Each field may be None (the `"unknown"` sentinel).
    """

    reaction_to_turn_id: str | None
    reaction_to_participant_id: str | None
    is_about_assistant_claim: bool | None


# === Bounded helpers ======================================================


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    """Return a float; non-numeric → default. NaN → default.

    Does NOT clamp to [0.0, 1.0]. The M18.7 normalize
    contract already clamps; the calibration harness
    assumes inputs are pre-clamped. This helper is a
    defensive guard against malformed inputs.
    """
    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float)):
        return default
    v = float(value)
    if v != v:  # NaN
        return default
    return v


def _is_known(value: Any) -> bool:
    """Return True when `value` is not the `"unknown"` sentinel
    and not None.
    """
    if value is None:
        return False
    if isinstance(value, str) and value == M18_7_1_GT_UNKNOWN_SENTINEL:
        return False
    return True


def _bin_index_for_confidence(confidence: float) -> int:
    """Return the bin index (0–9) for a confidence in [0.0, 1.0].

    Out-of-range confidences are clamped to [0.0, 1.0] at
    the bin-assignment level. The last bin (index 9) is
    closed on the right so confidence == 1.0 lands in
    the last bin.
    """
    if confidence < 0.0:
        return 0
    if confidence >= 1.0:
        return M18_7_1_N_BINS - 1
    # floor(confidence / 0.10) is the bin index, with
    # confidence == 0.0 → 0, confidence == 0.05 → 0,
    # confidence == 0.10 → 1, ..., confidence == 0.99 → 9.
    idx = int(math.floor(confidence / M18_7_1_BIN_WIDTH))
    if idx < 0:
        return 0
    if idx >= M18_7_1_N_BINS:
        return M18_7_1_N_BINS - 1
    return idx


# === Pure functions: bin computation ======================================


def compute_reliability_bins(
    confidences: Sequence[float],
    correct_flags: Sequence[bool],
) -> list[BinStats]:
    """Compute per-bin statistics for a reliability diagram.

    `confidences` and `correct_flags` must be the same
    length; an empty input yields 10 all-zero bins. Each
    bin is labeled per `M18_7_1_BIN_LABELS`. The last bin
    is closed on the right so `confidence == 1.0` lands
    in bin index 9 (label "0.90-1.00").
    """
    n = len(confidences)
    if n != len(correct_flags):
        # Defensive: mismatched lengths → fall back to
        # all-zero bins. The harness should never
        # construct mismatched lists.
        return _empty_bins()

    counts = [0] * M18_7_1_N_BINS
    conf_sums = [0.0] * M18_7_1_N_BINS
    correct_sums = [0.0] * M18_7_1_N_BINS

    for conf, correct in zip(confidences, correct_flags):
        idx = _bin_index_for_confidence(_bounded_float(conf))
        counts[idx] += 1
        conf_sums[idx] += _bounded_float(conf)
        correct_sums[idx] += 1.0 if bool(correct) else 0.0

    out: list[BinStats] = []
    for i, label in enumerate(M18_7_1_BIN_LABELS):
        count = counts[i]
        if count > 0:
            mean_conf = conf_sums[i] / count
            accuracy = correct_sums[i] / count
            gap = abs(accuracy - mean_conf)
        else:
            mean_conf = 0.0
            accuracy = 0.0
            gap = 0.0
        out.append(
            BinStats(
                label=label,
                count=count,
                mean_confidence=mean_conf,
                accuracy=accuracy,
                gap=gap,
            )
        )
    return out


def _empty_bins() -> list[BinStats]:
    return [
        BinStats(
            label=label,
            count=0,
            mean_confidence=0.0,
            accuracy=0.0,
            gap=0.0,
        )
        for label in M18_7_1_BIN_LABELS
    ]


# === Pure functions: aggregate metrics ====================================


def compute_ece(bins: Sequence[BinStats]) -> float:
    """Compute Expected Calibration Error (Guo et al. 2017).

    `ECE = sum_b (|bin_b| / N) * |acc_b - conf_b|`

    `N` is the total count across all bins. Empty
    input → 0.0. The result is in [0.0, 1.0].
    """
    n = sum(b.count for b in bins)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b.count > 0:
            ece += (b.count / n) * b.gap
    return ece


def compute_brier(
    confidences: Sequence[float],
    correct_flags: Sequence[bool],
) -> float:
    """Compute the Brier score (mean squared error).

    `Brier = (1/N) * sum_i (confidence_i - correct_i)^2`

    `correct_i` is 1.0 when `correct_flags[i]` is True
    and 0.0 otherwise. Empty input → 0.0. The result
    is in [0.0, 1.0] for `confidence ∈ [0.0, 1.0]`.
    """
    n = len(confidences)
    if n == 0 or n != len(correct_flags):
        return 0.0
    s = 0.0
    for conf, correct in zip(confidences, correct_flags):
        c = _bounded_float(conf)
        target = 1.0 if bool(correct) else 0.0
        s += (c - target) ** 2
    return s / n


def compute_accuracy(correct_flags: Sequence[bool]) -> float:
    """Mean of `correct_flags`; empty input → 0.0."""
    n = len(correct_flags)
    if n == 0:
        return 0.0
    return sum(1.0 for c in correct_flags if bool(c)) / n


# === Pure functions: drift signal taxonomy ================================


def derive_drift_signals(
    bins: Sequence[BinStats],
    n_present: int,
    *,
    min_present: int = M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL,
) -> list[str]:
    """Return a list of frozen-enum drift signals.

    Order of evaluation is deterministic: insufficient_data
    is checked first (it short-circuits the rest). The
    remaining signals are returned in the order:
    overconfidence_at_high_band,
    underconfidence_at_low_band, bimodal, flat_curve.
    """
    signals: list[str] = []

    if n_present < int(min_present):
        signals.append("insufficient_data")
        return signals

    populated = [b for b in bins if b.count > 0]

    # overconfidence_at_high_band
    high_bins = [
        b
        for b in populated
        if b.mean_confidence >= M18_7_1_HIGH_BAND_LOWER_BOUND
        and b.gap > M18_7_1_DRIFT_GAP_THRESHOLD
        and b.accuracy < b.mean_confidence
    ]
    if high_bins:
        signals.append("overconfidence_at_high_band")

    # underconfidence_at_low_band
    low_bins = [
        b
        for b in populated
        if b.mean_confidence <= M18_7_1_LOW_BAND_UPPER_BOUND
        and b.gap > M18_7_1_DRIFT_GAP_THRESHOLD
        and b.accuracy > b.mean_confidence
    ]
    if low_bins:
        signals.append("underconfidence_at_low_band")

    # bimodal: highest bin (>= 0.80) AND lowest bin
    # (<= 0.30) both have count > 0, AND middle bins
    # (0.30, 0.80) have count == 0 for at least 4 of
    # the 5 middle bins. The middle band is identified
    # by bin INDEX (not mean_confidence), because
    # empty bins have mean_confidence == 0.0 which
    # would otherwise be excluded from the "middle"
    # range.
    high_indices = {8, 9}  # bin labels 0.80-0.90, 0.90-1.00
    low_indices = {0, 1, 2}  # bin labels 0.00-0.10, 0.10-0.20, 0.20-0.30
    middle_indices = {3, 4, 5, 6, 7}  # 5 middle bins
    has_high = any(
        b.count > 0 and i in high_indices
        for i, b in enumerate(bins)
    )
    has_low = any(
        b.count > 0 and i in low_indices
        for i, b in enumerate(bins)
    )
    middle_empty = sum(
        1
        for i, b in enumerate(bins)
        if i in middle_indices and b.count == 0
    )
    if has_high and has_low and middle_empty >= 4:
        signals.append("bimodal")

    # flat_curve: every populated bin has
    # |gap| < M18_7_1_FLAT_CURVE_GAP, AND at least
    # M18_7_1_FLAT_CURVE_MIN_POPULATED_BINS are populated.
    if (
        len(populated) >= M18_7_1_FLAT_CURVE_MIN_POPULATED_BINS
        and all(b.gap < M18_7_1_FLAT_CURVE_GAP for b in populated)
    ):
        signals.append("flat_curve")

    return signals


# === Pure functions: threshold recommendation =============================


def recommend_thresholds(
    bins: Sequence[BinStats],
    *,
    current_admit_min: float = M18_7_1_ADMIT_MIN_CURRENT,
    current_tie_breaker_min: float = M18_7_1_TIE_BREAKER_MIN_CURRENT,
    min_present: int = M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL,
) -> dict[str, object]:
    """Return the data-driven threshold recommendation.

    The function NEVER mutates the M20.4 module
    constants. It only surfaces `candidate_admit_min` and
    `candidate_tie_breaker_min` for M20.4's review.

    The candidate is the bin boundary in
    `{0.1, 0.2, ..., 0.9}` closest to the current
    threshold that has the smallest `|gap|` among
    populated bins within
    `± M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH` of the
    threshold. If no populated bin falls in the
    neighborhood, the candidate is `None`.

    When `n_present < min_present`, both candidates
    are `None` and the caller is expected to include
    `insufficient_data` in the field's drift_signals.
    """
    n_present = sum(b.count for b in bins)
    if n_present < int(min_present):
        return {
            "current_admit_min": float(current_admit_min),
            "current_tie_breaker_min": float(current_tie_breaker_min),
            "candidate_admit_min": None,
            "candidate_tie_breaker_min": None,
            "caveat": M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT,
        }

    candidate_admit = _pick_candidate(
        bins, target=float(current_admit_min)
    )
    candidate_tb = _pick_candidate(
        bins, target=float(current_tie_breaker_min)
    )

    return {
        "current_admit_min": float(current_admit_min),
        "current_tie_breaker_min": float(current_tie_breaker_min),
        "candidate_admit_min": candidate_admit,
        "candidate_tie_breaker_min": candidate_tb,
        "caveat": M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT,
    }


def _pick_candidate(
    bins: Sequence[BinStats],
    *,
    target: float,
) -> float | None:
    """Pick the bin boundary in {0.1, ..., 0.9} closest to
    `target` whose neighborhood contains a populated bin
    with the smallest `|gap|`.

    Returns the bin boundary, not the bin's
    mean_confidence (so the candidate is a clean
    round number, not a noisy empirical estimate).
    """
    neighborhood_low = target - M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH
    neighborhood_high = target + M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH

    # Find the populated bin with the smallest gap in
    # the neighborhood.
    best_gap = None
    best_boundary = None
    for b in bins:
        if b.count == 0:
            continue
        if not (neighborhood_low <= b.mean_confidence <= neighborhood_high):
            continue
        if best_gap is None or b.gap < best_gap:
            best_gap = b.gap
            # Map the bin's mean_confidence to the
            # nearest bin boundary in {0.1, ..., 0.9}.
            best_boundary = _nearest_bin_boundary(b.mean_confidence)

    return best_boundary


def _nearest_bin_boundary(mean_confidence: float) -> float:
    """Round `mean_confidence` to the nearest bin boundary
    in {0.1, 0.2, ..., 0.9}.

    `mean_confidence == 0.05` → `0.1` (round-half-up);
    `mean_confidence == 0.50` → `0.5`; `mean_confidence == 0.95`
    → `0.9` (clamped to the largest boundary). The
    boundaries are chosen to be the round tenths that
    fit `[0.0, 1.0]` cleanly.
    """
    boundaries = [round(i * M18_7_1_BIN_WIDTH, 6) for i in range(1, M18_7_1_N_BINS)]
    # `boundaries = [0.1, 0.2, ..., 0.9]`
    best = min(boundaries, key=lambda b: abs(b - mean_confidence))
    return float(best)


# === Field-specific calibrators ===========================================


def calibrate_addressee_field(
    predictions: Sequence[AddresseePrediction],
    ground_truth: Sequence[AddresseeGroundTruth],
) -> CalibrationFieldReport:
    """Calibrate the M18.7 `addressee_hypothesis` field.

    Per-turn outcome rules (DECIDED 5 / 6):

    - prediction empty AND ground truth is "unknown" →
      `n_unknown++`; the prediction is not scored.
    - prediction empty AND ground truth is decidable →
      `n_incorrect++`; confidence is treated as 0.0.
    - prediction non-empty AND ground truth is
      "unknown" → `n_unknown++`; the prediction is
      not scored.
    - prediction non-empty AND ground truth is
      decidable → `n_correct++` if
      `prediction.addressed_to_assistant ==
      ground_truth.addressed_to_assistant`, else
      `n_incorrect++`.

    Outcome judgment is strict bool equality
    (`addressed_to_assistant` only). The
    `addressee_participant_id` field is recorded for
    traceability in the report's `reliability_bins` but
    is NOT used for outcome scoring.
    """
    n_total = len(predictions)
    if n_total != len(ground_truth):
        n_total = min(n_total, len(ground_truth))

    confidences: list[float] = []
    correct_flags: list[bool] = []
    n_present = 0
    n_unknown = 0
    n_correct = 0
    n_incorrect = 0

    for pred, gt in zip(predictions, ground_truth):
        if not _is_known(gt.addressed_to_assistant):
            n_unknown += 1
            if pred.present:
                # Non-empty prediction against "unknown"
                # ground truth → skip; do not score.
                continue
            # Both empty + "unknown" → skip.
            continue
        if not pred.present:
            # Empty prediction against decidable ground
            # truth → incorrect; confidence = 0.0.
            n_incorrect += 1
            n_present += 1
            confidences.append(0.0)
            correct_flags.append(False)
            continue
        n_present += 1
        is_correct = bool(
            pred.addressed_to_assistant == gt.addressed_to_assistant
        )
        if is_correct:
            n_correct += 1
            correct_flags.append(True)
        else:
            n_incorrect += 1
            correct_flags.append(False)
        confidences.append(_bounded_float(pred.confidence))

    bins = compute_reliability_bins(confidences, correct_flags)
    accuracy = compute_accuracy(correct_flags)
    brier = compute_brier(confidences, correct_flags)
    ece = compute_ece(bins)
    drift = derive_drift_signals(bins, n_present)
    threshold_rec = recommend_thresholds(bins)

    return CalibrationFieldReport(
        n_total=n_total,
        n_present=n_present,
        n_unknown=n_unknown,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=accuracy,
        brier=brier,
        ece=ece,
        reliability_bins=list(bins),
        drift_signals=drift,
        threshold_recommendation=threshold_rec,
    )


def calibrate_reaction_field(
    predictions: Sequence[ReactionPrediction],
    ground_truth: Sequence[ReactionGroundTruth],
) -> CalibrationFieldReport:
    """Calibrate the M18.7
    `reaction_attribution_hypothesis` field.

    Per-turn outcome rules (mirror `calibrate_addressee_field`):

    - prediction empty AND ground truth is "unknown" →
      `n_unknown++`; not scored.
    - prediction empty AND ground truth is decidable →
      `n_incorrect++`; confidence = 0.0.
    - prediction non-empty AND ground truth is
      "unknown" → `n_unknown++`; not scored.
    - prediction non-empty AND ground truth is
      decidable → `n_correct++` if
      `prediction.reaction_to_turn_id ==
      ground_truth.reaction_to_turn_id`, else
      `n_incorrect++`.

    Outcome judgment is strict string equality on
    `reaction_to_turn_id`. The
    `reaction_to_participant_id` and
    `is_about_assistant_claim` fields are recorded for
    traceability in the per-turn ground truth but are
    NOT used for outcome scoring.
    """
    n_total = len(predictions)
    if n_total != len(ground_truth):
        n_total = min(n_total, len(ground_truth))

    confidences: list[float] = []
    correct_flags: list[bool] = []
    n_present = 0
    n_unknown = 0
    n_correct = 0
    n_incorrect = 0

    for pred, gt in zip(predictions, ground_truth):
        if not _is_known(gt.reaction_to_turn_id):
            n_unknown += 1
            if pred.present:
                continue
            continue
        if not pred.present:
            n_incorrect += 1
            n_present += 1
            confidences.append(0.0)
            correct_flags.append(False)
            continue
        n_present += 1
        is_correct = bool(
            pred.reaction_to_turn_id == gt.reaction_to_turn_id
        )
        if is_correct:
            n_correct += 1
            correct_flags.append(True)
        else:
            n_incorrect += 1
            correct_flags.append(False)
        confidences.append(_bounded_float(pred.confidence))

    bins = compute_reliability_bins(confidences, correct_flags)
    accuracy = compute_accuracy(correct_flags)
    brier = compute_brier(confidences, correct_flags)
    ece = compute_ece(bins)
    drift = derive_drift_signals(bins, n_present)
    threshold_rec = recommend_thresholds(bins)

    return CalibrationFieldReport(
        n_total=n_total,
        n_present=n_present,
        n_unknown=n_unknown,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=accuracy,
        brier=brier,
        ece=ece,
        reliability_bins=list(bins),
        drift_signals=drift,
        threshold_recommendation=threshold_rec,
    )


# === Top-level runner =====================================================


def run_m18_7_1_calibration_harness(
    *,
    runtime: Any,
    fixture: Sequence[Mapping[str, Any]],
    fixture_name: str = "<inline>",
    now_base: int = 100000,
    time_step: int = 60,
    at: str = "2026-06-07T00:00:00Z",
    addressee_field: str = "addressee_hypothesis",
    reaction_field: str = "reaction_attribution_hypothesis",
) -> CalibrationHarnessReport:
    """Replay `fixture` through `runtime` and compute
    per-field calibration reports.

    The runner does NOT call the LLM directly; the LLM
    is invoked by `runtime.run_turn` (which uses the
    runtime's configured `llm` attribute). The runner
    observes the M18.7 state surface after each turn
    to extract the LLM's emitted `confidence` and
    `addressed_to_assistant` / `reaction_to_turn_id`
    predictions.

    The runner is deterministic for a deterministic LLM
    (e.g., `FakeJSONLLM` in tests). For a non-
    deterministic LLM, the runner is non-deterministic.

    The runner does NOT mutate the runtime's state
    beyond what `run_turn` itself writes. It does NOT
    mutate M20.4 module constants.
    """
    addressee_predictions: list[AddresseePrediction] = []
    addressee_ground_truth: list[AddresseeGroundTruth] = []
    reaction_predictions: list[ReactionPrediction] = []
    reaction_ground_truth: list[ReactionGroundTruth] = []

    for i, step in enumerate(fixture):
        # Replay the step through the runtime.
        runtime.run_turn(
            step["text"],
            speaker_name=step.get("speaker_name", ""),
            turn_index=int(step.get("turn_index", i)),
            now=now_base + i * time_step,
            group_turn_envelope=dict(step.get("group_turn_envelope", {})),
        )

        # Read the M18.7 state surface.
        state_dict = runtime.store.load()
        surface = state_dict.get("m18_7_attribution_hypotheses", [])
        current_turn_index = int(step.get("turn_index", i))
        current_entries = [
            e
            for e in surface
            if e.get("turn_index") == current_turn_index
        ]

        # Parse the LLM predictions from the state
        # surface entries. The surface holds at most
        # 2 entries per turn (addressee + reaction).
        addressee_entry = next(
            (
                e
                for e in current_entries
                if e.get("kind") == "addressee"
            ),
            None,
        )
        reaction_entry = next(
            (
                e
                for e in current_entries
                if e.get("kind") == "reaction"
            ),
            None,
        )

        addressee_predictions.append(
            _addressee_prediction_from_entry(addressee_entry)
        )
        addressee_ground_truth.append(
            _addressee_ground_truth_from_fixture(
                step.get("ground_truth", {})
            )
        )
        reaction_predictions.append(
            _reaction_prediction_from_entry(reaction_entry)
        )
        reaction_ground_truth.append(
            _reaction_ground_truth_from_fixture(
                step.get("ground_truth", {})
            )
        )

    addressee_report = calibrate_addressee_field(
        addressee_predictions, addressee_ground_truth
    )
    reaction_report = calibrate_reaction_field(
        reaction_predictions, reaction_ground_truth
    )
    union_drift = _union_preserving_order(
        addressee_report.drift_signals, reaction_report.drift_signals
    )

    return CalibrationHarnessReport(
        fixture_name=fixture_name,
        n_fixtures=len(fixture),
        addressee=addressee_report,
        reaction=reaction_report,
        drift_signals=union_drift,
    )


def _addressee_prediction_from_entry(
    entry: Mapping[str, Any] | None,
) -> AddresseePrediction:
    """Convert a state-surface entry to an AddresseePrediction.

    A `None` entry means the M18.7 state surface had no
    addressee entry for this turn (the LLM returned
    `{}`). In that case the prediction is `present=False`.
    """
    if not entry:
        return AddresseePrediction(
            present=False,
            addressed_to_assistant=False,
            participant_id="",
            confidence=0.0,
        )
    return AddresseePrediction(
        present=True,
        addressed_to_assistant=bool(entry.get("addressed_to_assistant", False)),
        participant_id=str(entry.get("participant_id", "") or ""),
        confidence=float(entry.get("confidence", 0.0) or 0.0),
    )


def _addressee_ground_truth_from_fixture(
    raw: Mapping[str, Any],
) -> AddresseeGroundTruth:
    """Convert a fixture `ground_truth` dict to an
    AddresseeGroundTruth.

    `"unknown"` and `None` are both treated as the
    `"unknown"` sentinel.
    """
    raw_addr = raw.get("addressed_to_assistant")
    raw_pid = raw.get("addressee_participant_id")
    return AddresseeGroundTruth(
        addressed_to_assistant=_normalize_gt_bool(raw_addr),
        addressee_participant_id=_normalize_gt_str(raw_pid),
    )


def _reaction_prediction_from_entry(
    entry: Mapping[str, Any] | None,
) -> ReactionPrediction:
    if not entry:
        return ReactionPrediction(
            present=False,
            reaction_to_turn_id="",
            reaction_to_participant_id="",
            is_about_assistant_claim=False,
            confidence=0.0,
        )
    return ReactionPrediction(
        present=True,
        reaction_to_turn_id=str(entry.get("reaction_to_turn_id", "") or ""),
        reaction_to_participant_id=str(
            entry.get("reaction_to_participant_id", "") or ""
        ),
        is_about_assistant_claim=bool(
            entry.get("is_about_assistant_claim", False)
        ),
        confidence=float(entry.get("confidence", 0.0) or 0.0),
    )


def _reaction_ground_truth_from_fixture(
    raw: Mapping[str, Any],
) -> ReactionGroundTruth:
    raw_turn = raw.get("reaction_to_turn_id")
    raw_pid = raw.get("reaction_to_participant_id")
    raw_about = raw.get("is_about_assistant_claim")
    return ReactionGroundTruth(
        reaction_to_turn_id=_normalize_gt_str(raw_turn),
        reaction_to_participant_id=_normalize_gt_str(raw_pid),
        is_about_assistant_claim=_normalize_gt_bool(raw_about),
    )


def _normalize_gt_bool(value: Any) -> bool | None:
    """Return True / False / None (the `"unknown"` sentinel)."""
    if value is None:
        return None
    if isinstance(value, str) and value == M18_7_1_GT_UNKNOWN_SENTINEL:
        return None
    if isinstance(value, bool):
        return value
    return None  # defensive default for non-bool scalars


def _normalize_gt_str(value: Any) -> str | None:
    """Return a string or None (the `"unknown"` sentinel)."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == M18_7_1_GT_UNKNOWN_SENTINEL:
            return None
        return value
    return None  # defensive default for non-strings


def _union_preserving_order(*lists: Sequence[str]) -> list[str]:
    """Union of string lists, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


# === State surface writer =================================================


def record_m18_7_1_calibration(
    state: dict,
    harness_report: CalibrationHarnessReport,
    *,
    at: str,
) -> None:
    """Write the calibration report to
    `state["m18_7_1_calibration"]`.

    The state surface is a single dict, NOT a rolling
    list. Each call overwrites the previous value. The
    M18.7 surface is a rolling list of per-turn
    hypotheses; the M18.7.1 surface is a one-shot
    calibration analysis.

    The function does NOT mutate M20.4 module constants;
    it only writes the `threshold_recommendation` field
    with the frozen caveat string.
    """
    if not isinstance(state, dict):
        return
    threshold_rec = {
        "current_admit_min": float(
            harness_report.addressee.threshold_recommendation.get(
                "current_admit_min",
                M18_7_1_ADMIT_MIN_CURRENT,
            )
        ),
        "current_tie_breaker_min": float(
            harness_report.addressee.threshold_recommendation.get(
                "current_tie_breaker_min",
                M18_7_1_TIE_BREAKER_MIN_CURRENT,
            )
        ),
        "candidate_admit_min": harness_report.addressee.threshold_recommendation.get(
            "candidate_admit_min"
        ),
        "candidate_tie_breaker_min": harness_report.reaction.threshold_recommendation.get(
            "candidate_tie_breaker_min"
        ),
        "caveat": M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT,
    }
    state[M18_7_1_CALIBRATION_SURFACE_KEY] = {
        "last_run_at": str(at),
        "fixture_name": str(harness_report.fixture_name),
        "n_fixtures": int(harness_report.n_fixtures),
        "addressee": harness_report.addressee.to_dict(),
        "reaction": harness_report.reaction.to_dict(),
        "drift_signals": list(harness_report.drift_signals),
        "threshold_recommendation": threshold_rec,
        "engineering_proxy_label": M18_7_1_ENGINEERING_PROXY_LABEL,
    }


# === Fixture schema validation ============================================


def validate_calibration_fixture_shape(
    fixture: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate a calibration fixture's shape.

    Returns a list of error messages; empty list means
    the fixture is well-formed. The function does NOT
    parse the `note` field; it only checks structural
    fields.
    """
    errors: list[str] = []
    if not isinstance(fixture, (list, tuple)):
        return [f"fixture must be a list, got {type(fixture).__name__}"]
    for i, step in enumerate(fixture):
        if not isinstance(step, Mapping):
            errors.append(f"step[{i}] must be a mapping, got {type(step).__name__}")
            continue
        # Required text / envelope.
        if "text" not in step:
            errors.append(f"step[{i}] missing 'text'")
        if "group_turn_envelope" not in step:
            errors.append(f"step[{i}] missing 'group_turn_envelope'")
        # Optional but expected.
        assertion_kind = step.get("assertion_kind", "")
        if assertion_kind not in ALLOWED_M18_7_1_ASSERTION_KIND:
            errors.append(
                f"step[{i}] 'assertion_kind' must be one of "
                f"{sorted(ALLOWED_M18_7_1_ASSERTION_KIND)}; "
                f"got {assertion_kind!r}"
            )
        band = step.get("expected_confidence_band", "")
        if band not in ALLOWED_M18_7_1_CONFIDENCE_BAND_VALUES:
            errors.append(
                f"step[{i}] 'expected_confidence_band' must be one of "
                f"{sorted(ALLOWED_M18_7_1_CONFIDENCE_BAND_VALUES)}; "
                f"got {band!r}"
            )
        # Ground truth shape.
        gt = step.get("ground_truth", {})
        if not isinstance(gt, Mapping):
            errors.append(f"step[{i}] 'ground_truth' must be a mapping")
            continue
        for key in M18_7_1_ADDRESSEE_GT_KEYS | M18_7_1_REACTION_GT_KEYS:
            if key not in gt:
                errors.append(f"step[{i}] ground_truth missing key {key!r}")
    return errors


__all__ = [
    "ALLOWED_M18_7_1_ASSERTION_KIND",
    "ALLOWED_M18_7_1_CONFIDENCE_BAND",
    "ALLOWED_M18_7_1_CONFIDENCE_BAND_VALUES",
    "ALLOWED_M18_7_1_DRIFT_SIGNALS",
    "AddresseeGroundTruth",
    "AddresseePrediction",
    "BinStats",
    "CalibrationFieldReport",
    "CalibrationHarnessReport",
    "M18_7_1_ADMIT_MIN_CURRENT",
    "M18_7_1_BIN_LABELS",
    "M18_7_1_BIN_WIDTH",
    "M18_7_1_CALIBRATION_SURFACE_KEY",
    "M18_7_1_DRIFT_GAP_THRESHOLD",
    "M18_7_1_ENGINEERING_PROXY_LABEL",
    "M18_7_1_FLAT_CURVE_GAP",
    "M18_7_1_FLAT_CURVE_MIN_POPULATED_BINS",
    "M18_7_1_GT_UNKNOWN_SENTINEL",
    "M18_7_1_HIGH_BAND_LOWER_BOUND",
    "M18_7_1_LOW_BAND_UPPER_BOUND",
    "M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL",
    "M18_7_1_N_BINS",
    "M18_7_1_TIE_BREAKER_MIN_CURRENT",
    "M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH",
    "M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT",
    "ReactionGroundTruth",
    "ReactionPrediction",
    "calibrate_addressee_field",
    "calibrate_reaction_field",
    "compute_accuracy",
    "compute_brier",
    "compute_ece",
    "compute_reliability_bins",
    "derive_drift_signals",
    "record_m18_7_1_calibration",
    "recommend_thresholds",
    "run_m18_7_1_calibration_harness",
    "validate_calibration_fixture_shape",
]
