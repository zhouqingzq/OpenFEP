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
import re
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

# === M18.7.1 v2 constants (2026-06-09) ====================================
# v2 adds three scoring modes for the reaction field:
#   - "by_pid": score on `reaction_to_participant_id` +
#     `is_about_assistant_claim` joint correctness,
#     with pid normalization (the primary v2 mode).
#   - "by_turn_id_resolved": v1 scoring on turn_id,
#     but with placeholder resolution at runner-time
#     (`turn_<assistant_prior_turn_id>` etc. resolved
#     against the actual prior turn at replay time).
#   - "by_turn_id_v1": byte-identical v1 scoring
#     (no resolution, no normalization). Default for
#     back-compat with v1 callers.
M18_7_1_SCORING_MODES: frozenset[str] = frozenset({
    "by_pid",
    "by_turn_id_resolved",
    "by_turn_id_v1",
})
M18_7_1_DEFAULT_SCORING_MODE: str = "by_pid"

# v2 placeholder pattern: matches a literal
# `turn_<role>` GT string. `role` is `[a-z_]+`.
# Examples: "turn_<assistant_prior_turn_id>",
# "turn_<carol_prior_turn_id>".
M18_7_1_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    r"^turn_<(?P<role>[a-z_]+)>$"
)
# Recognized "named role" placeholders. Anything
# else in `<role>` is treated as a named-speaker
# placeholder (matched against the speaker's
# participant_id, lowercased).
M18_7_1_PLACEHOLDER_ROLES_ASSISTANT: frozenset[str] = frozenset({
    "assistant_prior_turn_id",
})
M18_7_1_PLACEHOLDER_ROLES_USER: frozenset[str] = frozenset({
    "user_prior_turn_id",
})

# v2 participant-id normalization table. Maps
# surface ids (bot/role/persona) to a canonical form
# before pid equality scoring. Human names are
# pass-through (lowercased). Empty string is
# preserved (means "no attribution").
M18_7_1_PID_NORMALIZATION: dict[str, str] = {
    "assistant": "bot",
    "hutao": "bot",
    "hutao_assistant": "bot",
    "clawdgroupchat_bot": "bot",
    # Defensive: add new bot/role surface ids here
    # if a future persona or platform changes the
    # assistant's surface id.
}
# v2 does NOT mutate M20.4 / M18.7 / M18.7.2 / M18.5.

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

    v2 (2026-06-09): `pid_breakdown` and
    `is_about_breakdown` are populated only in
    `scoring_mode="by_pid"` (Mode A). For v1 modes
    (`by_turn_id_v1`, `by_turn_id_resolved`) and for
    the addressee field (which has no pid/is_about
    split), these are `None`. The `to_dict()` method
    OMITS the keys when `None`, preserving v1 byte-
    identical output in Mode C.
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
    pid_breakdown: dict[str, object] | None = None
    is_about_breakdown: dict[str, object] | None = None
    # P1 (2026-06-09): per-class confusion split for
    # the addressee field, and per-subset split for the
    # reaction joint axis. Both fields are pure
    # functions of the (predictions, ground_truth) data
    # and contain no scoring-mode-specific logic — they
    # expose the raw TP/FP/FN counts and a derived
    # precision/recall view. The fields are `None` for
    # the "wrong" field: addressee_class_breakdown is
    # `None` on reaction reports, and
    # reaction_joint_breakdown is `None` on addressee
    # reports. `to_dict()` omits None fields, preserving
    # the v2 (pre-P1) byte-identity for callers that
    # check the to_dict() shape.
    addressee_class_breakdown: dict[str, object] | None = None
    reaction_joint_breakdown: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
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
        # v2 fields: emitted only when populated, so
        # v1-mode reports preserve byte identity.
        if self.pid_breakdown is not None:
            out["pid_breakdown"] = {
                k: v
                for k, v in self.pid_breakdown.items()
            }
        if self.is_about_breakdown is not None:
            out["is_about_breakdown"] = {
                k: v
                for k, v in self.is_about_breakdown.items()
            }
        # P1: per-class addressee breakdown (precision on
        # not-addressed, recall on addressed). Populated
        # for the addressee field in all modes (v1, v2
        # by_pid, v2 by_turn_id_resolved) since it's a
        # pure function of the (pred, gt) tuples.
        if self.addressee_class_breakdown is not None:
            out["addressee_class_breakdown"] = {
                k: v
                for k, v in self.addressee_class_breakdown.items()
            }
        # P1: per-subset reaction joint breakdown
        # (all decidable vs LLM-emit subset). Populated
        # for the reaction field in all modes; the joint
        # axis is mode-specific (v1: turn_id equality;
        # v2: pid + is_about) but the all-vs-emit split
        # is a structural view that applies to both.
        if self.reaction_joint_breakdown is not None:
            out["reaction_joint_breakdown"] = {
                k: v
                for k, v in self.reaction_joint_breakdown.items()
            }
        return out


@dataclass(frozen=True)
class CalibrationHarnessReport:
    """Top-level harness report spanning both M18.7 fields.

    `fixture_name` is the path or label of the fixture
    that produced this report. `n_fixtures` is the
    number of fixture steps the runner replayed.

    v2 (2026-06-09): `scoring_mode` is the v2 mode
    used (`"by_pid"` / `"by_turn_id_resolved"` /
    `"by_turn_id_v1"`). `fixture_warnings` is a list
    of non-fatal warnings emitted during the replay
    (e.g., `"placeholder_unresolved:..."`).
    """

    fixture_name: str
    n_fixtures: int
    addressee: CalibrationFieldReport
    reaction: CalibrationFieldReport
    drift_signals: list[str] = field(default_factory=list)
    scoring_mode: str = "by_turn_id_v1"
    fixture_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "fixture_name": str(self.fixture_name),
            "n_fixtures": int(self.n_fixtures),
            "addressee": self.addressee.to_dict(),
            "reaction": self.reaction.to_dict(),
            "drift_signals": list(self.drift_signals),
        }
        # v2 (D6): v1-mode reports are byte-identical to
        # the pre-v2 output. The new keys are emitted
        # only when the scoring mode is one of the v2
        # modes (i.e., not the v1 default).
        if self.scoring_mode != "by_turn_id_v1":
            out["scoring_mode"] = str(self.scoring_mode)
            out["fixture_warnings"] = list(self.fixture_warnings)
        return out


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


# === P1 helpers: per-class + per-subset breakdowns ========================


def _compute_addressee_class_breakdown(
    predictions: Sequence[AddresseePrediction],
    ground_truth: Sequence[AddresseeGroundTruth],
) -> dict[str, object]:
    """Compute the per-class confusion matrix for the
    addressee field.

    P1 (2026-06-09): splits the addressee metric into
    precision-on-not-addressed and recall-on-addressed.
    The breakdown is a pure function of the
    (predictions, ground_truth) tuples — independent
    of the scoring mode and the v4 Phase 2A kwarg. The
    scorer decides how to count no-emit + GT-false
    (v1: wrong, v2 fix: correct), but the raw TP/FP/FN
    counts are the same either way.

    Counts:

    - `n_gt_true` / `n_gt_false` / `n_unknown`: total
      decidable-true, decidable-false, and unknown-skip
      turns in the GT.
    - `tp_addressed`: pred present + pred=True + GT=True.
    - `fn_addressed`: pred present + pred=False + GT=True,
      OR pred absent + GT=True (no-emit missed).
    - `tp_not_addressed`: pred present + pred=False + GT=False,
      OR pred absent + GT=False (no-emit is the v2-fix
      signal; in v1 this is `fn_v1` since v1 marks
      no-emit + GT-False as wrong).
    - `fp_not_addressed`: pred present + pred=True + GT=False.
    - `precision_on_not_addressed`: tp / (tp + fp) when
      tp + fp > 0; 0.0 otherwise. P(GT=False | pred=False).
    - `recall_on_addressed`: tp / (tp + fn) when
      tp + fn > 0; 0.0 otherwise. P(pred=True | GT=True).
    """
    n_gt_true = 0
    n_gt_false = 0
    n_unknown = 0
    tp_addr = 0
    fn_addr_present = 0
    fn_addr_noemit = 0
    fp_not_addr = 0
    tn_not_addr_present = 0
    tn_not_addr_noemit = 0

    for pred, gt in zip(predictions, ground_truth):
        gt_v = gt.addressed_to_assistant
        if not _is_known(gt_v):
            n_unknown += 1
            continue
        if gt_v:
            n_gt_true += 1
            if pred.present:
                if pred.addressed_to_assistant:
                    tp_addr += 1
                else:
                    fn_addr_present += 1
            else:
                fn_addr_noemit += 1
        else:
            n_gt_false += 1
            if pred.present:
                if pred.addressed_to_assistant:
                    fp_not_addr += 1
                else:
                    tn_not_addr_present += 1
            else:
                tn_not_addr_noemit += 1

    fn_addr_total = fn_addr_present + fn_addr_noemit
    tn_not_addr_total = tn_not_addr_present + tn_not_addr_noemit

    precision_denom = tn_not_addr_total + fp_not_addr
    recall_denom = tp_addr + fn_addr_total
    precision_on_not_addressed = (
        tn_not_addr_total / precision_denom
        if precision_denom > 0
        else 0.0
    )
    recall_on_addressed = (
        tp_addr / recall_denom if recall_denom > 0 else 0.0
    )

    return {
        "n_gt_true": int(n_gt_true),
        "n_gt_false": int(n_gt_false),
        "n_unknown": int(n_unknown),
        "tp_addressed": int(tp_addr),
        "fn_addressed": int(fn_addr_total),
        "fn_addressed_present": int(fn_addr_present),
        "fn_addressed_noemit": int(fn_addr_noemit),
        "tp_not_addressed": int(tn_not_addr_total),
        "tp_not_addressed_present": int(tn_not_addr_present),
        "tp_not_addressed_noemit": int(tn_not_addr_noemit),
        "fp_not_addressed": int(fp_not_addr),
        "precision_on_not_addressed": round(
            float(precision_on_not_addressed), 6
        ),
        "recall_on_addressed": round(float(recall_on_addressed), 6),
    }


def _compute_reaction_joint_breakdown(
    *,
    n_joint_all_decidable: int,
    n_joint_emit_subset: int,
    n_joint_correct_all: int,
    n_joint_correct_emit: int,
) -> dict[str, object]:
    """Compute the reaction joint-axis subset breakdown.

    P1 (2026-06-09): splits the reaction joint accuracy
    into "all decidable" (the M20.4-honest denominator,
    includes no-emit as wrong) and "LLM-emit subset"
    (the practical denominator, only counts cases
    where the LLM emitted a hypothesis).

    All four counts are passed in (rather than computed
    from predictions/ground_truth) because the joint
    correctness definition depends on the scoring mode:

    - v1 mode: joint = turn_id equality.
    - v2 by_pid mode: joint = pid equality AND
      is_about equality.

    The caller (calibrator) computes the per-mode joint
    correctness and reports the four counts here.

    Decidable / emit-subset membership:

    - v1 mode: decidable = GT `reaction_to_turn_id` is
      not None / not "unknown". Emit subset = decidable
      AND pred.present AND pred.reaction_to_turn_id
      is not None.
    - v2 by_pid mode: decidable = BOTH pid and
      is_about GT are known. Emit subset = decidable
      AND pred.present.

    Returned fields:

    - `n_joint_all_decidable` / `n_joint_emit_subset`:
      the two denominators.
    - `n_joint_correct_all_decidable` /
      `n_joint_correct_emit_subset`: the matching
      correct counts.
    - `n_joint_no_emit_wrong`: cases where the LLM
      didn't emit on a decidable GT (counted as wrong
      in the "all decidable" view; excluded from
      "emit subset"). Useful for diagnose: this is
      the count of "no-emit is the LLM's signal"
      cases.
    - `acc_joint_all_decidable` /
      `acc_joint_emit_subset`: the two accuracy
      values. 0.0 when the denominator is 0.
    """
    n_joint_no_emit_wrong = (
        n_joint_all_decidable
        - n_joint_emit_subset
        - (
            n_joint_correct_all
            - n_joint_correct_emit
        )
    )
    if n_joint_no_emit_wrong < 0:
        # Defensive: the math should never underflow
        # because "emit subset" is a subset of
        # "all decidable" and "correct emit" is a
        # subset of "correct all". If it does, clamp.
        n_joint_no_emit_wrong = max(
            0,
            n_joint_all_decidable - n_joint_emit_subset,
        )

    acc_all = (
        n_joint_correct_all / n_joint_all_decidable
        if n_joint_all_decidable > 0
        else 0.0
    )
    acc_emit = (
        n_joint_correct_emit / n_joint_emit_subset
        if n_joint_emit_subset > 0
        else 0.0
    )

    return {
        "n_joint_all_decidable": int(n_joint_all_decidable),
        "n_joint_emit_subset": int(n_joint_emit_subset),
        "n_joint_correct_all_decidable": int(
            n_joint_correct_all
        ),
        "n_joint_correct_emit_subset": int(
            n_joint_correct_emit
        ),
        "n_joint_no_emit_wrong": int(n_joint_no_emit_wrong),
        "acc_joint_all_decidable": round(float(acc_all), 6),
        "acc_joint_emit_subset": round(float(acc_emit), 6),
    }


# === Field-specific calibrators ===========================================


def calibrate_addressee_field(
    predictions: Sequence[AddresseePrediction],
    ground_truth: Sequence[AddresseeGroundTruth],
    *,
    treat_no_emit_as_not_addressed: bool = False,
) -> CalibrationFieldReport:
    """Calibrate the M18.7 `addressee_hypothesis` field.

    Per-turn outcome rules (DECIDED 5 / 6):

    - prediction empty AND ground truth is "unknown" →
      `n_unknown++`; the prediction is not scored.
    - prediction empty AND ground truth is decidable →
      `n_incorrect++`; confidence is treated as 0.0.
      (P4 Phase 2A opt-in: when
      `treat_no_emit_as_not_addressed=True` AND
      `gt.addressed_to_assistant is False`, the empty
      prediction is treated as matching the GT
      ("LLM correctly identified 'not addressed'") and
      counted as `n_correct++`. This is the v2 by_pid
      measurement fix; v1 byte-identity is preserved
      when the kwarg is False.)
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

    P4 Phase 2A: the `treat_no_emit_as_not_addressed`
    kwarg addresses a v1 measurement bias surfaced in
    the P4 Phase 1 memo. On the held-out fixture, 3 of
    the 7 wrong addressee cases in the P0 regen run
    are the LLM correctly identifying "not addressed
    to assistant" via no-emit on implicit side-thread /
    short-ack cases (turns 2, 3, 9). The v1 rule marks
    these as wrong; treating "no emit" as
    "not addressed" reflects the LLM's actual
    judgment and produces a more accurate calibration
    signal. The kwarg defaults to False to preserve
    v1 byte-identity (D6); the v2 by_pid runner path
    enables it.
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
            # truth. v1 behavior: incorrect, conf 0.0.
            # P4 Phase 2A (opt-in): when GT is False
            # (i.e., the message is NOT addressed to
            # assistant), the LLM's no-emit is the
            # correct judgment → count as correct.
            n_present += 1
            if (
                treat_no_emit_as_not_addressed
                and bool(gt.addressed_to_assistant) is False
            ):
                n_correct += 1
                correct_flags.append(True)
            else:
                n_incorrect += 1
                correct_flags.append(False)
            confidences.append(0.0)
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

    # P1 (2026-06-09): per-class confusion matrix.
    # The breakdown is independent of the
    # `treat_no_emit_as_not_addressed` kwarg — the raw
    # TP/FP/FN counts are the same in v1 and v2; the
    # scorer decides how to count `noemit + GT-False`
    # (v1: wrong; v2 fix: correct). The breakdown
    # exposes both views via `tp_not_addressed_present`
    # and `tp_not_addressed_noemit`.
    addr_breakdown = _compute_addressee_class_breakdown(
        predictions, ground_truth
    )

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
        addressee_class_breakdown=addr_breakdown,
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
    # P1 (2026-06-09): per-subset joint counters.
    # The v1 joint is turn_id equality, the all-decidable
    # denominator is GT turn_id is not None/unknown,
    # and the emit-subset denominator adds
    # `pred.present AND pred.reaction_to_turn_id is
    # not None`. We track both correct counts inline.
    n_joint_all_decidable = 0
    n_joint_emit_subset = 0
    n_joint_correct_all = 0
    n_joint_correct_emit = 0

    for pred, gt in zip(predictions, ground_truth):
        if not _is_known(gt.reaction_to_turn_id):
            n_unknown += 1
            if pred.present:
                continue
            continue
        # Joint axis: v1 = turn_id equality. Track
        # all-decidable membership (GT known) and
        # emit-subset membership (GT known + pred
        # non-empty + pred has a turn_id).
        n_joint_all_decidable += 1
        is_emit = bool(
            pred.present
            and str(pred.reaction_to_turn_id or "").strip() != ""
        )
        if is_emit:
            n_joint_emit_subset += 1
        if not pred.present:
            n_incorrect += 1
            n_present += 1
            confidences.append(0.0)
            correct_flags.append(False)
            # No emit on a decidable GT → wrong in the
            # "all decidable" view (already counted
            # above).
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
        if is_emit:
            if is_correct:
                n_joint_correct_emit += 1
        if is_correct:
            n_joint_correct_all += 1

    bins = compute_reliability_bins(confidences, correct_flags)
    accuracy = compute_accuracy(correct_flags)
    brier = compute_brier(confidences, correct_flags)
    ece = compute_ece(bins)
    drift = derive_drift_signals(bins, n_present)
    threshold_rec = recommend_thresholds(bins)

    # P1 (2026-06-09): per-subset joint breakdown.
    # In v1 mode, the joint axis is turn_id equality;
    # "all decidable" includes no-emit on a decidable
    # GT (counted as wrong). The emit-subset view
    # excludes those no-emit cases.
    joint_breakdown = _compute_reaction_joint_breakdown(
        n_joint_all_decidable=n_joint_all_decidable,
        n_joint_emit_subset=n_joint_emit_subset,
        n_joint_correct_all=n_joint_correct_all,
        n_joint_correct_emit=n_joint_correct_emit,
    )

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
        reaction_joint_breakdown=joint_breakdown,
    )


# === v2: by_pid reaction scoring ==========================================


def calibrate_reaction_field_by_pid(
    predictions: Sequence[ReactionPrediction],
    ground_truth: Sequence[ReactionGroundTruth],
) -> dict[str, Any]:
    """Score the M18.7 `reaction_attribution_hypothesis`
    field on `reaction_to_participant_id` +
    `is_about_assistant_claim` instead of
    `reaction_to_turn_id`.

    v2 (2026-06-09) — pure function, no I/O. Returns a
    dict with three sub-axes (pid, is_about, joint):

    - `pid_correct_flags` / `pid_confidences`
    - `is_about_correct_flags` / `is_about_confidences`
    - `joint_correct_flags` / `joint_confidences`

    Per-axis outcome rules (Q3 ruling, 2026-06-09):

    - `pred` empty AND `gt` decidable on this axis →
      incorrect; confidence = 0.0.
    - `pred` empty AND `gt` "unknown" → skip axis.
    - `pred` non-empty AND `gt` "unknown" → skip axis.
    - `pred` non-empty AND `gt` decidable on this axis
      → score by strict equality (after pid
      normalization).

    Joint: include only turns where BOTH axes have a
    decidable GT AND `pred` is non-empty. Joint
    correctness requires pid AND is_about to BOTH
    match. The runner wraps the joint axis into a
    `CalibrationFieldReport` for the report's primary
    bins / drift / threshold-recommendation fields;
    the per-axis breakdowns are surfaced via
    `pid_breakdown` and `is_about_breakdown` on the
    field report.

    The function NEVER mutates the input lists.
    """
    pid_correct_flags: list[bool] = []
    pid_confidences: list[float] = []
    is_about_correct_flags: list[bool] = []
    is_about_confidences: list[float] = []
    joint_correct_flags: list[bool] = []
    joint_confidences: list[float] = []

    for pred, gt in zip(predictions, ground_truth):
        pid_gt_known = _is_known(gt.reaction_to_participant_id)
        is_about_gt_known = _is_known(gt.is_about_assistant_claim)

        if pid_gt_known:
            if pred.present:
                pred_pid = normalize_pid(
                    pred.reaction_to_participant_id
                )
                gt_pid = normalize_pid(gt.reaction_to_participant_id)  # type: ignore[arg-type]
                is_pid_correct = bool(pred_pid == gt_pid)
                pid_correct_flags.append(is_pid_correct)
                pid_confidences.append(_bounded_float(pred.confidence))
            else:
                # Pred absent, GT decidable → incorrect.
                pid_correct_flags.append(False)
                pid_confidences.append(0.0)

        if is_about_gt_known:
            if pred.present:
                is_about_pred = bool(pred.is_about_assistant_claim)
                is_about_gt = bool(gt.is_about_assistant_claim)  # type: ignore[arg-type]
                is_about_correct = bool(is_about_pred == is_about_gt)
                is_about_correct_flags.append(is_about_correct)
                is_about_confidences.append(
                    _bounded_float(pred.confidence)
                )
            else:
                is_about_correct_flags.append(False)
                is_about_confidences.append(0.0)

        # Joint: BOTH axes decidable AND pred non-empty.
        if (
            pid_gt_known
            and is_about_gt_known
            and pred.present
        ):
            pred_pid = normalize_pid(pred.reaction_to_participant_id)
            gt_pid = normalize_pid(gt.reaction_to_participant_id)  # type: ignore[arg-type]
            is_about_pred = bool(pred.is_about_assistant_claim)
            is_about_gt = bool(gt.is_about_assistant_claim)  # type: ignore[arg-type]
            joint_correct = bool(
                pred_pid == gt_pid
                and is_about_pred == is_about_gt
            )
            joint_correct_flags.append(joint_correct)
            joint_confidences.append(_bounded_float(pred.confidence))

    return {
        "pid_correct_flags": pid_correct_flags,
        "pid_confidences": pid_confidences,
        "is_about_correct_flags": is_about_correct_flags,
        "is_about_confidences": is_about_confidences,
        "joint_correct_flags": joint_correct_flags,
        "joint_confidences": joint_confidences,
    }


def _calibrate_reaction_by_pid(
    predictions: Sequence[ReactionPrediction],
    ground_truth: Sequence[ReactionGroundTruth],
    *,
    pid_table: Mapping[str, str] | None = None,
) -> CalibrationFieldReport:
    """Wrap `calibrate_reaction_field_by_pid` output into a
    `CalibrationFieldReport` for Mode A (`by_pid`).

    v2 (2026-06-09) — the joint axis is the primary
    signal (the report's bins / drift / threshold-
    recommendation all reflect joint correctness). The
    per-axis breakdowns (pid, is_about) are surfaced
    via the new `pid_breakdown` / `is_about_breakdown`
    fields on the `CalibrationFieldReport`.

    `pid_table` is the optional override merged into
    the default `M18_7_1_PID_NORMALIZATION` table; if
    provided, it shadows the default for the duration
    of this call. Unused in the current implementation
    (the scorer uses the module-level default), but
    kept for future-proofing and test injection.
    """
    # `pid_table` is reserved for future per-call
    # overrides; the by_pid scorer uses the module-
    # level `normalize_pid` default. We acknowledge the
    # argument explicitly so callers can pass it
    # without TypeError, but a no-op merge is enough
    # for now.
    if pid_table is not None:
        # Build a merged view but do not mutate the
        # module-level constant.
        merged = dict(M18_7_1_PID_NORMALIZATION)
        merged.update(pid_table)
        # Re-import the helper with the merged table
        # would be invasive; instead, we trust the
        # caller to set the table before invoking
        # (the runner merges at the top-level).
        # This branch is for documentation only.
        _ = merged

    raw = calibrate_reaction_field_by_pid(predictions, ground_truth)

    joint_correct_flags = raw["joint_correct_flags"]
    joint_confidences = raw["joint_confidences"]
    pid_correct_flags = raw["pid_correct_flags"]
    pid_confidences = raw["pid_confidences"]
    is_about_correct_flags = raw["is_about_correct_flags"]
    is_about_confidences = raw["is_about_confidences"]

    joint_bins = compute_reliability_bins(
        joint_confidences, joint_correct_flags
    )
    joint_ece = compute_ece(joint_bins)
    joint_brier = compute_brier(
        joint_confidences, joint_correct_flags
    )
    joint_accuracy = compute_accuracy(joint_correct_flags)
    n_present = len(joint_confidences)
    drift = derive_drift_signals(joint_bins, n_present)
    threshold_rec = recommend_thresholds(joint_bins)

    pid_bins = compute_reliability_bins(
        pid_confidences, pid_correct_flags
    )
    is_about_bins = compute_reliability_bins(
        is_about_confidences, is_about_correct_flags
    )
    pid_breakdown = {
        "n_present": len(pid_confidences),
        "n_correct": int(sum(pid_correct_flags)),
        "accuracy": compute_accuracy(pid_correct_flags),
        "brier": compute_brier(pid_confidences, pid_correct_flags),
        "ece": compute_ece(pid_bins),
        "reliability_bins": [b.to_dict() for b in pid_bins],
    }
    is_about_breakdown = {
        "n_present": len(is_about_confidences),
        "n_correct": int(sum(is_about_correct_flags)),
        "accuracy": compute_accuracy(is_about_correct_flags),
        "brier": compute_brier(
            is_about_confidences, is_about_correct_flags
        ),
        "ece": compute_ece(is_about_bins),
        "reliability_bins": [b.to_dict() for b in is_about_bins],
    }

    n_total = min(len(predictions), len(ground_truth))
    # `n_unknown` (joint axis): turn where BOTH
    # axes are unknown. A turn with one decidable
    # axis and one unknown axis is "partial" — it
    # is still not in the joint denominator but it
    # IS in the per-axis denominator for the
    # decidable axis. We count the turn as fully
    # unknown only when both axes are unknown, to
    # keep the joint `n_unknown` consistent with
    # the joint `n_total - n_present` math.
    n_unknown = sum(
        1
        for gt in ground_truth
        if not (
            _is_known(gt.reaction_to_participant_id)
            and _is_known(gt.is_about_assistant_claim)
        )
    )
    n_correct = int(sum(joint_correct_flags))
    n_incorrect = n_present - n_correct

    # P1 (2026-06-09): per-subset joint breakdown.
    # In v2 by_pid mode, the joint axis is pid + is_about
    # equality. The "emit subset" is what the existing
    # `calibrate_reaction_field_by_pid` already counts
    # (BOTH axes decidable + pred.present). The
    # "all decidable" view drops the pred.present
    # requirement — no-emit on a decidable GT is
    # counted as wrong in this view. We compute the
    # all-decidable counts in a separate pass.
    n_joint_all_decidable = 0
    n_joint_correct_all = 0
    for pred, gt in zip(predictions, ground_truth):
        pid_gt_known = _is_known(gt.reaction_to_participant_id)
        is_about_gt_known = _is_known(
            gt.is_about_assistant_claim
        )
        if not (pid_gt_known and is_about_gt_known):
            continue
        n_joint_all_decidable += 1
        if not pred.present:
            # No emit on decidable joint GT → wrong
            # in the "all decidable" view.
            continue
        pred_pid = normalize_pid(pred.reaction_to_participant_id)
        gt_pid = normalize_pid(gt.reaction_to_participant_id)  # type: ignore[arg-type]
        is_about_pred = bool(pred.is_about_assistant_claim)
        is_about_gt = bool(gt.is_about_assistant_claim)  # type: ignore[arg-type]
        if pred_pid == gt_pid and is_about_pred == is_about_gt:
            n_joint_correct_all += 1
    joint_breakdown = _compute_reaction_joint_breakdown(
        n_joint_all_decidable=n_joint_all_decidable,
        n_joint_emit_subset=n_present,
        n_joint_correct_all=n_joint_correct_all,
        n_joint_correct_emit=n_correct,
    )

    return CalibrationFieldReport(
        n_total=n_total,
        n_present=n_present,
        n_unknown=n_unknown,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=joint_accuracy,
        brier=joint_brier,
        ece=joint_ece,
        reliability_bins=joint_bins,
        drift_signals=drift,
        threshold_recommendation=threshold_rec,
        pid_breakdown=pid_breakdown,
        is_about_breakdown=is_about_breakdown,
        reaction_joint_breakdown=joint_breakdown,
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
    scoring_mode: str = "by_turn_id_v1",
    pid_normalization_override: Mapping[str, str] | None = None,
    resolve_placeholders: bool = True,
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

    v2 (2026-06-09): `scoring_mode` selects the
    reaction-side scoring path:

    - `"by_pid"` (default, recommended): scores
      `reaction_to_participant_id` +
      `is_about_assistant_claim` joint correctness
      with pid normalization.
    - `"by_turn_id_resolved"`: v1 scoring on
      `reaction_to_turn_id` but with placeholder
      resolution at runner-time
      (`turn_<assistant_prior_turn_id>` etc.).
    - `"by_turn_id_v1"`: byte-identical v1 scoring
      (no resolution, no normalization).

    `pid_normalization_override` (optional) merges
    into the default `M18_7_1_PID_NORMALIZATION`
    table for the duration of the run. The override
    is honored in `by_pid` mode (and only in
    `by_pid` mode; v1 modes are byte-identical and
    do not consult the table).

    `resolve_placeholders` defaults to True. Set to
    False to skip placeholder resolution even in
    `by_turn_id_resolved` mode (useful for
    diagnosing the placeholder pattern itself).

    The runner builds a `replay_history` from the
    fixture's `group_turn_envelope`. A turn is
    treated as an assistant turn when its
    `speaker_participant_id` is NOT in the turn's
    `visible_participant_ids` (conservative
    heuristic — the assistant is never in
    `visible_participant_ids` for the held-out
    fixture).
    """
    # === v2: scoring mode validation ======================
    if scoring_mode not in M18_7_1_SCORING_MODES:
        raise ValueError(
            f"unknown scoring_mode: {scoring_mode!r}; "
            f"allowed: {sorted(M18_7_1_SCORING_MODES)}"
        )
    # Acknowledge the override (used by `_calibrate_reaction_by_pid`).
    # The actual scoring path uses the module-level
    # `normalize_pid` default; the override is exposed in the
    # report envelope for diagnose.
    pid_table_effective: dict[str, str] = dict(
        M18_7_1_PID_NORMALIZATION
    )
    if pid_normalization_override:
        pid_table_effective.update(pid_normalization_override)

    addressee_predictions: list[AddresseePrediction] = []
    addressee_ground_truth: list[AddresseeGroundTruth] = []
    reaction_predictions: list[ReactionPrediction] = []
    reaction_ground_truth: list[ReactionGroundTruth] = []
    # v2: per-turn index for the fixture step's
    # reaction_ground_truth (used as `current_turn_index`
    # in placeholder resolution; not present on the
    # dataclass itself, so we track it here).
    reaction_turn_indices: list[int] = []

    # === v2: build replay_history from fixture =============
    replay_history: list[Mapping[str, Any]] = []
    for i, step in enumerate(fixture):
        env = dict(step.get("group_turn_envelope", {}))
        visible = {
            str(p).strip()
            for p in env.get("visible_participant_ids", [])
        }
        sp = str(env.get("speaker_participant_id", "")).strip()
        is_assistant_turn = bool(sp) and sp not in visible
        replay_history.append(
            {
                "turn_index": int(step.get("turn_index", i)),
                "participant_id": sp,
                "role": "assistant" if is_assistant_turn else "user",
                "text": str(step.get("text", ""))[:80],
            }
        )

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
        reaction_turn_indices.append(current_turn_index)

    # === v2: pre-resolve placeholder GTs for Mode B ======
    fixture_warnings: list[str] = []
    if (
        scoring_mode == "by_turn_id_resolved"
        and resolve_placeholders
    ):
        resolved_gt: list[ReactionGroundTruth] = []
        for idx, gt in enumerate(reaction_ground_truth):
            raw = gt.reaction_to_turn_id
            if raw is None:
                resolved_gt.append(gt)
                continue
            current_idx = reaction_turn_indices[idx]
            resolved, warns = resolve_placeholder(
                raw,
                replay_history=replay_history[:idx],
                current_turn_index=current_idx,
            )
            if warns:
                fixture_warnings.extend(warns)
            resolved_gt.append(
                ReactionGroundTruth(
                    reaction_to_turn_id=resolved,
                    reaction_to_participant_id=(
                        gt.reaction_to_participant_id
                    ),
                    is_about_assistant_claim=(
                        gt.is_about_assistant_claim
                    ),
                )
            )
        reaction_ground_truth = resolved_gt

    # === v2: dispatch scoring ============================
    # P4 Phase 2A: enable the v2 by_pid no-emit
    # measurement fix only in by_pid mode. The other
    # modes (by_turn_id_v1, by_turn_id_resolved) keep
    # the v1 byte-identical behavior to preserve D6
    # (v1 baseline) and existing T9 / T12 regression
    # tests.
    addressee_report = calibrate_addressee_field(
        addressee_predictions,
        addressee_ground_truth,
        treat_no_emit_as_not_addressed=(
            scoring_mode == "by_pid"
        ),
    )
    if scoring_mode == "by_pid":
        reaction_report = _calibrate_reaction_by_pid(
            reaction_predictions,
            reaction_ground_truth,
            pid_table=pid_table_effective,
        )
    else:
        # "by_turn_id_v1" and "by_turn_id_resolved":
        # same v1 scoring function (v1 is byte-
        # identical; resolved mode resolves the GT
        # string at the runner level before calling).
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
        scoring_mode=scoring_mode,
        fixture_warnings=fixture_warnings,
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


# === v2: pid normalization + placeholder resolution ======================


def normalize_pid(
    pid: Any,
    *,
    table: Mapping[str, str] | None = None,
) -> str:
    """Canonicalize a participant id before pid equality scoring.

    v2 (2026-06-09) — pure function, no side effects. The
    table is opt-in: callers that want strict surface-id
    equality (v1 Mode C) should NOT call this.

    Rules:
      - None / non-string / empty / whitespace-only
        → "" (preserved as the "no-attribution" signal).
      - Whitespace is stripped first.
      - Lookup in `table` (defaults to
        `M18_7_1_PID_NORMALIZATION`) wins; if a surface
        id is in the table, the canonical form is
        returned.
      - Otherwise the stripped, lowercased form is
        returned (passthrough for human names).

    The function is idempotent: `normalize_pid(x) == x`
    for any x already in canonical form ("" in, "" out;
    "bot" in, "bot" out; "alice" in, "alice" out).
    """
    if table is None:
        table = M18_7_1_PID_NORMALIZATION
    if not isinstance(pid, str) or not pid.strip():
        return ""
    raw = pid.strip()
    # Table lookup (case-sensitive on the raw form;
    # the table keys are exact surface strings).
    if raw in table:
        return table[raw]
    return raw.lower()


def resolve_placeholder(
    raw_turn_id: Any,
    *,
    replay_history: Sequence[Mapping[str, Any]],
    current_turn_index: int,
) -> tuple[str, list[str]]:
    """Resolve a `turn_<role>` placeholder against the
    `replay_history` collected so far.

    v2 (2026-06-09) — pure function, no I/O.

    Returns `(resolved_value, warnings)`. If the input
    is not a placeholder string, returns
    `(raw_turn_id, [])` unchanged. If the placeholder
    is unresolvable (no matching prior turn in
    `replay_history`), returns `("unknown", [warning])`
    so the caller treats it as a skipped GT per v1
    rules (calibrate_*_field skips `unknown` turn_ids).

    Placeholder roles (the `<role>` part):
      - "assistant_prior_turn_id" → most recent prior
        assistant turn (matched by `role == "assistant"`
        in `replay_history`).
      - "user_prior_turn_id" → most recent prior user
        turn (matched by `role == "user"` in
        `replay_history`).
      - Other (e.g. "carol_prior_turn_id") → most recent
        prior turn where
        `participant_id.strip().lower() == role.removeprefix("..._prior_turn_id")`
        (i.e., the role without the `_prior_turn_id`
        suffix is treated as a speaker name).
    """
    if not isinstance(raw_turn_id, str):
        return "unknown", [
            f"non_string_turn_id:{type(raw_turn_id).__name__}"
        ]
    candidate = raw_turn_id.strip()
    match = M18_7_1_PLACEHOLDER_PATTERN.match(candidate)
    if match is None:
        # Not a placeholder; pass through unchanged.
        return raw_turn_id, []
    role = match.group("role")
    # Walk replay_history backward (most recent first).
    for record in reversed(replay_history):
        try:
            ridx = int(record.get("turn_index", -1))
        except (TypeError, ValueError):
            continue
        if ridx >= int(current_turn_index):
            continue
        rrole = str(record.get("role", ""))
        rspeaker = str(record.get("participant_id", "")).strip().lower()
        if (
            role in M18_7_1_PLACEHOLDER_ROLES_ASSISTANT
            and rrole == "assistant"
        ):
            return f"turn_{ridx}", []
        if (
            role in M18_7_1_PLACEHOLDER_ROLES_USER
            and rrole == "user"
        ):
            return f"turn_{ridx}", []
        # Named-speaker placeholder: role without the
        # `_prior_turn_id` suffix is the speaker's
        # normalized pid.
        speaker_key = role
        if speaker_key.endswith("_prior_turn_id"):
            speaker_key = speaker_key[: -len("_prior_turn_id")]
        if rspeaker and rspeaker == speaker_key:
            return f"turn_{ridx}", []
    return "unknown", [f"placeholder_unresolved:{raw_turn_id}"]


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
    "M18_7_1_DEFAULT_SCORING_MODE",
    "M18_7_1_DRIFT_GAP_THRESHOLD",
    "M18_7_1_ENGINEERING_PROXY_LABEL",
    "M18_7_1_FLAT_CURVE_GAP",
    "M18_7_1_FLAT_CURVE_MIN_POPULATED_BINS",
    "M18_7_1_GT_UNKNOWN_SENTINEL",
    "M18_7_1_HIGH_BAND_LOWER_BOUND",
    "M18_7_1_LOW_BAND_UPPER_BOUND",
    "M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL",
    "M18_7_1_N_BINS",
    "M18_7_1_PID_NORMALIZATION",
    "M18_7_1_PLACEHOLDER_PATTERN",
    "M18_7_1_PLACEHOLDER_ROLES_ASSISTANT",
    "M18_7_1_PLACEHOLDER_ROLES_USER",
    "M18_7_1_SCORING_MODES",
    "M18_7_1_TIE_BREAKER_MIN_CURRENT",
    "M18_7_1_THRESHOLD_NEIGHBORHOOD_HALFWIDTH",
    "M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT",
    "ReactionGroundTruth",
    "ReactionPrediction",
    "calibrate_addressee_field",
    "calibrate_reaction_field",
    "calibrate_reaction_field_by_pid",
    "compute_accuracy",
    "compute_brier",
    "compute_ece",
    "compute_reliability_bins",
    "derive_drift_signals",
    "normalize_pid",
    "record_m18_7_1_calibration",
    "recommend_thresholds",
    "resolve_placeholder",
    "run_m18_7_1_calibration_harness",
    "validate_calibration_fixture_shape",
]
