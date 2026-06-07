"""Tests for M18.7.1 held-out calibration harness.

The pure-function tests use synthetic data; the
integration tests use a deterministic FakeJSONLLM
subclass that returns controlled M18.7 hypothesis
fields. The M20.4 module constants are not mutated
by the runner; an explicit test asserts this.

The fixture shape tests load
`tests/fixtures/m18_7_1_held_out_calibration.json`
and verify the schema is consistent with
`validate_calibration_fixture_shape`.
"""

from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime import m18_7_1_calibration as cal
from segmentum.dialogue.runtime import m20_4_attribution as m20_4
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
)
from tests.test_mvp_dialogue_runtime import (
    FakeJSONLLM,
    _maybe_m12_extractor_response,
)


# === Pure-function tests: bin computation ================================


def test_compute_reliability_bins_assigns_to_correct_bin() -> None:
    bins = cal.compute_reliability_bins(
        confidences=[0.05, 0.15, 0.25, 0.35, 0.45],
        correct_flags=[True, False, True, False, True],
    )
    assert [b.label for b in bins] == list(cal.M18_7_1_BIN_LABELS)
    assert bins[0].count == 1 and bins[0].mean_confidence == 0.05
    assert bins[1].count == 1 and bins[1].mean_confidence == 0.15
    assert bins[2].count == 1 and bins[2].mean_confidence == 0.25
    assert bins[3].count == 1 and bins[3].mean_confidence == 0.35
    assert bins[4].count == 1 and bins[4].mean_confidence == 0.45


def test_compute_reliability_bins_handles_boundary_value_1_0() -> None:
    bins = cal.compute_reliability_bins(
        confidences=[1.0],
        correct_flags=[True],
    )
    # Last bin (label "0.90-1.00") is closed on the
    # right so confidence == 1.0 lands there.
    assert bins[-1].count == 1
    assert bins[-1].mean_confidence == 1.0
    assert bins[-1].accuracy == 1.0
    assert bins[-1].gap == 0.0
    for b in bins[:-1]:
        assert b.count == 0


def test_compute_reliability_bins_handles_boundary_value_0_0() -> None:
    bins = cal.compute_reliability_bins(
        confidences=[0.0],
        correct_flags=[False],
    )
    assert bins[0].count == 1
    assert bins[0].mean_confidence == 0.0
    assert bins[0].accuracy == 0.0
    assert bins[0].gap == 0.0
    for b in bins[1:]:
        assert b.count == 0


def test_compute_reliability_bins_empty_input_returns_all_zero_bins() -> None:
    bins = cal.compute_reliability_bins([], [])
    assert len(bins) == cal.M18_7_1_N_BINS
    for b in bins:
        assert b.count == 0
        assert b.mean_confidence == 0.0
        assert b.accuracy == 0.0
        assert b.gap == 0.0


def test_compute_reliability_bins_aggregates_within_a_bin() -> None:
    bins = cal.compute_reliability_bins(
        confidences=[0.04, 0.06, 0.08],
        correct_flags=[True, False, True],
    )
    # All three confidences are in [0.0, 0.10).
    assert bins[0].count == 3
    assert abs(bins[0].mean_confidence - (0.04 + 0.06 + 0.08) / 3) < 1e-9
    assert abs(bins[0].accuracy - 2 / 3) < 1e-9


# === Pure-function tests: aggregate metrics ===============================


def test_compute_ece_perfectly_calibrated_returns_zero() -> None:
    # In each bin, accuracy == mean_confidence, so the
    # gap is 0 and ECE = 0.
    bins = cal.compute_reliability_bins(
        confidences=[0.05, 0.05, 0.55, 0.55, 0.95, 0.95],
        correct_flags=[False, False, True, True, True, True],
    )
    # Bins 0 (mean 0.05, acc 0.0 → gap 0.05), 5
    # (mean 0.55, acc 1.0 → gap 0.45), 9 (mean 0.95,
    # acc 1.0 → gap 0.05). Not perfectly calibrated.
    # Build a perfectly-calibrated bin set instead.
    bins = [
        cal.BinStats(
            label=b.label,
            count=b.count,
            mean_confidence=b.mean_confidence,
            accuracy=b.mean_confidence,
            gap=0.0,
        )
        for b in bins
    ]
    assert cal.compute_ece(bins) == 0.0


def test_compute_ece_systematic_overconfidence_is_positive() -> None:
    # Each bin's accuracy is 0.0, but mean_confidence
    # is in (0, 1). The gap is the mean_confidence.
    bins = [
        cal.BinStats(label="0.50-0.60", count=4,
                     mean_confidence=0.55, accuracy=0.0, gap=0.55),
        cal.BinStats(label="0.60-0.70", count=4,
                     mean_confidence=0.65, accuracy=0.0, gap=0.65),
    ]
    ece = cal.compute_ece(bins)
    # Both bins are equally weighted (4 each → 0.5).
    # ECE = 0.5 * 0.55 + 0.5 * 0.65 = 0.6.
    assert abs(ece - 0.6) < 1e-6


def test_compute_ece_empty_input_returns_zero() -> None:
    assert cal.compute_ece([]) == 0.0
    assert cal.compute_ece(cal._empty_bins()) == 0.0


def test_compute_brier_perfect_calibration_is_zero() -> None:
    # confidence matches the outcome exactly for every
    # row (0.0 with False, 1.0 with True). All
    # (c - t)^2 terms are 0.
    brier = cal.compute_brier(
        confidences=[0.0, 0.0, 1.0, 1.0],
        correct_flags=[False, False, True, True],
    )
    assert brier == 0.0


def test_compute_brier_symmetric_for_under_and_overconfidence() -> None:
    # (0.7 - 0)^2 == (0.3 - 1)^2 == 0.49
    brier = cal.compute_brier(
        confidences=[0.7, 0.3],
        correct_flags=[False, True],
    )
    assert abs(brier - 0.49) < 1e-9


def test_compute_brier_empty_input_returns_zero() -> None:
    assert cal.compute_brier([], []) == 0.0


def test_compute_accuracy_empty_input_returns_zero() -> None:
    assert cal.compute_accuracy([]) == 0.0


def test_compute_accuracy_simple_mean() -> None:
    assert cal.compute_accuracy([True, False, True, True]) == 0.75


# === Pure-function tests: drift signals ===================================


def test_derive_drift_signals_insufficient_data_when_n_below_threshold() -> None:
    bins = cal._empty_bins()
    bins[5] = cal.BinStats(
        label=bins[5].label, count=2,
        mean_confidence=0.55, accuracy=1.0, gap=0.45,
    )
    signals = cal.derive_drift_signals(bins, n_present=2)
    assert signals == ["insufficient_data"]


def test_derive_drift_signals_overconfidence_at_high_band() -> None:
    bins = cal._empty_bins()
    # High-band bin (>= 0.80): count 5, mean 0.85,
    # accuracy 0.4 → gap 0.45 > 0.15, accuracy <
    # mean_confidence → overconfidence.
    bins[8] = cal.BinStats(
        label="0.80-0.90", count=5,
        mean_confidence=0.85, accuracy=0.4, gap=0.45,
    )
    # Add some populated bins in the middle so
    # insufficient_data is not emitted.
    bins[4] = cal.BinStats(
        label="0.40-0.50", count=3,
        mean_confidence=0.45, accuracy=0.6, gap=0.15,
    )
    bins[2] = cal.BinStats(
        label="0.20-0.30", count=2,
        mean_confidence=0.25, accuracy=0.5, gap=0.25,
    )
    signals = cal.derive_drift_signals(bins, n_present=10)
    assert "overconfidence_at_high_band" in signals
    assert "insufficient_data" not in signals


def test_derive_drift_signals_underconfidence_at_low_band() -> None:
    bins = cal._empty_bins()
    # Low-band bin (<= 0.40): count 5, mean 0.25,
    # accuracy 0.8 → gap 0.55 > 0.15, accuracy >
    # mean_confidence → underconfidence.
    bins[2] = cal.BinStats(
        label="0.20-0.30", count=5,
        mean_confidence=0.25, accuracy=0.8, gap=0.55,
    )
    bins[5] = cal.BinStats(
        label="0.50-0.60", count=3,
        mean_confidence=0.55, accuracy=0.4, gap=0.15,
    )
    bins[8] = cal.BinStats(
        label="0.80-0.90", count=2,
        mean_confidence=0.85, accuracy=0.5, gap=0.35,
    )
    signals = cal.derive_drift_signals(bins, n_present=10)
    assert "underconfidence_at_low_band" in signals
    assert "insufficient_data" not in signals


def test_derive_drift_signals_bimodal() -> None:
    bins = cal._empty_bins()
    # Highest bin (0.80-0.90) populated, lowest bin
    # (0.20-0.30) populated, middle 6 bins all empty.
    bins[8] = cal.BinStats(
        label="0.80-0.90", count=4,
        mean_confidence=0.85, accuracy=0.6, gap=0.25,
    )
    bins[2] = cal.BinStats(
        label="0.20-0.30", count=4,
        mean_confidence=0.25, accuracy=0.4, gap=0.15,
    )
    signals = cal.derive_drift_signals(bins, n_present=8)
    assert "bimodal" in signals


def test_derive_drift_signals_flat_curve() -> None:
    bins = cal._empty_bins()
    # All populated bins have |gap| < 0.10, and at
    # least 3 are populated.
    bins[1] = cal.BinStats(
        label="0.10-0.20", count=3,
        mean_confidence=0.15, accuracy=0.2, gap=0.05,
    )
    bins[4] = cal.BinStats(
        label="0.40-0.50", count=4,
        mean_confidence=0.45, accuracy=0.5, gap=0.05,
    )
    bins[7] = cal.BinStats(
        label="0.70-0.80", count=3,
        mean_confidence=0.75, accuracy=0.7, gap=0.05,
    )
    signals = cal.derive_drift_signals(bins, n_present=10)
    assert "flat_curve" in signals
    assert "insufficient_data" not in signals


def test_derive_drift_signals_empty_returns_insufficient_data() -> None:
    bins = cal._empty_bins()
    signals = cal.derive_drift_signals(bins, n_present=0)
    assert signals == ["insufficient_data"]


# === Pure-function tests: threshold recommendation ========================


def test_recommend_thresholds_returns_none_when_insufficient_data() -> None:
    bins = cal._empty_bins()
    bins[5] = cal.BinStats(
        label=bins[5].label, count=2,
        mean_confidence=0.55, accuracy=0.5, gap=0.05,
    )
    rec = cal.recommend_thresholds(bins)
    assert rec["current_admit_min"] == 0.4
    assert rec["current_tie_breaker_min"] == 0.85
    assert rec["candidate_admit_min"] is None
    assert rec["candidate_tie_breaker_min"] is None
    assert "M20.4" in rec["caveat"]


def test_recommend_thresholds_picks_bin_with_smallest_gap() -> None:
    bins = cal._empty_bins()
    # Bin 4 (mean 0.45) is in the admit neighborhood
    # (0.4 ± 0.15 = [0.25, 0.55]); gap 0.05.
    bins[4] = cal.BinStats(
        label="0.40-0.50", count=4,
        mean_confidence=0.45, accuracy=0.5, gap=0.05,
    )
    # Bin 6 (mean 0.65) is in the admit neighborhood
    # too; gap 0.15.
    bins[6] = cal.BinStats(
        label="0.60-0.70", count=3,
        mean_confidence=0.65, accuracy=0.5, gap=0.15,
    )
    # Bin 8 (mean 0.85) is in the tie-breaker
    # neighborhood (0.85 ± 0.15 = [0.70, 1.00]);
    # gap 0.05.
    bins[8] = cal.BinStats(
        label="0.80-0.90", count=3,
        mean_confidence=0.85, accuracy=0.9, gap=0.05,
    )
    rec = cal.recommend_thresholds(bins)
    # The candidate is the bin boundary in
    # {0.1, ..., 0.9} nearest the populated bin's
    # mean_confidence. Ties on `abs` resolve to the
    # lower boundary (min() returns the first min).
    # bin 4 (mean 0.45) → 0.4 (|0.45-0.4|=0.05; ties
    # with 0.5 → 0.4 wins).
    assert rec["candidate_admit_min"] == 0.4
    # bin 8 (mean 0.85) → 0.8 (|0.85-0.8|=0.05; ties
    # with 0.9 → 0.8 wins).
    assert rec["candidate_tie_breaker_min"] == 0.8


def test_recommend_thresholds_caveat_string_is_frozen() -> None:
    bins = cal._empty_bins()
    rec = cal.recommend_thresholds(bins)
    assert rec["caveat"] == cal.M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT


# === Pure-function tests: field calibrators ===============================


def test_calibrate_addressee_field_skips_unknown_ground_truth() -> None:
    predictions = [
        cal.AddresseePrediction(
            present=True, addressed_to_assistant=True,
            participant_id="alice", confidence=0.9,
        ),
    ]
    ground_truth = [
        cal.AddresseeGroundTruth(
            addressed_to_assistant=None,  # "unknown"
            addressee_participant_id=None,
        ),
    ]
    report = cal.calibrate_addressee_field(predictions, ground_truth)
    # The prediction is not scored; n_unknown increments.
    assert report.n_unknown == 1
    assert report.n_present == 0
    assert report.n_correct == 0
    assert report.n_incorrect == 0


def test_calibrate_addressee_field_treats_empty_prediction_as_incorrect() -> None:
    predictions = [
        cal.AddresseePrediction(
            present=False, addressed_to_assistant=False,
            participant_id="", confidence=0.0,
        ),
    ]
    ground_truth = [
        cal.AddresseeGroundTruth(
            addressed_to_assistant=True,  # decidable
            addressee_participant_id="alice",
        ),
    ]
    report = cal.calibrate_addressee_field(predictions, ground_truth)
    # Empty prediction against decidable ground truth:
    # n_incorrect=1, confidence treated as 0.0.
    assert report.n_present == 1
    assert report.n_correct == 0
    assert report.n_incorrect == 1
    # Brier = (0.0 - 0)^2 = 0.0 (incorrect ⇒ correct flag False;
    # contribution is (0.0 - 0)^2 = 0).
    assert report.brier == 0.0
    # The reliability bin for confidence 0.0 has count 1
    # and accuracy 0.0 → gap = 0.0.
    assert report.reliability_bins[0].count == 1


def test_calibrate_addressee_field_counts_only_present_predictions() -> None:
    predictions = [
        cal.AddresseePrediction(
            present=True, addressed_to_assistant=True,
            participant_id="alice", confidence=0.9,
        ),
        cal.AddresseePrediction(
            present=True, addressed_to_assistant=False,
            participant_id="bob", confidence=0.6,
        ),
        cal.AddresseePrediction(
            present=True, addressed_to_assistant=True,
            participant_id="carol", confidence=0.3,
        ),
    ]
    ground_truth = [
        cal.AddresseeGroundTruth(
            addressed_to_assistant=True,
            addressee_participant_id="alice",
        ),
        cal.AddresseeGroundTruth(
            addressed_to_assistant=True,  # wrong
            addressee_participant_id="bob",
        ),
        cal.AddresseeGroundTruth(
            addressed_to_assistant=True,
            addressee_participant_id="carol",
        ),
    ]
    report = cal.calibrate_addressee_field(predictions, ground_truth)
    assert report.n_present == 3
    assert report.n_correct == 2
    assert report.n_incorrect == 1
    assert abs(report.accuracy - 2 / 3) < 1e-9


def test_calibrate_reaction_field_string_equality_only() -> None:
    predictions = [
        cal.ReactionPrediction(
            present=True, reaction_to_turn_id="turn_42",
            reaction_to_participant_id="alice",
            is_about_assistant_claim=True, confidence=0.9,
        ),
        cal.ReactionPrediction(
            present=True, reaction_to_turn_id="turn_42_user_utterance",
            reaction_to_participant_id="alice",
            is_about_assistant_claim=True, confidence=0.9,
        ),
    ]
    ground_truth = [
        cal.ReactionGroundTruth(
            reaction_to_turn_id="turn_42",
            reaction_to_participant_id="alice",
            is_about_assistant_claim=True,
        ),
        cal.ReactionGroundTruth(
            reaction_to_turn_id="turn_42",  # different shape → mismatch
            reaction_to_participant_id="alice",
            is_about_assistant_claim=True,
        ),
    ]
    report = cal.calibrate_reaction_field(predictions, ground_truth)
    # First is correct; second is not (string equality only).
    assert report.n_correct == 1
    assert report.n_incorrect == 1


def test_calibrate_reaction_field_treats_empty_as_incorrect() -> None:
    predictions = [
        cal.ReactionPrediction(
            present=False, reaction_to_turn_id="",
            reaction_to_participant_id="",
            is_about_assistant_claim=False, confidence=0.0,
        ),
    ]
    ground_truth = [
        cal.ReactionGroundTruth(
            reaction_to_turn_id="turn_42",
            reaction_to_participant_id="alice",
            is_about_assistant_claim=True,
        ),
    ]
    report = cal.calibrate_reaction_field(predictions, ground_truth)
    assert report.n_present == 1
    assert report.n_incorrect == 1


# === Fixture shape tests ==================================================


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "m18_7_1_held_out_calibration.json"
    )


def test_calibration_fixture_loads_and_has_required_ground_truth_fields() -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    errors = cal.validate_calibration_fixture_shape(fixture)
    assert errors == [], f"fixture shape errors: {errors}"


def test_calibration_fixture_covers_low_medium_and_high_confidence_bands() -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    bands = {step.get("expected_confidence_band") for step in fixture}
    assert bands == {"low", "medium", "high"}


def test_calibration_fixture_includes_probe_turns() -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    assertion_kinds = {step.get("assertion_kind") for step in fixture}
    assert "probe" in assertion_kinds


def test_calibration_fixture_assertion_kind_is_one_of_frozen_enum() -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    for step in fixture:
        kind = step.get("assertion_kind", "")
        assert kind in cal.ALLOWED_M18_7_1_ASSERTION_KIND


def test_validate_calibration_fixture_shape_reports_errors_on_invalid_fixture() -> None:
    bad = [
        {"text": "hi", "group_turn_envelope": {},  # missing assertion_kind / band / ground_truth
         "ground_truth": {}},
    ]
    errors = cal.validate_calibration_fixture_shape(bad)
    assert len(errors) >= 3


# === Integration / runner tests ===========================================


class _CalibrationFakeLLM(FakeJSONLLM):
    """Deterministic LLM that returns controlled M18.7
    hypothesis fields.

    The base class is the standard `FakeJSONLLM` from
    `tests/test_mvp_dialogue_runtime.py`. We override
    `complete_json` to return a payload that includes
    the M18.7 fields for the conscious-loop system
    prompt.
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-turn M18.7 hypothesis responses, keyed
        # by the user_text → most-recently-seen
        # hypothesis. The test sets the
        # `_responses_by_text` map before each test.
        self._responses_by_text: dict[str, dict[str, object]] = {}

    def complete_json(
        self, *, system_prompt: str, user_prompt: str
    ) -> dict[str, object]:
        # The conscious-loop system prompt contains
        # the M18.7 fields block.
        if "addressee_hypothesis" in system_prompt:
            # Look up the per-user-text response.
            for text, response in self._responses_by_text.items():
                if text in user_prompt:
                    return response
            # Default: empty M18.7 fields.
            return {
                "addressee_hypothesis": {},
                "reaction_attribution_hypothesis": {},
            }
        m12_hit = _maybe_m12_extractor_response(system_prompt)
        if m12_hit is not None:
            return m12_hit
        return super().complete_json(
            system_prompt=system_prompt, user_prompt=user_prompt
        )


def _runtime(tmp_path: Path) -> MVPDialogueRuntime:
    return MVPDialogueRuntime(
        store=MVPStateStore(tmp_path / "persona"),
        llm=_CalibrationFakeLLM(),
        persona_name="hutao",
    )


def test_runner_replays_fixture_and_returns_per_field_report(tmp_path: Path) -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path)
    # Program the LLM to return M18.7 fields for
    # every step. The "perfect" calibration scenario
    # — every prediction matches ground truth.
    runtime.llm._responses_by_text = {
        step["text"]: {
            "addressee_hypothesis": {
                "participant_id": "carol",
                "addressed_to_assistant": (
                    step["ground_truth"]["addressed_to_assistant"]
                    if isinstance(
                        step["ground_truth"]["addressed_to_assistant"], bool
                    )
                    else False
                ),
                "confidence": 0.7,
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "carol",
                "reaction_to_turn_id": (
                    step["ground_truth"]["reaction_to_turn_id"]
                    if isinstance(
                        step["ground_truth"]["reaction_to_turn_id"], str
                    )
                    and step["ground_truth"]["reaction_to_turn_id"]
                    not in ("", "unknown")
                    else ""
                ),
                "reaction_to_participant_id": "carol",
                "is_about_assistant_claim": False,
                "confidence": 0.6,
            },
        }
        for step in fixture
    }

    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
        fixture_name="tests/fixtures/m18_7_1_held_out_calibration.json",
    )

    assert report.n_fixtures == len(fixture)
    assert report.fixture_name.endswith("m18_7_1_held_out_calibration.json")
    # Both fields have non-empty predictions.
    assert report.addressee.n_present > 0
    assert report.reaction.n_present > 0


def test_runner_writes_calibration_to_state_surface(tmp_path: Path) -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path)
    runtime.llm._responses_by_text = {
        step["text"]: {
            "addressee_hypothesis": {
                "participant_id": "carol",
                "addressed_to_assistant": True,
                "confidence": 0.7,
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "carol",
                "reaction_to_turn_id": "turn_0",
                "reaction_to_participant_id": "carol",
                "is_about_assistant_claim": False,
                "confidence": 0.6,
            },
        }
        for step in fixture
    }
    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
        fixture_name="tests/fixtures/m18_7_1_held_out_calibration.json",
    )
    state: dict = {}
    cal.record_m18_7_1_calibration(
        state, report, at="2026-06-07T00:00:00Z"
    )
    assert cal.M18_7_1_CALIBRATION_SURFACE_KEY in state
    surface = state[cal.M18_7_1_CALIBRATION_SURFACE_KEY]
    assert surface["last_run_at"] == "2026-06-07T00:00:00Z"
    assert surface["n_fixtures"] == len(fixture)
    assert surface["engineering_proxy_label"] == cal.M18_7_1_ENGINEERING_PROXY_LABEL
    assert surface["threshold_recommendation"]["current_admit_min"] == 0.4
    assert surface["threshold_recommendation"]["current_tie_breaker_min"] == 0.85
    assert surface["threshold_recommendation"]["caveat"] == cal.M18_7_1_THRESHOLD_RECOMMENDATION_CAVEAT


def test_runner_handles_empty_m18_7_hypothesis_in_predictions(tmp_path: Path) -> None:
    """When the LLM returns empty M18.7 fields for every
    turn, the runner must not crash. The calibrator
    counts empty predictions as scored (n_present++
    with confidence=0.0 and `correct=False`). For 12
    fixture turns with 8 decidable addressee and 6
    decidable reaction ground truths:

    - addressee: n_present=8, n_unknown=4,
      n_correct=0, n_incorrect=8
    - reaction: n_present=6, n_unknown=6,
      n_correct=0, n_incorrect=6
    - the only populated bin is bin 0
      (count=N, mean_confidence=0.0, accuracy=0.0,
      gap=0.0); no drift signal fires (gap==0
      ⇒ not over/under; only one populated bin
      ⇒ no flat_curve or bimodal)
    - candidate thresholds are None (no populated
      bin in either neighborhood)
    """
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path)
    runtime.llm._responses_by_text = {}  # default: empty

    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
        fixture_name="<empty-llm>",
    )
    # The runner does not crash; the report is well-formed.
    assert report.n_fixtures == len(fixture)
    # 8 of 12 fixture turns have decidable addressee
    # ground truth; the calibrator counts them as
    # n_present (scored as incorrect with
    # confidence=0.0).
    assert report.addressee.n_present == 8
    assert report.addressee.n_unknown == 4
    assert report.addressee.n_correct == 0
    assert report.addressee.n_incorrect == 8
    # 6 of 12 fixture turns have decidable reaction
    # ground truth.
    assert report.reaction.n_present == 6
    assert report.reaction.n_unknown == 6
    assert report.reaction.n_correct == 0
    assert report.reaction.n_incorrect == 6
    # Brier is 0.0 because all confidences are 0.0
    # and all outcomes are 0 (incorrect) → (0-0)^2 = 0.
    assert report.addressee.brier == 0.0
    # The only populated bin is bin 0 (the 0.00-0.10
    # bin) with count == n_present, mean_confidence
    # == 0.0, accuracy == 0.0, gap == 0.0.
    bins = report.addressee.reliability_bins
    assert bins[0].count == 8
    assert bins[0].mean_confidence == 0.0
    assert bins[0].accuracy == 0.0
    assert bins[0].gap == 0.0
    for b in bins[1:]:
        assert b.count == 0
    # No drift signals fire (gap==0 ⇒ not
    # over/under; only one populated bin ⇒ no
    # flat_curve or bimodal). n_present=8 is also
    # above the insufficient_data threshold (5).
    assert report.addressee.drift_signals == []
    # Candidate thresholds are None (no populated bin
    # in either threshold's neighborhood).
    assert (
        report.addressee.threshold_recommendation["candidate_admit_min"]
        is None
    )
    assert (
        report.reaction.threshold_recommendation["candidate_tie_breaker_min"]
        is None
    )


def test_runner_threshold_recommendation_caveat_mentions_m20_4(tmp_path: Path) -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    runtime = _runtime(tmp_path)
    runtime.llm._responses_by_text = {
        step["text"]: {
            "addressee_hypothesis": {
                "participant_id": "carol",
                "addressed_to_assistant": True,
                "confidence": 0.7,
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "carol",
                "reaction_to_turn_id": "turn_0",
                "reaction_to_participant_id": "carol",
                "is_about_assistant_claim": False,
                "confidence": 0.6,
            },
        }
        for step in fixture
    }
    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
    )
    assert (
        "M20.4"
        in report.addressee.threshold_recommendation["caveat"]
    )
    assert (
        "M20.4"
        in report.reaction.threshold_recommendation["caveat"]
    )


def test_runner_does_not_modify_m20_4_constants() -> None:
    """Sanity: the runner does not mutate the
    `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN` or
    `M20_4_TIE_BREAKER_CONFIDENCE_MIN` constants.
    """
    admit_before = m20_4.M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN
    tb_before = m20_4.M20_4_TIE_BREAKER_CONFIDENCE_MIN
    # Importing the calibration module should not have
    # mutated M20.4 either.
    importlib = __import__("importlib")
    importlib.reload(cal)
    assert m20_4.M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN == admit_before
    assert m20_4.M20_4_TIE_BREAKER_CONFIDENCE_MIN == tb_before


def test_runner_drift_signals_insufficient_data_when_fixture_too_small(
    tmp_path: Path,
) -> None:
    """A 2-step fixture yields n_present < 5; drift
    signals collapse to `insufficient_data`.
    """
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))[:2]
    runtime = _runtime(tmp_path)
    runtime.llm._responses_by_text = {
        step["text"]: {
            "addressee_hypothesis": {
                "participant_id": "carol",
                "addressed_to_assistant": True,
                "confidence": 0.7,
            },
            "reaction_attribution_hypothesis": {
                "participant_id": "carol",
                "reaction_to_turn_id": "turn_0",
                "reaction_to_participant_id": "carol",
                "is_about_assistant_claim": False,
                "confidence": 0.6,
            },
        }
        for step in fixture
    }
    report = cal.run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
    )
    assert report.addressee.n_present == 2
    assert "insufficient_data" in report.addressee.drift_signals
    assert "insufficient_data" in report.reaction.drift_signals
    # Candidate thresholds are `None` for both fields.
    assert (
        report.addressee.threshold_recommendation["candidate_admit_min"]
        is None
    )
    assert (
        report.reaction.threshold_recommendation["candidate_tie_breaker_min"]
        is None
    )


# === Constants ============================================================


def test_m18_7_1_constants_are_frozen() -> None:
    assert cal.M18_7_1_BIN_WIDTH == 0.10
    assert cal.M18_7_1_N_BINS == 10
    assert len(cal.M18_7_1_BIN_LABELS) == 10
    assert cal.M18_7_1_ADMIT_MIN_CURRENT == 0.4
    assert cal.M18_7_1_TIE_BREAKER_MIN_CURRENT == 0.85
    assert cal.M18_7_1_MIN_PRESENT_FOR_DRIFT_SIGNAL == 5
    assert cal.M18_7_1_CALIBRATION_SURFACE_KEY == "m18_7_1_calibration"
    assert (
        cal.M18_7_1_ENGINEERING_PROXY_LABEL
        == "mvp_local_group_attribution_calibration"
    )
    assert "insufficient_data" in cal.ALLOWED_M18_7_1_DRIFT_SIGNALS
    assert "probe" in cal.ALLOWED_M18_7_1_ASSERTION_KIND


def test_drift_signal_enum_is_closed() -> None:
    """The drift signal set is exactly the 5 frozen
    enum values, no more no less.
    """
    assert cal.ALLOWED_M18_7_1_DRIFT_SIGNALS == frozenset({
        "overconfidence_at_high_band",
        "underconfidence_at_low_band",
        "bimodal",
        "flat_curve",
        "insufficient_data",
    })


def test_assertion_kind_enum_is_closed() -> None:
    assert cal.ALLOWED_M18_7_1_ASSERTION_KIND == frozenset({
        "addressee_only",
        "reaction_only",
        "both",
        "neither",
        "probe",
    })
