"""Tests for the 4-sub-capability real-LLM group chat
acceptance framework (`scripts/run_group_chat_real_llm_acceptance.py`).

These tests are **pure-function tests** — they do not
require the real LLM. They validate the 4 sub-capability
metric functions and the verdict logic against synthetic
inputs. The real-LLM 5-run end-to-end test is a separate
operator-only run (no automated CI), per the M18.7.x
P0 gate pattern (P0-7 etc. are operator-only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# pylint: disable=wrong-import-position
from run_group_chat_real_llm_acceptance import (  # type: ignore[import-not-found]
    SUB1_PRECISION_ON_NOT_ADDRESSED_MIN,
    SUB1_RECALL_ON_ADDRESSED_MIN,
    SUB2_SPEAKER_PID_EXACT_MATCH_MIN,
    SUB3_P04_DIR_TRUE_ADMIT_MIN,
    SUB3_PER_PERSONA_CHANNELS_MIN,
    SUB3_PRODUCER_ADMIT_TOTAL_MIN,
    SUB4_M12_1_PROFILES_NONEMPTY_MIN,
    _aggregate_runs,
    _subcap1_addressee_target,
    _subcap2_speaker_identity,
    _subcap2_speaker_identity_from_harness,
    _subcap2_speaker_identity_from_turn_log,
    _subcap3_bidirectional_fep,
    _producer_admits_from_bus_messages,
    _subcap4_persona_consistency,
    _verdict,
)
from segmentum.dialogue.runtime.m18_7_1_calibration import (  # type: ignore[import-not-found]
    AddresseePrediction,
    CalibrationFieldReport,
    CalibrationHarnessReport,
)


# === Frozen: bar constants (T1) ============================================


def test_bars_are_documented_thresholds():
    """The bar constants must be conservative first-cut
    thresholds. They are the acceptance gate for the
    user's 4-sub-capability bar. If the user revises
    them, this test will fail and force a documentation
    update.
    """
    assert SUB1_RECALL_ON_ADDRESSED_MIN == 0.60
    assert SUB1_PRECISION_ON_NOT_ADDRESSED_MIN == 0.90
    assert SUB2_SPEAKER_PID_EXACT_MATCH_MIN == 0.70
    assert SUB3_PRODUCER_ADMIT_TOTAL_MIN == 1
    assert SUB3_P04_DIR_TRUE_ADMIT_MIN == 1
    assert SUB3_PER_PERSONA_CHANNELS_MIN == 2
    assert SUB4_M12_1_PROFILES_NONEMPTY_MIN == 1


# === Sub-capability 1: addressee target (T2-T3) ============================


class _FakeAddrReport:
    def __init__(self, breakdown: dict) -> None:
        self.addressee = _FakeAddr(breakdown)


class _FakeAddr:
    def __init__(self, breakdown: dict) -> None:
        self.addressee_class_breakdown = breakdown
        self.n_present = (
            breakdown.get("n_gt_true", 0) + breakdown.get("n_gt_false", 0)
        )


def test_sub1_acceptable_when_recall_and_precision_meet_bars():
    """T2: sub-1 acceptable iff recall >= 0.6 AND
    precision >= 0.9 simultaneously.
    """
    breakdown = {
        "n_gt_true": 4, "n_gt_false": 4, "n_unknown": 0,
        "tp_addressed": 3, "fn_addressed": 1,
        "tp_not_addressed": 4, "fp_not_addressed": 0,
        "recall_on_addressed": 0.75, "precision_on_not_addressed": 1.0,
    }
    r = _subcap1_addressee_target(_FakeAddrReport(breakdown))
    assert r["verdict"] == "acceptable", r


def test_sub1_fails_when_precision_below_bar():
    """T3: sub-1 verdict is `overemit_false_positives`
    when precision drops below 0.9 even if recall is fine.
    """
    breakdown = {
        "n_gt_true": 2, "n_gt_false": 4, "n_unknown": 0,
        "tp_addressed": 2, "fn_addressed": 0,
        "tp_not_addressed": 2, "fp_not_addressed": 2,
        "recall_on_addressed": 1.0, "precision_on_not_addressed": 0.5,
    }
    r = _subcap1_addressee_target(_FakeAddrReport(breakdown))
    assert r["verdict"] == "overemit_false_positives", r


# === Sub-capability 2: speaker identity (T4-T5) ============================


def test_sub2_acceptable_when_pids_match():
    """T4: 4/4 exact matches on 4 decidable emits → acceptable.
    Uses the on-disk flat surface schema:
    `kind: "addressee"` entries with `participant_id` at
    the entry root, NOT nested under `addressee_hypothesis`.
    """
    state = {
        "m18_7_attribution_hypotheses": [
            {"kind": "addressee", "turn_index": 0, "participant_id": "carol"},
            {"kind": "addressee", "turn_index": 1, "participant_id": "dave"},
            {"kind": "addressee", "turn_index": 2, "participant_id": "carol"},
            {"kind": "addressee", "turn_index": 3, "participant_id": "dave"},
        ]
    }
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "dave"}},
        {"turn_index": 2, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 3, "group_turn_envelope": {"speaker_participant_id": "dave"}},
    ]
    r = _subcap2_speaker_identity(state, fixture)
    assert r["n_decidable_emits"] == 4
    assert r["n_exact_match"] == 4
    assert r["speaker_pid_exact_match_rate"] == 1.0
    assert r["verdict"] == "acceptable", r


def test_sub2_no_emits_is_not_below_bar():
    """T5: zero decidable emits → verdict `no_emits`
    (not `below_bar`); the LLM chose not to attribute
    rather than being wrong.
    """
    state = {"m18_7_attribution_hypotheses": []}
    fixture: list[dict] = []
    r = _subcap2_speaker_identity(state, fixture)
    assert r["n_decidable_emits"] == 0
    assert r["verdict"] == "no_emits", r


def test_sub2_below_bar_when_mismatch():
    """T5b: 2/3 matches (rate 0.667) below the 0.7 bar.
    Surface uses flat `kind: "addressee"` entries.
    """
    state = {
        "m18_7_attribution_hypotheses": [
            {"kind": "addressee", "turn_index": 0, "participant_id": "carol"},
            {"kind": "addressee", "turn_index": 1, "participant_id": "dave"},
            {"kind": "addressee", "turn_index": 2, "participant_id": "carol"},
        ]
    }
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "dave"}},
        {"turn_index": 2, "group_turn_envelope": {"speaker_participant_id": "dave"}},
    ]
    r = _subcap2_speaker_identity(state, fixture)
    assert r["n_decidable_emits"] == 3
    assert r["n_exact_match"] == 2
    assert r["verdict"] == "below_bar", r


def test_sub2_flat_surface_schema_is_required():
    """T5c: the on-disk surface is FLAT entries
    (`kind: "addressee"` discriminator), NOT nested
    under `addressee_hypothesis`. If a future change
    nests the surface, this test will fail and force
    a re-write of `_subcap2_speaker_identity`.
    """
    state = {
        "m18_7_attribution_hypotheses": [
            # legacy nested — must be ignored
            {"turn_index": 0,
             "addressee_hypothesis": {"participant_id": "carol"}},
            # new flat — must be counted
            {"kind": "addressee", "turn_index": 1, "participant_id": "dave"},
        ]
    }
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "dave"}},
    ]
    r = _subcap2_speaker_identity(state, fixture)
    # Only the flat entry counts.
    assert r["n_decidable_emits"] == 1
    assert r["n_exact_match"] == 1
    assert r["speaker_pid_exact_match_rate"] == 1.0
    assert r["verdict"] == "acceptable", r


def test_sub2_case_insensitive_and_alias_collapse():
    """T5d: pid normalization collapses "Carol"=="carol"
    (case-insensitive) and "bot"=="assistant" (alias
    via `_pid_eq`).
    """
    state = {
        "m18_7_attribution_hypotheses": [
            {"kind": "addressee", "turn_index": 0, "participant_id": "Carol"},
            {"kind": "addressee", "turn_index": 1, "participant_id": "bot"},
            {"kind": "addressee", "turn_index": 2, "participant_id": "Assistant"},
        ]
    }
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "bot"}},
        {"turn_index": 2, "group_turn_envelope": {"speaker_participant_id": "assistant"}},
    ]
    r = _subcap2_speaker_identity(state, fixture)
    assert r["n_decidable_emits"] == 3
    assert r["n_exact_match"] == 3
    assert r["speaker_pid_exact_match_rate"] == 1.0


def test_sub2_turn_log_scores_structured_speaker_not_addressee():
    """Sub-2 measures the ingress speaker identity contract."""
    fixture = [
        {
            "turn_index": 0,
            "group_turn_envelope": {"speaker_participant_id": "carol"},
        },
        {
            "turn_index": 1,
            "group_turn_envelope": {"speaker_participant_id": "dave"},
        },
    ]
    turn_log = [
        {
            "event": "turn",
            "turn_index": 0,
            "speaker_participant_id": "carol",
            "diagnostics": {
                "bus_messages": [
                    {
                        "type": "M18_7_2_AddresseeHypothesisAdmitted",
                        "participant_id": "assistant",
                    }
                ]
            },
        },
        {
            "event": "turn",
            "turn_index": 1,
            "group_turn_binding": {
                "current_speaker_participant_id": "dave",
            },
        },
    ]
    r = _subcap2_speaker_identity_from_turn_log(turn_log, fixture)
    assert r["n_decidable_emits"] == 2
    assert r["n_exact_match"] == 2
    assert r["speaker_pid_exact_match_rate"] == 1.0
    assert r["source"] == "structured_turn_log"


# === Sub-capability 3: bidirectional FEP (T6-T7) ===========================


def test_sub3_acceptable_when_producer_alive_and_channels_wide():
    """T6: producer admit >=1 + dir_true admit >=1 +
    persona channels >=2 → acceptable.
    """
    state = {"m11_user_models": {"carol": {}, "dave": {}, "eve": {}}}
    diag = {
        "producer_admit_total": 5,
        "producer_admit_addressee_directed_total": 2,
        "producer_admit_addressee_not_directed_total": 3,
        "producer_reject_low_confidence_addressee_directed_total": 0,
    }
    r = _subcap3_bidirectional_fep(state, diag)
    assert r["producer_admit_total"] == 5
    assert r["producer_admit_dir_true"] == 2
    assert r["n_persona_channels"] == 3
    assert r["verdict"] == "acceptable", r


def test_sub3_producer_dormant_blocks_acceptance():
    """T7: producer_admit_total == 0 → verdict mentions
    `producer_dormant`.
    """
    state = {"m11_user_models": {"carol": {}, "dave": {}}}
    diag: dict = {
        "producer_admit_total": 0,
        "producer_admit_addressee_directed_total": 0,
        "producer_admit_addressee_not_directed_total": 0,
    }
    r = _subcap3_bidirectional_fep(state, diag)
    assert "producer_dormant" in r["verdict"], r


def test_sub3_dir_true_zero_blocks_acceptance():
    """T7b: producer admit >=1 but dir_true admit == 0
    → verdict mentions `p04_dir_true_admit_zero`.
    """
    state = {"m11_user_models": {"carol": {}, "dave": {}}}
    diag = {
        "producer_admit_total": 3,
        "producer_admit_addressee_directed_total": 0,
        "producer_admit_addressee_not_directed_total": 3,
    }
    r = _subcap3_bidirectional_fep(state, diag)
    assert "p04_dir_true_admit_zero" in r["verdict"], r


def test_sub3_bus_events_override_stale_diagnostic_admit_counts():
    """Producer admission counts match the emitted audit envelopes."""
    bus = [
        {
            "type": "AddresseeTargetMatchAdmitted",
            "hypothesis": {
                "addressed_to_assistant": True,
                "confidence": 0.8,
            },
            "aggregation_kind": "single_strong",
        },
        {
            "type": "AddresseeTargetMatchAdmitted",
            "hypothesis": {
                "addressed_to_assistant": False,
                "confidence": 0.95,
            },
            "aggregation_kind": "single_strong",
        },
    ]
    counts = _producer_admits_from_bus_messages(bus)
    assert counts["producer_admit_total"] == 2
    assert counts["producer_admit_addressee_directed_total"] == 1

    state = {"m11_user_models": {"carol": {}, "dave": {}}}
    r = _subcap3_bidirectional_fep(
        state,
        {
            "producer_admit_total": 0,
            "producer_admit_addressee_directed_total": 0,
        },
        bus_messages=bus,
    )
    assert r["producer_admit_total"] == 2
    assert r["producer_admit_dir_true"] == 1
    assert r["source"] == "turn_log_bus_events"
    assert r["verdict"] == "acceptable"


# === Sub-capability 4: persona consistency (T8) ============================


def test_sub4_surface_alive_when_profiles_nonempty():
    """T8: 2 profiles, 2 latest reports → surface_alive."""
    state = {
        "m12_1_user_personality": {
            "profiles_by_user": {"carol": {}, "dave": {}},
            "latest_reports_by_user": {
                "carol": {"confidence": 0.7},
                "dave": {"confidence": 0.85},
            },
        }
    }
    r = _subcap4_persona_consistency(state)
    assert r["n_profiles"] == 2
    assert r["n_latest_reports"] == 2
    assert r["verdict"] == "surface_alive", r


def test_sub4_no_surface_when_empty():
    """T8b: empty m12.1 surface → no_m12_1_surface."""
    state: dict = {"m12_1_user_personality": {}}
    r = _subcap4_persona_consistency(state)
    assert r["verdict"] == "no_m12_1_surface", r


# === Aggregate + verdict (T9-T10) =========================================


def test_aggregate_5run_sums_sub1_counts():
    """T9: aggregate sums TP/FN/FP across runs and means
    the rates per the means-not-sums rule.
    """
    run1 = {
        "sub1_addressee_target": {
            "recall_on_addressed": 0.5,
            "precision_on_not_addressed": 1.0,
            "tp_addressed": 2, "fn_addressed": 2,
            "tp_not_addressed": 4, "fp_not_addressed": 0,
        },
        "sub2_speaker_identity": {
            "n_decidable_emits": 3, "n_exact_match": 2,
            "speaker_pid_exact_match_rate": 0.667,
        },
        "sub3_bidirectional_fep": {
            "producer_admit_total": 5,
            "producer_admit_dir_true": 1,
            "producer_reject_total": 1,
            "write_path_skip_dir_true": 0,
            "tie_breaker_engaged_dir_true": 0,
            "n_persona_channels": 2,
        },
        "sub4_persona_consistency": {
            "n_profiles": 2, "n_latest_reports": 2,
        },
    }
    agg = _aggregate_runs([run1])
    assert agg["sub1_means"]["recall_on_addressed"] == 0.5
    assert agg["sub1_means"]["tp_addressed_total"] == 2
    assert agg["sub2_means"]["n_decidable_emits_total"] == 3
    assert agg["sub2_means"]["n_exact_match_total"] == 2


def test_verdict_all_acceptable():
    """T10: when all 4 sub-capabilities are acceptable
    on every run, the verdict is `all_4_subcap_acceptable`.
    """
    good_run = {
        "sub1_addressee_target": {
            "verdict": "acceptable",
            "recall_on_addressed": 0.75,
            "precision_on_not_addressed": 1.0,
        },
        "sub2_speaker_identity": {
            "verdict": "acceptable",
            "n_decidable_emits": 4, "n_exact_match": 4,
            "speaker_pid_exact_match_rate": 1.0,
        },
        "sub3_bidirectional_fep": {
            "verdict": "acceptable",
            "producer_admit_total": 5,
            "producer_admit_dir_true": 1,
            "n_persona_channels": 3,
        },
        "sub4_persona_consistency": {
            "verdict": "surface_alive",
            "n_profiles": 2, "n_latest_reports": 2,
        },
    }
    agg = _aggregate_runs([good_run])
    v = _verdict([good_run], agg)
    assert v == "all_4_subcap_acceptable", v


def test_verdict_lists_failing_subs():
    """T10b: when sub1 and sub2 fail, the verdict is
    `failed:sub1+sub2`.
    """
    bad_run = {
        "sub1_addressee_target": {
            "verdict": "under_recall_dir_true",
            "recall_on_addressed": 0.5,
            "precision_on_not_addressed": 1.0,
        },
        "sub2_speaker_identity": {
            "verdict": "below_bar",
            "n_decidable_emits": 3, "n_exact_match": 2,
            "speaker_pid_exact_match_rate": 0.667,
        },
        "sub3_bidirectional_fep": {
            "verdict": "acceptable",
            "producer_admit_total": 5,
            "producer_admit_dir_true": 1,
            "n_persona_channels": 3,
        },
        "sub4_persona_consistency": {
            "verdict": "surface_alive",
            "n_profiles": 2, "n_latest_reports": 2,
        },
    }
    agg = _aggregate_runs([bad_run])
    v = _verdict([bad_run], agg)
    assert v == "failed:sub1+sub2", v


# === Sub-2: harness-source variant (P0-8 follow-up) =====================


def _make_stub_harness_report(
    *,
    addressee_preds: list,
    turn_indices: list[int],
) -> CalibrationHarnessReport:
    """Build a minimal CalibrationHarnessReport for sub-2 tests.

    Only `addressee_predictions` + `fixture_turn_indices` matter
    to `_subcap2_speaker_identity_from_harness`; the addressee
    and reaction reports can be empty placeholders. We pass
    `threshold_recommendation={}` and `reliability_bins=[]` to
    satisfy the CalibrationFieldReport required fields.
    """
    empty_field = CalibrationFieldReport(
        n_total=0, n_present=0, n_unknown=0,
        n_correct=0, n_incorrect=0,
        accuracy=0.0, brier=0.0, ece=0.0,
        reliability_bins=[],
        drift_signals=[],
        threshold_recommendation={},
    )
    return CalibrationHarnessReport(
        fixture_name="<test>",
        n_fixtures=len(turn_indices),
        addressee=empty_field,
        reaction=empty_field,
        drift_signals=[],
        scoring_mode="by_pid",
        addressee_predictions=tuple(addressee_preds),
        fixture_turn_indices=tuple(turn_indices),
    )


def test_sub2_from_harness_sees_full_in_memory_sequence():
    """T5e: harness report exposes in-memory predictions, so
    `_subcap2_speaker_identity_from_harness` sees the full
    sequence (e.g. 4/4 emits) — even when the on-disk
    state has been evicted by the rolling-window cap=8.

    The state-based variant would see only the last 2
    entries; the harness variant sees all 4.
    """
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "dave"}},
        {"turn_index": 2, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 3, "group_turn_envelope": {"speaker_participant_id": "dave"}},
    ]
    preds = [
        AddresseePrediction(present=True, addressed_to_assistant=False,
                            participant_id="carol", confidence=0.8),
        AddresseePrediction(present=True, addressed_to_assistant=True,
                            participant_id="dave", confidence=0.7),
        AddresseePrediction(present=True, addressed_to_assistant=False,
                            participant_id="dave", confidence=0.6),  # wrong: carol speaker, dave emit
        AddresseePrediction(present=True, addressed_to_assistant=False,
                            participant_id="dave", confidence=0.9),
    ]
    report = _make_stub_harness_report(
        addressee_preds=preds, turn_indices=[0, 1, 2, 3]
    )
    r = _subcap2_speaker_identity_from_harness(report, fixture)
    assert r["n_decidable_emits"] == 4  # all 4 emits
    assert r["n_exact_match"] == 3      # turn 2 mismatch
    assert r["speaker_pid_exact_match_rate"] == 0.75
    assert r["source"] == "harness_report"
    assert r["verdict"] == "acceptable", r


def test_sub2_from_harness_no_emits_is_not_below_bar():
    """T5f: harness report with zero `present=True` emits
    → verdict `no_emits` (not `below_bar`); the LLM
    chose not to attribute rather than being wrong.
    """
    fixture: list[dict] = []
    preds = [
        AddresseePrediction(present=False, addressed_to_assistant=False,
                            participant_id="", confidence=0.0),
        AddresseePrediction(present=False, addressed_to_assistant=False,
                            participant_id="", confidence=0.0),
    ]
    report = _make_stub_harness_report(
        addressee_preds=preds, turn_indices=[0, 1]
    )
    r = _subcap2_speaker_identity_from_harness(report, fixture)
    assert r["n_decidable_emits"] == 0
    assert r["verdict"] == "no_emits", r


def test_sub2_from_harness_uses_pid_normalization():
    """T5g: harness-source sub-2 still applies the
    `_pid_eq` alias-collapse + case-insensitive
    normalization (e.g. 'Carol' == 'carol',
    'bot' == 'assistant').
    """
    fixture = [
        {"turn_index": 0, "group_turn_envelope": {"speaker_participant_id": "carol"}},
        {"turn_index": 1, "group_turn_envelope": {"speaker_participant_id": "bot"}},
    ]
    preds = [
        AddresseePrediction(present=True, addressed_to_assistant=False,
                            participant_id="Carol", confidence=0.8),
        AddresseePrediction(present=True, addressed_to_assistant=True,
                            participant_id="assistant", confidence=0.7),
    ]
    report = _make_stub_harness_report(
        addressee_preds=preds, turn_indices=[0, 1]
    )
    r = _subcap2_speaker_identity_from_harness(report, fixture)
    assert r["n_decidable_emits"] == 2
    assert r["n_exact_match"] == 2
    assert r["speaker_pid_exact_match_rate"] == 1.0
    assert r["verdict"] == "acceptable", r


# === M12.1 enable in group chat acceptance path (P0-8 fix #2) ============


def test_init_store_and_runtime_enables_m12_1():
    """T5h: `_init_store_and_runtime` primes
    `m12_1_personality_enabled=True` + the 4-key
    `m12_1_user_personality` shape BEFORE the
    runtime is constructed. The runtime's
    `_m12_1_enabled_for_state` reads `True` on the
    first `run_turn`, and the M12.1 state loader
    sees a 4-sub-key dict (matching the runtime
    default at mvp_loop.py:344-351).

    Without this prime, sub-4 reports
    `no_m12_1_surface` 5/5 because the runtime
    never enables M12.1 (the persona-init path is
    the only one that flips the flag, and the
    acceptance script does not call it).
    """
    import json
    import tempfile
    from pathlib import Path
    from run_group_chat_real_llm_acceptance import (  # type: ignore[import-not-found]
        _init_store_and_runtime,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "test_run"
        # Use a dummy LLM client (the runtime is
        # constructed but `run_turn` is not called
        # in this test).
        class _DummyClient:
            pass
        store, _runtime = _init_store_and_runtime(root, _DummyClient())
        state = store.load()
        assert state["m12_1_personality_enabled"] is True, state
        m12_1 = state["m12_1_user_personality"]
        # The 4-sub-key shape must match the runtime
        # default exactly (mvp_loop.py:344-351).
        assert set(m12_1.keys()) == {
            "profiles_by_user",
            "latest_reports_by_user",
            "run_records_by_user",
            "consecutive_step1_insufficient_by_user",
        }, m12_1
        # And the persona dicts must be empty
        # initially (no profiles written yet).
        assert m12_1["profiles_by_user"] == {}
        assert m12_1["latest_reports_by_user"] == {}
        assert m12_1["run_records_by_user"] == {}
        assert m12_1["consecutive_step1_insufficient_by_user"] == {}
