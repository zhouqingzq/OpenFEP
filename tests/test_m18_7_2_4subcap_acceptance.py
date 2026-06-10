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
    _subcap3_bidirectional_fep,
    _subcap4_persona_consistency,
    _verdict,
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
