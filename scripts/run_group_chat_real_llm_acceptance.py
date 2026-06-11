"""Real-LLM end-to-end 5-run group chat acceptance framework.

This script is the **4-sub-capability acceptance gate** for
Hu Tao's group-chat competence. It is the data backbone
for the user's group-chat bar:

  1. 区分说话人目标 (distinguish addressee target)
  2. 区分说话人 (distinguish speakers)
  3. 针对每个人不同建立双向自由能评估渠道
     (per-person bidirectional FEP channels)
  4. 群内表现的人格具有一致性
     (consistent persona across group)

The script wires:

  secrets/openrouter.json (or OPENAI_API_KEY / OPENROUTER_API_KEY)
  -> OpenRouterJSONClient (real LLM)
  -> MVPStateStore (per-run session)
  -> MVPDialogueRuntime (conscious loop, M18.7 hypothesis extraction,
                        M11/M12 user modeling, M20.4 attribution)
  -> run_m18_7_1_calibration_harness (v2 by_pid scoring)
  -> 4-sub-capability metric surface

Each run uses a fresh session directory; the M18.7.1 v2
calibration harness runs in the by_pid mode (Q1 default).
The 5 runs are independent (no state reuse across runs).

Output: a structured JSON summary to stdout with per-run
4-sub-capability breakdowns + 5-run aggregated means +
a per-sub-capability verdict.

Usage:

    python scripts/run_group_chat_real_llm_acceptance.py \\
        --fixture tests/fixtures/m18_7_1_held_out_calibration.json \\
        --n-runs 5 \\
        --session-root-base tmp_group_chat_real_llm_5run

CAVEAT — what this script does NOT do:

- It does NOT mutate the M18.7.2 prompt, the M20.4
  thresholds, or the M11/M12 surfaces. The 4 metrics
  are observational, not intervention.
- It does NOT propose a v3 prompt or a v2.1 calibration
  harness. The script is the **measurement** layer; the
  intervention is a separate milestone.
- The `acceptable` verdict is intentionally conservative
  — it requires **all 4 sub-capabilities** to meet a
  minimum bar simultaneously. Until that happens, the
  script will surface which sub-capability is the
  bottleneck.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from segmentum.dialogue.runtime.m18_7_1_calibration import (
    CalibrationHarnessReport,
    record_m18_7_1_calibration,
    run_m18_7_1_calibration_harness,
    validate_calibration_fixture_shape,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    default_openrouter_client,
)


# === Frozen verdict thresholds (4 sub-capability bars) ====================
# These are conservative first-cut bars. They will be
# revised when the user ratifies a productized criterion.
#
# The bars are documented in the plan
# `prompts/M18.7.1_Harness_V2_Design.md` companion
# (this script is the data source, not the criteria).

# Sub-1 (区分说话人目标): the LLM must catch >=60% of
# GT-addressed turns (recall_on_addressed) AND must not
# over-emit (precision_on_not_addressed >= 0.9).
SUB1_RECALL_ON_ADDRESSED_MIN: float = 0.60
SUB1_PRECISION_ON_NOT_ADDRESSED_MIN: float = 0.90

# Sub-2 (区分说话人): the LLM's emitted `participant_id`
# in addressee_hypothesis must exactly match the fixture
# `speaker_participant_id` for >=70% of decidable turns
# (decidable = LLM emitted a non-empty participant_id).
SUB2_SPEAKER_PID_EXACT_MATCH_MIN: float = 0.70

# Sub-3 (双向 FEP 渠道): the M20.4 producer must have
# fired at least once in the 5-run (proves the producer
# pipeline is alive). 0 fires = the M20.4 producer
# silently never engaged.
SUB3_PRODUCER_ADMIT_TOTAL_MIN: int = 1
# P0-4 dir=True admit >=1 (proves the LLM emitted
# at least one dir=True that passed the 0.7 admit
# threshold).
SUB3_P04_DIR_TRUE_ADMIT_MIN: int = 1
# Per-persona channel coverage: the M11 user-models
# surface must cover >=2 distinct persona ids (not
# just one repeated).
SUB3_PER_PERSONA_CHANNELS_MIN: int = 2

# Sub-4 (人格一致性): the M12.1 personality surface
# must be non-empty and the LLM's emitted reports
# must not contain a `confidence` field that drops
# below 0.2 for any tracked persona (this is a
# placeholder bar — full M12.1 criteria are a
# separate M12.1 milestone).
SUB4_M12_1_PROFILES_NONEMPTY_MIN: int = 1


# === Pid normalization (sub-2) =============================================
# The LLM emits surface ids that are role/alias-equivalent:
# - "bot" / "assistant" / "hutao" → the assistant role
#   (the GT speaker when the assistant speaks, but here we
#   only score addressee pid which can include "talking to
#   the assistant" claims).
# - Case-insensitive: "Carol" == "carol"
# - "speaker_n" / "person_n" → no match (LLM hallucinated
#   a generic id; not a real persona).
# This table is the same role-collapse as the M18.7.1 v2
# `M18_7_1_PID_NORMALIZATION` table (kept in sync with
# the calibration harness).

_PID_NORMALIZATION_TABLE: dict[str, str] = {
    "bot": "bot",
    "assistant": "bot",
    "hutao": "bot",
    "hutao_assistant": "bot",
    "clawdgroupchat_bot": "bot",
}


def _pid_eq(emit: str, gt: str) -> bool:
    """True if two participant ids refer to the same
    persona (case-insensitive, alias-collapsed).
    """
    e = str(emit or "").strip().lower()
    g = str(gt or "").strip().lower()
    if not e or not g:
        return False
    if e == g:
        return True
    e_norm = _PID_NORMALIZATION_TABLE.get(e, e)
    g_norm = _PID_NORMALIZATION_TABLE.get(g, g)
    return e_norm == g_norm


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _load_fixture(path: Path) -> list[Mapping[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError(
            f"fixture must be a JSON list, got {type(raw).__name__}"
        )
    return raw


def _safe_count(mapping: Any, key: str) -> int:
    """Read an int counter from a mapping, defaulting to 0."""
    if not isinstance(mapping, Mapping):
        return 0
    raw = mapping.get(key, 0)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    return 0


def _subcap1_addressee_target(
    harness_report: Any,
) -> dict[str, Any]:
    """Sub-capability 1 — 区分说话人目标.

    Reuse the M18.7.1 P1 split. Read the v2 by_pid
    harness's `addressee_class_breakdown` dict, which
    exposes `recall_on_addressed` and
    `precision_on_not_addressed`. Also surface the
    raw TP/FN/FP counts for transparency.
    """
    addr = harness_report.addressee
    if not isinstance(addr, object) or not hasattr(addr, "addressee_class_breakdown"):
        return {
            "n_present": 0,
            "n_gt_true": 0,
            "n_gt_false": 0,
            "tp_addressed": 0,
            "fn_addressed": 0,
            "tp_not_addressed": 0,
            "fp_not_addressed": 0,
            "recall_on_addressed": 0.0,
            "precision_on_not_addressed": 0.0,
            "verdict": "no_addressee_breakdown",
        }
    bd = addr.addressee_class_breakdown or {}
    recall = float(bd.get("recall_on_addressed", 0.0))
    precision = float(bd.get("precision_on_not_addressed", 0.0))
    # Bar check.
    if (recall >= SUB1_RECALL_ON_ADDRESSED_MIN
            and precision >= SUB1_PRECISION_ON_NOT_ADDRESSED_MIN):
        verdict = "acceptable"
    elif precision < SUB1_PRECISION_ON_NOT_ADDRESSED_MIN:
        verdict = "overemit_false_positives"
    elif recall < SUB1_RECALL_ON_ADDRESSED_MIN:
        verdict = "under_recall_dir_true"
    else:
        verdict = "marginal"
    return {
        "n_present": int(getattr(addr, "n_present", 0)),
        "n_gt_true": int(bd.get("n_gt_true", 0)),
        "n_gt_false": int(bd.get("n_gt_false", 0)),
        "tp_addressed": int(bd.get("tp_addressed", 0)),
        "fn_addressed": int(bd.get("fn_addressed", 0)),
        "tp_not_addressed": int(bd.get("tp_not_addressed", 0)),
        "fp_not_addressed": int(bd.get("fp_not_addressed", 0)),
        "recall_on_addressed": round(recall, 6),
        "precision_on_not_addressed": round(precision, 6),
        "verdict": verdict,
    }


def _subcap2_speaker_identity(
    state: Mapping[str, Any],
    fixture: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sub-capability 2 — 区分说话人.

    The fixture's `group_turn_envelope.speaker_participant_id`
    is the GT. The LLM's emitted `addressee_hypothesis.participant_id`
    is the prediction. We compare the **emitted** pid
    (when the LLM chose to emit one) to the GT speaker
    pid of the same turn. A non-emit is treated as
    "the LLM chose not to attribute" — that is **not**
    a speaker-id mistake, it's a no-emit (sub-cap-1
    territory). For sub-cap-2 we only count turns where
    the LLM emitted a non-empty pid.

    Two metrics:
    - `n_decidable_emits`: LLM emitted non-empty pid.
    - `n_exact_match`: emitted pid == GT speaker pid.
    - `speaker_pid_exact_match_rate`: match / decidable.
    """
    hypotheses = state.get("m18_7_attribution_hypotheses", [])
    if not isinstance(hypotheses, list):
        hypotheses = []
    # The on-disk surface is a flat rolling-window list
    # of entries with `kind: "addressee" | "reaction"`
    # discriminator (NOT nested under
    # `addressee_hypothesis` / `reaction_attribution_hypothesis`).
    # Each entry is shape:
    #   {
    #     "kind": "addressee" | "reaction",
    #     "turn_index": int,
    #     "participant_id": str,  # addressee: whom the speaker
    #                            # is addressing; reaction: whom
    #                            # the assistant is reacting to
    #     "addressed_to_assistant": bool,  # addressee only
    #     "confidence": float,
    #     ...
    #   }
    # We only count `kind == "addressee"` entries (the
    # LLM's "I think the user is talking to X" claim).
    n_decidable = 0
    n_exact_match = 0
    n_close_match = 0
    exact_pids: list[str] = []
    for entry in hypotheses:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("kind", "")) != "addressee":
            continue
        emit_pid = str(entry.get("participant_id", "")).strip()
        if not emit_pid:
            continue
        n_decidable += 1
        ti = entry.get("turn_index", -1)
        # Find the matching fixture turn.
        gt_speaker = ""
        for step in fixture:
            if int(step.get("turn_index", -1)) == int(ti):
                env = step.get("group_turn_envelope", {}) or {}
                gt_speaker = str(env.get("speaker_participant_id", "")).strip()
                break
        if not gt_speaker:
            continue
        if _pid_eq(emit_pid, gt_speaker):
            n_exact_match += 1
            exact_pids.append(emit_pid)
        elif emit_pid.lower() == gt_speaker.lower():
            # Case-insensitive match (e.g. LLM emits
            # "Carol" but GT is "carol") — counts as
            # exact match for v1 (the pid is the same
            # name; capitalization is a surface detail).
            n_exact_match += 1
            exact_pids.append(emit_pid)
        elif (
            emit_pid.lower() in gt_speaker.lower()
            or gt_speaker.lower() in emit_pid.lower()
        ):
            n_close_match += 1
    rate = (n_exact_match / n_decidable) if n_decidable > 0 else 0.0
    if rate >= SUB2_SPEAKER_PID_EXACT_MATCH_MIN:
        verdict = "acceptable"
    elif n_decidable == 0:
        verdict = "no_emits"
    else:
        verdict = "below_bar"
    return {
        "n_decidable_emits": int(n_decidable),
        "n_exact_match": int(n_exact_match),
        "n_close_match": int(n_close_match),
        "speaker_pid_exact_match_rate": round(rate, 6),
        "exact_pid_set": sorted(set(exact_pids)),
        "verdict": verdict,
    }


def _subcap2_speaker_identity_from_harness(
    harness_report: CalibrationHarnessReport,
    fixture: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sub-capability 2 — 区分说话人 (harness-source variant).

    P0-8 follow-up (2026-06-11): the on-disk state
    surface is a rolling window capped at 8 entries
    (M18_7_STATE_SURFACE_CAP), so early turns get
    evicted before the surface is persisted. The
    harness's in-memory view, exposed via
    `harness_report.addressee_predictions` +
    `harness_report.fixture_turn_indices`, has the
    full 1:1 per-turn sequence.

    The M18.7.2 v2 P0-8 5-run stability report
    (commit 39d2ef0) already documented this
    surface-vs-harness discrepancy and concluded
    "the harness's view is the **correct** one for
    scoring — it reads the surface during each
    turn." This function is the P0-8 follow-up that
    re-routes sub-2 to that source.

    Scoring is identical to `_subcap2_speaker_identity`
    (the state-based variant kept for unit tests):
    - Iterate harness predictions.
    - For each `present=True` prediction, look up
      GT speaker pid by turn index from the fixture.
    - Apply `_pid_eq` (alias-collapse) +
      case-insensitive matching.
    - `present=False` (LLM returned `{}`) is a
      no-emit, not a sub-2 mistake.
    """
    n_decidable = 0
    n_exact_match = 0
    n_close_match = 0
    exact_pids: list[str] = []
    for pred, turn_index in zip(
        harness_report.addressee_predictions,
        harness_report.fixture_turn_indices,
    ):
        if not getattr(pred, "present", False):
            continue
        emit_pid = str(getattr(pred, "participant_id", "") or "").strip()
        if not emit_pid:
            continue
        n_decidable += 1
        # Find the matching fixture turn.
        gt_speaker = ""
        for step in fixture:
            if int(step.get("turn_index", -1)) == int(turn_index):
                env = step.get("group_turn_envelope", {}) or {}
                gt_speaker = str(env.get("speaker_participant_id", "")).strip()
                break
        if not gt_speaker:
            continue
        if _pid_eq(emit_pid, gt_speaker):
            n_exact_match += 1
            exact_pids.append(emit_pid)
        elif emit_pid.lower() == gt_speaker.lower():
            # Case-insensitive match (e.g. LLM emits
            # "Carol" but GT is "carol") — counts as
            # exact match (the pid is the same name;
            # capitalization is a surface detail).
            n_exact_match += 1
            exact_pids.append(emit_pid)
        elif (
            emit_pid.lower() in gt_speaker.lower()
            or gt_speaker.lower() in emit_pid.lower()
        ):
            n_close_match += 1
    rate = (n_exact_match / n_decidable) if n_decidable > 0 else 0.0
    if rate >= SUB2_SPEAKER_PID_EXACT_MATCH_MIN:
        verdict = "acceptable"
    elif n_decidable == 0:
        verdict = "no_emits"
    else:
        verdict = "below_bar"
    return {
        "n_decidable_emits": int(n_decidable),
        "n_exact_match": int(n_exact_match),
        "n_close_match": int(n_close_match),
        "speaker_pid_exact_match_rate": round(rate, 6),
        "exact_pid_set": sorted(set(exact_pids)),
        "verdict": verdict,
        "n_decidable_from_harness": int(n_decidable),
        "source": "harness_report",
    }


def _subcap3_bidirectional_fep(
    state: Mapping[str, Any],
    m20_4_diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Sub-capability 3 — 双向自由能评估渠道.

    For now (the framework milestone), we surface:
    - M20.4 producer admit / reject / write-skip / tie-breaker
      counters (proves the producer pipeline is alive).
    - M11 user-models per-persona coverage (proves the
      bidirectional channel — LLM tracks per-persona
      state — exists in the surface).

    The `bidirectional` part (LLM → user and user → LLM
    FEP channels per persona) is a separate M13.6 +
    M11.0 milestone. This script surfaces the
    observable per-persona state; the channel itself
    is a future M-side work item.
    """
    diag = dict(m20_4_diagnostics or {})
    producer_admit_total = _safe_count(diag, "producer_admit_total")
    producer_admit_dir_true = _safe_count(
        diag, "producer_admit_addressee_directed_total"
    )
    producer_admit_dir_false = _safe_count(
        diag, "producer_admit_addressee_not_directed_total"
    )
    producer_reject_total = _safe_count(
        diag, "producer_reject_low_confidence_total"
    )
    producer_reject_dir_true = _safe_count(
        diag, "producer_reject_low_confidence_addressee_directed_total"
    )
    write_path_skip = _safe_count(
        diag, "write_path_skip_addressee_directed_low_confidence_total"
    )
    tie_breaker_engaged = _safe_count(
        diag, "tie_breaker_engaged_addressee_directed_total"
    )
    # M11 user-models per-persona coverage.
    m11_models = state.get("m11_user_models", {}) or {}
    if not isinstance(m11_models, Mapping):
        m11_models = {}
    persona_channels = sorted(str(k) for k in m11_models.keys() if str(k).strip())
    n_persona_channels = len(persona_channels)
    # Verdict: producer alive + at least one dir=True
    # admit + >=2 persona channels.
    failures: list[str] = []
    if producer_admit_total < SUB3_PRODUCER_ADMIT_TOTAL_MIN:
        failures.append("producer_dormant")
    if producer_admit_dir_true < SUB3_P04_DIR_TRUE_ADMIT_MIN:
        failures.append("p04_dir_true_admit_zero")
    if n_persona_channels < SUB3_PER_PERSONA_CHANNELS_MIN:
        failures.append("per_persona_channels_too_few")
    verdict = "acceptable" if not failures else ("+".join(failures) or "marginal")
    return {
        "producer_admit_total": int(producer_admit_total),
        "producer_admit_dir_true": int(producer_admit_dir_true),
        "producer_admit_dir_false": int(producer_admit_dir_false),
        "producer_reject_total": int(producer_reject_total),
        "producer_reject_dir_true": int(producer_reject_dir_true),
        "write_path_skip_dir_true": int(write_path_skip),
        "tie_breaker_engaged_dir_true": int(tie_breaker_engaged),
        "n_persona_channels": int(n_persona_channels),
        "persona_channels": persona_channels,
        "verdict": verdict,
    }


def _subcap4_persona_consistency(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Sub-capability 4 — 群内表现的人格一致性.

    For now (the framework milestone), we surface the
    M12.1 personality surface state. The full
    consistency metric (cross-persona LLM-emitted
    reports must agree on the assistant's persona
    trait distributions) is a future M12.1 milestone.
    We surface:
    - `n_profiles`: how many distinct personas the
      M12.1 surface has profiled.
    - `n_latest_reports`: how many distinct personas
      have a `latest_reports_by_user` entry.
    - `low_confidence_count`: count of personas whose
      `latest_reports` `confidence` field (if any) is
      below 0.2.
    """
    payload = state.get("m12_1_user_personality", {}) or {}
    if not isinstance(payload, Mapping):
        payload = {}
    profiles = payload.get("profiles_by_user", {}) or {}
    latest = payload.get("latest_reports_by_user", {}) or {}
    n_profiles = sum(1 for _ in profiles.keys() if str(_).strip()) if isinstance(profiles, Mapping) else 0
    n_latest = sum(1 for _ in latest.keys() if str(_).strip()) if isinstance(latest, Mapping) else 0
    low_confidence_count = 0
    for _user, report in (latest.items() if isinstance(latest, Mapping) else []):
        if not isinstance(report, Mapping):
            continue
        conf = report.get("confidence", None)
        if isinstance(conf, (int, float)) and float(conf) < 0.2:
            low_confidence_count += 1
    if n_profiles < SUB4_M12_1_PROFILES_NONEMPTY_MIN:
        verdict = "no_m12_1_surface"
    else:
        verdict = "surface_alive"
    return {
        "n_profiles": int(n_profiles),
        "n_latest_reports": int(n_latest),
        "low_confidence_count": int(low_confidence_count),
        "verdict": verdict,
    }


def _run_one(
    *,
    runtime: MVPDialogueRuntime,
    store: MVPStateStore,
    fixture: Sequence[Mapping[str, Any]],
    fixture_path: Path,
    now_base: int,
    time_step: int,
    at: str,
    run_index: int,
) -> dict[str, Any]:
    """Execute one full 5-run-loop iteration.

    The M18.7.1 v2 calibration harness drives the runtime
    through the fixture itself (it calls
    `runtime.run_turn` per fixture step and reads the
    state surface at each turn). Calling `run_turn` here
    AND letting the harness do it would double-process
    the fixture, doubling LLM calls. We let the harness
    drive; the 4 sub-capability metrics read the final
    state surface after the harness returns.
    """
    # v2 by_pid harness (Q1 default scoring mode).
    # The harness internally calls `runtime.run_turn`
    # for each fixture step and accumulates predictions
    # from the state surface.
    harness_report = run_m18_7_1_calibration_harness(
        runtime=runtime,
        fixture=fixture,
        fixture_name=str(fixture_path),
        now_base=now_base,
        time_step=time_step,
        at=at,
        scoring_mode="by_pid",
    )

    state = store.load()
    record_m18_7_1_calibration(state, harness_report, at=at)
    store.save(state)

    m20_4_diag = runtime.get_m20_4_diagnostics() or {}

    sub1 = _subcap1_addressee_target(harness_report)
    sub2 = _subcap2_speaker_identity_from_harness(harness_report, fixture)
    sub3 = _subcap3_bidirectional_fep(state, m20_4_diag)
    sub4 = _subcap4_persona_consistency(state)

    return {
        "run_index": run_index,
        "scoring_mode": harness_report.scoring_mode,
        "verdict": harness_report.drift_signals,  # informational
        "addressee_ece": float(harness_report.addressee.ece),
        "reaction_ece": float(harness_report.reaction.ece),
        "sub1_addressee_target": sub1,
        "sub2_speaker_identity": sub2,
        "sub3_bidirectional_fep": sub3,
        "sub4_persona_consistency": sub4,
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate 5-run means per sub-capability metric."""
    if not run_summaries:
        return {}

    def _mean(key_path: Sequence[str], default: float = 0.0) -> float:
        values: list[float] = []
        for r in run_summaries:
            cur: Any = r
            for k in key_path:
                if not isinstance(cur, Mapping):
                    cur = None
                    break
                cur = cur.get(k)
            if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                values.append(float(cur))
        if not values:
            return default
        return round(sum(values) / len(values), 6)

    def _total(key_path: Sequence[str], default: int = 0) -> int:
        total = 0
        for r in run_summaries:
            cur: Any = r
            for k in key_path:
                if not isinstance(cur, Mapping):
                    cur = None
                    break
                cur = cur.get(k)
            if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                total += int(cur)
        return int(total)

    return {
        "n_runs": len(run_summaries),
        "sub1_means": {
            "recall_on_addressed": _mean(
                ["sub1_addressee_target", "recall_on_addressed"]
            ),
            "precision_on_not_addressed": _mean(
                ["sub1_addressee_target", "precision_on_not_addressed"]
            ),
            "tp_addressed_total": _total(
                ["sub1_addressee_target", "tp_addressed"]
            ),
            "fn_addressed_total": _total(
                ["sub1_addressee_target", "fn_addressed"]
            ),
            "tp_not_addressed_total": _total(
                ["sub1_addressee_target", "tp_not_addressed"]
            ),
            "fp_not_addressed_total": _total(
                ["sub1_addressee_target", "fp_not_addressed"]
            ),
        },
        "sub2_means": {
            "n_decidable_emits_total": _total(
                ["sub2_speaker_identity", "n_decidable_emits"]
            ),
            "n_exact_match_total": _total(
                ["sub2_speaker_identity", "n_exact_match"]
            ),
            "speaker_pid_exact_match_rate_mean": _mean(
                ["sub2_speaker_identity", "speaker_pid_exact_match_rate"]
            ),
        },
        "sub3_means": {
            "producer_admit_total": _total(
                ["sub3_bidirectional_fep", "producer_admit_total"]
            ),
            "producer_admit_dir_true_total": _total(
                ["sub3_bidirectional_fep", "producer_admit_dir_true"]
            ),
            "producer_reject_total": _total(
                ["sub3_bidirectional_fep", "producer_reject_total"]
            ),
            "write_path_skip_dir_true_total": _total(
                ["sub3_bidirectional_fep", "write_path_skip_dir_true"]
            ),
            "tie_breaker_engaged_dir_true_total": _total(
                ["sub3_bidirectional_fep", "tie_breaker_engaged_dir_true"]
            ),
            "n_persona_channels_max": max(
                (
                    r.get("sub3_bidirectional_fep", {}).get(
                        "n_persona_channels", 0
                    )
                    for r in run_summaries
                ),
                default=0,
            ),
        },
        "sub4_means": {
            "n_profiles_max": max(
                (
                    r.get("sub4_persona_consistency", {}).get("n_profiles", 0)
                    for r in run_summaries
                ),
                default=0,
            ),
            "n_latest_reports_max": max(
                (
                    r.get("sub4_persona_consistency", {}).get(
                        "n_latest_reports", 0
                    )
                    for r in run_summaries
                ),
                default=0,
            ),
        },
    }


def _verdict(runs: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    """Compute a 4-sub-capability aggregate verdict.

    Conservative: requires ALL 4 sub-capabilities to
    be `acceptable` (or `surface_alive` for sub-4)
    on every run.
    """
    if not runs:
        return "no_runs"
    failed_subs: set[str] = set()
    for r in runs:
        s1 = r.get("sub1_addressee_target", {}).get("verdict", "")
        if s1 not in ("acceptable",):
            failed_subs.add("sub1")
        s2 = r.get("sub2_speaker_identity", {}).get("verdict", "")
        if s2 not in ("acceptable",):
            failed_subs.add("sub2")
        s3 = r.get("sub3_bidirectional_fep", {}).get("verdict", "")
        if s3 not in ("acceptable",):
            failed_subs.add("sub3")
        s4 = r.get("sub4_persona_consistency", {}).get("verdict", "")
        if s4 not in ("acceptable", "surface_alive"):
            failed_subs.add("sub4")
    if not failed_subs:
        return "all_4_subcap_acceptable"
    if len(failed_subs) == 4:
        return "all_4_subcap_below_bar"
    return f"failed:{'+'.join(sorted(failed_subs))}"


def _init_store_and_runtime(
    root: Path,
    client: Any,
) -> tuple[Any, Any]:
    """Initialize a fresh `MVPStateStore` + `MVPDialogueRuntime`.

    P0-8 follow-up (2026-06-11): the 5-run baseline
    surfaced sub-4 = `no_m12_1_surface` because
    `m12_1_personality_enabled` defaults to `False`
    on a fresh store, and the runtime never sets it.
    Without the flag, M12.1 never runs in
    `MVPDialogueRuntime.run_turn` and the
    `m12_1_user_personality` surface stays empty
    (`profiles_by_user == {}`), so sub-4 reports
    `n_profiles == 0` even when the fixture is long
    enough for M12.1 to write a profile.

    The acceptance gate's sub-4 bar is "M12.1 surface
    alive", which is a **state-surface** check, not a
    LLM verdict. To validate that bar we need M12.1
    actually enabled. This helper primes the store
    BEFORE the runtime is constructed so the runtime
    reads the enabled flag on its first `run_turn`.

    Out of scope: this is the **measurement** path
    only. M12.1's `n_turns_threshold` and "is the
    fixture long enough to surface a profile"
    question is a separate investigation.
    """
    store = MVPStateStore(root=root)
    # MVPStateStore.__post_init__ already wrote the
    # default `m12_1_personality_enabled=False` file.
    # Flip it to True so the runtime's
    # `_m12_1_enabled_for_state` returns True on
    # the first `run_turn`. The 4 sub-key shape of
    # `m12_1_user_personality` matches the runtime
    # default (mvp_loop.py:344-351) so the M12.1
    # state loader's `to_dict()` round-trips cleanly.
    (root / "m12_1_personality_enabled.json").write_text(
        "true", encoding="utf-8"
    )
    (root / "m12_1_user_personality.json").write_text(
        json.dumps(
            {
                "profiles_by_user": {},
                "latest_reports_by_user": {},
                "run_records_by_user": {},
                "consecutive_step1_insufficient_by_user": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    runtime = MVPDialogueRuntime(store=store, llm=client)
    return store, runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 4-sub-capability real-LLM end-to-end "
            "5-run group chat acceptance gate. "
            "Surfaces (1) addressee target, (2) speaker identity, "
            "(3) bidirectional FEP channels, (4) persona consistency."
        )
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Path to a held-out group chat fixture (e.g. "
             "tests/fixtures/m18_7_1_held_out_calibration.json).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of independent runs (default: 5).",
    )
    parser.add_argument(
        "--session-root-base",
        required=True,
        type=Path,
        help=(
            "Base path for per-run session directories. "
            "Each run uses `<base>/run_<i>/` as the "
            "fresh MVPStateStore root."
        ),
    )
    parser.add_argument(
        "--now-base",
        type=int,
        default=1_780_000_000,
        help="Base unix timestamp for the replay (default: 2026-06-01 UTC).",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=60,
        help="Seconds between successive fixture turns (default: 60).",
    )
    args = parser.parse_args()

    fixture_path: Path = args.fixture.resolve()
    base: Path = args.session_root_base.resolve()
    n_runs: int = max(1, int(args.n_runs))

    fixture = _load_fixture(fixture_path)
    shape_errors = validate_calibration_fixture_shape(fixture)
    if shape_errors:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "fixture_shape_validation",
                    "errors": shape_errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    client = default_openrouter_client()
    if client is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "llm_configuration",
                    "error": (
                        "real LLM unavailable: no secrets/openrouter.json "
                        "and no OPENAI_API_KEY / OPENROUTER_API_KEY env var. "
                        "Refusing to run a no-op fake-LLM replay under the "
                        "P0 gate."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3

    base.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    for i in range(1, n_runs + 1):
        run_root = base / f"run_{i}"
        # Wipe any stale state from a prior run.
        if run_root.exists():
            import shutil as _shutil
            _shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        # Connection-reset retry: the underlying
        # `complete_json` raises `RuntimeError` with
        # "OpenRouter chat completion failed" when the
        # upstream closes the connection (HTTP 10054).
        # This is intermittent on `deepseek-v4-flash`
        # (observed 3+ connection resets in one evening,
        # with minimal 30-60s gaps between them — the
        # upstream is flaky for sustained traffic).
        # We retry the whole run up to 5 times with a
        # fresh session directory per attempt, with a
        # short sleep between attempts to let the
        # upstream settle.
        import time as _time
        max_attempts = 5
        last_err: Exception | None = None
        run_summary: dict[str, Any] | None = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                # Wipe and retry with a fresh session
                # directory under run_root/retry_N/.
                import shutil as _shutil
                retry_root = run_root / f"retry_{attempt}"
                if retry_root.exists():
                    _shutil.rmtree(retry_root)
                retry_root.mkdir(parents=True, exist_ok=True)
                store, runtime = _init_store_and_runtime(
                    retry_root, client
                )
            else:
                store, runtime = _init_store_and_runtime(
                    run_root, client
                )
            at = _now_iso8601()
            try:
                run_summary = _run_one(
                    runtime=runtime,
                    store=store,
                    fixture=fixture,
                    fixture_path=fixture_path,
                    now_base=args.now_base,
                    time_step=args.time_step,
                    at=at,
                    run_index=i,
                )
                # Success.
                break
            except RuntimeError as exc:
                last_err = exc
                msg = str(exc)
                if "OpenRouter chat completion failed" not in msg:
                    # Not a connection error — re-raise.
                    raise
                # Connection-reset retry: continue to next
                # attempt. Print a one-line notice to stderr.
                print(
                    f"[run {i}] attempt {attempt}/{max_attempts} "
                    f"connection-reset: {msg[:200]!r}; "
                    f"sleeping 30s and retrying...",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt >= max_attempts:
                    raise
                _time.sleep(30)
        if run_summary is None:
            # All attempts failed.
            raise last_err  # type: ignore[misc]
        run_summary["session_root"] = str(run_root)
        run_summaries.append(run_summary)

    agg = _aggregate_runs(run_summaries)
    overall = _verdict(run_summaries, agg)

    summary: dict[str, Any] = {
        "ok": True,
        "stage": "group_chat_5run_acceptance_complete",
        "n_runs": n_runs,
        "fixture_name": str(fixture_path),
        "scoring_mode": "by_pid",
        "sub_capability_bars": {
            "sub1_recall_on_addressed_min": SUB1_RECALL_ON_ADDRESSED_MIN,
            "sub1_precision_on_not_addressed_min": (
                SUB1_PRECISION_ON_NOT_ADDRESSED_MIN
            ),
            "sub2_speaker_pid_exact_match_min": (
                SUB2_SPEAKER_PID_EXACT_MATCH_MIN
            ),
            "sub3_producer_admit_total_min": SUB3_PRODUCER_ADMIT_TOTAL_MIN,
            "sub3_p04_dir_true_admit_min": SUB3_P04_DIR_TRUE_ADMIT_MIN,
            "sub3_per_persona_channels_min": SUB3_PER_PERSONA_CHANNELS_MIN,
            "sub4_m12_1_profiles_nonempty_min": (
                SUB4_M12_1_PROFILES_NONEMPTY_MIN
            ),
        },
        "runs": run_summaries,
        "aggregate_5run": agg,
        "verdict": overall,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
