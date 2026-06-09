"""Run the M18.7.1 held-out calibration harness against the
real OpenRouter LLM (via `MVPDialogueRuntime.run_turn`).

M18.7.1 is a pure analysis layer; this script is a thin
CLI wrapper that wires:

  secrets/openrouter.json (or OPENAI_API_KEY / OPENROUTER_API_KEY)
  -> OpenRouterJSONClient (real LLM, default model: deepseek/deepseek-v4-flash)
  -> MVPDialogueRuntime (conscious loop, M18.7 hypothesis extraction)
  -> run_m18_7_1_calibration_harness (M18.7.1 pure function, v2 modes)
  -> record_m18_7_1_calibration (writes state["m18_7_1_calibration"])

The script does NOT modify the M18.7 prompt / normalize, does
NOT mutate the M20.4 threshold constants, and does NOT
auto-revise 0.4 / 0.85. It only surfaces
`candidate_admit_min` / `candidate_tie_breaker_min` with the
frozen caveat `"decision belongs to M20.4; M18.7.1 only
surfaces"`.

Real-LLM replay of the held-out fixture is the M18.7.1 P0
acceptance gate (replacing the structural-only STRUCTURAL
status in `reports/m18_7_1_calibration_summary.md`).

v2 (2026-06-09) scoring modes (Q1 default = `by_pid`):

- `by_pid` (default): score on
  `reaction_to_participant_id` +
  `is_about_assistant_claim` joint correctness with pid
  normalization. Primary v2 mode.
- `by_turn_id_resolved`: v1 scoring on
  `reaction_to_turn_id` but with placeholder
  resolution at runner-time
  (`turn_<assistant_prior_turn_id>` etc.).
- `by_turn_id_v1`: byte-identical v1 scoring.
- `all`: run the 3 modes on the same fixture and
  return all 3 sub-reports (state surface write uses
  the by_pid report as the canonical primary).

Usage:

    python scripts/run_m18_7_1_real_llm_calibration.py \\
        --fixture tests/fixtures/m18_7_1_held_out_calibration.json \\
        --session-root tmp_m18_7_1_real_llm \\
        --scoring-mode by_pid

The script always prints the calibration summary as JSON to
stdout, and exits non-zero if the LLM is unavailable so
operators notice the gate failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.m18_7_1_calibration import (
    CalibrationHarnessReport,
    M18_7_1_SCORING_MODES,
    record_m18_7_1_calibration,
    run_m18_7_1_calibration_harness,
    validate_calibration_fixture_shape,
)
from segmentum.dialogue.runtime.mvp_loop import (
    MVPDialogueRuntime,
    MVPStateStore,
    default_openrouter_client,
)


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _load_fixture(path: Path) -> list[Mapping[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError(
            f"fixture must be a JSON list, got {type(raw).__name__}"
        )
    return raw


def _load_pid_override(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"pid-normalization-override must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"pid-normalization-override keys/values must be "
                f"strings; got {type(k).__name__}={type(v).__name__}"
            )
        out[k] = v
    return out


def _classify_calibration(ece: float, brier: float) -> str:
    """Map (ECE, Brier) to a coarse verdict per the M18.7.1 P0
    bands agreed in the planning conversation:

      ECE < 0.05 AND Brier < 0.10 -> "well_calibrated"
      ECE < 0.15 AND Brier < 0.25 -> "moderate_drift"
      ECE >= 0.15 OR  Brier >= 0.25 -> "severe_drift_recommend_m20_4"
    """
    if ece < 0.05 and brier < 0.10:
        return "well_calibrated"
    if ece < 0.15 and brier < 0.25:
        return "moderate_drift"
    return "severe_drift_recommend_m20_4"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the M18.7.1 held-out calibration harness against "
            "the real LLM (secrets/openrouter.json or env vars)."
        )
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Path to the M18.7.1 held-out calibration fixture JSON.",
    )
    parser.add_argument(
        "--session-root",
        required=True,
        type=Path,
        help=(
            "Empty session directory for the MVPStateStore. "
            "The directory must NOT already contain state files "
            "(use a fresh path per replay)."
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
    parser.add_argument(
        "--scoring-mode",
        choices=["by_pid", "by_turn_id_resolved",
                 "by_turn_id_v1", "all"],
        default="by_pid",
        help=(
            "Reaction-side scoring mode (v2). Default: by_pid "
            "(Q1). `all` runs the 3 modes and returns the triple."
        ),
    )
    parser.add_argument(
        "--pid-normalization-override",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with pid normalization overrides "
            "(merged into the default M18_7_1_PID_NORMALIZATION "
            "table for the duration of the run; honored in by_pid "
            "mode only)."
        ),
    )
    parser.add_argument(
        "--no-resolve-placeholders",
        action="store_true",
        help=(
            "Skip placeholder resolution even in by_turn_id_resolved "
            "mode. Useful for diagnosing the placeholder pattern."
        ),
    )
    args = parser.parse_args()

    fixture_path: Path = args.fixture.resolve()
    session_root: Path = args.session_root.resolve()

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

    pid_override = _load_pid_override(args.pid_normalization_override)

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

    session_root.mkdir(parents=True, exist_ok=True)
    store = MVPStateStore(root=session_root)
    runtime = MVPDialogueRuntime(store=store, llm=client)

    at = _now_iso8601()
    resolve_placeholders = not args.no_resolve_placeholders

    reports: dict[str, CalibrationHarnessReport] | None = None
    if args.scoring_mode == "all":
        # Run 3 modes on the same fixture. The runner is
        # idempotent (pure function over the runtime's
        # accumulated state), so we replay the fixture
        # 3 times against the same runtime. With a
        # real LLM the 3 sub-reports may differ slightly
        # (non-determinism); with FakeJSONLLM they
        # match exactly. `all` is the "comparison"
        # mode for diagnosing the measurement fix.
        reports = {}
        for mode in (
            "by_pid",
            "by_turn_id_resolved",
            "by_turn_id_v1",
        ):
            sub = run_m18_7_1_calibration_harness(
                runtime=runtime,
                fixture=fixture,
                fixture_name=str(fixture_path),
                now_base=args.now_base,
                time_step=args.time_step,
                at=at,
                scoring_mode=mode,
                pid_normalization_override=pid_override,
                resolve_placeholders=resolve_placeholders,
            )
            reports[mode] = sub
        # Use the by_pid report as the canonical primary
        # for state surface writing.
        harness_report = reports["by_pid"]
    else:
        if args.scoring_mode not in M18_7_1_SCORING_MODES:
            # Defensive: argparse should have caught this.
            print(
                json.dumps(
                    {
                        "ok": False,
                        "stage": "scoring_mode_validation",
                        "error": (
                            f"unknown scoring_mode: {args.scoring_mode!r}; "
                            f"allowed: {sorted(M18_7_1_SCORING_MODES)}"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        harness_report = run_m18_7_1_calibration_harness(
            runtime=runtime,
            fixture=fixture,
            fixture_name=str(fixture_path),
            now_base=args.now_base,
            time_step=args.time_step,
            at=at,
            scoring_mode=args.scoring_mode,
            pid_normalization_override=pid_override,
            resolve_placeholders=resolve_placeholders,
        )

    state = store.load()
    record_m18_7_1_calibration(state, harness_report, at=at)
    store.save(state)

    addr = harness_report.addressee
    react = harness_report.reaction
    ece = float(addr.ece) + float(react.ece)  # coarse aggregate
    brier = float(addr.brier) + float(react.brier)
    verdict = _classify_calibration(ece / 2.0, brier / 2.0)

    summary: dict[str, Any] = {
        "ok": True,
        "stage": "real_llm_replay_complete",
        "scoring_mode": harness_report.scoring_mode,
        "fixture_name": harness_report.fixture_name,
        "n_fixtures": harness_report.n_fixtures,
        "at": at,
        "verdict": verdict,
        "addressee": addr.to_dict(),
        "reaction": react.to_dict(),
        "fixture_warnings": list(harness_report.fixture_warnings),
        "drift_signals": list(harness_report.drift_signals),
        "threshold_recommendation": state[
            "m18_7_1_calibration"
        ]["threshold_recommendation"],
        "session_root": str(session_root),
    }
    # P0-7 (2026-06-09) — surface the M20.4 per-sub-class
    # diagnostic counters (P0-4 producer admit + P0-5
    # write-path skip + P0-6 tie-breaker engagement) that
    # `runtime.run_turn` accumulates on the state surface
    # during the fixture replay. The harness reads the
    # counters from the runtime's `_last_m20_4_diagnostics`
    # cache (the M20.4 diagnostics key is in-memory only
    # because it is not in `MVPStateStore.SYSTEM_FILE_DEFAULTS`).
    # The surface is what the M20.4 owner needs to validate
    # the producer / write / tie-breaker sub-class split on
    # a 5-run stability data set.
    if harness_report.m20_4_diagnostics:
        summary["m20_4_attribution_diagnostics"] = dict(
            harness_report.m20_4_diagnostics
        )
    if reports is not None:
        summary["scoring_mode_reports"] = {
            mode: {
                "addressee": r.addressee.to_dict(),
                "reaction": r.reaction.to_dict(),
                "fixture_warnings": list(r.fixture_warnings),
                "drift_signals": list(r.drift_signals),
            }
            for mode, r in reports.items()
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
