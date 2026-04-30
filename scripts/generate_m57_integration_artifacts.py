"""Generate M5.7 integration-trial acceptance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.dialogue.integration_trial import (
    IntegrationTrialConfig,
    run_integration_trial,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M5.7 end-to-end integration trial")
    parser.add_argument("--raw-input", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--personas", type=int, default=5)
    parser.add_argument("--turns-per-persona", type=int, default=200)
    parser.add_argument("--simulated-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--skip-cross-context", action="store_true")
    args = parser.parse_args()

    config = IntegrationTrialConfig(
        personas=args.personas,
        turns_per_persona=args.turns_per_persona,
        simulated_days=args.simulated_days,
        seed=args.seed,
        run_cross_context=not args.skip_cross_context,
    )
    report = run_integration_trial(
        output_dir=args.artifacts_dir,
        report_dir=args.reports_dir,
        raw_input_path=args.raw_input,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "report": report["artifacts"]["technical_report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
