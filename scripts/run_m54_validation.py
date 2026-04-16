from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.dialogue.lifecycle import ImplantationConfig
from segmentum.dialogue.validation.pipeline import ValidationConfig, run_batch_validation
from segmentum.dialogue.validation.report import generate_report
from segmentum.dialogue.validation.splitter import SplitStrategy


def _load_user_datasets(user_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(user_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
            rows.append(payload)
    return rows


def _parse_strategies(raw: str) -> list[SplitStrategy]:
    out: list[SplitStrategy] = []
    for item in [piece.strip() for piece in raw.split(",") if piece.strip()]:
        out.append(SplitStrategy(item))
    return out


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5.4 consistency validation.")
    parser.add_argument("--user-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m54_validation"))
    parser.add_argument("--strategies", type=str, default="random,temporal,partner,topic")
    parser.add_argument("--min-users", type=int, default=10)
    parser.add_argument("--pilot-users", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = _load_user_datasets(args.user_dir)
    strategies = _parse_strategies(args.strategies)
    config = ValidationConfig(
        strategies=strategies,
        min_users=int(args.min_users),
        pilot_user_count=int(args.pilot_users),
        seed=int(args.seed),
        implantation_config=ImplantationConfig(),
    )
    reports = run_batch_validation(datasets, config)
    md_path = generate_report(reports, args.output)
    aggregate_path = args.output / "aggregate_report.json"
    agg = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    acceptance = {
        "milestone": "M5.4",
        "user_count": len(reports),
        "strategies": [item.value for item in strategies],
        "metric_version": agg.get("metric_version"),
        "hard_pass": agg.get("hard_pass"),
        "overall_conclusion": agg.get("overall_conclusion"),
        "aggregate_report_json": str(aggregate_path.as_posix()),
        "aggregate_report_md": str(md_path.as_posix()),
    }
    _write_json(Path("artifacts/m54_acceptance.json"), acceptance)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

