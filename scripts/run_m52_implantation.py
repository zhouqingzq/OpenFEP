from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.lifecycle import ImplantationConfig, implant_personality
from segmentum.dialogue.maturity import maturity_report
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.world import DialogueWorld


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5.2 personality implantation")
    parser.add_argument("--user-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sleep-every", type=int, default=10)
    parser.add_argument("--maturity-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = _load_json(args.user_data)
    observer = DialogueObserver()
    world = DialogueWorld(dataset, observer, seed=args.seed)
    agent = SegmentAgent()
    config = ImplantationConfig(
        sleep_every_n_sessions=max(1, int(args.sleep_every)),
        maturity_threshold=float(args.maturity_threshold),
    )
    result = implant_personality(agent, world, config)
    uid = int(dataset.get("uid", 0))
    base = args.output
    _write_json(base / f"{uid}_agent_state.json", result.final_agent_state)
    _write_json(base / f"{uid}_snapshots.json", [item.to_dict() for item in result.snapshots])
    _write_json(
        base / f"{uid}_maturity_report.json",
        maturity_report(
            result.snapshots,
            threshold=float(config.maturity_threshold),
            window=int(config.maturity_window),
        ),
    )
    _write_json(
        Path("artifacts/m52_acceptance.json"),
        {
            "milestone": "M5.2",
            "uid": uid,
            "ticks": result.total_ticks,
            "sleep_cycles": result.total_sleep_cycles,
            "matured": result.matured,
            "maturity_tick": result.maturity_tick,
        },
    )


if __name__ == "__main__":
    main()
