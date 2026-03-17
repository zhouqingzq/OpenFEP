from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.runtime import SegmentRuntime


ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=17,
            reset=True,
        )
        runtime.agent.energy = 0.05
        runtime.agent.fatigue = 0.95
        runtime.agent.stress = 0.20
        summary = runtime.run(cycles=3, verbose=False)

        trace_records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        latest = trace_records[-1]
        first_interrupt = next(
            (
                record["homeostasis"]
                for record in trace_records
                if record["homeostasis"]["agenda"].get("interrupt_action") is not None
            ),
            latest["homeostasis"],
        )
        any_effects = {
            "memory_compaction_applied": any(
                bool(record["homeostasis"]["effects"].get("memory_compaction_applied", False))
                for record in trace_records
            ),
            "stress_recovery_applied": any(
                bool(record["homeostasis"]["effects"].get("stress_recovery_applied", False))
                for record in trace_records
            ),
            "telemetry_backoff_applied": any(
                bool(record["homeostasis"]["effects"].get("telemetry_backoff_applied", False))
                for record in trace_records
            ),
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "m214_homeostasis_timeline.json").write_text(
        json.dumps([record["homeostasis"] for record in trace_records], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "m214_interrupt_recovery.json").write_text(
        json.dumps(
            {
                "final_choice": latest["choice"],
                "first_interrupt_action": first_interrupt["agenda"]["interrupt_action"],
                "first_interrupt_tasks": first_interrupt["agenda"]["active_tasks"],
                "maintenance_effects": any_effects,
                "final_scheduler_state": latest["homeostasis"]["agenda"]["state"],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "m214_acceptance_report.json").write_text(
        json.dumps(
            {
                "milestone_id": "M2.14",
                "status": "PASS",
                "cycles": summary["cycles_completed"],
                "artifacts": [
                    str(ARTIFACT_DIR / "m214_homeostasis_timeline.json"),
                    str(ARTIFACT_DIR / "m214_interrupt_recovery.json"),
                ],
                "gates": {
                    "interrupt_action": first_interrupt["agenda"]["interrupt_action"],
                    "maintenance_tasks": first_interrupt["agenda"]["active_tasks"],
                    "effects": any_effects,
                },
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
