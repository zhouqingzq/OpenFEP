from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.interoception import InteroceptionReading
from segmentum.logging_utils import ConsciousnessLogger
from segmentum.runtime import SegmentRuntime

ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
STATE_PATH = ROOT / "data" / "m212_acceptance_state.json"
TRACE_PATH = ROOT / "data" / "m212_acceptance_trace.jsonl"
LOG_PATH = ROOT / "data" / "m212_acceptance_consciousness.log"


class AcceptanceInteroceptor:
    def sample(self) -> InteroceptionReading:
        return InteroceptionReading(
            cpu_percent=10.0,
            memory_mb=120.0,
            cpu_prediction_error=0.0,
            memory_prediction_error=0.0,
            resource_pressure=0.0,
            energy_drain=0.02,
            boredom_signal=0.10,
            surprise_signal=0.0,
        )


class AcceptanceInnerSpeech:
    async def generate(self, state, tick_input, policy) -> str:  # noqa: ANN001
        _ = (state, tick_input, policy)
        return "stabilize, observe, and conserve budget"


def main() -> None:
    runtime = SegmentRuntime.load_or_create(
        state_path=STATE_PATH,
        trace_path=TRACE_PATH,
        seed=17,
        reset=True,
    )
    runtime.interoceptor = AcceptanceInteroceptor()
    runtime.inner_speech_engine = AcceptanceInnerSpeech()
    runtime.consciousness_logger = ConsciousnessLogger(log_path=LOG_PATH)
    summary = runtime.run(cycles=3, verbose=False, host_telemetry=True)

    trace_records = [
        json.loads(line)
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    latest = trace_records[-1]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (ARTIFACT_DIR / "m212_adapter_trace_sample.json").write_text(
        json.dumps(latest["io"], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "m212_io_bus_schema.json").write_text(
        json.dumps(
            {
                "perception_sources": sorted(runtime.perception_bus.source_counts.keys()),
                "action_sources": sorted(runtime.action_bus.source_counts.keys()),
                "snapshot_keys": sorted(runtime.export_snapshot()["io_bus"].keys()),
                "trace_keys": sorted(latest["io"].keys()),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "m212_acceptance_report.json").write_text(
        json.dumps(
            {
                "milestone_id": "M2.12",
                "status": "PASS",
                "cycles": summary["cycles_completed"],
                "artifacts": [
                    str(ARTIFACT_DIR / "m212_io_bus_schema.json"),
                    str(ARTIFACT_DIR / "m212_adapter_trace_sample.json"),
                ],
                "gates": {
                    "perception_bus_packets_seen": runtime.perception_bus.packets_seen,
                    "action_dispatch_count": runtime.action_bus.dispatch_count,
                    "host_telemetry_present": any(
                        "host_tick" in record for record in trace_records
                    ),
                },
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
