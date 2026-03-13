from __future__ import annotations

import json
import tempfile
from pathlib import Path

from segmentum.runtime import SegmentRuntime


def test_trace_includes_memory_hit_and_prediction_error_sources() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "state.json"
        trace_path = Path(tmp_dir) / "trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=17,
            reset=True,
        )
        runtime.run(cycles=5, verbose=False)

        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]

    assert records
    assert all(isinstance(record["memory_hit"], bool) for record in records)
    assert all("prediction_errors" in record for record in records)
