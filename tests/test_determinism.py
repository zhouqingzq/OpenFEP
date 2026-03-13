from __future__ import annotations

import json
import tempfile
from pathlib import Path

from segmentum.runtime import SegmentRuntime


def test_snapshot_reload_is_identical_after_seeded_run() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"

        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=17,
            reset=True,
        )
        runtime.run(cycles=20, verbose=False)
        exported = runtime.export_snapshot()

        restored = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=18,
        )
        reloaded = restored.export_snapshot()

        assert json.dumps(exported, sort_keys=True) == json.dumps(reloaded, sort_keys=True)
