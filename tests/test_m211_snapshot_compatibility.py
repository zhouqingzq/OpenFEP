from __future__ import annotations

import json
import tempfile
from pathlib import Path

from segmentum.m211_readiness import build_snapshot_compatibility_payload
from segmentum.runtime import SegmentRuntime


def test_m211_legacy_snapshot_loads_with_safe_defaults() -> None:
    payload = build_snapshot_compatibility_payload()
    assert payload["passed"] is True
    assert payload["checks"]["attention_defaults_restored"] is True
    assert payload["checks"]["personality_profile_defaulted"] is True


def test_missing_new_fields_do_not_break_snapshot_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "legacy_state.json"
        runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=17, reset=True)
        runtime.run(cycles=1, verbose=False)
        snapshot = json.loads(state_path.read_text(encoding="utf-8"))
        snapshot["state_version"] = "0.1"
        snapshot["agent"].pop("attention_bottleneck", None)
        snapshot["agent"].pop("self_model", None)
        state_path.write_text(json.dumps(snapshot, ensure_ascii=True), encoding="utf-8")

        restored = SegmentRuntime.load_or_create(state_path=state_path, seed=21)

    assert restored.agent.attention_bottleneck.enabled is True
    assert restored.agent.self_model.personality_profile is not None
    assert restored.agent.self_model.identity_narrative is not None
