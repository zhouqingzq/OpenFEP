from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m28_acceptance_artifacts import generate_artifacts


def test_generate_m28_acceptance_artifacts(tmp_path: Path) -> None:
    _ = tmp_path
    generate_artifacts(seed=44)
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"

    required = [
        artifacts_dir / "m28_attention_on_off.json",
        artifacts_dir / "m28_personality_anova.json",
        artifacts_dir / "m28_transfer_benchmark.json",
        artifacts_dir / "m28_world_rollout_foraging_valley.json",
        artifacts_dir / "m28_world_rollout_predator_river.json",
        artifacts_dir / "m28_world_rollout_social_shelter.json",
    ]
    for path in required:
        assert path.exists(), str(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload
