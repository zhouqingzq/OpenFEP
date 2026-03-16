from __future__ import annotations

import json
from pathlib import Path

from segmentum.m29_benchmarks import run_transfer_acceptance_suite, transfer_gate_met


def test_m29_transfer_benchmark_schema_is_stable() -> None:
    result = run_transfer_acceptance_suite()

    assert result["milestone"] == "M2.9"
    assert len(result["world_rollouts"]) == 3
    assert len(result["benchmarks"]) == 2
    assert result["acceptance"]["required_transfer_paths"] == 2
    for benchmark in result["benchmarks"]:
        assert "train_world" in benchmark
        assert "comparisons" in benchmark
        assert len(benchmark["comparisons"]) == 1
        comparison = benchmark["comparisons"][0]
        assert "eval_seed" in comparison
        assert comparison["protocol"]["fresh_baseline"] == "fresh_agent_same_eval_seed"
        assert comparison["protocol"]["transfer_agent"] == "pre_exposed_agent_snapshot_same_eval_seed"


def test_m29_required_transfer_directions_pass_acceptance_gates() -> None:
    result = run_transfer_acceptance_suite()

    assert result["acceptance"]["passed"] is True
    assert result["acceptance"]["transfer_paths_passing"] == 2
    for record in result["acceptance"]["comparison_records"]:
        assert transfer_gate_met(record["improvements"])


def test_m29_benchmark_artifact_is_json_stable(tmp_path: Path) -> None:
    artifact_path = tmp_path / "m29_transfer_benchmark.json"
    payload = run_transfer_acceptance_suite()
    artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    restored = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert restored["acceptance"]["passed"] is True
    assert restored["benchmarks"][0]["comparisons"][0]["world_id"] == "foraging_valley"
    assert restored["benchmarks"][1]["comparisons"][0]["world_id"] == "social_shelter"
