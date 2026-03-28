from __future__ import annotations

from segmentum.m345_process_benchmarks import (
    M345_REPORT_PATH,
    run_m345_process_benchmark,
    write_m345_process_benchmark_artifacts,
)


def test_m345_process_benchmark_is_deterministic() -> None:
    left = run_m345_process_benchmark()
    right = run_m345_process_benchmark()
    assert left["metrics"] == right["metrics"]
    assert left["gates"] == right["gates"]


def test_m345_process_benchmark_detects_process_surface_gains() -> None:
    payload = run_m345_process_benchmark()
    metrics = payload["metrics"]
    assert metrics["focus_bonus_gain"] > 0.12
    assert metrics["closure_penalty_gain"] > 0.10
    assert metrics["boredom_bonus_gain"] > 0.07
    assert metrics["selective_unknown_scan_gain"] > 0.01
    assert metrics["explorer_unknown_scan_gain"] > 0.03


def test_m345_process_benchmark_separates_all_seeded_styles() -> None:
    payload = run_m345_process_benchmark()
    assert payload["metrics"]["style_label_diversity"] >= 4
    assert payload["findings"] == []


def test_m345_artifacts_are_written() -> None:
    written = write_m345_process_benchmark_artifacts()
    assert M345_REPORT_PATH.exists()
    assert written["report"].endswith("m345_process_benchmark_report.json")
