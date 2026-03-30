# M4.2 Milestone Specification

## Title

`M4.2: Cognitive Benchmark Environment Setup`

## Scope

- Add a benchmark task protocol with standardized observation, action, feedback, and confidence interfaces.
- Add a Confidence Database benchmark slice, preprocessing flow, and evaluation metrics.
- Add an Iowa Gambling Task adapter skeleton and protocol placeholder.
- Generate protocol, preprocessing, trace, ablation, stress, and acceptance artifacts.

## Non-Goals

- Full cross-task fitting
- Full Iowa Gambling Task behavioral study
- Training a task-specific benchmark-only agent

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`
- `artifact_freshness`
- `benchmark_closed_loop`

## Canonical Files

- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `scripts/generate_m42_acceptance_artifacts.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_acceptance.py`
