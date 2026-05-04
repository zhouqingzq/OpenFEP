# M4.2 Work Prompt

## Goal

Connect the M4.1 cognitive-parameter and logging protocol to a real-task benchmark adapter so the same agent structure can run on an external cognitive task with reproducible evaluation.

## Non-Goals

- Claiming cross-task cognitive-style proof from a single benchmark
- Completing full Iowa Gambling Task fitting in this round
- Hardcoding task-specific logic inside the core agent

## Engineering Scope

- Standardize a benchmark trial schema and adapter protocol
- Add a Confidence Database benchmark slice with preprocessing, split manifests, and evaluation
- Add an Iowa Gambling Task adapter skeleton and schema placeholder
- Reuse the M4.1 parameter/logging bridge for benchmark execution
- Produce canonical, ablation, stress, preprocessing, and protocol artifacts

## Canonical Files

- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `scripts/generate_m42_acceptance_artifacts.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_acceptance.py`

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`
- `artifact_freshness`
- `benchmark_closed_loop`

## Required Regressions

- `tests/test_m41_acceptance.py`
- `tests/test_m36_acceptance.py`

## Exit Condition

The M4.2 acceptance report is `PASS`, at least one benchmark runs end to end with reproducible metrics, ablation weakens benchmark performance, malformed inputs are contained, and M4.1 plus M3.6 acceptance still hold.
