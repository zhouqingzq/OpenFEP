# M4.2 Milestone Specification

## Title

`M4.2: Cognitive Benchmark Environment Setup`

## Scope

- Add a benchmark registry and manifest-validation path that distinguishes smoke fixtures from acceptance-grade external bundles.
- Add benchmark task protocols with standardized observation, action, feedback, confidence, and trace-export interfaces.
- Add runnable benchmark adapters for the current task set.
- Add deterministic replay, leakage checks, provenance reporting, and acceptance artifacts for the benchmark environment.
- Make it possible to run benchmark tasks reproducibly without yet claiming strong behavioral fit.

## Non-Goals

- Strong claims about human alignment or task quality
- Baseline superiority claims
- Latent-parameter identifiability claims
- Cross-task parameter sharing claims

## Acceptance Gates

- `bundle_provenance`
- `protocol_schema`
- `adapter_execution`
- `determinism_and_replay`
- `leakage_checks`
- `smoke_fixture_rejection`
- `artifact_freshness`
- `report_honesty`

## Deferred Work

Moved to `M4.3`:

- benchmark-quality metrics
- non-circular task evaluation
- human-alignment reporting
- baseline comparison on held-out task slices

## Canonical Files

- `segmentum/benchmark_registry.py`
- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `scripts/generate_m42_acceptance_artifacts.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_external_bundle_integration.py`
- `tests/test_m42_reproducibility.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_acceptance.py`
