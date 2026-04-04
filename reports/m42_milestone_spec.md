# M4.2 Milestone Specification

## Title

`M4.2: Cognitive Benchmark Environment Setup`

## Scope

- Enter benchmark/task environments using the parameter, observable, and logging interfaces defined in `M4.1`.
- Add a benchmark registry and manifest-validation path that distinguishes smoke fixtures from acceptance-grade external bundles.
- Add benchmark task protocols with standardized observation, action, feedback, confidence, and trace-export interfaces.
- Add runnable benchmark adapters for the current task set.
- Add deterministic replay, leakage checks, provenance reporting, and acceptance artifacts for the benchmark environment.
- Make it possible to run benchmark tasks reproducibly without yet claiming strong behavioral fit.
- Start recovery-on-task, replay, and provenance questions only at the benchmark/task layer rather than through `M4.1` toy inference sidecars.

## Non-Goals

- Strong claims about human alignment or task quality
- Baseline superiority claims
- Latent-parameter identifiability claims
- Cross-task parameter sharing claims
- Treating same-framework synthetic sidecars as completed benchmark-task recovery evidence

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

Synthetic inference, blind classification, falsification, and same-framework
recoverability modules near `M4.1` may still exist, but they do not satisfy
this milestone unless the evidence is grounded in benchmark tasks or
independently designed task scenarios with replay and provenance.

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
