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

## What M4.2 inherits from M4.1

M4.2 depends on the following M4.1 deliverables being stable:

- `CognitiveStyleParameters` — 8-parameter frozen dataclass with roundtrip
- `DecisionLogRecord` — 18-field decision log schema
- `observable_parameter_contracts()` — parameter-to-metric mapping
- `compute_observable_metrics()` — metric computation from logs
- `run_cognitive_style_trial()` — minimal simulator for smoke testing

M4.2 does NOT inherit the following as completed work:

- `m41_inference.py` — toy parameter recovery must be re-evaluated on benchmark tasks in M4.3
- `m41_blind_classifier.py` — cross-generator synthetic classifier, not benchmark-validated
- `m41_baselines.py` — same-framework baselines, not benchmark baselines
- `m41_falsification.py` — internal sensitivity, not benchmark falsification
- `m41_identifiability.py` — same-framework recoverability, not benchmark identifiability
- `data/m41_external/` — synthetic holdout data, not external human data

These sidecar modules may be useful references, but M4.2 must build its own
benchmark-grounded evidence rather than re-packaging sidecar results.

## Non-Goals

- Strong claims about human alignment or task quality
- Baseline superiority claims
- Latent-parameter identifiability claims
- Cross-task parameter sharing claims
- Treating same-framework synthetic sidecars as completed benchmark-task recovery evidence

## Acceptance Gates

- `bundle_provenance` — benchmark registry can distinguish smoke from acceptance-grade
- `protocol_schema` — Confidence DB, IGT, bandit protocols have complete fields
- `adapter_execution` — benchmark adapters produce valid trace exports
- `determinism_and_replay` — same seed reproduces identical results
- `leakage_checks` — subjects/sessions don't cross split boundaries
- `smoke_fixture_rejection` — acceptance path rejects repo smoke fixtures
- `artifact_freshness` — artifacts are regenerated this round
- `report_honesty` — report distinguishes blocked/smoke-only/pass/fail

## Canonical Files

- `segmentum/benchmark_registry.py`
- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_external_bundle_integration.py`
- `tests/test_m42_reproducibility.py`
- `tests/test_m42_acceptance.py`

## Deferred Work

Moved to `M4.3`:

- benchmark-quality metrics
- non-circular task evaluation
- human-alignment reporting
- baseline comparison on held-out task slices
- parameter recovery on benchmark data
- blind classification on external data
- falsification on benchmark data

Synthetic inference, blind classification, falsification, and same-framework
recoverability modules near `M4.1` may still exist, but they do not satisfy
this milestone unless the evidence is grounded in benchmark tasks or
independently designed task scenarios with replay and provenance.
