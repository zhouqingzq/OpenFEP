# M4.3 Agent Work Prompt — Single-Task Behavioral Fit

## Project Context

You are working on the Segmentum project — a cognitive-style simulation framework.
The core engine lives in `segmentum/m4_cognitive_style.py` (CognitiveStyleParameters, 8 parameters,
DecisionLogRecord, CognitiveParameterBridge with score_action).

**M4.2 delivered:**
- BenchmarkAdapter protocol: load_trials → observation_from_trial → action_space → apply_action → export_trial_record
- Three adapters: ConfidenceDatabaseAdapter, IowaGamblingTaskAdapter, TwoArmedBanditAdapter
- Real external data bundles in `external_benchmark_registry/`:
  - Confidence Database: 825,344 trials, 2426 subjects, 45 source files
  - Iowa Gambling Task: 11,800 trials, 59 subjects
- Leakage detection (subject/session cross-split)
- Deterministic replay (same_seed_triple_replay, compute_behavioral_seed_summaries)
- Benchmark registry with smoke/external separation

**M4.3 goal:** Demonstrate that CognitiveStyleParameters can produce non-trivial
behavioral fit on **real external data**, one task at a time. This is NOT cross-task
transfer (M4.4) — fit each task independently.

---

## Strict Prohibitions

### 1. No Fake External Validation
- **NEVER** set `external_validation: true` on any artifact produced by code inside this repo.
- Both the primary generator (m4_cognitive_style.py) and any sidecar generator
  (m41_external_generator.py) are **same-codebase synthetic** — labeling them "external" is fraud.
- The ONLY external data is what lives in `external_benchmark_registry/`.

### 2. No Synthetic Data Claims
- Do not generate synthetic trials and call them "benchmark results."
- Every fit metric reported in acceptance must come from running the adapter pipeline
  against **real external bundle data**.
- If you need synthetic data for debugging, label it `claim_envelope: "synthetic_diagnostic"`.

### 3. No Fake Tests
- Every test assertion must be falsifiable — it must be possible for the test to fail
  with a realistic bad implementation.
- Do not write tests that assert `True` or check only that output is not None.
- Tests against real data must use `@unittest.skipUnless(external_bundle_available, ...)`.

### 4. No Toy Implementations
- Do not write stub functions that return hardcoded values.
- If a fitting method is not ready, raise `NotImplementedError` — do not return fake results.

### 5. No Scope Bloat
- M4.3 is single-task fit. Do NOT implement cross-task transfer, longitudinal stability,
  or population-level clustering. Those are M4.4–M4.6.
- Do not add new parameters to CognitiveStyleParameters.
- Do not modify the BenchmarkAdapter protocol.

---

## Required Tasks

### T1: Confidence Database Fit
1. Load real Confidence Database trials via `ConfidenceDatabaseAdapter`.
2. Run `CognitiveStyleParameters` agent through the adapter pipeline.
3. Compute held-out metrics: accuracy, Brier score, calibration error, confidence correlation.
4. Compare against baseline ladder (see below).
5. Export trial-level predictions with `claim_envelope: "benchmark_eval"`.

### T2: Iowa Gambling Task Fit
1. Load real IGT trials via `IowaGamblingTaskAdapter` with `protocol_mode: "standard_100"`.
2. Run agent through 100-trial protocol.
3. Compute: deck selection accuracy, learning curve slope, advantageous-deck ratio, reward correlation.
4. Compare against baseline ladder (see below).
5. Export with `claim_envelope: "benchmark_eval"`.

### T3: Parameter Sensitivity on Real Data
1. For each of the 8 CognitiveStyleParameters, sweep ±1σ from default while holding others fixed.
2. Measure which parameters actually change behavioral metrics on **real data**.
3. Report: which parameters are active (produce measurable metric change) vs inert.
4. Label: `claim_envelope: "benchmark_eval"`, NOT "external_validation".

### T4: Honest Failure Analysis
1. Identify where the agent systematically fails (e.g., specific stimulus types, confidence ranges, IGT phases).
2. Report failure modes with concrete examples from real trials.
3. If the agent cannot beat random baseline on a task, **report this honestly** — do not hide it.
4. Include a `failure_modes` section in the acceptance report.

### T5: Acceptance Reporting
1. Generate `reports/m43_acceptance_report.json` with all metrics, baselines, gates.
2. Generate `artifacts/m43_confidence_fit.json` and `artifacts/m43_igt_fit.json`.
3. Every artifact must include: `source_type`, `claim_envelope`, `external_validation: false`,
   `benchmark_state`, `trial_count`, `leakage_check`.

---

## Baseline Ladder

### Confidence Database Baselines
| Tier | Baseline | Description |
|------|----------|-------------|
| Lower (must beat) | Random | Random choice + uniform confidence |
| Lower (must beat) | Stimulus-only | Choose based on stimulus strength, ignore confidence |
| Competitive | Human-match ceiling | Per-condition human majority vote + mean confidence |

### IGT Baselines
| Tier | Baseline | Description |
|------|----------|-------------|
| Lower (must beat) | Random | Uniform deck selection |
| Lower (must beat) | Frequency-matching | Select decks proportional to observed reward frequency |
| Competitive | Human behavior | Actual human deck selection patterns from the dataset |

### Acceptance Rules
- All lower baselines must be beaten or M4.3 **fails**.
- At least one competitive baseline must be matched or exceeded on primary metric.
  Brier score difference < 5% counts as parity.
- If every competitive baseline is worse by >15%, block M4.4 entry pending architecture review.

---

## Acceptance Gates

```
G1: fit_confidence_db        — Agent beats both lower baselines on Confidence DB
G2: fit_igt                  — Agent beats both lower baselines on IGT
G3: baseline_ladder          — At least one competitive baseline matched per task
G4: parameter_sensitivity    — ≥4 of 8 parameters show measurable effect on real data
G5: honest_failure_analysis  — Failure modes documented with real examples
G6: no_synthetic_claims      — Zero artifacts with external_validation:true or inflated claim_envelope
```

---

## Labeling Rules

| Data source | `claim_envelope` | `external_validation` | `source_type` |
|-------------|------------------|-----------------------|---------------|
| external_benchmark_registry | `benchmark_eval` | `false` | `external_bundle` |
| Same-codebase synthetic | `synthetic_diagnostic` | `false` | `synthetic_protocol` |
| Mixed/debug | `synthetic_diagnostic` | `false` | `synthetic_protocol` |

**Never** use `external_validation: true`. The real external data validates the benchmark adapter,
not the cognitive model itself. Only independent replication by another team on different data
would qualify as external validation.

---

## File Structure (suggested)

```
segmentum/m43_fit.py              — Core fitting logic for single-task behavioral fit
segmentum/m43_baselines.py        — Baseline implementations (random, stimulus-only, etc.)
segmentum/m43_audit.py            — Acceptance artifact generation
tests/test_m43_fit.py             — Fit tests against real data (skip without bundle)
tests/test_m43_baselines.py       — Baseline correctness tests
tests/test_m43_acceptance.py      — Acceptance gate tests
artifacts/m43_confidence_fit.json — Confidence DB fit results
artifacts/m43_igt_fit.json        — IGT fit results
reports/m43_acceptance_report.json — Final acceptance report
```

---

## Regression Requirements

Before M4.3 acceptance, verify:
- All M4.1 tests still pass (`tests/test_m41_*.py`)
- All M4.2 tests still pass (`tests/test_m42_*.py`)
- No modifications to `segmentum/m4_cognitive_style.py` CognitiveStyleParameters interface
- No modifications to `segmentum/m4_benchmarks.py` BenchmarkAdapter protocol
