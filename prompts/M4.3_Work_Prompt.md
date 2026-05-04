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

## Existing M4.3 Code — What Must Change

There is already M4.3 code in the repo. It is **mostly scaffold** and must be substantially
reworked. Here is what exists and what's wrong with it:

### `segmentum/m43_modeling.py` (current state)
- **Problem 1:** Every function calls `allow_smoke_test=True`, running on 4 trials / 1 subject.
  Must be changed to use `external_benchmark_registry/` via `benchmark_root` parameter.
- **Problem 2:** No IGT fitting at all — only Confidence DB.
- **Problem 3:** `candidate_parameter_grid()` is a brute-force 81-point grid. This worked on
  4 trials but is impractical on 825K trials. Need efficient fitting (e.g., random search
  with early stopping, or coordinate descent on a few key parameters).
- **Problem 4:** Baselines (`run_signal_detection_baseline`, `run_no_persona_baseline`,
  `run_task_optimal_baseline`) use hardcoded coefficients (slope=3.6 etc.), not proper
  independent fits. Move to `m43_baselines.py` and implement properly.
- **What to keep:** `_fit_logistic_baseline` is a real online SGD logistic regressor — this
  is honest and useful as a statistical baseline. `_bootstrap_likelihood_margin` is correct.
  `_subject_folds` CV logic is correct.

### `segmentum/m43_audit.py` (current state)
- **Problem 1:** Relies entirely on `m43_modeling.py`'s smoke-only functions.
- **Problem 2:** No IGT track in the acceptance report.
- **Problem 3:** No parameter sensitivity gate.
- **Problem 4:** Calls `write_m41_acceptance_artifacts` and `write_m42_acceptance_artifacts`
  which overwrites production artifacts during M4.3 test runs.
- **What to keep:** Gate evaluation structure and artifact layout are reasonable.

### Tests (current state)
- `test_m43_acceptance.py` line 14 asserts `assertEqual(report["status"], "FAIL")` —
  this will break when M4.3 actually passes. Must restructure to test both blocked and
  pass paths (like M4.2 tests do).
- `test_m43_single_task_fit.py` and `test_m43_baselines.py` test smoke-only path.
  Must add `@unittest.skipUnless(external_bundle_available, ...)` tests for real data.

### Current report: `reports/m43_acceptance_report.json`
- Status: FAIL, trial_count: 4, subject_count: 1, claim_envelope: smoke_only
- Gates failing: benchmark_fit, baseline_competitive, sample_size_sufficient
- This is correct for the current smoke-only state.

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
  or population-level clustering. Those are M4.4–M4.9.
- Do not add new parameters to CognitiveStyleParameters.
- Do not modify the BenchmarkAdapter protocol.

---

## Required Tasks

### T1: Confidence Database Fit (modify `m43_modeling.py`)
1. Add `benchmark_root` parameter to all fitting functions. Default to `default_acceptance_benchmark_root()`.
2. When `benchmark_root` is available, load real Confidence Database trials via adapter.
3. Replace brute-force grid with efficient fitting (random search + early stopping, or
   coordinate descent on top-4 parameters).
4. Compute held-out metrics: accuracy, Brier score, calibration error, confidence correlation.
5. Compare against baseline ladder (see below).
6. Export trial-level predictions with `claim_envelope: "benchmark_eval"`.
7. When `benchmark_root` is None, produce blocked/smoke-only report (not fake PASS).

### T2: Iowa Gambling Task Fit (new code in `m43_modeling.py`)
1. Load real IGT trials via `IowaGamblingTaskAdapter` with `protocol_mode: "standard_100"`.
2. Run agent through 100-trial protocol for multiple subjects.
3. Compute: deck selection accuracy, learning curve slope, advantageous-deck ratio.
4. Compare against baseline ladder (see below).
5. Export with `claim_envelope: "benchmark_eval"`.

### T3: Parameter Sensitivity on Real Data (new)
1. For each of the 8 CognitiveStyleParameters, sweep ±1σ from default while holding others fixed.
2. Measure which parameters actually change behavioral metrics on **real data** (subsample if needed for speed).
3. Report: which parameters are active (produce measurable metric change) vs inert.
4. Gate: at least 4 of 8 must be active.
5. Label: `claim_envelope: "benchmark_eval"`.

### T4: Honest Failure Analysis (extend existing)
1. Identify where the agent systematically fails (e.g., specific stimulus types, confidence ranges, IGT phases).
2. Report failure modes with concrete examples from real trials.
3. If the agent cannot beat random baseline on a task, **report this honestly** — do not hide it.
4. Include a `failure_modes` section in the acceptance report.

### T5: Acceptance Reporting (update `m43_audit.py`)
1. Add IGT track to acceptance report.
2. Add parameter_sensitivity gate.
3. Add blocked path when external bundle is missing (like M4.2 does).
4. Do NOT call `write_m41_acceptance_artifacts()` or `write_m42_acceptance_artifacts()` —
   regression should be tested separately, not by overwriting production artifacts.
5. Generate `artifacts/m43_confidence_fit.json`, `artifacts/m43_igt_fit.json`,
   `artifacts/m43_parameter_sensitivity.json`.

### T6: Fix Tests
1. `test_m43_acceptance.py`: Add blocked-path test (mock `default_acceptance_benchmark_root` to None).
   Add real-bundle test with `@unittest.skipUnless`. Remove hardcoded `assertEqual(status, "FAIL")`.
2. `test_m43_single_task_fit.py`: Add external-bundle tests. Keep smoke tests as sanity checks.
3. `test_m43_baselines.py`: Test baseline implementations independently on known inputs.

---

## Baseline Ladder

### Confidence Database Baselines (implement in `m43_baselines.py`)
| Tier | Baseline | Description |
|------|----------|-------------|
| Lower (must beat) | Random | Random choice + uniform confidence |
| Lower (must beat) | Stimulus-only | Choose based on stimulus strength sign only, fixed confidence |
| Competitive | Statistical logistic | `_fit_logistic_baseline` from current code — keep this, it's honest |
| Competitive | Human-match ceiling | Per-condition human majority vote + mean confidence |

### IGT Baselines (implement in `m43_baselines.py`)
| Tier | Baseline | Description |
|------|----------|-------------|
| Lower (must beat) | Random | Uniform deck selection |
| Lower (must beat) | Frequency-matching | Select decks proportional to cumulative reward history |
| Competitive | Human behavior | Actual human deck selection distribution from the dataset |

### Acceptance Rules
- All lower baselines must be beaten or M4.3 **fails**.
- At least one competitive baseline must be matched or exceeded on primary metric.
  Brier score difference < 5% counts as parity.
- If every competitive baseline is worse by >15%, block M4.4 entry pending architecture review.

---

## Acceptance Gates

```
F1: fit_confidence_db        — Agent beats both lower baselines on real Confidence DB
F2: fit_igt                  — Agent beats both lower baselines on real IGT
F3: baseline_ladder          — At least one competitive baseline matched per task
F4: parameter_sensitivity    — ≥4 of 8 parameters show measurable effect on real data
F5: honest_failure_analysis  — Failure modes documented with real examples
F6: non_circular_scoring     — Train/test separation, no subject leakage
F7: no_synthetic_claims      — Zero artifacts with external_validation:true or inflated claim_envelope
F8: sample_size_sufficient   — trial_count ≥ 1000 (Confidence DB), subject_count ≥ 3 (IGT)
F9: regression               — M4.1 and M4.2 tests still pass
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

## File Structure

Modify existing files rather than creating new ones where possible:

```
segmentum/m43_modeling.py     — Core fitting logic (MODIFY: add external bundle path, IGT fit)
segmentum/m43_baselines.py    — Baseline implementations (CREATE: moved from m43_modeling.py)
segmentum/m43_audit.py        — Acceptance artifact generation (MODIFY: add IGT track, blocked path)
tests/test_m43_single_task_fit.py  — Fit tests (MODIFY: add external bundle tests)
tests/test_m43_baselines.py        — Baseline tests (MODIFY: test real baselines)
tests/test_m43_acceptance.py       — Gate tests (MODIFY: blocked + pass paths)
```

---

## Regression Requirements

Before M4.3 acceptance, verify:
- All M4.1 tests still pass (`tests/test_m41_*.py`)
- All M4.2 tests still pass (`tests/test_m42_*.py`)
- No modifications to `segmentum/m4_cognitive_style.py` CognitiveStyleParameters interface
- No modifications to `segmentum/m4_benchmarks.py` BenchmarkAdapter protocol

## Key API Reference

```python
# Load external bundle
from segmentum.m4_benchmarks import default_acceptance_benchmark_root, EXTERNAL_BENCHMARK_ROOT

# Confidence DB adapter
adapter = ConfidenceDatabaseAdapter()
trials = adapter.load_trials(benchmark_root=EXTERNAL_BENCHMARK_ROOT)

# IGT adapter
adapter = IowaGamblingTaskAdapter()
trials = adapter.load_trials(benchmark_root=EXTERNAL_BENCHMARK_ROOT, protocol_mode="standard_100",
                             selected_subject_id="s-01")

# Run full benchmark
result = run_confidence_database_benchmark(parameters, seed=42, benchmark_root=EXTERNAL_BENCHMARK_ROOT)
result = run_iowa_gambling_benchmark(parameters, seed=44, benchmark_root=EXTERNAL_BENCHMARK_ROOT,
                                     selected_subject_id="s-01", protocol_mode="standard_100")

# Leakage check
from segmentum.m4_benchmarks import detect_subject_leakage
check = detect_subject_leakage(trials, key_field="subject_id")
```
