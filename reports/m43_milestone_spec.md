# M4.3 Milestone Specification

## Title

`M4.3: Single-Task Behavioral Fit and Initial Falsification`

## Scope

- Use the M4.2 benchmark environment to evaluate single-task behavioral fit on **real external data**.
- Fit the cognitive-style agent to Confidence Database and Iowa Gambling Task independently.
- Compare the agent against a tiered baseline ladder (lower, competitive, ceiling).
- Assess which of the 8 cognitive parameters are active on real data.
- Produce honest failure analysis when the agent systematically fails.
- Generate acceptance artifacts grounded in external bundle evidence, not smoke fixtures.

## Current Interpretation

M4.3 is the first milestone where behavioral claims must be backed by real
external human data. Prior milestones validated interfaces (M4.1) and
environments (M4.2). M4.3 asks: does the parameterized agent actually explain
human behavior on real tasks?

The current answer is **no**. The existing `m43_modeling.py` runs entirely on
smoke fixtures (4 trials, 1 subject) and correctly reports FAIL. The milestone
requires substantial implementation work to reach acceptance.

It is useful for:

- establishing whether CognitiveStyleParameters map to real behavioral patterns
- identifying which parameters are active vs inert on human data
- building an honest baseline ladder for the cognitive-style architecture
- documenting failure modes that inform M4.4+ architecture decisions

It should not be read as evidence that:

- the cognitive-style agent matches human behavior across tasks (M4.4)
- parameters transfer between tasks (M4.4)
- style effects are stable over time (M4.9)

## What M4.3 inherits from M4.2

Stable benchmark environment (can use directly):

- `benchmark_registry.py` — bundle discovery, manifest validation, smoke/external separation
- `m4_benchmarks.py` — ConfidenceDatabaseAdapter, IowaGamblingTaskAdapter, TwoArmedBanditAdapter
- `m42_audit.py` — environment acceptance artifacts
- `external_benchmark_registry/` — Confidence Database (825,344 trials), IGT (11,800 trials)
- Leakage detection, deterministic replay, protocol validation

M4.3 does NOT inherit the following as completed work:

- `_score_action_candidates` heuristic scoring — this is benchmark scaffold, not behavioral model
- M4.1 sidecar conclusions (parameter recovery, blind classification, falsification)
- Current `m43_modeling.py` smoke-only results (trial_count=4, subject_count=1)

## Non-Goals

- Cross-task shared-parameter credibility (M4.4)
- Controlled transfer claims (M4.8)
- Longitudinal stability claims (M4.9)
- Modifying CognitiveStyleParameters parameter count
- Modifying BenchmarkAdapter protocol
- Architecture-level rewrite of `_score_action_candidates` (parameter tuning is fine)

## Acceptance Gates

- `fit_confidence_db` — agent beats lower baselines on real Confidence DB data
- `fit_igt` — agent beats lower baselines on real IGT data
- `baseline_ladder` — tiered baseline comparison with honest reporting
- `parameter_sensitivity` — ≥4 of 8 parameters show measurable effect on real data
- `non_circular_scoring` — train/test separation, no subject leakage
- `honest_failure_analysis` — failure modes documented with real examples
- `sample_size_sufficient` — adequate trial and subject counts for claims made
- `no_synthetic_claims` — zero artifacts with inflated claim_envelope
- `regression` — M4.1 and M4.2 tests still pass

## Canonical Files

- `segmentum/m43_modeling.py`
- `segmentum/m43_baselines.py`
- `segmentum/m43_audit.py`
- `tests/test_m43_single_task_fit.py`
- `tests/test_m43_baselines.py`
- `tests/test_m43_acceptance.py`

## Known Issues in Current Implementation

1. **Smoke-only execution**: All fitting runs on 4 trials, 1 subject. Must add external bundle path.
2. **No IGT fitting**: Only Confidence DB is implemented. IGT fit is entirely missing.
3. **Brute-force grid search**: 81-point grid over 4 trials works; over 825K trials it's impractical.
4. **Hand-tuned baselines**: Signal detection baseline uses fixed slope=3.6, not a proper SDT fit.
5. **Test asserts FAIL**: `test_m43_acceptance.py` asserts `status == "FAIL"`, will break when milestone passes.
6. **Artifact overwrite**: Tests write to production artifact paths, same issue as M4.2.

## Deferred Work

Moved to `M4.4`:

- Cross-task parameter consistency (Confidence DB params vs IGT params)
- Shared-parameter credibility checks
- Task-to-task transfer metrics

Moved to `M4.8` or later:

- Complex environment transfer
- Longitudinal stability
- Open-world deployment
