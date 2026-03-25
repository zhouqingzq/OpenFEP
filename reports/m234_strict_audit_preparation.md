# M2.34 Strict Audit Preparation

## Goal

Freeze `M2.34` around the implemented semantics: bounded experiment portfolios under finite resource constraints, interpreted as approximate Bayesian-style weighted ranking plus staged inquiry rather than single-probe minimalism.

## Interpretation Rule

The audit should evaluate whether the system performs resource-bounded free-energy reduction through ranked inquiry plans. It should not require the globally cheapest single action if multiple high-value probes fit within the active experiment budget, and it should allow staged or batched probing when that is more realistic under bounded cognition.

## What Must Be True Before Audit Sign-Off

1. Decision-relevant competing hypotheses are turned into explicit discrimination targets.
2. Each target yields hypothesis-linked predictions and governed inquiry candidates.
3. Candidate ranking reflects information gain, falsification value, cost, risk, and active-goal alignment.
4. Only a bounded number of plans become `active_experiment`; lower-priority options are queued, deferred, rejected, or left for later batches.
5. Experiment plans change downstream prediction, verification, subject state, or decision scoring.
6. Experiment state survives snapshot round-trip.

## Minimum Test Plan

- `tests/test_m234_experiment_design.py`
  Verifies competition translation, goal/risk-sensitive ranking, downstream causality, governance bounds, and snapshot persistence.

## Minimum Artifact Plan

- `artifacts/m234_experiment_trace.jsonl`
- `artifacts/m234_experiment_ablation.json`
- `artifacts/m234_experiment_stress.json`
- `reports/m234_acceptance_report.json`
- `reports/m234_acceptance_summary.md`

## Regression Boundary

- `tests/test_m233_uncertainty_decomposition.py`
- `tests/test_m227_snapshot_roundtrip.py`
- `tests/test_m228_prediction_ledger.py`
- `tests/test_runtime.py`

## Exit Criteria

- The milestone spec explicitly permits bounded parallel inquiry.
- The audit language distinguishes finite-resource approximate inference from strict resource minimization or exact single-probe optimality.
- A fresh acceptance bundle can be generated in strict mode.
- The audit still rejects unbounded or fake concurrency.
