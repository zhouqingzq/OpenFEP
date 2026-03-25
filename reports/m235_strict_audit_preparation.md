# M2.35 Strict Audit Preparation

## Milestone

- `M2.35`
- Title: `Inquiry Budget Scheduler`
- Schema version: `m235_audit_v1`

## Frozen Test Bundle

- Milestone tests:
  - `tests/test_m235_inquiry_scheduler.py`
  - `tests/test_m235_acceptance.py`
  - `tests/test_m235_audit_preparation.py`
- Regression tests:
  - `tests/test_m234_experiment_design.py`
  - `tests/test_m228_prediction_ledger.py`
  - `tests/test_m229_verification_loop.py`
  - `tests/test_runtime.py`

## Frozen Gates

- `cross_surface_ranking`
- `verification_budgeting`
- `workspace_allocation`
- `action_biasing`
- `downstream_causality`
- `snapshot_roundtrip`
- `regression`
- `artifact_freshness`

## Required Audit Objects

- Specification: `reports/m235_milestone_spec.md`
- Deterministic evidence: scheduler ranking, replay signatures, snapshot state
- Adversarial evidence: cooldown, slot saturation, continuity pressure
- Regression evidence: prior experiment, ledger, verification, and runtime suites
- Interpretation: `reports/m235_acceptance_report.json` and `reports/m235_acceptance_summary.md`
- Decision: strict report recommendation

## Strict Mode Rules

- Strict mode refuses injected execution records.
- Strict mode requires current-round artifacts, authentic suite execution timestamps, and a frozen baseline.
- Generated `m235` artifacts and `.pytest_m235_*` logs are non-blocking; code and specification changes are blocking until frozen.
