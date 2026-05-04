# M3.6 Work Prompt

## Goal

Validate that semantic growth, process motivation, cognitive style emergence, and bounded continuity can hold together in a longer open-world growth trial.

## Non-Goals

- Treating a short scripted replay as open-world evidence
- Allowing semantic growth to inflate without bounds
- Skipping restart continuity checks

## Engineering Scope

- Build a deterministic multi-subject, multi-narrative, multi-task trial.
- Verify semantic schema survival, process motivation observability, style stability, and restart continuity.
- Produce strict ablation, stress, snapshot, and acceptance artifacts.

## Canonical Files

- `segmentum/m3_open_world_trial.py`
- `segmentum/m3_audit.py`
- `scripts/generate_m36_acceptance_artifacts.py`
- `tests/test_m36_open_world_growth.py`
- `tests/test_m36_acceptance.py`

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`

## Audit Objects

- Specification
- Deterministic evidence
- Adversarial evidence
- Regression evidence
- Interpretation
- Decision

## Required Regressions

- `tests/test_m31_acceptance.py`
- `tests/test_m32_acceptance.py`
- `tests/test_m33_acceptance.py`
- `tests/test_m34_acceptance.py`
- `tests/test_m35_acceptance.py`
- `tests/test_m345_process_benchmark.py`

## Exit Condition

Acceptance report is `PASS`, open-world non-scripted evidence is present, semantic growth remains bounded, and restart continuity holds.
