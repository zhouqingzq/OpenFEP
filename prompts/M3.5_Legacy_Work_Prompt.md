# M3.5 Work Prompt

## Goal

Let effort allocation history produce stable cognitive style differences without introducing any dedicated laziness drive.

## Non-Goals

- Hard-coded personality labels
- A fixed laziness variable
- Style labels without behavioral surface differences

## Engineering Scope

- Record effort allocation history and derive style surfaces from it.
- Make style affect action or inquiry allocation.
- Preserve style continuity through serialization and acceptance evidence.

## Canonical Files

- `segmentum/slow_learning.py`
- `segmentum/m35_audit.py`

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

## Exit Condition

Acceptance report is `PASS`, at least two stable styles differentiate, and no lazy-drive field exists.
