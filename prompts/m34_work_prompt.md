# M3.4 Work Prompt

## Goal

Add a bounded process-valence loop so unresolved wanting, closure, satiation, and boredom can redirect inquiry and action over time.

## Non-Goals

- One-shot scripted focus bonuses
- Endless unresolved tension without closure
- Process state that only appears in logs

## Engineering Scope

- Maintain a persistent process-valence state.
- Let unresolved focus, closure, and boredom affect action and inquiry.
- Keep all process phases serializable and auditable.

## Canonical Files

- `segmentum/drives.py`
- `segmentum/inquiry_scheduler.py`
- `segmentum/m34_audit.py`

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

## Exit Condition

Acceptance report is `PASS`, unresolved focus and boredom causally affect behavior, and closure remains measurable.
