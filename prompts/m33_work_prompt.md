# M3.3 Work Prompt

## Goal

Expand semantic schemas into prediction, verification, and inquiry prioritization so semantics causally changes downstream behavior.

## Non-Goals

- Trace-only semantics
- Prediction conditioning without provenance
- Skipping verification and inquiry effects

## Engineering Scope

- Condition predictive coding on semantic schemas.
- Carry schema provenance into prediction hypotheses and discrepancies.
- Raise verification and inquiry priority when semantic uncertainty matters.

## Canonical Files

- `segmentum/predictive_coding.py`
- `segmentum/prediction_ledger.py`
- `segmentum/verification.py`
- `segmentum/m33_audit.py`

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

- `tests/test_m32_acceptance.py`
- `tests/test_m228_prediction_ledger.py`
- `tests/test_m229_verification_loop.py`

## Exit Condition

Acceptance report is `PASS`, semantic conditioning degrades under ablation, and provenance survives replay.
