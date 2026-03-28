# M3.2 Work Prompt

## Goal

Derive bounded semantic schemas from memory-grounded episodes and preserve conflict evidence when schemas split or weaken.

## Non-Goals

- Manual rule tables standing in for semantic growth
- Unlimited schema proliferation
- Ignoring conflict history

## Engineering Scope

- Refresh semantic schemas from grounded memory episodes.
- Track support, confidence, contexts, and conflict outcomes.
- Preserve round-trip compatibility through memory serialization.

## Canonical Files

- `segmentum/memory.py`
- `segmentum/semantic_schema.py`
- `segmentum/m32_audit.py`

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

- `tests/test_memory.py`
- `tests/test_narrative_sleep_consolidation.py`
- `tests/test_m31_acceptance.py`

## Exit Condition

Acceptance report is `PASS`, schema growth is bounded, and conflict evidence is preserved.
