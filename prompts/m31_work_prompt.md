# M3.1 Work Prompt

## Goal

Introduce deterministic episodic semantic grounding for open narrative text and carry grounded motifs into downstream appraisal and memory ingestion.

## Non-Goals

- Stochastic parsing
- Schema growth claims
- Replacing uncertainty decomposition

## Engineering Scope

- Keep semantic grounding deterministic and provenance-preserving.
- Separate surface, paraphrase, and implicit evidence.
- Ensure grounded motifs survive compilation into narrative episodes.

## Canonical Files

- `segmentum/semantic_grounding.py`
- `segmentum/narrative_compiler.py`
- `segmentum/narrative_types.py`
- `segmentum/m31_audit.py`

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

- `tests/test_narrative_compiler.py`
- `tests/test_m233_narrative_robustness.py`

## Exit Condition

Acceptance report is `PASS`, artifacts are fresh, and no blocking findings remain.
