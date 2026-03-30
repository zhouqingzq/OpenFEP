# M4.1 Milestone Specification

## Title

`M4.1: Cognitive Variable Operationalization and Unified Parameter Interface`

## Scope

- Introduce a stable cognitive-parameter family for style-sensitive control.
- Add a versioned decision-log schema for trial or episode reconstruction.
- Add a bridge that maps cognitive parameters into action-schema scoring.
- Produce schema, trace, ablation, stress, mapping, and acceptance artifacts.

## Non-Goals

- Real benchmark fitting
- Cross-task stability claims
- Replacing the existing M3 runtime

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`
- `artifact_freshness`

## Canonical Files

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`
