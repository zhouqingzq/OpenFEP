# M4.1 Milestone Specification

## Title

`M4.1: Cognitive Variable Operationalization and Unified Parameter Interface`

## Scope

- Introduce a stable eight-parameter cognitive-style family for style-sensitive control.
- The canonical parameter family is:
  - `uncertainty_sensitivity`
  - `error_aversion`
  - `exploration_bias`
  - `attention_selectivity`
  - `confidence_gain`
  - `update_rigidity`
  - `resource_pressure_sensitivity`
  - `virtual_prediction_error_gain`
- Add a versioned decision-log schema for trial or episode reconstruction.
- Add a bridge that maps cognitive parameters into action-schema scoring.
- Add an observability registry with at least two indirect behavioral metrics per parameter.
- Add a blind classification experiment across at least three canonical profiles.
- Produce schema, trace, ablation, stress, observability, blind-classification, and acceptance artifacts.

## Non-Goals

- Real benchmark fitting
- Cross-task stability claims
- Replacing the existing M3 runtime

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `observability`
- `distinguishability`
- `log_completeness`
- `stress`
- `regression`
- `artifact_freshness`

## Canonical Files

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_blind_classification.py`
- `tests/test_m41_acceptance.py`
