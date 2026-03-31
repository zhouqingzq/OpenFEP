# M4.1 Milestone Specification

## Title

`M4.1: Toy Cognitive-Style Benchmark With Falsifiable Gates`

## Scope

- Introduce a stable eight-parameter cognitive-style family for toy decision experiments.
- Keep the canonical parameter family:
  - `uncertainty_sensitivity`
  - `error_aversion`
  - `exploration_bias`
  - `attention_selectivity`
  - `confidence_gain`
  - `update_rigidity`
  - `resource_pressure_sensitivity`
  - `virtual_prediction_error_gain`
- Produce decision logs that are rich enough to reconstruct indirect behavioral metrics.
- Sample trial episodes from a scenario family pool so seed changes affect behavior traces.
- Validate intervention sensitivity, observability, blind profile separability, and log completeness.

## Current Interpretation

M4.1 should currently be interpreted as an executable toy benchmark / internal simulator layer.

It is useful for:

- testing whether parameter changes alter behavior inside a controlled generator family
- checking whether logs are rich enough to support indirect observables
- probing whether internally defined profiles are distinguishable under the same synthetic setup

It should not be read as evidence that:

- real cognitive styles have been identified in the wild
- those styles have been externally validated across independent tasks or datasets
- the underlying behavioral mechanism has been fully explained

## Non-Goals

- Scientific causal inference claims
- Real-world benchmark fitting
- Cross-task generalization claims
- Replacing the existing M3 runtime

## Acceptance Definition

M4.1 passes only if the current round demonstrates all of the following:

1. Parameter changes produce measurable shifts in behavioral distributions.
2. Logged behavior is sufficient to compute executable indirect observables.
3. Canonical profiles can be separated in a blind experiment using only behavior-derived metrics.

## Acceptance Gates

- `schema_integrity`
- `trial_variation`
- `observability`
- `intervention_sensitivity`
- `blind_distinguishability`
- `log_completeness`
- `stress_behavior`
- `regression`

## Canonical Files

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_blind_classification.py`
- `tests/test_m41_acceptance.py`
