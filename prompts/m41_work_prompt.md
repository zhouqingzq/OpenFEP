# M4.1 Work Prompt

## Goal

Turn digital personality and cognitive style into a measurable, serializable, cross-task parameter interface that can be injected into action selection and exported through a unified decision log.

## Non-Goals

- Fitting real human datasets in this milestone
- Introducing task-specific parameter explosions
- Rewriting the existing M3 agent core

## Engineering Scope

- Define a stable cognitive parameter family and versioned schema
- Add a decision log schema that records evidence, candidate actions, resource state, confidence, selected action, prediction error, update magnitude, task context, seed, and tick
- Provide a pluggable bridge from cognitive parameters into planner or action-schema scoring
- Provide a behavior-metric to parameter mapping table
- Produce canonical, ablation, stress, and mapping artifacts

## Canonical Files

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`
- `artifact_freshness`

## Required Regressions

- `tests/test_m35_acceptance.py`
- `tests/test_m36_acceptance.py`

## Exit Condition

The M4.1 acceptance report is `PASS`, the parameter and log schemas round-trip, action selection changes under parameter ablation, stress replay avoids silent corruption, and M3.5-M3.6 acceptance remains intact.
