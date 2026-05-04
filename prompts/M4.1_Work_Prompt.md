# M4.1 Work Prompt

## Goal

Turn digital personality and cognitive style into a measurable, serializable, cross-task parameter interface that can be injected into action selection and exported through a unified decision log for internal toy-benchmark use.

## Non-Goals

- Fitting real human datasets in this milestone
- Claiming that real cognitive styles have already been identified or externally validated
- Introducing task-specific parameter explosions
- Rewriting the existing M3 agent core

## Engineering Scope

- Define a stable cognitive parameter family and versioned schema
- Explicitly split source trust into two parameters:
  - `source_precision_gain`, the relative precision given to virtual prediction error from deliberative or linguistic inputs versus direct perceptual error,
  - `source_authority_weighting`, the degree to which source reliability history changes weighting across conflicting indirect sources
- Add a decision log schema that records evidence, candidate actions, resource state, confidence, selected action, prediction error, update magnitude, task context, seed, and tick
- Provide a pluggable bridge from cognitive parameters into planner or action-schema scoring
- Add behavior probes for language-warning compliance, language-versus-perception conflict resolution, and source-preference switching under reliability updates
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
- `source_trust_observability`

## Required Regressions

- `tests/test_m35_acceptance.py`
- `tests/test_m36_acceptance.py`

## Exit Condition

The milestone produces a coherent internal toy-benchmark bundle: schemas round-trip, action selection changes under parameter ablation, source-trust effects are behaviorally observable inside the synthetic setup, stress replay avoids silent corruption, and M3.5-M3.6 acceptance remains intact. This exit condition does not imply external validation or real-world identification of cognitive style.
