# M4.1 Milestone Specification

## Title

`M4.1: Cognitive Variable Operationalization and Unified Interfaces`

## Scope

- Define a stable cognitive-parameter schema for the current M4 style family.
- Define a unified decision-log schema that records parameter snapshots, candidate actions, evidence, confidence, and updates.
- Define an executable observable registry that maps each parameter to indirect behavioral metrics.
- Provide a minimal internal simulator that exercises those contracts and makes interface-level audits executable.
- Provide acceptance reporting for schema completeness, trial variability, observability, intervention sensitivity, log completeness, and stress-mode interface behavior.

## Current Interpretation

M4.1 is the interface layer for cognitive-style work in this repository.

It is useful for:

- giving downstream tasks a common parameter language
- giving downstream audits a common observable language
- ensuring logs are rich enough for later benchmark and transfer work

It should not be read as evidence that:

- real cognitive styles have been identified on human data
- benchmark environments have been fully built out
- benchmark-task behavioral fit has already been established
- latent parameters have already passed blind classification, falsification, or recovery audits

## Non-Goals

- Benchmark registry and task-adapter setup
- External-bundle integration and provenance claims
- Single-task fit claims
- Baseline superiority claims
- Cross-task parameter credibility claims
- Open-world deployment claims

## Acceptance Definition

M4.1 passes only if the current round demonstrates all of the following:

1. The parameter schema roundtrips cleanly and remains stable.
2. The decision-log schema is complete enough to support downstream analysis.
3. Each parameter is linked to executable observable contracts.
4. Parameter interventions move their intended observables inside the minimal simulator.
5. Logs remain auditable under normal and stress-mode runs.

## Acceptance Gates

- `g1_schema_completeness`
- `g2_trial_variability`
- `g3_observability`
- `g4_intervention_sensitivity`
- `g5_log_completeness`
- `g6_stress_behavior`
- `r1_report_structure`

## Deferred Work

Moved to `M4.2`:

- benchmark task adapters
- benchmark bundle provenance
- replayability and leakage checks in benchmark environments

Moved to `M4.3` or later:

- single-task behavioral fit
- baseline comparison
- benchmark-quality metrics

Retained only as synthetic sidecar diagnostics, not `M4.1` acceptance evidence:

- blind classification
- parameter recovery
- falsification
- synthetic external-generator comparisons

## Canonical Files

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`
