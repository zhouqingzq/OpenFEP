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
- `segmentum/m41_inference.py` already proves benchmark-task parameter recovery

## Non-Goals

- Benchmark registry and task-adapter setup
- External-bundle integration and provenance claims
- Single-task fit claims
- Baseline superiority claims
- Cross-task parameter credibility claims
- Open-world deployment claims
- Parameter recovery, blind classification, or falsification claims of any kind

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

## Canonical Files

Only these files are M4.1 acceptance evidence:

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `segmentum/m41_explanations.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`

## Sidecar Modules (NOT acceptance evidence)

The following carry an `m41_` prefix but are synthetic diagnostic sidecars
for M4.3+ pre-research. They must not be cited as M4.1 acceptance evidence
and must not carry `external_validation: true` labels.

- `segmentum/m41_inference.py` — toy parameter recovery (ridge regression + candidate bank)
- `segmentum/m41_blind_classifier.py` — cross-generator synthetic classifier
- `segmentum/m41_baselines.py` — same-framework baseline models
- `segmentum/m41_falsification.py` — internal intervention sensitivity
- `segmentum/m41_identifiability.py` — same-framework recoverability
- `segmentum/m41_external_generator.py` — second synthetic generator (NOT external)
- `segmentum/m41_external_dataset.py` — holdout data loader
- `segmentum/m41_external_observables.py` — alternative observable computation
- `segmentum/m41_external_validation.py` — task eval wrapper
- `segmentum/m41_external_task_eval.py` — belongs to M4.2 scope
- `scripts/generate_m41_external_data.py` — synthetic data generator
- `data/m41_external/` — synthetic holdout data (not external human data)

## Deferred Work

Moved to `M4.2`:

- benchmark task adapters
- benchmark bundle provenance
- replayability and leakage checks in benchmark environments

Moved to `M4.3` or later:

- single-task behavioral fit
- baseline comparison
- benchmark-quality metrics
- blind classification on external data
- parameter falsification on benchmark data
- parameter recovery on non-synthetic data
