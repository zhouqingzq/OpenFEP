# M2.33 Strict Audit Preparation

Milestone: M2.33 Narrative Uncertainty Decomposition

Audit scope:
- validate structured unknown extraction
- validate competing hypothesis generation
- validate surface cue versus latent cause separation
- validate downstream causal influence on subject state, prediction ledger, and verification
- validate multilingual/noisy degradation behavior
- validate snapshot persistence

Primary milestone tests:
- `tests/test_m233_uncertainty_decomposition.py`
- `tests/test_m233_narrative_robustness.py`
- `tests/test_m233_acceptance.py`
- `tests/test_m233_audit_preparation.py`

Regression tests:
- `tests/test_narrative_compiler.py`
- `tests/test_m220_narrative_initialization.py`
- `tests/test_m227_snapshot_roundtrip.py`
- `tests/test_m228_prediction_ledger.py`
- `tests/test_runtime.py`

Strict gates:
- schema
- competition
- downstream_causality
- surface_latent_separation
- snapshot_roundtrip
- regression
- artifact_freshness

Expected evidence artifacts:
- canonical trace showing retained unknowns and downstream effects
- ablation showing degraded downstream propagation without uncertainty retention
- stress artifact showing multilingual/noisy robustness and snapshot continuity
