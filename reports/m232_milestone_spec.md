# M2.32 Milestone Specification

## Title

`M2.32: Threat Trace Protection And Trauma-Like Encoding`

## Scope

- Persistent or repeated threat should leave a structural memory trace instead of collapsing into an undifferentiated routine episode.
- High-value memory anchors must be explicitly protectable even when they are not identity-critical commitments.
- Sensitive memory patterns must influence both:
  - attention allocation,
  - memory-conditioned prediction shaping.
- Protected anchors must survive restart continuity handling, compression pressure, and snapshot round-trip.

## Non-Goals

- Replacing the existing preference model or value hierarchy.
- Rewriting the full narrative system.
- Declaring `M2.33` readiness from implementation alone without a fresh audit bundle.

## Acceptance Gates

- `schema`: protected-anchor metadata survives serialization and restart-anchor export.
- `protection`: explicit `protect_episode_ids(...)` protection works for non-identity structural traces.
- `causality`: threat-trace retrieval changes downstream prediction or action scoring.
- `attention_prediction_influence`: sensitive patterns measurably change attention selection and prediction deltas.
- `regression`: affected prior memory/threat suites still pass.
- `artifact_freshness`: acceptance report and artifacts are generated in the current audit round.

## Canonical Files

- `segmentum/memory.py`
- `segmentum/agent.py`
- `segmentum/attention.py`
- `segmentum/world_model.py`
- `segmentum/m232_audit.py`
- `tests/test_restart_memory_protection.py`
- `tests/test_m28_attention.py`
- `tests/test_m2_targeted_repair.py`
- `tests/test_m232_acceptance.py`

## Required Evidence

- Canonical trace showing structural trace creation and protected-anchor export.
- Ablation showing prediction/attention degradation without threat-trace protection.
- Stress artifact showing non-identity anchors remain protected.
- Machine-readable acceptance report in `reports/m232_acceptance_report.json`.

## Audit Focus

- No crashing protection path.
- No silent loss of explicit protection metadata.
- No decorative threat traces that fail to affect prediction or attention.
- No regression where learned sleep policy bias disappears because cluster binding is absent.
