# M2.27 Milestone Specification

## Title

`M2.27: Unified Subject State`

## Scope

- Introduce a persistent `SubjectState` that summarizes the agent's current phase, dominant goal, continuity anchors, tensions, priorities, and self-maintenance flags.
- Make subject state affect downstream behavior in three places:
  - action selection,
  - episodic memory write sensitivity,
  - maintenance agenda prioritization.
- Persist subject state through snapshot save/restore and expose it in trace and conscious report surfaces.

## Non-Goals

- Long-horizon identity therapy or value retraining.
- Replacing commitment evaluation, workspace broadcast, or homeostasis subsystems.
- Claiming organism-level soak stability beyond the local subject-state contract.

## Acceptance Gates

- `schema`: `SubjectState` round-trips through runtime snapshot and restart.
- `determinism`: canonical seeds `227` and `342` replay to equivalent subject-state signatures.
- `causality`: subject state changes at least one downstream behavior under controlled comparison.
- `ablation`: removing subject-state modulation degrades safety reroute and memory protection behavior.
- `stress`: acute continuity and maintenance stress do not silently corrupt subject-state persistence.
- `regression`: relevant prior milestone suites still pass.

## Canonical Files

- `segmentum/subject_state.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/m227_audit.py`
- `tests/test_m227_subject_state.py`
- `tests/test_m227_subject_state_causality.py`
- `tests/test_m227_snapshot_roundtrip.py`
- `tests/test_m227_subject_state_ablation.py`
- `tests/test_m227_subject_state_stress.py`
- `tests/test_m227_acceptance.py`
- `scripts/generate_m227_acceptance_artifacts.py`

## Risks

- Subject state could become decorative if it only appears in explanations.
- Snapshot compatibility could drift if the payload is present only on fresh runs.
- Stress handling could bias maintenance without preserving continuity anchors across restart.

## Audit Bundle

- Canonical trace artifact: `artifacts/m227_subject_state_trace.jsonl`
- Ablation artifact: `artifacts/m227_subject_state_ablation.json`
- Stress artifact: `artifacts/m227_subject_state_stress.json`
- Machine-readable report: `reports/m227_acceptance_report.json`
- Human summary: `reports/m227_acceptance_summary.md`
