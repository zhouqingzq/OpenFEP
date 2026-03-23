# M2.31 Milestone Spec

## Name

M2.31 Long-Horizon Reconciliation

## Scope

- Promote repeated or identity-critical conflict signals into persistent reconciliation threads.
- Require thread-bound evidence before a local repair can be upgraded into deep reconciliation.
- Write reconciliation state back into the autobiographical narrative rather than leaving it as a sidecar runtime mechanism.
- Preserve reconciliation state across runtime trace emission, sleep review, and snapshot restore.

## In Scope Files

- `segmentum/reconciliation.py`
- `segmentum/verification.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/subject_state.py`
- `segmentum/self_model.py`

## Gating Claims

- Repeated conflicts must become persistent threads that span chapters when the same contradiction recurs.
- Verification evidence must bind to a single matching thread; unrelated evidence must not contaminate other threads.
- Repair attempts must bind only when attribution is explicit; unmatched repair attempts must not update any thread.
- Reconciliation must write back into `IdentityNarrative` fields that survive serialization and influence downstream summaries.

## Required Evidence

- Canonical deterministic trace showing reconciliation payload and narrative writeback.
- Ablation comparing writeback-enabled reconciliation against writeback removal.
- Stress artifact showing containment under unrelated verification evidence and unmatched repair attempts.
- Machine-readable acceptance report in `reports/m231_acceptance_report.json`.

## Non-Goals

- Full narrative claim regeneration from reconciliation alone.
- Large-scale redesign of chapter generation.
- Automated promotion to `M2.32` without strict audit acceptance.

## Residual Risks To Watch

- Narrative writeback may remain too summary-level if future milestones require claim-level reconciliation edits.
- Cross-thread attribution still depends on current commitment and identity-anchor metadata quality.
