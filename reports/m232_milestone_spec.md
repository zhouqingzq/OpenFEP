# M2.32 Milestone Specification

## Title

`M2.32: Claim-Level Narrative Reconciliation Hardening`

## Assumption

`M2.31` already introduced an initial claim-level reconciliation path:

- reconciled threads can write bounded metadata into `NarrativeClaim`,
- claim updates can influence `IdentityNarrative.contradiction_summary`,
- commitment confidence can be recalibrated from reconciled claims.

`M2.32` therefore must not re-claim those baseline capabilities. Its role is to harden claim-level narrative reconciliation into a strict, bounded, and adversarially-auditable mechanism.

## Scope

- Constrain claim selection so one reconciliation thread only edits claims with explicit anchor overlap or explicit thread attribution.
- Distinguish three claim outcomes under reconciliation:
  - reaffirmed,
  - contested,
  - downgraded.
- Preserve explicit provenance from reconciliation thread to:
  - updated claim ids,
  - source evidence ids,
  - source chapter ids,
  - repair and verification evidence.
- Prove that claim-level reconciliation changes downstream behavior in at least three places:
  - `IdentityNarrative.contradiction_summary`,
  - commitment derivation or confidence,
  - generated autobiographical or core summary text.
- Preserve claim-level reconciliation state through snapshot save/restore, runtime trace emission, and sleep review replay.
- Add containment guarantees showing unrelated threads cannot mutate unrelated claims or silently launder mixed claims into resolved ones.

## Non-Goals

- Re-introducing summary-level narrative writeback already accepted in `M2.31`.
- Full free-text regeneration of the autobiographical narrative.
- Replacing the existing `NarrativeClaim` scoring model.
- Open-ended narrative editing without thread-bound attribution.
- Declaring `M2.33` readiness from implementation alone without a fresh strict audit bundle.

## Acceptance Gates

- `schema`: claim-level reconciliation metadata round-trips through agent snapshot and runtime trace.
- `determinism`: canonical seeds `232` and `464` replay to equivalent claim-revision signatures.
- `causality`: hardened claim-level reconciliation changes at least one downstream behavior under controlled comparison.
- `ablation`: removing bounded claim-level writeback degrades contradiction closure, commitment recalibration, or summary grounding.
- `stress`: unrelated reconciliation threads cannot mutate unrelated claims or leak provenance across claims.
- `mixed_state_visibility`: unresolved or mixed claims remain explicitly contested instead of being silently marked reconciled.
- `regression`: relevant prior milestone suites still pass.
- `artifact_freshness`: acceptance report and artifacts are generated in the current round.

## Canonical Files

- `segmentum/self_model.py`
- `segmentum/reconciliation.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/subject_state.py`
- `segmentum/m232_audit.py`
- `tests/test_m232_claim_reconciliation.py`
- `tests/test_m232_claim_containment.py`
- `tests/test_m232_acceptance.py`

## Required Evidence

- Canonical deterministic trace showing bounded claim-level reconciliation edits and downstream narrative consequences.
- Ablation comparing hardened claim-level writeback against the current baseline claim writeback path.
- Stress artifact showing cross-claim containment under unrelated thread pressure.
- Machine-readable acceptance report in `reports/m232_acceptance_report.json`.

## Audit Focus

- No decorative claim edits: revised claims must change commitment or summary behavior.
- No free-floating reconciliation: every changed claim must point back to one bounded reconciliation thread.
- No silent laundering: unresolved or mixed claims must remain visibly contested.
- No provenance loss: updated claims must survive snapshot round-trip with repair and verification evidence intact.
- No overreach: one thread must not rewrite an unrelated claim set without explicit anchor evidence.

## Candidate Artifacts

- Canonical trace artifact: `artifacts/m232_claim_reconciliation_trace.jsonl`
- Ablation artifact: `artifacts/m232_claim_reconciliation_ablation.json`
- Stress artifact: `artifacts/m232_claim_reconciliation_stress.json`
- Machine-readable report: `reports/m232_acceptance_report.json`
- Human summary: `reports/m232_acceptance_summary.md`

## Recommended Regression Suites

- `tests/test_m229_acceptance.py`
- `tests/test_m230_acceptance.py`
- `tests/test_m231_acceptance.py`
- `tests/test_narrative_evolution.py`

## Risks

- Claim edits may drift from the actual evidence ledger if reconciliation writes into claims without bounded provenance.
- Commitment derivation could overreact if one reconciled thread rewrites too many narrative claims at once.
- Summary regeneration could still mask unresolved contradictions if mixed claims are collapsed too aggressively.
- Existing M2.31 baseline behavior may make the M2.32 causal delta too small unless the new bounded-selection rules are explicitly ablated.
