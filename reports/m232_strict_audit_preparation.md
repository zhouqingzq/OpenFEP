# M2.32 Strict Audit Preparation

## Goal

Prepare `M2.32` so implementation starts from the real post-`M2.31` baseline instead of from an outdated summary-only assumption.

## Working Assumption

`M2.31` already ships a proto claim-level reconciliation path. `M2.32` is therefore prepared as `Claim-Level Reconciliation Hardening`: bounded selection, contested-state visibility, provenance completeness, and adversarial containment.

## Current Codebase Touchpoints

- `[self_model.py](/E:/workspace/segments/segmentum/self_model.py)` already contains:
  - `NarrativeClaim`
  - `IdentityNarrative.claims`
  - claim-level reconciliation metadata
  - commitment derivation from narrative claims
- `[reconciliation.py](/E:/workspace/segments/segmentum/reconciliation.py)` already contains:
  - persistent reconciliation threads
  - verification binding
  - summary-level narrative writeback
  - initial claim-level claim updates and commitment recalibration
- `[m231_acceptance_report.json](/E:/workspace/segments/reports/m231_acceptance_report.json)` is now the immediate regression boundary.

## What Must Exist Before Audit Execution

1. A bounded mapping from reconciliation thread to affected claim ids that rejects weak or ambiguous matches.
2. A claim-edit payload that survives:
   snapshot save and restore,
   runtime trace emission,
   sleep review replay.
3. An explicit distinction between reaffirmed, contested, and downgraded claims.
4. A containment rule proving unrelated threads cannot mutate unrelated claims.
5. A freshness rule proving acceptance artifacts come from a real current-round pytest execution and not a simulated report payload.

## Minimum Test Plan To Add

- `tests/test_m232_claim_reconciliation.py`
  Verifies bounded claim edits, explicit provenance, and downstream commitment or summary change.
- `tests/test_m232_claim_containment.py`
  Verifies unrelated threads, weak anchors, and unmatched repair evidence do not mutate unrelated claims.
- `tests/test_m232_acceptance.py`
  Verifies the machine-readable report, artifact freshness, deterministic seeds, and executed regression evidence.

## Minimum Artifact Plan

- `artifacts/m232_claim_reconciliation_trace.jsonl`
  Canonical replay with claim edit payloads, claim outcome classes, and downstream summary effects.
- `artifacts/m232_claim_reconciliation_ablation.json`
  Controlled comparison between hardened bounded claim writeback and the current baseline claim writeback path.
- `artifacts/m232_claim_reconciliation_stress.json`
  Negative-control artifact for cross-claim contamination and provenance leakage.
- `reports/m232_acceptance_report.json`
  Final machine-readable decision object.
- `reports/m232_acceptance_summary.md`
  Human-readable summary.

## Proposed Canonical Seeds

- `232`
- `464`

These satisfy the strict audit requirement for multi-seed deterministic evidence and keep continuity with the existing milestone pattern.

## Proposed Regression Boundary

- `tests/test_m229_acceptance.py`
- `tests/test_m230_acceptance.py`
- `tests/test_m231_acceptance.py`
- `tests/test_narrative_evolution.py`

## High-Risk Failure Modes To Audit For

- Summary updates imply reconciliation succeeded while the wrong claims were edited.
- One reconciliation thread edits multiple claims without evidence-level attribution.
- Mixed or contested claims are silently rewritten as fully resolved.
- Snapshot restore keeps thread state but loses claim-edit provenance.
- Acceptance report records inherited or simulated pytest output instead of real execution evidence.

## Exit Criteria For `Ready To Implement`

- The milestone spec is frozen against the actual post-`M2.31` baseline.
- The seed set is fixed.
- Artifact names and report path are frozen.
- Regression suites are named.
- The causal claim is explicit:
  hardened claim-level reconciliation must alter at least one downstream behavior, not only logs.
