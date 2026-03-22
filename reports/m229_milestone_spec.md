# M2.29 Milestone Specification

## Title

`M2.29: Prediction Verification Loop`

## Scope

- Introduce an explicit `VerificationLoop` that turns active predictions into concrete verification targets.
- Make verification pressure affect downstream behavior in four places:
  - action selection,
  - workspace focus,
  - episodic memory write sensitivity,
  - maintenance agenda prioritization.
- Update the prediction ledger when evidence confirms, falsifies, contradicts, or times out a prediction.
- Persist verification state through snapshot save/restore and expose it in runtime trace and explanation payloads.

## Non-Goals

- Replacing the prediction ledger or global workspace.
- Claiming long-horizon epistemic calibration beyond the local verification contract.
- Declaring M2.30 readiness from code integration alone without a fresh acceptance bundle.

## Acceptance Gates

- `schema`: verification loop state round-trips through agent snapshot and runtime trace.
- `determinism`: canonical seeds `229` and `431` replay to equivalent verification signatures.
- `causality`: verification pressure changes at least one downstream behavior under controlled comparison.
- `ablation`: removing verification pressure degrades evidence-seeking behavior and maintenance prioritization.
- `stress`: missing evidence and snapshot restore do not silently corrupt verification state.
- `regression`: relevant prior milestone suites still pass.
- `artifact_freshness`: acceptance report and artifacts are generated in the current round.

## Canonical Files

- `segmentum/verification.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/types.py`
- `segmentum/m229_audit.py`
- `tests/test_m229_verification_loop.py`
- `tests/test_m229_acceptance.py`
- `scripts/generate_m229_acceptance_artifacts.py`

## Risks

- Verification could become decorative if it appears only in explanation payloads.
- Timeout handling could escalate predictions without preserving the causal evidence chain.
- Snapshot restore could keep ledger state while losing active verification targets.

## Audit Bundle

- Canonical trace artifact: `artifacts/m229_verification_trace.jsonl`
- Ablation artifact: `artifacts/m229_verification_ablation.json`
- Stress artifact: `artifacts/m229_verification_stress.json`
- Machine-readable report: `reports/m229_acceptance_report.json`
- Human summary: `reports/m229_acceptance_summary.md`
