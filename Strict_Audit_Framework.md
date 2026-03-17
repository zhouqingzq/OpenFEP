# Strict Audit Framework

## Purpose

This framework applies to every milestone from `M2.12` onward.

The goal is to prevent three recurring failure modes:

- shipping narrative claims without causal evidence,
- accepting benchmark wins without adversarial replay,
- declaring readiness from stale or partial artifacts.

---

## Audit Objects

Each milestone must define and preserve six audit objects:

1. `Specification`
   The milestone contract: scope, files, gates, risks, and non-goals.
2. `Deterministic Evidence`
   Fixed-seed traces, snapshots, and benchmark outputs.
3. `Adversarial Evidence`
   Stress injections, ablation runs, failure injections, and negative controls.
4. `Regression Evidence`
   Proof that current milestone work did not silently damage prior guarantees.
5. `Interpretation`
   A machine-readable acceptance report plus a short human summary.
6. `Decision`
   `PASS`, `PASS_WITH_RESIDUAL_RISK`, `FAIL`, or `BLOCKED`.

No audit is valid without all six.

---

## Required Evidence Categories

Every milestone must produce evidence for all categories below:

- `schema`: serialized format is versioned and round-trippable.
- `determinism`: fixed seeds and configs replay to equivalent outputs.
- `causality`: the new mechanism changes behavior, not only traces.
- `ablation`: removing the mechanism degrades a declared downstream capability.
- `stress`: failure injections or resource pressure do not produce silent corruption.
- `regression`: critical prior milestone suites still pass.
- `artifact_freshness`: reports are generated in the current round, not inherited silently.

---

## Minimum Acceptance Bundle

Each milestone acceptance bundle must include:

- at least 1 canonical trace artifact,
- at least 1 ablation artifact,
- at least 1 stress or failure-injection artifact,
- at least 1 machine-readable acceptance report in `reports/`,
- at least 2 new milestone-specific test files,
- explicit reuse of relevant prior regression suites.

---

## Severity Model

Audit findings use four severities:

- `S0 critical`
  Safety, silent corruption, fake evidence, or broken causal claim.
- `S1 major`
  Acceptance gate not met, but failure is localized and diagnosable.
- `S2 moderate`
  Important evidence missing or incomplete; claim weakened.
- `S3 minor`
  Documentation, trace readability, or non-gating hygiene issue.

Rules:

- Any unresolved `S0` means automatic `FAIL`.
- Two or more unresolved `S1` also mean `FAIL`.
- `PASS_WITH_RESIDUAL_RISK` is allowed only when no `S0` exists and every `S1` has a documented containment plan.

---

## Audit Procedure

For every milestone:

1. Freeze the milestone specification and acceptance gates.
2. Generate canonical artifacts.
3. Run milestone test suite.
4. Run adversarial and ablation suite.
5. Run required historical regressions.
6. Generate a machine-readable acceptance report.
7. Review freshness, reproducibility, and unresolved findings.
8. Record final disposition.

---

## Freshness Rules

- Every acceptance report must contain the generation timestamp, commit hash if available, seed set, and artifact paths.
- A stale artifact cannot satisfy a current acceptance gate by itself.
- If evidence is inherited, it must be labeled as inherited and non-gating unless explicitly replayed this round.

---

## Determinism Rules

- Every milestone must declare its canonical seed set.
- At least one audit path must use more than one seed.
- If exact replay is not possible, the report must define tolerance bounds and justify them.
- Any hidden nondeterminism must be classified as an audit finding.

---

## Causality Rules

A milestone only passes if the newly introduced mechanism is shown to alter at least one downstream behavior under controlled comparison.

Examples:

- workspace affects memory prioritization or action selection,
- identity commitments affect policy ranking,
- social memory affects future partner behavior,
- governance blocks actions that would otherwise execute.

If the mechanism appears only in logs or explanations, the milestone fails.

---

## Required Report Fields

Every machine-readable acceptance report must contain:

- `milestone_id`
- `status`
- `generated_at`
- `seed_set`
- `artifacts`
- `tests`
- `gates`
- `findings`
- `residual_risks`
- `freshness`
- `recommendation`

---

## Exit Criteria

A milestone is accepted only when:

- all gating tests pass,
- all required artifacts exist,
- all mandatory evidence categories are present,
- no unresolved `S0` remains,
- the final report recommendation is not `BLOCK`.

If any condition fails, the milestone remains open.
