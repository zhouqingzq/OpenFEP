# M2.35 Milestone Specification

## Title

`M2.35: Inquiry Budget Scheduler`

## Scope

- Convert inquiry competition into an explicit bounded scheduler rather than letting every unresolved ambiguity stay equally live.
- Rank inquiry candidates across narrative uncertainty, experiment plans, prediction verification, subject tensions, and reconciliation threads on a shared bounded priority surface.
- Allocate scarce verification slots, workspace emphasis, and action budget using the scheduler state rather than independent local heuristics.
- Cool down repeatedly low-yield predictions so inconclusive probing does not silently consume verification capacity forever.
- Make scheduler decisions causally affect experiment plan status, verification targeting, workspace focus, action scoring, runtime traces, and snapshot persistence.

## Non-Goals

- Proving the globally optimal inquiry schedule over the full future horizon.
- Treating every unresolved unknown as equally urgent merely because it is uncertain.
- Allowing verification capacity to grow without an explicit slot budget.
- Preserving stale or low-yield probes indefinitely when stronger inquiry candidates exist.

## Acceptance Gates

- `cross_surface_ranking`: high-value inquiry candidates outrank merely uncertain but low-value candidates on a shared priority surface.
- `verification_budgeting`: verification slots stay bounded and repeated low-yield predictions can be cooled down or displaced.
- `workspace_allocation`: scheduler decisions alter workspace inquiry focus instead of staying local to logs.
- `action_biasing`: scheduler decisions alter downstream action scoring instead of remaining descriptive metadata.
- `downstream_causality`: removing scheduler state degrades at least one downstream surface among verification targeting, experiment plan activation, workspace focus, or action bias.
- `snapshot_roundtrip`: scheduler state survives runtime snapshot save and restore.
- `artifact_freshness`: milestone artifacts and acceptance report are generated in the current audit round.

## Canonical Files

- `segmentum/inquiry_scheduler.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/verification.py`
- `segmentum/types.py`
- `tests/test_m235_inquiry_scheduler.py`

## Required Evidence

- Canonical trace showing inquiry scheduling decisions and downstream runtime consumption.
- Evidence that high-value candidates outrank low-value but highly uncertain candidates.
- Evidence that verification slots are bounded and low-yield predictions can enter cooldown.
- Evidence that scheduler state changes workspace focus, action bias, and verification targeting.
- Snapshot round-trip evidence preserving scheduler state.
- Machine-readable acceptance report in `reports/m235_acceptance_report.json`.

## Audit Focus

- No decorative scheduler that ranks candidates but does not affect verification, workspace, or action selection.
- No unbounded verification slot growth under competing predictions.
- No low-yield prediction loop that silently consumes scarce inquiry budget forever.
- No loss of scheduler state on snapshot restore.
- No stale or inherited artifact accepted as current-round evidence.
