# M4.3-M4.6 Execution Control

## Round Objective

Execute M4.3, M4.4, M4.5, and M4.6 through the standard loop:

1. Work prompt
2. Code generation or refresh
3. Strict audit acceptance
4. Repair against findings
5. Re-audit until pass

## Milestone Ledger

| Milestone | Spec | Prompt | Code | Audit | Repair Loop | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| M4.3 | ready | ready | implemented | pass | completed | pass |
| M4.4 | ready | ready | implemented | pass | completed | pass |
| M4.5 | ready | ready | implemented | pass | completed | pass |
| M4.6 | ready | ready | implemented | pass | completed | pass |

## Global Constraints

- Do not overwrite unrelated dirty worktree changes.
- Every acceptance report must be generated in the current round.
- Every milestone must preserve strict audit fields and evidence categories.
- M4.5 is the controlled complex-environment bridge milestone; open-world tool integration is deferred to M5.
- M4.6 must regress M4.5, M4.4, M4.3, and continuity/restart acceptance evidence before merge.
