# M4.1-M4.2 Execution Control

## Round Objective

Execute M4.1 and M4.2 through the standard loop:

1. Work prompt
2. Code generation or refresh
3. Strict audit acceptance
4. Repair against findings
5. Re-audit until pass

## Milestone Ledger

| Milestone | Spec | Prompt | Code | Audit | Repair Loop | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| M4.1 | ready | ready | implemented | pending | pending | pending |
| M4.2 | ready | ready | implemented | pending | pending | pending |

## Global Constraints

- Do not overwrite unrelated dirty worktree changes.
- Every acceptance report must be generated in the current round.
- Every milestone must preserve strict audit fields and evidence categories.
- M4.2 must regress M4.1 and M3.6 before merge.
