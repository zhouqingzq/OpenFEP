# M3.1-M3.6 Execution Control

## Round Objective

Execute every milestone through the same loop:

1. Work prompt
2. Code generation or refresh
3. Strict audit acceptance
4. Repair against findings
5. Re-audit until pass

## Milestone Ledger

| Milestone | Spec | Prompt | Code | Audit | Repair Loop | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| M3.1 | ready | ready | refreshed | rerun | closed with pass | PASS |
| M3.2 | ready | ready | refreshed | rerun | closed with pass | PASS |
| M3.3 | ready | ready | refreshed | rerun | closed with pass | PASS |
| M3.4 | ready | ready | refreshed | rerun | closed with pass | PASS |
| M3.5 | ready | ready | refreshed | rerun | closed with pass | PASS |
| M3.6 | ready | ready | implemented | strict acceptance | closed with pass | PASS |

## Global Constraints

- Do not overwrite unrelated dirty worktree changes.
- Every acceptance report must be generated this round.
- Every milestone must keep strict audit framework fields and evidence categories.
- M3.6 must regress M3.1-M3.5 before merge.
