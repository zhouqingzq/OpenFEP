# M3.6 Milestone Specification

## Title

`M3.6: Open-World Growth Trial`

## Scope

- Add a deterministic long-horizon open-world trial that composes semantic growth, process motivation, cognitive style, and bounded maintenance evidence.
- Run multi-subject, multi-narrative, multi-task trajectories with restart continuity checks.
- Produce a strict acceptance bundle with trace, ablation, stress, snapshot, failure-audit, report, and summary artifacts.

## Non-Goals

- Claiming open-world general intelligence from a single short benchmark.
- Allowing semantic growth to expand without bounded schema checks.
- Treating scripted replay logs as sufficient evidence of open-world composition.

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`

## Canonical Files

- `segmentum/m3_open_world_trial.py`
- `segmentum/m3_audit.py`
- `scripts/generate_m36_acceptance_artifacts.py`
- `tests/test_m36_open_world_growth.py`
- `tests/test_m36_acceptance.py`

## Audit Bundle

- `artifacts/m36_open_world_growth_trace.json`
- `artifacts/m36_open_world_growth_ablation.json`
- `artifacts/m36_open_world_growth_stress.json`
- `artifacts/m36_open_world_growth_snapshots.json`
- `artifacts/m36_open_world_growth_failure_audit.json`
- `reports/m36_acceptance_report.json`
- `reports/m36_acceptance_summary.md`
