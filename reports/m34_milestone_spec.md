# M3.4 Milestone Specification

## Title

`M3.4: Process-Valence Motivation Loop`

## Scope

- Add a bounded process-valence state for unresolved wanting, closure, satiation, and boredom.
- Make unresolved targets persist across cycles and resolved targets cool off after closure.
- Let low-tension states redirect inquiry toward new unresolved targets.

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`

## Canonical Files

- `segmentum/drives.py`
- `segmentum/preferences.py`
- `segmentum/inquiry_scheduler.py`
- `segmentum/agent.py`
- `segmentum/m34_audit.py`
- `tests/test_m34_process_valence_motivation.py`
- `tests/test_m34_acceptance.py`

## Audit Bundle

- `artifacts/m34_process_valence_trace.json`
- `artifacts/m34_process_valence_ablation.json`
- `artifacts/m34_process_valence_stress.json`
- `reports/m34_acceptance_report.json`
- `reports/m34_acceptance_summary.md`
