# M2.32 Strict Audit Preparation

## Goal

Freeze `M2.32` around the actual implemented theme: threat-trace persistence, explicit anchor protection, and memory-driven attention/prediction influence.

## What Must Be True Before Audit Sign-Off

1. `protect_episode_ids(...)` works for non-identity structural traces.
2. Explicit protection metadata survives continuity synchronization.
3. A protected or chronic-threat trace can change downstream prediction shaping.
4. A sensitive memory pattern can change downstream attention selection.
5. Regression evidence covers memory, restart protection, and threat learning.

## Minimum Test Plan

- `tests/test_restart_memory_protection.py`
  Verifies structural trace anchors remain restart-protected.
- `tests/test_m28_attention.py`
  Verifies memory-sensitive pattern bias can promote threat channels.
- `tests/test_m2_targeted_repair.py`
  Verifies memory context changes prediction surface with explicit chronic-threat summary.
- `tests/test_m232_acceptance.py`
  Verifies audit artifacts and freshness behavior.

## Minimum Artifact Plan

- `artifacts/m232_threat_memory_trace.jsonl`
- `artifacts/m232_threat_memory_ablation.json`
- `artifacts/m232_threat_memory_stress.json`
- `reports/m232_acceptance_report.json`
- `reports/m232_acceptance_summary.md`

## Regression Boundary

- `tests/test_memory.py`
- `tests/test_threat_profile_learning.py`

## Exit Criteria

- The milestone spec reflects threat-trace protection rather than claim reconciliation.
- The seed set is fixed at `232` and `464`.
- A machine-readable acceptance report can be produced.
- Strict mode blocks injected or stale execution records.
