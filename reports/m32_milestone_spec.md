# M3.2 Milestone Specification

## Title

`M3.2: Memory-Derived Semantic Schemas`

## Scope

- Build semantic schemas from repeated grounded episodes.
- Track support, confidence, contexts, protected anchors, and conflict history.
- Persist semantic schemas through long-term memory state and sleep refresh.

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`

## Canonical Files

- `segmentum/semantic_schema.py`
- `segmentum/memory.py`
- `segmentum/agent.py`
- `segmentum/m32_audit.py`
- `tests/test_m32_semantic_schema_growth.py`
- `tests/test_m32_schema_conflict_resolution.py`
- `tests/test_m32_acceptance.py`

## Audit Bundle

- `artifacts/m32_semantic_schema_trace.json`
- `artifacts/m32_semantic_schema_ablation.json`
- `artifacts/m32_semantic_schema_stress.json`
- `reports/m32_acceptance_report.json`
- `reports/m32_acceptance_summary.md`
