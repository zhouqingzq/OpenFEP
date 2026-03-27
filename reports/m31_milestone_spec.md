# M3.1 Milestone Specification

## Title

`M3.1: Episodic Semantic Grounding`

## Scope

- Introduce a deterministic semantic grounding layer for open narrative text.
- Separate surface, paraphrase, and implicit evidence from downstream appraisal.
- Persist grounded motifs and provenance through embodied narrative episodes and memory ingestion.

## Non-Goals

- Stochastic semantic parsing.
- Replacing uncertainty decomposition.
- Claiming schema growth before memory-derived compression exists.

## Acceptance Gates

- `schema`
- `determinism`
- `causality`
- `ablation`
- `stress`
- `regression`

## Canonical Files

- `segmentum/semantic_grounding.py`
- `segmentum/narrative_compiler.py`
- `segmentum/narrative_types.py`
- `segmentum/m31_audit.py`
- `tests/test_m31_episodic_semantic_grounding.py`
- `tests/test_m31_grounding_causality.py`
- `tests/test_m31_acceptance.py`

## Audit Bundle

- `artifacts/m31_semantic_grounding_trace.json`
- `artifacts/m31_semantic_grounding_ablation.json`
- `artifacts/m31_semantic_grounding_stress.json`
- `reports/m31_acceptance_report.json`
- `reports/m31_acceptance_summary.md`
