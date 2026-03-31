# M1 Milestone Specification

## Title

`M1: Core Survival Agent Foundations`

## Scope Origin

No authored `M1` spec, acceptance report, or dedicated `M1` test suite was found in this repository.
This document defines the smallest executable M1 scope that can be justified from current repo evidence:

- the core architecture and principles described in `README.md`,
- the earliest pre-M2 regression-style tests under `tests/`,
- the M2 evaluation report's description of what already existed before M2-specific expansion.

This is therefore a reconstructed spec, not a recovered original contract.

## Goal

Establish a deterministic survival-agent core that can:

- maintain and persist runtime state,
- classify self-vs-world failures through a self model,
- write and filter episodic memory based on surprise/value,
- consolidate repeated experience through sleep,
- compile narrative inputs into embodied episodes with provenance.

## Acceptance Gates

- `determinism`
  Fixed-seed runtime execution is replayable in the current round.
- `persistence`
  Runtime snapshots preserve cycle state and episodic memory.
- `self_model`
  Resource exhaustion and fatal events are classified through the self model.
- `memory`
  High-surprise episodes are stored and low-value low-error ticks are skipped.
- `sleep`
  Sleep extraction changes downstream surprise or policy behavior under repeat exposure.
- `narrative`
  Narrative compilation and ingestion emit provenance-bearing episodic memory.
- `regression`
  Core foundational test suites pass in the current round.

## Canonical Files

- `README.md`
- `reports/m2_evaluation_report.md`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/memory.py`
- `segmentum/self_model.py`
- `segmentum/sleep_consolidator.py`
- `segmentum/narrative_compiler.py`
- `segmentum/narrative_ingestion.py`
- `segmentum/m1_audit.py`
- `scripts/generate_m1_acceptance_artifacts.py`
- `tests/test_runtime.py`
- `tests/test_memory.py`
- `tests/test_self_model.py`
- `tests/test_sleep_consolidation_loop.py`
- `tests/test_narrative_compiler.py`
- `tests/test_narrative_ingestion.py`
- `tests/test_m1_acceptance.py`

## Non-Goals

- Claiming that an original authored M1 package has been recovered.
- Treating later M2/M3/M4 benchmark work as part of M1.
- Declaring real-world benchmark validation or external-tool readiness.

## Audit Bundle

- `artifacts/m1_runtime_trace.json`
- `artifacts/m1_memory_gate.json`
- `artifacts/m1_sleep_consolidation.json`
- `artifacts/m1_narrative_trace.json`
- `reports/m1_acceptance_report.json`
- `reports/m1_acceptance_summary.md`
