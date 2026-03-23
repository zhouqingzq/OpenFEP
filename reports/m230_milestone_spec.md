# M2.30 Milestone Specification

## Title

`M2.30: Slow Variable Learning and Continuity Hardening`

## Scope

- Introduce slow-variable consolidation that updates long-horizon traits, values, and identity stability from repeated experience.
- Make slow-learning pressure affect downstream behavior in five places:
  - action selection,
  - episodic memory write sensitivity,
  - verification target prioritization,
  - continuity auditing,
  - subject-state interpretation.
- Preserve protected anchors and per-cycle drift budgets so repeated evidence can change the agent without abrupt collapse.
- Persist slow-learning state, audits, and explanations through snapshot save/restore and runtime trace emission.

## Non-Goals

- Replacing the existing personality profile, prediction ledger, or continuity audit.
- Claiming open-ended lifelong learning beyond bounded sleep-cycle consolidation.
- Allowing slow-variable pressure to silently override protected continuity anchors.

## Acceptance Gates

- `schema`: slow-learning state and audits round-trip through agent snapshot and runtime trace.
- `determinism`: canonical seeds `230` and `460` replay to equivalent slow-learning signatures.
- `causality`: slow-variable consolidation changes at least one downstream behavior under controlled comparison.
- `ablation`: removing slow-learning consolidation degrades declared downstream capabilities.
- `stress`: divergent pressure, anchor protection, and snapshot restore do not silently corrupt continuity state.
- `regression`: relevant prior milestone suites still pass.
- `artifact_freshness`: acceptance report and artifacts are generated in the current round.

## Canonical Files

- `segmentum/slow_learning.py`
- `segmentum/agent.py`
- `segmentum/runtime.py`
- `segmentum/self_model.py`
- `segmentum/subject_state.py`
- `segmentum/verification.py`
- `segmentum/m230_audit.py`

## Audit Focus

- No fake slow learning: repeated experience must change policy or verification behavior, not only logs.
- No silent continuity inflation: subject-state continuity must remain aligned with the canonical continuity audit.
- No collapse by accumulation: multi-variable pressure must remain bounded by explicit drift budgets and protected anchors.
