# M4.8 Acceptance Criteria — Ablation Contrast

## Gate 1: memory_enabled flag exists and round-trips

- `SegmentAgent(memory_enabled=False)` produces an agent where `agent.memory_enabled is False`.
- `to_dict()` includes `memory_enabled`; `from_dict()` restores it faithfully.
- `SegmentRuntime.load_or_create(memory_enabled=False)` produces a runtime whose agent has `memory_enabled=False`.
- `export_snapshot()` includes `memory_enabled`.

## Gate 2: memory_enabled=False cleanly suppresses memory influence

- `_retrieve_decision_memories()` returns `[]` when disabled.
- `_build_memory_context()` returns a context with all-zero `state_delta`, zero `chronic_threat_bias`, zero `protected_anchor_bias`.
- `memory_bias` and `pattern_bias` in the decision loop are `0.0` when disabled.
- Episode recording still occurs: `long_term_memory.episodes` grows even when disabled.
- `last_memory_context["memory_enabled"]` reflects the flag state.
- `last_memory_context["memory_bias"]` and `last_memory_context["pattern_bias"]` are present and correct in both modes.

## Gate 3: Ablation contrast is significant and reproducible

- **Divergence**: A 20-cycle same-seed rollout with `memory_enabled=True` vs `False` produces different decision sequences (at least 1 action differs).
- **Entropy**: Decision entropy (measured as action distribution entropy over the 20 cycles) differs between enabled and disabled runs.
- **Bias directionality**: In the enabled run, when `chronic_threat_bias > 0.1`, the proportion of avoidance actions (`hide`, `rest`) is higher than in the disabled run.
- **state_delta causation**: Enabled run has at least one cycle where `max(abs(state_delta.values())) > 0.05`; disabled run has `state_delta` all-zero every cycle.
- **Negative control**: Two `memory_enabled=True` runs with the same seed produce **identical** decision sequences, proving divergence is caused by the flag and not by nondeterminism.

## Gate 4: memory_bias / pattern_bias exposed in last_memory_context

- `last_memory_context["memory_bias"]` is a `float` reflecting the per-action memory bias of the chosen action.
- `last_memory_context["pattern_bias"]` is a `float` reflecting the per-action pattern bias of the chosen action.
- Both are `0.0` when `memory_enabled=False`.

## Gate 5: m47_runtime snapshot demoted to diagnostic

- `m47_audit.py` or `m47_reacceptance.py` contains `diagnostic_only: true` (or equivalent field) on the shared runtime snapshot.
- `reports/m4_milestone_boundaries.md` states that M4.7 behavioral claims depend on M4.8 ablation evidence, and M4.5-M4.7 satisfy only structural self-consistency (layer a).

## Gate 6: Three-layer acceptance model documented

- `reports/m4_milestone_boundaries.md` defines layers (a) structural, (b) behavioral, (c) phenomenological.
- Each M4 milestone row in the boundary table is annotated with which layers it currently satisfies.

## NOT in scope

- No changes to scoring formulas, thresholds, or multipliers.
- No changes to encoding, promotion, retrieval, or consolidation logic.
- No new memory features — this milestone is purely about proving existing memory works.

## 2026-04-10 Status Alignment

- M4.8 official acceptance is `NOT_ISSUED` until all six gates pass.
