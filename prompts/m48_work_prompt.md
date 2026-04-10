# M4.8 Work Prompt — Ablation Contrast (Memory Causal Verification)

## Goal

Prove that the memory system **causally changes** default agent behavior, not just audit snapshots. The deliverable is an ablation framework and a test suite that demonstrates measurable, reproducible behavioral divergence when memory is enabled vs disabled.

## Context

After the H1-H3 hotfix, the default `SegmentAgent` runs with `memory_backend="memory_store"`. Diagnostics confirm:
- `memory_hit=True`, retrieval returns real episode IDs
- `state_delta` is nonzero (retrieval modifies predicted state)
- `memory_bias` and `pattern_bias` are computed and enter `policy_score`
- `chronic_threat_bias` and `protected_anchor_bias` shape state projection

But no test currently proves that **disabling** memory changes behavior. This milestone closes that gap.

## Implementation Plan

### 1. Add `memory_enabled` flag to SegmentAgent

- Add `self.memory_enabled: bool = True` to `SegmentAgent.__init__` (agent.py ~line 700).
- Persist in `to_dict()` / `from_dict()` roundtrip.
- When `memory_enabled=False`:
  - `_retrieve_decision_memories()` returns `[]` (agent.py ~line 1721).
  - `_build_memory_context()` returns a zero-valued context (no state_delta, no chronic_threat_bias, no protected_anchor_bias).
  - `memory_bias` and `pattern_bias` in the decision loop are forced to `0.0` (agent.py ~lines 2339-2343).
  - Episode recording still happens (so the agent accumulates history), but retrieval and bias are suppressed.
- When `memory_enabled=True`: no behavior change from current code.

### 2. Expose `memory_enabled` on SegmentRuntime

- `SegmentRuntime.load_or_create(..., memory_enabled=True)` passes through to agent.
- `SegmentRuntime.export_snapshot()` includes `memory_enabled` in the snapshot.

### 3. Add `memory_bias` and `pattern_bias` to `last_memory_context`

Currently these values are computed in the decision loop but not stored. Add them to `last_memory_context` after the decision loop completes, alongside `memory_enabled`.

### 4. Ablation contrast test

Write `tests/test_m48_ablation_contrast.py`:

- **Same-seed divergence test**: Run two 20-cycle rollouts with identical seed, one `memory_enabled=True`, one `memory_enabled=False`. Assert:
  - Decision sequences diverge (not all actions identical).
  - Decision entropy differs (memory should reduce entropy by biasing toward known-safe actions).
  - The `memory_bias` and `pattern_bias` fields in `last_memory_context` are nonzero for enabled, zero for disabled.

- **Valence alignment test**: In the enabled run, when `chronic_threat_bias > 0`, the agent should show higher avoidance (more `hide`/`rest` vs `explore`/`seek_contact`). Compare the action distributions between enabled and disabled runs.

- **state_delta causal test**: Assert that `state_delta` in the enabled run has nonzero entries, while the disabled run has all-zero `state_delta`.

- **Negative control**: Run two `memory_enabled=True` rollouts with the same seed. Assert they produce **identical** decision sequences (proving divergence comes from the flag, not from nondeterminism).

### 5. Demote m47_runtime shared snapshot

- In `m47_audit.py` and `m47_reacceptance.py`, add a comment/field marking the shared runtime snapshot as `diagnostic_only: true`, not acceptance evidence.
- The existing m47 acceptance tests remain as structural-self-consistency checks, but the milestone boundaries doc should note that M4.7 behavioral claims now depend on M4.8 ablation evidence.

### 6. Document the three-layer acceptance model

Update `reports/m4_milestone_boundaries.md` to define:
- **(a) Structural self-consistency**: artifact matches evaluator truth (existing M4.5-M4.7 standard).
- **(b) Behavioral causation**: ablation contrast (M4.8 standard).
- **(c) Phenomenological alignment**: classical memory effects emerge naturally (future M4.10+).

State that M4.5-M4.7 currently satisfy only (a), and M4.8 targets (b).

## Constraints

- Do NOT change any scoring formula, threshold, or multiplier.
- Do NOT modify the memory encoding, promotion, or retrieval logic.
- The `memory_enabled=False` path must be a clean suppression, not a separate code path with its own logic.
- Episode recording must continue even when memory is disabled (the agent still builds history; it just can't use it for decisions).

## Acceptance Criteria

See `prompts/m48_acceptance_criteria.md`.
