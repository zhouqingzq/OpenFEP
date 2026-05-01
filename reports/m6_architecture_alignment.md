# M6.0 Architecture Alignment

## Milestone Position

M6 extends the existing M5 dialogue runtime as an incremental closure of the FEP
loop. It does not replace `SegmentAgent`, `AttentionBottleneck`,
`GlobalWorkspace`, `MetaCognitiveLayer`, `DecisionDiagnostics`, or
`FEPPromptCapsule`. M6.0 freezes the local architecture map so M6.1-M6.7 can add
event, state, conscious-artifact, and meta-control adapters without building a
parallel cognition runtime.

## Current Dialogue Chain

The current M5 dialogue chain is implemented in
`segmentum/dialogue/conversation_loop.py` and uses these repository anchors:

1. `DialogueObserver.observe` in `segmentum/dialogue/observer.py` converts the
   partner turn and transcript context into `DialogueObservation.channels`.
2. `SegmentAgent.decision_cycle_from_dict` in `segmentum/agent.py` runs the
   existing predictive/FEP decision cycle over observed channels.
3. `build_fep_prompt_capsule` in `segmentum/dialogue/fep_prompt.py` derives the
   prompt-safe `FEPPromptCapsule` from `DecisionDiagnostics` and observation.
4. `ResponseGenerator.generate` in `segmentum/dialogue/generator.py` renders the
   selected dialogue action, either rule-based or through the runtime LLM wrapper.
5. `classify_dialogue_outcome` in `segmentum/dialogue/outcome.py` labels the
   previous agent turn from the next observation without claiming causation.
6. `SegmentAgent.integrate_outcome` in `segmentum/agent.py` persists the actual
   outcome into the existing learning, memory, metacognitive, and trace surfaces.

Supporting existing modules:

- `segmentum/attention.py`: `AttentionBottleneck`, `AttentionTrace`, and
  attention allocation.
- `segmentum/workspace.py`: `GlobalWorkspace`, `GlobalWorkspaceState`, conscious
  report payloads, and workspace action bias.
- `segmentum/metacognitive.py`: `MetaCognitiveLayer` and self-consistency review.
- `segmentum/types.py`: `DecisionDiagnostics` and ranked intervention scores.
- `segmentum/dialogue/runtime/chat.py`: runtime prompt injection around
  `run_conversation`.
- `segmentum/dialogue/runtime/prompts.py`: `PromptBuilder` system prompt assembly.

## M6 Concept To Local Module Map

| M6 concept | Local anchor | Role | Producer | Consumer | Trace surface | Test owner |
| --- | --- | --- | --- | --- | --- | --- |
| Cognitive event infrastructure | New adapter planned under `segmentum/dialogue/` or `segmentum/tracing.py` | Records bounded turn events, not cognition itself | `run_conversation` around observe, decide, generate, outcome | state derivation, conscious trace writer, audit tests | `conscious_trace.jsonl` plus M6 test fixtures | M6.1 event tests |
| Derived cognitive state | New `CognitiveStateMVP` adapter over existing outputs | Read model for current turn state; not a policy truth source | event reducer plus `DecisionDiagnostics` and observation | meta-control guidance, prompt capsule conditioning, conscious projection | `TurnTrace.cognitive_state` | M6.2 state tests |
| Lightweight affective state | Section inside `CognitiveStateMVP` | Maintenance signal derived from observation, body state, prior outcome, and social tension | observation channels, existing body/homeostasis state, previous outcome | prompt tone guidance and meta-control hints | `TurnTrace.cognitive_state.affective` | M6.2 state tests |
| Cognitive path view | New read-only `CognitivePath` adapter over `DecisionDiagnostics.ranked_options` | Explains ranked choices without re-ranking | `DecisionDiagnostics` | conscious projection, diagnostics UI, meta-control | `TurnTrace.cognitive_path` | M6.2 path tests |
| Meta-control guidance | New `MetaControlGuidance` adapter | First affects prompt conditioning, not core policy selection | cognitive state, path view, metacognitive review | `PromptBuilder` and `FEPPromptCapsule` extension fields | `TurnTrace.meta_control` | M6.3 tests |
| Prompt guidance | Existing `FEPPromptCapsule` in `segmentum/dialogue/fep_prompt.py` plus bounded M6 extension | Prompt-safe summary of decision constraints | `build_fep_prompt_capsule` plus meta-control adapter | `ResponseGenerator.generate` and runtime `PromptBuilder` | generation diagnostics `fep_prompt_capsule` | M6.4 prompt tests |
| Human-readable conscious context | New `Conscious.md` projection | Session-scoped readable current context; not a decision source | conscious artifact writer from `TurnTrace` | humans, audit review, optional prompt-safe excerpt after filtering | `artifacts/conscious/.../Conscious.md` | M6.5 artifact tests |
| Long-term self-conscious prior | New `Self-consciousness.md` projection | Persona-scoped slow self-prior; not memory, policy, or diagnostics truth | slow consolidation gate from session summaries | prompt-safe identity continuity excerpt after persona isolation checks | `artifacts/conscious/.../Self-consciousness.md` | M6.6 consolidation tests |
| Persisted memory | Existing memory modules, especially `segmentum/memory.py`, `segmentum/memory_store.py`, and `SegmentAgent.integrate_outcome` | Source of episodic/semantic persistence | `SegmentAgent.integrate_outcome` and existing memory consolidation | decision cycle, retrieval, validation | existing memory snapshots and reports | existing memory regression tests |
| Outcome feedback | Existing `classify_dialogue_outcome` and `SegmentAgent.integrate_outcome` | Prior-turn outcome correlation and learning signal | `run_conversation` using next observation | memory, policy diagnostics, M6 event/state reducers | episode outcome semantics and `TurnTrace.outcome` | M6.7 outcome tests |

There are no orphan M6 concepts. Every new object is an adapter or projection
whose producer and consumer are named above.

## First Integration Points

Events:

- Add event emission around the existing `run_conversation` chain, after
  `DialogueObserver.observe`, after `SegmentAgent.decision_cycle_from_dict`,
  after `build_fep_prompt_capsule`, after `ResponseGenerator.generate`, after
  `classify_dialogue_outcome`, and after `SegmentAgent.integrate_outcome`.
- Events must be consumed by the state reducer or artifact writer. A
  `CognitiveEventBus` cannot be write-only logging.

State:

- Derive `CognitiveStateMVP` from the current observation, previous outcome,
  `DecisionDiagnostics`, attention trace, workspace state, metacognitive review,
  and generation diagnostics.
- `CognitiveStateMVP` is derived state only. It is not prompt text, policy truth,
  or a replacement for `DecisionDiagnostics`.

Affective maintenance:

- Add `AffectiveStateMVP` as a bounded section inside cognitive state.
- It is a lightweight maintenance signal derived from observation, body state,
  prior outcome, and social tension. It is not a full emotion simulator,
  personality replacement, or unbounded mood narrative.

Path view:

- Implement `CognitivePath` as a read-only adapter over
  `DecisionDiagnostics.ranked_options`.
- It may summarize choice order, margins, and dominant components, but must not
  re-rank or mutate the policy output.

Meta-control:

- Add `MetaControlGuidance` after diagnostics and state derivation.
- Its first effect is prompt conditioning through bounded guidance fields; it
  must not override core policy selection in M6.0-M6.3.

Prompt capsule:

- Extend the existing `FEPPromptCapsule` path rather than creating a second
  prompt capsule type.
- Prompt capsule constraints forbid raw event streams, full memory dumps, full
  diagnostics, sensitive dumps, and unfiltered hidden-intent speculation.

Conscious artifacts:

- Write human-readable projections from `TurnTrace`, not from ad hoc prompt text.
- `Conscious.md` is a current/session context projection and not a decision
  source.
- `Self-consciousness.md` is persona-scoped, slow-updated, and isolated from
  other personas.

Trace:

- `TurnTrace` is the machine-readable turn join point. It should include bounded
  event ids, state snapshot, path view, meta-control guidance, prompt capsule
  summary, generation diagnostics summary, outcome label, artifact paths, and
  schema version.

Outcome feedback:

- Keep previous outcome ownership in the existing `run_conversation` /
  `classify_dialogue_outcome` / `SegmentAgent.integrate_outcome` path.
- Do not compute previous outcome in multiple owners without a single source of
  truth.
- Treat outcome labels as correlation evidence unless validated by ablation.

## Boundary Distinctions

Event infrastructure:

- Transport and ordering layer for small JSON-safe event records.
- It does not decide, attend, remember, infer, or generate.

Derived cognitive state:

- A reducer output built from events and existing diagnostics.
- It is not prompt text, policy truth, memory truth, or a second
  `DecisionDiagnostics`.

Lightweight affective state:

- A bounded maintenance section inside cognitive state.
- It summarizes pressure, regulation need, and social tension from current
  signals.

Policy diagnostics:

- Existing `DecisionDiagnostics` remains the policy explanation source.
- M6 adapters can read diagnostics but must not duplicate its authority.

Prompt guidance:

- Prompt-safe bounded fields routed through the existing `FEPPromptCapsule` and
  `PromptBuilder`.
- It is not raw trace, raw memory, raw diagnostics, or private speculation.

Human-readable conscious context:

- `Conscious.md` and turn summaries are readable projections for review and
  continuity.
- They do not own policy, memory, diagnostics, or event truth.

Long-term self-conscious prior:

- `Self-consciousness.md` is a slow-moving persona-scoped self-prior.
- It is cross-session within one persona and isolated from other personas.

Persisted memory:

- Existing memory stores remain the source of episodic and semantic persistence.
- Conscious artifacts may cite memory-derived summaries but are not memory
  stores.

## Conscious Artifact Layout

M6 conscious artifacts use this MVP layout:

```text
artifacts/conscious/
  personas/
    {persona_id}/
      profile.json
      Self-consciousness.md
      sessions/
        {session_id}/
          Conscious.md
          conscious_trace.jsonl
          turn_summaries/
            turn_0001.md
```

File roles:

- `profile.json` stores stable `persona_id`, display name, schema version, and
  artifact metadata.
- `Self-consciousness.md` is a long-term, persona-scoped self-prior. It is
  slow-moving and cross-session.
- `Conscious.md` is a session-scoped current conscious context. It may be
  rewritten or rolled forward each turn.
- `conscious_trace.jsonl` is machine-readable evidence. Markdown files are
  human-readable projections.
- `turn_summaries/turn_0001.md` stores one bounded human-readable projection per
  turn.

Identity rules:

- `persona_id` must be stable and must not depend only on display name.
- `session_id` must be scoped under a persona.
- No persona may read or write another persona's `Self-consciousness.md`.
- Prompt assembly must resolve `persona_id` before reading any conscious artifact.
- Display names may change; directory identity must not.

## Non-goals And Boundary Constraints

- Do not build a parallel cognition runtime.
- Do not replace `SegmentAgent.decision_cycle`.
- Do not replace `SegmentAgent.decision_cycle_from_dict`.
- Do not replace `AttentionBottleneck` or `GlobalWorkspace`.
- Do not replace `MetaCognitiveLayer`.
- Do not replace `DecisionDiagnostics`.
- Do not create a second prompt capsule type.
- Do not replace `FEPPromptCapsule`.
- Do not insert raw event streams, full memory dumps, full diagnostics, or
  unfiltered hidden-intent speculation into prompts.
- Do not let `CognitiveEventBus` become write-only logging; every event type
  needs at least one planned consumer.
- Do not let `CognitiveStateMVP` become a second truth source duplicating
  `DecisionDiagnostics`.
- Do not let `AffectiveStateMVP` become a full emotion simulator, personality
  replacement, or unbounded mood narrative.
- Do not let `Conscious.md` or `Self-consciousness.md` become the source of truth
  for policy, memory, or diagnostics.
- Do not let multiple personas share one `Self-consciousness.md`.
- Do not let `Self-consciousness.md` update every turn without
  slow-consolidation gates.
- Do not treat outcome correlation as causal proof without ablation.
- Do not compute previous outcome in multiple owners without a single source of
  truth.

## M6.0 Acceptance Lock

- M6 augments the M5 dialogue runtime rather than replacing it.
- The event layer is infrastructure, not cognition itself.
- `CognitiveStateMVP` is derived state, not prompt text or policy truth.
- `AffectiveStateMVP` is a lightweight maintenance signal.
- `CognitivePath` is an adapter over existing `ranked_options`.
- `Conscious.md` is a human-readable projection, not a decision source.
- `Self-consciousness.md` is persona-scoped, slow-updated, and isolated from
  other personas.
- Meta-control first affects prompt conditioning, not core policy selection.
- Prompt capsule constraints forbid raw event streams and sensitive dumps.
