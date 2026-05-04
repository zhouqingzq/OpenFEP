# MVP Architecture Contract: Message Bus, Cognitive Loop, Memory Dynamics

Date: 2026-05-04

## 中文摘要

当前工程总体在正确路线上：已经有 `CognitiveEventBus`、`CognitiveLoop`、`AttentionGate`、`CognitiveStateMVP`、meta-control、prompt 压缩、memory dynamics 和 anchored memory。它不是“大方向错了”，更像是已经完成了 M6/M8 的关键积木，但还缺一份更硬的状态所有权契约。

最重要的架构原则是：

```text
主循环是当前自我认知的唯一编排者。
FEP、记忆、自我思考、meta-control 都可以计算或提出更新。
但当前自我认知和 durable self-state 的变化，必须经由事件、proposal、commit 或明确的状态 owner 落地。
```

当前最符合目标的地方：

1. 事件总线和主循环已经存在，并且测试覆盖了 bus -> attention -> state update。
2. `CognitiveStateMVP` 已经维护 task、memory、gap、affect、meta-control、resource、user、world、candidate path 和 self agenda。
3. prompt 侧已经明确禁止 raw event、raw diagnostics、full memory dump 和 full markdown 进入生成上下文。
4. anchored memory 和 citation audit 已经对“少编细节”有直接帮助。

当前最需要修的地方：

1. `SelfThoughtEvent` 还不是一等事件，自我思考还没有完全落成“高优先级消息源”。
2. 对话事实抽取现在会直接写 anchored memory，绕过了 `ChatEvent -> CognitiveLoop -> MemoryWriteIntent -> MemoryStore` 的理想口径。
3. `CognitiveStateMVP`、`SubjectState`、`SelfModel`、memory、slow learning、sleep consolidation 都在维护不同层次的“我”，但还缺明确的 state ownership 和 patch/commit 审计。
4. retrieval evidence、anchored facts、current input 和 external evidence 还需要统一成一个 generation evidence contract，保证可证伪细节都有来源或被降级成“不知道/不确定”。

下一步优先级：

```text
P0: 冻结 state ownership 表。
P1: 把直接 dialogue fact storage 改成 MemoryWriteIntent。
P2: 增加 SelfThoughtEvent。
P3: 增加 StatePatchProposal / StateCommitEvent。
P4: 建 ResponseEvidenceContract，统一生成侧事实边界。
```

## 1. MVP Goal

This MVP builds a digital persona system for character-dialogue demos. It does not claim to simulate subjective consciousness. It should make the user experience three properties:

1. A persistent self-state, not one-shot prompt roleplay.
2. Selective memory, where high-value memories are retained longer and low-value memories decay or stay unavailable.
3. Self-thought intervention, where uncertainty, contradiction, emotional pressure, identity conflict, or a bad prior response can affect later expression.

The target closed loop is:

```text
external input / internal thought
-> message bus
-> cognitive loop
-> self-cognition and affect update
-> memory retrieval / write intent
-> response constraint
-> reply generation
-> response / outcome feedback back into the bus
```

## 2. Core Architectural Principle

The cognitive loop is the only orchestrator of current self-cognition. Other modules may calculate, propose, retrieve, or warn, but durable self-cognition must be changed through an auditable update path.

Recommended wording:

```text
FEP may compute cognitive updates.
Memory may provide evidence or write suggestions.
Self-thought may publish intervention events.
Meta-control may produce bounded control signals.
But all current self-cognition changes must be represented as events, proposals, or derived state updates that are consumed by the cognitive loop and committed through a named state owner.
```

This avoids a false requirement that the cognitive loop personally computes every update, while still preventing multiple hidden "selves" from mutating the system.

## 3. Module Ownership

| Module | Contract role | Can directly write current self-cognition? |
| --- | --- | --- |
| MessageBus / CognitiveEventBus | Transport typed events with metadata. It does not infer, decide, remember, or generate. | No |
| CognitiveLoop | Consumes bus events, applies attention and budget, invokes state update, and hands compressed constraints downstream. | Orchestrates yes, direct hidden mutation no |
| AttentionGate | Selects which events enter state update under salience, priority, ttl, and budget. | No |
| FEPProcessor / diagnostics | Computes prediction error, expected free energy, uncertainty, and decision signals. | No, produces diagnostics or patch proposals |
| CognitiveStateStore / current CognitiveStateMVP | Owns current-turn derived cognitive state, including gaps, affect, resources, path uncertainty, and self agenda. | Yes, but only via reducer / commit function |
| SubjectState | Long-lived subject continuity read model over identity, commitments, tensions, and priorities. | Yes today, but should be treated as a durable state owner with explicit sync contract |
| SelfModel | Durable persona model: identity narrative, commitments, preferences, threats, repair history. | Yes today, but should be slow-updated and not bypass current-turn loop without trace |
| MemoryDynamics / MemoryStore | Stores, retrieves, decays, consolidates, and detects memory conflict. | No for current self-cognition; yes for memory store |
| SelfThought / MetaObserver | Detects possible inconsistency or problem and publishes high-priority events. | No direct prompt mutation and no direct state mutation |
| MetaControl | Produces bounded guidance for caution, memory reliance, exploration, repair, and style. | No direct self-state mutation |
| ResponseController / PromptBuilder | Turns compressed cognitive constraints into generation guidance. | No |
| ResponseGenerator | Generates reply and diagnostics. | No |

## 4. Required Event Types

Current engineering event names may differ, but the MVP contract requires these logical event classes:

| Logical event | Current equivalent or gap | Contract |
| --- | --- | --- |
| ChatEvent | ObservationEvent | External user input enters the bus as a bounded observation summary. Raw chat may remain in transcript, but state update uses bounded fields. |
| SelfThoughtEvent | Not present as a first-class event | High-priority self-thought intervention must go through the bus. It must not directly edit prompt or state. |
| RetrievalEvidenceEvent | Partially MemoryActivationEvent | Retrieval results should enter as evidence, then cognitive loop decides whether to rely on them. |
| MemoryRecallEvent | Partially MemoryActivationEvent | Recall must carry source ids, confidence, and conflict status. |
| MemoryWriteIntent | Not consistently present | Fact extraction, response facts, outcome learning, and consolidation should emit intent before store write when they affect dialogue memory. |
| CognitiveGapEvent | Derived inside CognitiveStateMVP | Unverified claims, contextual ambiguity, social tension, and blocking gaps should be auditable. |
| StatePatchProposal | Not present as explicit object | FEP, self-thought, memory conflict, and outcome feedback should propose state changes rather than silently mutate. |
| StateCommitEvent | Not present as explicit object | Durable self-state changes should record source, reason, confidence, and target state owner. |
| PromptAssemblyEvent | Present | Prompt must receive compressed guidance, not raw event dumps. |
| GenerationEvent | Present | Generated reply should re-enter trace as an event. |
| OutcomeEvent | Present | Prior response feedback should affect later state, memory, and control. |

## 5. Single Tick Lifecycle

Each dialogue tick should follow one main order:

```text
1. Publish
   Collect ChatEvent, SelfThoughtEvent, RetrievalEvidenceEvent, and prior OutcomeEvent.

2. Consume
   CognitiveLoop consumes turn-scoped events from the bus.

3. Attend
   AttentionGate selects events by priority, salience, ttl, and budget.

4. Evaluate
   FEP and diagnostics compute prediction_error, uncertainty_cost, update_value, decision margin, and conflicts.

5. Update
   CognitiveStateStore commits current-turn SelfState/Affect/Gaps/SelfAgenda or emits StatePatchProposal when durable state is involved.

6. Memory
   MemorySystem retrieves by cue and emits recall evidence. Memory writes go through MemoryWriteIntent unless explicitly classified as low-risk local bookkeeping.

7. Control
   MetaControl derives caution, repair, exploration, memory reliance, and assertiveness guidance.

8. Constrain
   ResponseController builds a compressed response contract: known facts, recalled memories, unverified claims, forbidden assumptions, tone and boundary guidance.

9. Generate
   ResponseGenerator produces the reply.

10. Feedback
   GenerationEvent, OutcomeEvent, memory update result, and citation audit return to trace or bus for the next tick.
```

## 6. Free Energy Engineering Contract

For MVP, do not treat FEP as a vague universal explanation. Implement it as observable fields:

| Field | Meaning | Typical action |
| --- | --- | --- |
| prediction_error | Current input conflicts with prediction, memory, identity, or dialogue goal. | Mark gap, lower confidence, trigger clarification, or repair. |
| uncertainty_cost | Cost of absorbing the input into self-cognition or memory. | Defer, ask, retrieve, or reject as belief. |
| update_value | Expected stability or future usefulness gained by updating. | Commit state patch or memory write if high enough. |
| resource_cost | Attention, prompt, memory, and control cost. | Compress context or drop low-salience events. |

"Reject" means:

```text
received as input
not accepted as current belief
may be stored as unverified / noise / deferred
may be politely surfaced in dialogue
```

It must not mean silent ignorance.

## 7. Memory Contract

Short-term memory:

```text
recent structured turn events
recent high-salience unresolved gaps
recent affect changes
recent uncommitted state proposals
```

Long-term memory:

```text
retrieved only through cues
stores source, confidence, value, salience, decay, last access, and conflict status
does not become prompt truth merely because it exists
```

Memory write policy:

```text
value_score =
  identity continuity value
+ relationship continuity value
+ future prediction usefulness
+ user explicit emphasis
+ affective salience
- privacy / safety risk
- low confidence penalty
- contradiction penalty
```

If long-term memory is not cued or recalled, the generator's factual stance is "unknown". It may say "I am not sure" or "I have an impression", but that is still an unknown stance, not a factual answer.

## 8. Response Contract

Before generation, the system must separate:

```text
Truth layer:
  known from current input
  known from retrieved memory
  known from external evidence
  unverified
  unknown
  forbidden assumptions

Style layer:
  warmth
  relational stance
  caution level
  repair posture
  boundary language
```

The style layer may soften or warm the reply. It may not promote an unknown or unverified detail into a fact.

## 9. Current Engineering Alignment

### Strongly aligned

1. There is a real event bus.
   `segmentum/cognitive_events.py` defines typed cognitive events and a `CognitiveEventBus` with publish, filter, consume, ttl, salience, priority, persona, session, and planned consumers.

2. There is a minimal cognitive loop.
   `segmentum/cognition/cognitive_loop.py` implements `bus -> AttentionGate -> update_cognitive_state`.

3. Current-turn cognitive state is structured.
   `segmentum/cognitive_state.py` has `TaskState`, `MemoryState`, `GapState`, `AffectiveState`, `MetaControlState`, `ResourceState`, `UserState`, `WorldState`, `CandidatePathState`, and `SelfAgenda`.

4. Affect is not only style text.
   `AffectiveState` is derived from observation, prior state, gaps, and outcome, then consumed by meta-control and prompt guidance.

5. Prompt compression is explicitly guarded.
   `segmentum/dialogue/cognitive_guidance.py` blocks raw events, diagnostics, prompt text, full markdown, payload dumps, and sensitive keys from entering prompt guidance.

6. Meta-control is bounded and deterministic.
   `segmentum/meta_control_guidance.py` derives caution, clarification, memory reliance, repair, exploration, warmth, and assertiveness from compact state instead of direct prompt mutation.

7. Memory conflict and overdominance are recognized.
   `segmentum/memory_dynamics.py` detects memory interference and reusable path patterns; `meta_control` can reduce memory reliance.

8. Anchored memory directly supports "less fabrication".
   `segmentum/memory_anchored.py` distinguishes asserted, corroborated, hypothesis, retracted, private, forbidden, and citation-audited memory items.

9. Current M6/M8 test surface is healthy.
   Targeted tests passed: `63 passed` for M6 event/state/path/meta-control/prompt/memory dynamics and M8 anchored memory.

### Partially aligned

1. Event names are implementation-specific rather than concept-specific.
   The bus has `ObservationEvent`, `MemoryActivationEvent`, `DecisionEvent`, `CandidatePathEvent`, `PathSelectionEvent`, `PromptAssemblyEvent`, `GenerationEvent`, and `OutcomeEvent`. This maps well to M6, but it does not yet expose first-class `SelfThoughtEvent`, `RetrievalEvidenceEvent`, `MemoryWriteIntent`, `StatePatchProposal`, or `StateCommitEvent`.

2. CognitiveStateMVP is current-turn derived state, not the whole self.
   This is good for MVP control, but your design language says "self-cognition maintenance". The current code also has `SubjectState` and `SelfModel`, so the architecture needs an explicit relation:

   ```text
   CognitiveStateMVP = current-turn working self-state
   SubjectState = durable subject continuity state
   SelfModel = slow persona / identity / commitment model
   ```

3. FEP computation is mostly inside existing diagnostics and agent decision logic.
   The state reducer consumes diagnostics, margins, prediction errors, and outcomes, but there is no named `FEPProcessor` or explicit `StatePatchProposal` boundary.

4. Meta-control now has more causal power than the early M6 wording.
   It no longer only affects prompt guidance. It also affects memory retrieval gain and path-scoring/control surfaces. This is probably desirable, but the contract should acknowledge it as bounded causal control.

5. Retrieval is cue-based in memory modules, but generation-facing memory can still arrive through multiple surfaces.
   Anchored memory, legacy memory, retrieval diagnostics, and prompt memory context need one common evidence boundary.

### Not aligned or risky

1. Anchored fact extraction directly writes memory before the cognitive loop.
   `run_conversation` calls `_extract_and_store_dialogue_facts` before `ObservationEvent` publication and later again after the agent reply. This helps anti-fabrication, but it bypasses the desired contract of `ChatEvent -> CognitiveLoop -> MemoryWriteIntent -> MemoryStore`.

2. Durable self-state has multiple write owners.
   `SegmentAgent`, `Runtime`, `SelfModel`, `SubjectState`, slow learning, sleep consolidation, memory, and metacognition can all update different parts of self-related state. Many are legitimate, but the ownership contract is not explicit enough.

3. Self-thought is not first-class in the dialogue bus.
   There are metacognitive and meta-control modules, but the architecture does not yet model "self-thought as high-priority event source" in the same bus path as chat.

4. There is no explicit state patch / commit audit.
   `update_cognitive_state` returns a derived immutable state, but durable updates such as `agent.subject_state = ...`, `self_model.update_identity_narrative`, and memory consolidation do not share a common patch record.

5. Retrieval evidence can still be too implicit.
   `MemoryActivationEvent` records hit/count/ids, but a generation-time factual claim should be traceable to current input, anchored memory, recalled memory, or external evidence with confidence and source id.

6. Some source files show mojibake in Chinese comments and regex literals.
   This may be only display/encoding in the current shell, but if real, it threatens Chinese fact extraction reliability and should be checked with an encoding-aware test or file normalization pass.

7. MessageBus could become turn-local rather than system-level.
   `run_conversation` creates `event_bus_for_turn = cognitive_event_bus or CognitiveEventBus()` each turn. Passing a shared bus works, but the default path is per-turn. That is acceptable for MVP if persistent state carries continuity, but self-thought and delayed events need a persistent bus or scheduled event queue.

## 10. Next Work Plan

### Phase 1: Freeze the ownership map

Deliverables:

1. Add this contract to project docs.
2. Add a short `StateOwnership` table in code docs or tests:
   `CognitiveStateMVP`, `SubjectState`, `SelfModel`, `MemoryStore`, `MetaControlSignal`, `PromptCapsule`.
3. Add tests that no prompt builder consumes raw `CognitiveEvent`, raw diagnostics, or full memory dump.

Acceptance:

```text
The team can answer who owns each self-related state and how it is updated.
```

### Phase 2: Introduce MemoryWriteIntent for dialogue facts

Deliverables:

1. Replace direct `_extract_and_store_dialogue_facts` memory writes with:
   `DialogueFactExtractionEvent -> MemoryWriteIntent -> MemoryStore commit`.
2. Keep existing anchored memory behavior, but add event ids and source ids to each item.
3. Emit write result events for created, merged, rejected, retracted, or pruned facts.

Acceptance:

```text
Any dialogue fact in anchored memory can be traced to a user or agent utterance and a write-intent event.
```

### Phase 3: Add SelfThoughtEvent

Deliverables:

1. Define `SelfThoughtEvent` with source, trigger, confidence, priority, salience, ttl, and proposed intervention.
2. Trigger it from high conflict, low margin, memory conflict, citation audit failure, or response/outcome mismatch.
3. Consume it through the same `CognitiveLoop -> AttentionGate -> update_cognitive_state` path.

Acceptance:

```text
Self-thought can change next-turn caution, repair stance, or memory reliance without direct prompt mutation.
```

### Phase 4: Add StatePatchProposal and StateCommitEvent for durable state

Deliverables:

1. Define a minimal patch schema:
   `target_state`, `operation`, `field_path`, `value_summary`, `source_event_id`, `reason`, `confidence`, `ttl`.
2. Use it first for `SubjectState` and selected `SelfModel` updates.
3. Log accepted, rejected, and deferred commits.

Acceptance:

```text
Durable self-cognition changes have source, reason, confidence, and owner.
```

### Phase 5: Unify evidence boundary for generation

Deliverables:

1. Build a `ResponseEvidenceContract`:
   current input facts, anchored facts, recalled memories, external evidence, unverified claims, unknowns, forbidden assumptions.
2. Make `PromptBuilder` and `ResponseGenerator` consume this contract rather than ad hoc memory surfaces.
3. Expand citation audit to check generated claims against this evidence contract.

Acceptance:

```text
Every specific factual detail in a reply is traceable or downgraded to uncertainty.
```

## 11. Recommended MVP Priority

Do not rewrite the whole architecture. The current engineering is already on the correct route. The next highest-value changes are:

1. Make self-thought a first-class bus event.
2. Turn direct dialogue fact storage into memory write intents.
3. Add a state ownership and patch/commit contract for durable self-state.
4. Add one generation evidence contract to unify current input, anchored memory, recalled memory, and unknown stance.

These four changes will align the implementation with the design goal without throwing away the existing M6 and M8 work.
