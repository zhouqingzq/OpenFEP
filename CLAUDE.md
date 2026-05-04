# Claude / Agent Notes

## Required Architecture Context

Before changing the post-M8 dialogue cognition stack, read:

- `reports/mvp_architecture_contract_2026-05-04.md`
- `prompts/README.md`
- `prompts/M8.9_Work_Prompt.md`

The MVP architecture contract is the current source of truth for the bridge from M8 to M9-M11.

## Core Contract

Do not bypass the main cognitive path:

```text
external input / internal thought / retrieval evidence
-> CognitiveEventBus
-> CognitiveLoop
-> attention and budget
-> cognitive state update
-> memory intent / retrieval evidence
-> response evidence contract
-> prompt-safe generation
-> feedback
```

The cognitive loop is the orchestrator of current self-cognition. FEP, memory, self-thought, and meta-control may calculate or propose updates, but durable self-cognition changes must pass through a named owner with source, reason, confidence, and traceability.

## Current Roadmap

```text
M6: Cognitive Loop
M7: Bounded Causal Control
M8: Anchored Memory Contract
M8.9: MVP Architecture Contract Hardening
M9.0: Memory Dynamics Integration
M10.0: Self-Initiated Exploration Agenda
M11.0: Conscious Projection Runtime
```

## Guardrails

- Do not let prompt text become the only cognition layer.
- Do not insert raw events, raw diagnostics, full memory dumps, full prompt text, or full conscious markdown into prompts.
- Do not let self-thought directly edit prompts, memory, or durable self-state; route it through events.
- Do not write anchored dialogue facts directly to memory when the work belongs to M8.9 or later; prefer `MemoryWriteIntent`.
- Do not promote unrecalled long-term memory, hypotheses, or unsupported details into facts. Use unknown or uncertain stance.
- Do not treat `Conscious.md` or `Self-consciousness.md` as policy, memory, or diagnostics truth.

## Prompt Files

Milestone work prompts belong under `prompts/` and should use:

```text
M{major}.{minor}_Work_Prompt.md
```

New implementation guidance should not be added as root-level `M*_Implementation_Prompt.md` files.
