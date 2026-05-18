# Claude / Agent Notes

## Active Product Path: Path B Only

All new dialogue cognition work targets **Path B** — the MVP UI chat stack:

```text
segmentum/dialogue/runtime/app.py
-> ChatInterface
-> MVPDialogueRuntime (mvp_loop.py)
-> MVPStateStore + persona/session JSON
```

**Path B is the only path to extend.** Do not plan features around Path A, do not
unify the two stacks, and do not carry forward historical experiments as
requirements.

### Path A — frozen / out of scope

Path A is the research stack built around `conversation_loop.py`, `SegmentAgent`,
`CognitiveLoop`, `SelfThoughtProducer`, and related M6–M10 wiring. Treat it as
**legacy experiment code**, not the product direction.

When touching shared modules, avoid expanding Path A integration. Do not add new
call sites, milestones, or acceptance criteria that require Path A ↔ Path B
bridges.

### M10 — do not use for new work

`M10.0` (Self-Initiated Exploration Agenda, `SelfThoughtEvent`,
`MetaObserver` / `SelfThoughtProducer`, gap-driven self-thought thresholds) was an
early experiment. **It is not part of the current architecture.**

- Do not design features that depend on M10 events, producers, or gap triggers.
- Do not propose “bridging M10 into MVP” or “wiring SelfThought into mvp_loop”.
- Do not use M10 acceptance reports or Path A conversation-loop hooks as templates
for Path B behavior.

Idle initiative, boredom, reward proxy, conscious planning, and memory dynamics
on Path B replace that experiment track (see M13.x and MVP `run_turn`).

---

## Required Architecture Context

Before changing the post-M8 dialogue cognition stack on Path B, read:

- `reports/mvp_architecture_contract_2026-05-04.md` (ownership and evidence rules)
- `prompts/README.md`
- `prompts/M8.9_Work_Prompt.md`
- `prompts/M13.0_Work_Prompt.md` (Path B orchestration and M13 bridge)

The MVP architecture contract remains the source of truth for memory evidence,
state ownership, and prompt-safe generation on Path B.

---

## Core Contract (Path B)

Do not bypass the MVP main turn path:

```text
external input / bounded internal tick (e.g. proactive surrogate)
-> per-turn bus messages (audit + TemporalContext / binding / M13 events)
-> conscious loop (plan: task, recall, expectations, temporal_assessment)
-> memory_dynamics + recall + evidence judgment
-> M13 drive evaluation (behavioral pull, boredom, reward proxy, initiative policy)
-> thinking + reply generation
-> reply validation + safety
-> optional post_reply_observer (same-turn followup only)
-> explicit patches (memory, self_cognition, m13_drive_state, open_items, …)
-> conversation_log + diagnostics
```

`MVPDialogueRuntime` is the orchestrator for this path. Submodule code may
**calculate or propose**; durable changes must use a **named owner**, `source`,
`reason`, `confidence`, and traceability (patches, intents, audit events)—not
prompt text alone.

`CognitiveEventBus` / `CognitiveLoop` on Path B are **not** the live orchestrator.
The lightweight per-turn `bus` list inside `mvp_loop.py` is an audit/trace
surface, not a background scheduler. True idle initiative still needs an explicit
UI/runtime tick (see M13.3).

---

## Dialogue Observation Channels (6 channels)

The six dialogue observation channels (`semantic_content`, `topic_novelty`,
`emotional_tone`, `conflict_tension`, `relationship_depth`, `hidden_intent`) are
**not** a full dialogue-understanding layer. They compress one turn into a small
set of scalars so that:

- FEP / decision scoring can **bias** among the bounded reply-action set (on Path
A this was explicit; on Path B, analogous guidance flows through
`control_guidance`, M13, and memory dynamics rather than a second personality),
- legacy research code could feed gap / exploration thresholds.

**Do not treat channels as sufficient semantics for planning, memory recall, or
user modeling.** Path B dialogue meaning lives in the **conscious loop**,
**memory_dynamics**, **retrieval/evidence judgment**, and **M11/M12** modules—not
in channel floats alone.

Do not add new features that depend on M10-style gap triggers driven only by
channel thresholds.

---

## Current Roadmap (Path B–relevant)

```text
M8:    Anchored Memory Contract
M8.9:  MVP Architecture Contract Hardening
M9.0:  Memory Dynamics Integration (mvp_loop)
M11.0: User Generative Model And Value Memory Dynamics
M12.0: User Identity Continuity Model
M12.1: Mechanistic Personality Model And Plain-Language Report
M13.0: MVP-Local Behavioral Pull
M13.1: Boredom / Exploration Bias
M13.2: Affective Reward Proxy And Settlement
M13.3: UI-Level Bounded Initiative
```

Deferred or non-active for new implementation unless explicitly revived:

```text
M6–M7:  CognitiveLoop / meta-control as primary orchestrator (Path A research)
M10.0:  Self-Initiated Exploration Agenda (superseded by M13 + conscious idle work)
```

M11 “Conscious Projection Runtime” remains deferred; M11.0 owns user modeling.
M12 is split into M12.0 (identity continuity) and M12.1 (mechanistic personality).

---

## Guardrails

- Do not let prompt text become the only cognition layer.
- Do not insert raw events, raw diagnostics, full memory dumps, full prompt text,
or full conscious markdown into user-visible replies.
- Do not write anchored dialogue facts directly to memory when the work belongs to
M8.9 or later; prefer `MemoryWriteIntent` and audited patches on Path B.
- Do not promote unrecalled long-term memory, hypotheses, or unsupported details
into facts. Use unknown or uncertain stance.
- Do not treat `Conscious.md` or `Self-consciousness.md` as policy, memory, or
diagnostics truth.
- Do not extend Path A (`conversation_loop.py`) or M10 (`exploration.py`
self-thought producers) for new Hu Tao / MVP chat behavior.
- Do not unify Path A and Path B in a single milestone.

---

## Prompt Files

Milestone work prompts belong under `prompts/` and should use:

```text
M{major}.{minor}_Work_Prompt.md
```

New implementation guidance should not be added as root-level
`M*_Implementation_Prompt.md` files.