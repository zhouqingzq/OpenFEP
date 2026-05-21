# Prompt Directory Conventions

This directory is the canonical home for milestone work prompts.

## Naming

Use:

```text
M{major}.{minor}_Work_Prompt.md
```

Examples:

```text
M6.0_Work_Prompt.md
M8.9_Work_Prompt.md
M10.0_Work_Prompt.md
```

For older milestones that historically used compact numbering, this directory now prefers expanded dotted milestone names. New or migrated implementation prompts should use `Work_Prompt`, not `Implementation_Prompt`.

## Current Roadmap Boundary

The post-M8 MVP architecture hardening path is:

```text
M8.9:  MVP Architecture Contract Hardening
M9.0:  Memory Dynamics Integration
M10.0: Self-Initiated Exploration Agenda
M11.0: User Generative Model And Value Memory Dynamics
M12.0: User Identity Continuity Model
M12.1: Mechanistic Personality Model And Plain-Language Report
M12.2: Bidirectional Free-Energy And Second-Order Role Cognition
M13.0: MVP-Local Behavioral Pull
M13.1: MVP-Local Boredom And Exploration Bias
M13.2: MVP-Local Affective Reward Proxy And Tolerance
M13.3: UI-Level Bounded Initiative
M13.4: UI Idle Tick And Introspection Entry Point
M13.6: Memory-Backed Expected Free Energy Bridge
M14.0: Conscious Idle Reflector And Self-Cognition Patch
M14.1: Background Self-Continuity And Persistent Idle Loop
M14.2: Decoupled Environment Loop And Durable Overnight Outreach
M14.3: Traceable Proactive Delivery Alignment And Safety Diagnostics
M14.4: Streamlit Implicit Idle Proactive Delivery
M14.5: Production Proactive Policy Hardening
```

M8.9 is a bridge milestone. It does not replace the original roadmap; it locks state ownership, memory write intent, and generation evidence boundaries before M9-M12 expand the system.

M11.0 was repurposed from the earlier "Conscious Projection Runtime" scope to the user-modeling track. M12 is split into M12.0 (identity continuity, alias claims, impersonation strangeness, entity binding), M12.1 (mechanistic personality model with the eight-step plain-language report), and M12.2 (reciprocal role cognition). M12.1 and M12.2 read upstream state read-only and never write back into M11 or M12.0 ledgers.

M13 is an MVP-local drive layer for the real UI chat path. It intentionally does not wait for full Path A / Path B unification. The sequence is M13.0 behavioral pull, M13.1 boredom and exploration bias, M13.2 affective reward proxy and tolerance, M13.3 bounded UI initiative, and M13.4 the UI idle tick plus introspection entry point.

M13.6 adds a memory-backed expected-free-energy bridge between traceable pending expectations/open items and bounded Path B policy guidance. Silence is not a direct boredom or outreach input: it only matters when it leaves a concrete memory-backed expectation due or unresolved. M14.0 builds on the M13.4 plumbing to add the conscious idle reflector: idle ticks can trigger a JSON-only conscious plan that proposes self-cognition patches, memory consolidation intents, and open-item updates, and may hand off to the existing M13.3 outreach pipeline only when M13.6 allows outreach. M14.1 makes that loop budgeted and persistent. M14.2 separates environment observation, event storage, self-loop preparation, and delivery; UI renders and reports runtime state, it is not the lifecycle owner for overnight self-continuity. Explicit later-message requests must become durable scheduled intents plus outbox entries, never just natural-language follow-up notes. M14.3 aligns proactive delivery around traceable expectations with evidence refs; vague open-item `next_check` tokens alone must not trigger M13.3 proposals, and blocked proactive delivery must emit structured `reason_code` audit events. M14.4 wires Streamlit reruns to `evaluate_proactive_initiative(implicit_idle_request=True)` so an open chat page can deliver a proactive bubble after silence without manual continue; it adds `proactive_policy_profile=streamlit_open_chat` to relax session/cooldown caps for demos only. M14.5 restores conservative defaults for production. Path A and M10 are explicitly out of scope.

## Migration Note

Root-level `M2.*_Implementation_Prompt.md` and `M6.*_Implementation_Prompt.md` files were moved into this directory and renamed to `M*_Work_Prompt.md` so milestone prompts have one canonical location.

Legacy lowercase M3/M4 prompt files were also renamed to expanded dotted names, for example:

```text
m410_work_prompt.md -> M4.10_Work_Prompt.md
m411_acceptance_criteria.md -> M4.11_Acceptance_Criteria.md
```

When two historical files already covered the same milestone, the older compact prompt was kept with `Legacy` in the filename rather than overwritten.
