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
M8.9: MVP Architecture Contract Hardening
M9.0: Memory Dynamics Integration
M10.0: Self-Initiated Exploration Agenda
M11.0: Conscious Projection Runtime
```

M8.9 is a bridge milestone. It does not replace the original roadmap; it locks state ownership, memory write intent, and generation evidence boundaries before M9-M11 expand the system.

## Migration Note

Root-level `M2.*_Implementation_Prompt.md` and `M6.*_Implementation_Prompt.md` files were moved into this directory and renamed to `M*_Work_Prompt.md` so milestone prompts have one canonical location.

Legacy lowercase M3/M4 prompt files were also renamed to expanded dotted names, for example:

```text
m410_work_prompt.md -> M4.10_Work_Prompt.md
m411_acceptance_criteria.md -> M4.11_Acceptance_Criteria.md
```

When two historical files already covered the same milestone, the older compact prompt was kept with `Legacy` in the filename rather than overwritten.
