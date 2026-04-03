# M4.1 Scope Definition

## Goal

Translate 'prior preference structure under finite energy constraints' into a unified
parameter interface, observable interface, and logging interface that provides a common
language for downstream benchmark tasks and open-world validation.

## In Scope (G1-G6)

- G1: Schema completeness — 8-parameter roundtrip, DecisionLogRecord field audit
- G2: Trial variability — seed determinism, parameter sensitivity
- G3: Observability — each parameter maps to >= 2 computable observable metrics
- G4: Intervention sensitivity — parameter changes cause expected metric changes
- G5: Log completeness — invalid record rate <= 0.05, all required fields present
- G6: Stress behavior — resource conservation under energy pressure

## Deferred to Later Milestones

- Cross-generator blind classification (requires independent external generator)
- Parameter falsification (requires robust control metrics)
- Cross-generator parameter recovery (requires external human data)
- Inference engine data-driven audit
- Baseline non-hardcoded audit

These items verify whether latent parameters correspond to real cognitive structure.
They require external human data and independent annotation, not interface-layer testing.
