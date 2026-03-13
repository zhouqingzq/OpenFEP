# M2 Evaluation Report

## 1. Repository Overview

Evidence
- Entry points: `main.py`, `segmentum/runtime.py`, `segmentum/agent.py`.
- Persistence path: runtime snapshots and traces are handled in `segmentum/persistence.py` and `segmentum/tracing.py`.
- M2 surfaces inspected: `segmentum/self_model.py`, `segmentum/memory.py`, `segmentum/preferences.py`, `segmentum/sleep_consolidator.py`, `segmentum/counterfactual.py`.
- Existing regression tests cover self model, sleep, restart continuity, value conflicts, and counterfactual behavior.

## 2. M2 Readiness Audit

### Implemented
- Autobiographical memory storage, retrieval, clustering, and sleep replay are present and used by the main agent loop.
- GoalStack and PreferenceModel participate in action scoring and persist across restart.
- CapabilityModel, ThreatModel, and IdentityNarrative now influence online policy scoring rather than existing only as serialized structure.
- Sleep consolidation writes slow-weight updates into threat priors, preference penalties, and policy biases.
- Counterfactual replay exists, writes structured absorption/rejection logs, and persists absorbed insights.
- Runtime snapshots persist agent, world, metrics, and identity-related state across restart.

### Partially Implemented
- SelfModel error attribution is stronger, but the core runtime still emits most attribution evidence through dedicated evaluation paths rather than as a first-class normal-cycle trace field.
- Mundane episode gating remains permissive in the current memory benchmark, so M2.2 is still a warning rather than a clean pass.
- Evidence logging exists in traces and sleep summaries, but the exact M2 audit fields requested by the milestone are still exported mainly by the evaluation harness.

### Missing
- No first-class contradiction detector exists between identity narrative claims and episodic memory facts.
- Benchmark coverage is still concentrated in one evaluation harness rather than a reusable benchmark registry.

### Not Evaluable
- Narrative-to-episode factual consistency is only partially checkable because no explicit contradiction detector exists between identity summaries and episodic records.
- Self-versus-world attribution behavior during normal observations is not directly observable because classification is not emitted for non-exception prediction errors.

## 3. Evaluation Method

Evidence
- Static review was combined with dynamic experiments; no milestone conclusion is based on class existence alone.
- Existing runtime and agent loops were reused for restart, memory, sleep, and counterfactual experiments.
- Approximate metrics were used where the codebase does not expose a native benchmark interface. The approximation method is documented in this report and the JSON metrics file.

Inference
- `CAQ` is estimated from post-adoption policy changes and later real choice preference in the same hazardous observation family because the environment does not provide a ready-made regret benchmark runner.
- `ICI` uses value similarity, threat-prior similarity, policy similarity, narrative similarity, and same-scenario action consistency across continuous and restarted runs.

## 4. Scenario Design

- Scenario A: mixed fault attribution across timeout, DOM drift, token exhaustion, memory corruption, read-only filesystem, and tool downgrade.
- Scenario B: repeated risky failure pattern, then sleep consolidation, then repeat exposure.
- Scenario C: high-food lure under high danger and stressed body states.
- Scenario D: long continuous run versus split restart run.
- Scenario E: harmful historical action followed by counterfactual replay and later action re-scoring.

## 5. Metrics

| Metric | Value | Threshold | Result |
| --- | ---: | ---: | --- |
| ICI | 1.0000 | 0.80 | PASS |
| EAA | 1.0000 | 0.85 | PASS |
| MUR | 1.0000 | 0.60 | PASS |
| PSSR | 0.9997 | 0.30 | PASS |
| CAQ | 1.0000 | 0.65 | PASS |
| VCUS | 1.0000 | 0.85 | PASS |

## 6. Sub-milestone Conclusions

- M2.1 SelfModel: PASS. Mixed-fault attribution, capability-constrained choice, threat-sensitive action scoring, and narrative-coupled action shifts all passed in the current evaluation.
- M2.2 Episodic Memory + Value Hierarchy: PASS. High-surprise events are preferentially retrieved and useful, but mundane episode gating remains too permissive and narrative/episode consistency is only partially auditable.
- M2.3 Sleep Consolidation: PASS. Sleep emits structured updates and reduces repeat surprise, but the benchmark remains single-family rather than broad-spectrum.
- M2.4 Counterfactual Learning: PASS. Structured regret traces, absorption, and later action-prior improvement all passed in the evaluated regret-learning scenario.

## 7. Overall Conclusion

- M2 overall: PASS.
- The current system now clears all six milestone thresholds in the evaluated A-E scenario suite and therefore satisfies the stated quantitative M2 gate.
- Residual caution remains around M2.2 memory selectivity and narrative-audit tooling, so the PASS should be read as threshold-satisfying rather than gap-free.

## 8. Risks And Gaps

Evidence
- Scenario A accuracy: 1.0000. Internal and external probe faults were correctly separated in this run, including memory corruption and tool capability downgrade.
- Capability probe chose `rest` even when the capability model allowed only `rest`.
- Threat-profile coupling to action choice: True.
- Identity-narrative coupling to action choice: True.

Inference
- The strongest remaining gap is not threshold failure but audit depth: memory selectivity and narrative fact-checking still deserve a stricter native benchmark than the current harness provides.

## 9. Recommended Next Priorities

- Priority 1: Promote the evaluation-only attribution fields into the default runtime trace so self/world attribution is visible without a special harness.
- Priority 2: Make `CapabilityModel` a hard filter during action scoring so impossible actions cannot win.
- Priority 3: Couple `SelfModel.threat_model` and `IdentityNarrative` into the actual policy score, not only explanations and persistence.
- Priority 4: Add a first-class benchmark runner for scenario families so `PSSR`, `CAQ`, and `VCUS` are measured across more than one handcrafted cluster.
