# M4.6 Strict Audit

Generated on 2026-04-08 in `E:\workspace\segments`.

## Verdict

- M4.6 under a strict acceptance standard: `BLOCK`
- M4.6 core implementation status: `PARTIALLY REAL, PARTIALLY TOY`
- M4.7 readiness: `NOT READY`

The main reason for the `BLOCK` verdict is not that every M4.6 code path is fake. Several real production paths exist and do run. The blocker is that the current acceptance story is self-certifying: the report is assembled from synthetic probes in [`segmentum/m46_acceptance_data.py`](/E:/workspace/segments/segmentum/m46_acceptance_data.py), the evaluator in [`segmentum/m46_audit.py`](/E:/workspace/segments/segmentum/m46_audit.py) trusts that synthetic payload, and the acceptance tests in [`tests/test_m46_acceptance.py`](/E:/workspace/segments/tests/test_m46_acceptance.py) mostly verify that the payload writer and evaluator agree with each other.

## Real Test Runs

- M4.6 targeted tests: `py -m pytest tests/test_m46_memory_core.py tests/test_m46_acceptance.py -q`
  Result: `12 passed in 0.74s`
- M4.1-M4.5 regression suite: `py -m pytest ... -q` over all `REGRESSION_TARGETS`
  Result: `128 passed, 1 skipped, 29 subtests passed in 1759.65s (0:29:19)`

This means the branch is not obviously regressing M4.1-M4.5, but regression pass alone is not enough to certify M4.6 acceptance honesty.

## Gate Matrix

| Gate | Standard requirement | Actual code path | Current test/report evidence | Strict audit judgement |
|---|---|---|---|---|
| G1 `retrieval_multi_cue` | Explicit weighted formula, mood/accessibility effects, dormant filtering, `recall_hypothesis`, score breakdowns | Weights and score breakdowns are real in [`segmentum/memory_retrieval.py:14`](/E:/workspace/segments/segmentum/memory_retrieval.py:14), [`segmentum/memory_retrieval.py:427`](/E:/workspace/segments/segmentum/memory_retrieval.py:427), [`segmentum/memory_retrieval.py:461`](/E:/workspace/segments/segmentum/memory_retrieval.py:461) | M4.6 tests use one hand-authored mentor/lab/reactor fixture set in [`tests/test_m46_memory_core.py:103`](/E:/workspace/segments/tests/test_m46_memory_core.py:103); acceptance report uses the same kind of synthetic fixture in [`segmentum/m46_acceptance_data.py:151`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:151) | Real code exists, but evidence is synthetic and same-source. `NOT INDEPENDENTLY AUDITABLE` |
| G2 `candidate_competition` | Thresholded dominance, low-confidence legal outputs, auditable competitors | Real logic exists in [`segmentum/memory_retrieval.py:288`](/E:/workspace/segments/segmentum/memory_retrieval.py:288) | Only two toy score arrangements are used in [`tests/test_m46_memory_core.py:219`](/E:/workspace/segments/tests/test_m46_memory_core.py:219) and mirrored in [`segmentum/m46_acceptance_data.py:278`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:278) | Core path is real, acceptance evidence is toy and duplicated. `WEAK / NOT INDEPENDENT` |
| G3 `reconstruction_mechanism` | Trigger A/B/C, borrow cap, anchor protection, version/hash change, trace | Triggering and borrow cap are real in [`segmentum/memory_consolidation.py:246`](/E:/workspace/segments/segmentum/memory_consolidation.py:246), [`segmentum/memory_consolidation.py:258`](/E:/workspace/segments/segmentum/memory_consolidation.py:258), [`segmentum/memory_consolidation.py:319`](/E:/workspace/segments/segmentum/memory_consolidation.py:319) | Tests cover A/B/C on handcrafted entries in [`tests/test_m46_memory_core.py:246`](/E:/workspace/segments/tests/test_m46_memory_core.py:246); acceptance probe duplicates the same pattern in [`segmentum/m46_acceptance_data.py:303`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:303) | Implementation is real enough, but evidence is still synthetic. `WEAK ACCEPTANCE EVIDENCE` |
| G4 `reconsolidation` | Numeric before/after updates, procedural protection, conflict handling without overwrite | Numeric updates are real in [`segmentum/memory_consolidation.py:401`](/E:/workspace/segments/segmentum/memory_consolidation.py:401) | Tests only assert update type and a few flags in [`tests/test_m46_memory_core.py:327`](/E:/workspace/segments/tests/test_m46_memory_core.py:327); report checks update type rather than actual numeric deltas in [`segmentum/m46_audit.py:183`](/E:/workspace/segments/segmentum/m46_audit.py:183) | Core path exists, but the acceptance evidence does not satisfy the standard's numeric-comparison requirement. `FAIL AS ACCEPTANCE EVIDENCE` |
| G5 `offline_consolidation_pipeline` | Full four-stage cycle, short-to-mid, skeleton/inference generation, cleanup, per-stage report | Pipeline exists in [`segmentum/memory_consolidation.py:804`](/E:/workspace/segments/segmentum/memory_consolidation.py:804) | Tests only show one synthetic cluster in [`tests/test_m46_memory_core.py:362`](/E:/workspace/segments/tests/test_m46_memory_core.py:362); report accepts any non-empty IDs in [`segmentum/m46_audit.py:196`](/E:/workspace/segments/segmentum/m46_audit.py:196) | Real path exists, but implementation has toy shortcuts: only the first qualifying pattern group is processed (`break`) in [`segmentum/memory_consolidation.py:625`](/E:/workspace/segments/segmentum/memory_consolidation.py:625), and replay is capped to at most 3 items regardless of `batch_size` in [`segmentum/memory_consolidation.py:642`](/E:/workspace/segments/segmentum/memory_consolidation.py:642). `TOY-LIKE / NOT ACCEPTANCE READY` |
| G6 `inference_validation_gate` | Explicit score formula, traceable inputs, pass/fail examples, long upgrade on pass | Formula exists in [`segmentum/memory_consolidation.py:690`](/E:/workspace/segments/segmentum/memory_consolidation.py:690) and can promote `MID -> LONG` in [`segmentum/memory_consolidation.py:721`](/E:/workspace/segments/segmentum/memory_consolidation.py:721) | Tests only check validated vs unvalidated outcomes in [`tests/test_m46_memory_core.py:413`](/E:/workspace/segments/tests/test_m46_memory_core.py:413); implementation stores the final score but not the decomposed inputs required for auditability | Core behavior exists, but input decomposition is not persisted. `PARTIAL IMPLEMENTATION / AUDIT TRACE INSUFFICIENT` |
| G7 `legacy_integration` | Callable bridge, replay batch, M4.1-M4.5 regression pass | Bridge path is real in [`segmentum/memory_store.py:513`](/E:/workspace/segments/segmentum/memory_store.py:513) and [`segmentum/memory.py:803`](/E:/workspace/segments/segmentum/memory.py:803) | Real regression was rerun and passed. However, acceptance tests still fake regression summaries through `_passing_regression_summary()` in [`tests/test_m46_acceptance.py:12`](/E:/workspace/segments/tests/test_m46_acceptance.py:12) | `PASS` for actual bridge behavior and real regression rerun, but acceptance test design is still weak |
| G8 `report_honesty` | Non-empty evidence for all gates and no fake pass | Current honesty check trusts payload completeness and static non-empty fields in [`segmentum/m46_audit.py:238`](/E:/workspace/segments/segmentum/m46_audit.py:238) | `integration` probes are aliases of `boundary` probes in [`segmentum/m46_acceptance_data.py:538`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:538); `failure_injection` is static text in [`segmentum/m46_acceptance_data.py:586`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:586); tests verify self-consistency in [`tests/test_m46_acceptance.py:29`](/E:/workspace/segments/tests/test_m46_acceptance.py:29) | `FAIL`. This is the clearest blocker. The report does not prove honesty; it proves internal agreement among synthetic helpers |

## Highest-Risk Findings

1. Acceptance evidence is synthetic and self-certified.
   The report is produced from `build_m46_acceptance_payload()` in [`segmentum/m46_acceptance_data.py:611`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:611), which itself constructs hand-authored probe scenarios rather than harvesting evidence from live milestone runs.

2. Boundary and integration probes are not independent.
   `INTEGRATION_PROBES` is just a renamed copy of `BOUNDARY_PROBES` in [`segmentum/m46_acceptance_data.py:528`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:528), [`segmentum/m46_acceptance_data.py:538`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:538). That means the "integration" channel adds no new evidence.

3. Acceptance tests fake the regression summary instead of exercising the real regression path.
   `_passing_regression_summary()` in [`tests/test_m46_acceptance.py:12`](/E:/workspace/segments/tests/test_m46_acceptance.py:12) fabricates a successful pytest result. This is useful as a unit-test seam, but it also means the acceptance test suite does not independently prove that regression execution is wired correctly.

4. `report_honesty` only checks that fields are populated, not that evidence is real.
   The honesty gate in [`segmentum/m46_audit.py:238`](/E:/workspace/segments/segmentum/m46_audit.py:238) accepts non-empty failure injection cases and a non-negative ablation gap. The failure injection itself is only a static dictionary, not an executed fault scenario, in [`segmentum/m46_acceptance_data.py:586`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:586).

5. Parts of the offline consolidation implementation are toy-like.
   `extract_patterns()` stops after the first qualifying group in [`segmentum/memory_consolidation.py:625`](/E:/workspace/segments/segmentum/memory_consolidation.py:625), and `constrained_replay()` ignores large `batch_size` values by taking at most 3 source entries in [`segmentum/memory_consolidation.py:642`](/E:/workspace/segments/segmentum/memory_consolidation.py:642). That looks like milestone scaffolding, not a mature general pipeline.

6. G4 and G6 are under-audited even when the code paths are real.
   `reconsolidate()` updates numeric fields in [`segmentum/memory_consolidation.py:420`](/E:/workspace/segments/segmentum/memory_consolidation.py:420), but neither the report nor the tests require before/after numeric evidence. `validate_inference()` computes a real score in [`segmentum/memory_consolidation.py:697`](/E:/workspace/segments/segmentum/memory_consolidation.py:697), but the implementation stores only the final score and status, not the input decomposition demanded by the acceptance criteria.

## Conceptual Or Toy Code Inventory

### Conceptual / synthetic acceptance scaffolding

- [`segmentum/m46_acceptance_data.py:36`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:36) to [`segmentum/m46_acceptance_data.py:525`](/E:/workspace/segments/segmentum/m46_acceptance_data.py:525)
  This entire module is effectively a curated acceptance theater: mentor/lab/reactor examples, fixed donor memories, fixed conflict cases, and fixed bridge seeds.

- [`tests/test_m46_memory_core.py:21`](/E:/workspace/segments/tests/test_m46_memory_core.py:21) to [`tests/test_m46_memory_core.py:468`](/E:/workspace/segments/tests/test_m46_memory_core.py:468)
  Useful as behavior checks, but still heavily synthetic. Nearly all evidence is derived from one tightly controlled vocabulary and a few handcrafted arrangements.

### Toy implementation shortcuts

- [`segmentum/memory_consolidation.py:625`](/E:/workspace/segments/segmentum/memory_consolidation.py:625)
  Pattern extraction breaks after the first qualifying group.

- [`segmentum/memory_consolidation.py:642`](/E:/workspace/segments/segmentum/memory_consolidation.py:642)
  Replay generation takes at most 3 source entries regardless of requested `batch_size`.

- [`segmentum/memory_consolidation.py:648`](/E:/workspace/segments/segmentum/memory_consolidation.py:648)
  Replay hypotheses are generic string templates rather than richer derived artifacts.

- [`segmentum/memory_consolidation.py:373`](/E:/workspace/segments/segmentum/memory_consolidation.py:373)
  Conflict resolution uses fixed numeric penalties by conflict type. This is serviceable milestone logic, but still conceptual and shallow.

### Self-Consistency Tests

- [`tests/test_m46_acceptance.py:29`](/E:/workspace/segments/tests/test_m46_acceptance.py:29)
  Checks that generated artifacts match `_evaluate_acceptance()` rebuilt from the same payload family.

- [`tests/test_m46_acceptance.py:56`](/E:/workspace/segments/tests/test_m46_acceptance.py:56)
  Proves a passing synthetic payload reaches `PASS`, not that the system produced that payload honestly.

- [`tests/test_m46_acceptance.py:73`](/E:/workspace/segments/tests/test_m46_acceptance.py:73)
  "Fake pass detection" only mutates a field inside the synthetic payload and asks the same evaluator to notice.

## Final Decision

- Can M4.6 be accepted now: `NO`
- Is there real work here: `YES`
- Is the current acceptance report trustworthy enough to declare the milestone done: `NO`
- Is it appropriate to move into M4.7 now: `NO`

The fair reading is:

- Retrieval, reconstruction, reconsolidation, inference validation, and bridge entry points are present and runnable.
- The M4.6 acceptance package is still too synthetic, too same-source, and too self-validating.
- Some production paths still contain milestone-grade shortcuts, especially offline consolidation and honesty checking.

## Recommended Exit Criteria Before M4.7

1. Replace synthetic acceptance payloads with evidence harvested from real execution of the production interfaces.
2. Make integration probes genuinely different from boundary probes.
3. Store inference score decomposition, not just the final score.
4. Add explicit before/after numeric assertions for reconsolidation.
5. Remove the first-group-only and max-3 replay shortcuts from offline consolidation, or explicitly downgrade the milestone claim.
6. Keep the real M4.1-M4.5 regression rerun as part of acceptance generation, not only as an external manual step.
