# M20.4 P0-4 — Sub-Class Admit Threshold (P1-Driven)

- **for**: M20.4 producer (P0 milestone for the P1 handoff)
- **from**: M18.7.1 P1 (commit 26d2157 +
  `reports/m18_7_1_p1_precision_recall_split.md`)
- **date**: 2026-06-09
- **read time**: ~5 min

## TL;DR

1. **M20.4 producer now weights the `addressee_target_match`
   admit rule by sub-class.** "Addressed to assistant" claims
   (`addressed_to_assistant == True`) admit at **conf >= 0.7**
   (P0-4 raised bar). "Not addressed" claims admit at the
   v1 **conf >= 0.4** default. Reaction claims are unchanged.
2. **The change is data-driven.** P1 (bqxsmofri, real LLM
   on the held-out fixture) surfaced
   `precision_on_not_addressed = 1.0` and
   `recall_on_addressed = 0.0` on the 12-turn fixture. The
   LLM is structurally asymmetric; the v1 uniform 0.4 admit
   threshold admitted too many unreliable "addressed" claims.
3. **The change is producer-only.** Settler, write path,
   tie-breaker, and M20.4.1 are unchanged. P0-4 is the
   minimum-scope M20.4 change that consumes the P1 signal.
4. **Per-sub-class diagnostic counters** are added
   (additive over the v1 aggregate). The aggregate
   `producer_reject_low_confidence_total` and
   `producer_admit_total` are preserved for back-compat.

## What P0-4 changes (1 paragraph)

`m20_4_attribution.py` adds the frozen constant

```python
M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED: float = 0.7
```

and the helper

```python
def _admit_threshold_for(
    *, kind: str, addressed_to_assistant: bool | None = None
) -> float:
    if kind == "addressee" and addressed_to_assistant is True:
        return M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED
    return M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN  # 0.4 v1 default
```

`produce_m20_4_attribution_commitments` reads the LLM's
`addressed_to_assistant` boolean from the M18.7 entry,
dispatches to the per-sub-class threshold, and bumps
the new per-sub-class diagnostic counters. The M18.7
state surface is unchanged (M18.7 is the upstream
source; M20.4 is the consumer).

## P1 numbers driving P0-4

| metric | bqxsmofri value | P0-4 read |
|---|---|---|
| `precision_on_not_addressed` | **1.0** | LLM is perfect on "not addressed" claims; the v1 0.4 admit threshold is appropriate. |
| `recall_on_addressed` | **0.0** | LLM misses all 4 "addressed" cases; the v1 0.4 admit threshold admits too many unreliable claims. |
| addressee conf=0.85 bin | gap 0.85 | High-band overconfidence drift starting point. |
| addressee conf=0.95 bin | gap 0.283 | Above the 0.7 P0-4 threshold; admitted. |
| addressee conf=0.50-0.60 bin | gap 0.6 | Below the 0.7 P0-4 threshold; rejected. |

**The 0.7 value is calibrated against the bqxsmofri drift
signature.** At conf=0.7, the 0.50-0.60 band is rejected
(the overconfidence drift starting point), while the
0.80-0.90 and 0.90-1.00 bands (the M20.4 actionable
signal) still admit. The 0.7 value is **directional**,
not definitive: a future M18.7.1 stability rerun with
n=30+ on the reaction axis may surface a tighter value.

## What P0-4 does NOT change

| layer | frozen? | reason |
|---|---|---|
| Settler (LLM judge) | unchanged | The LLM judge's job is "is the hypothesis consistent with the inbound turn", not "is the hypothesis correct". The judge's accuracy is independent of the producer's admit rule. P0-4 filters out the unreliable admits *before* the settler processes them. |
| Tie-breaker (per-field 0.9 / 0.7) | unchanged | The tie-breaker flips M18.5's decision on `microadjust + confirmed + high + structural empty + conf > 0.9 (addressee)`. P0-4 is upstream of the settler; the tie-breaker sees fewer false "addressed" admits, but the engagement rule itself is unchanged. |
| M20.4.1 (same-turn gate) | unchanged | M20.4.1 is in P3 kill-switch (`M20_4_1_OVERRIDE_ENABLED = False`, audit-only). P1 surfaces a separate threshold decision for M20.4.1 (`conf > 0.85` is the M20.4.1 trigger; P1 data shows 2 wrong at conf > 0.85). M20.4.1 is a separate M20.4.x milestone; P0-4 does not touch it. |
| Write path (`write_addressee_graph_microadjust`) | unchanged | The write path writes to the addressee graph on `microadjust + confirmed`. P0-4 reduces the number of "addressed" admits that reach the graph, but the write rule itself is unchanged. |
| Reaction admit rule | unchanged | The reaction joint-axis asymmetry is in the LLM's emit decision (50% no-emit rate, P1), not in the admit calibration. The 0.4 admit threshold admits all LLM-emit reaction hypotheses; the no-emit cases are filtered at the M18.7 layer (no admit because `participant_id = ""`). |
| M18.7 state surface | unchanged | M18.7 is the upstream source. M20.4 reads the existing `state["m18_7_attribution_hypotheses"]` surface; P0-4 only changes how M20.4 interprets the `addressed_to_assistant` boolean at admit time. |

## Diagnostic surface (P0-4 additive)

P0-4 preserves the v1 aggregate counters for back-compat
and adds the following per-sub-class counters:

| key | meaning |
|---|---|
| `producer_admit_addressee_directed_total` | # of addressee admits with `addressed_to_assistant == True` |
| `producer_admit_addressee_not_directed_total` | # of addressee admits with `addressed_to_assistant == False` |
| `producer_admit_reaction_total` | # of reaction admits (the v1 reaction bucket; the field is the same name, so existing diagnostic readers keep working) |
| `producer_reject_low_confidence_addressee_directed_total` | # of addressee rejects with `addressed_to_assistant == True` (P0-4 raise) |
| `producer_reject_low_confidence_addressee_not_directed_total` | # of addressee rejects with `addressed_to_assistant == False` (v1 default) |
| `producer_reject_low_confidence_reaction_total` | # of reaction rejects (the v1 reaction bucket) |

The v1 `producer_admit_total` and
`producer_reject_low_confidence_total` are unchanged
in semantics. Existing diagnostic readers do not need
to migrate.

## Test surface (P0-4)

7 new tests in `tests/test_m20_4_producer.py`:

| test | pins |
|---|---|
| `test_p0_4_subclass_threshold_constant_is_frozen` | The 0.7 / 0.4 split is the v1 → P0-4 contract. |
| `test_p0_4_addressee_directed_admits_at_threshold_above_0_7` | Boundary: conf=0.71 admits (just above 0.7). |
| `test_p0_4_addressee_directed_rejects_at_0_4_under_new_rule` | v1 → P0-4 behavior change at the 0.4 point. |
| `test_p0_4_addressee_not_directed_admits_at_v1_threshold` | "Not addressed" sub-class keeps the 0.4 default. |
| `test_p0_4_reaction_admit_rule_unchanged` | Reaction admit is unchanged. |
| `test_p0_4_mixed_batch_subclass_split` | Mixed surface with both sub-classes + reaction. |
| `test_p0_4_admit_threshold_helper_for_kind_and_subclass` | `_admit_threshold_for` is the single source of truth. |

Plus 1 updated test (the existing
`test_producer_filters_hypotheses_by_threshold` was
updated to use `addressed_to_assistant = False` for
the addressee entries, since the v1 default `True`
no longer applies at conf=0.4).

Cross-M18.7.1 regression: 319/319 pass (was 312/312
in P1; +7 new P0-4 tests).

## M20.4-relevant reads from the P1 handoff doc

The P1 handoff doc (`reports/m18_7_1_p1_m20_4_handoff.md`)
lists 4 M20.4-relevant reads. P0-4 addresses read 1
directly and is consistent with reads 2-4:

| read | P0-4 stance |
|---|---|
| 1. Addressee precision/recall is the structural story | **Addressed**: sub-class admit threshold. "Not addressed" claims (precision 1.0) admit at the v1 0.4; "addressed" claims (recall 0.0) admit at the P0-4 0.7. |
| 2. Reaction joint "all decidable" accuracy is the M20.4-honest signal | Out of scope: this is about the reaction axis, not addressee. P0-4 does not change the reaction admit rule; a future M20.4 milestone may add a reaction sub-axis admit if P1's joint stability requires it. |
| 3. High-band overconfidence drift is the actionable signal | **Consistent**: P0-4's 0.7 threshold sits below the 0.80-0.90 and 0.90-1.00 high bands. The high bands are still admitted; the overconfidence drift starting point (0.50-0.60) is rejected. |
| 4. Low-conf emits are unstable across runs | **Consistent**: P0-4 rejects the 0.50-0.60 band (unstable across bxg45ar4h / bqxsmofri). The 0.20-0.30 band (very low conf) was already rejected by the v1 0.4 threshold. P0-4 expands the rejection to the 0.50-0.60 band for the "addressed" sub-class. |

## Open questions for M20.4 owner

1. **Is the 0.7 threshold right?** P1 data is a single
   fresh replay (bqxsmofri). The 5-run v2 stability
   report has 5 by_pid replays, but the addressee
   acc/ECE spread is wide (0.25-0.50 acc; P4 is the
   right milestone for tightening the addressee
   calibration). P0-4 ships 0.7 as a directional
   raise; future M18.7.1 stability reruns may
   recommend a different value.
2. **Should the write path also filter "addressed"
   claims?** P0-4 is producer-only. A future M20.4
   milestone may add a write-path filter (e.g.,
   don't write "addressed" rows to the
   `addressee_graph` at conf < 0.9). This is a
   separate decision; P0-4 does not do it.
3. **Should the tie-breaker also split by sub-class?**
   P0-4 is producer-only. The tie-breaker engages on
   `conf > 0.9 (addressee)` and flips M18.5's
   `no_reply` / `clarify_addressee` to
   `reply_to_current_speaker`. With P0-4, the
   tie-breaker sees fewer false "addressed" admits,
   but the engagement rule is unchanged. A future
   M20.4 milestone may raise the tie-breaker for
   "addressed" claims (e.g., to 0.95) to prevent
   bad flips.
4. **What about the 5-run v2 stability on the new
   threshold?** P0-4 ships with the 0.7 value from
   bqxsmofri alone. A 5-run stability rerun with
   the P0-4 producer is a future P2 task. If the
   5-run data shows the 0.7 threshold is too lax
   (e.g., the 0.50-0.60 band recurs in later
   runs), P0-4 may need to be raised to 0.8 or 0.85.

## CAVEAT (frozen, binding)

**M20.4 surfaces candidates. M20.4 owner sets the
threshold.** The 0.7 value is a P1-data-driven
recommendation, not a binding threshold. M20.4
owner can revise `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED`
up or down with a documented decision. The change
log lives in M20.4's own change log (P0-4 in
`reports/m20_4_p0_4_subclass_admit.md` and follow-ups
in subsequent M20.4 docs).

## Pointers

- P1 implementation report:
  `reports/m18_7_1_p1_precision_recall_split.md`
- P1 M20.4 handoff: `reports/m18_7_1_p1_m20_4_handoff.md`
- P0-4 implementation: `segmentum/dialogue/runtime/m20_4_attribution.py`
  (commit pending; ~80 lines added: constant, helper,
  producer changes, diagnostic counters)
- P0-4 tests: `tests/test_m20_4_producer.py` (+7 new tests,
  1 updated test)
- Memory: `project_m18_7_1_p1_landed.md` (P1 status,
  references this P0-4 follow-up)
