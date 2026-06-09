# M20.4 P0-6 — Tie-Breaker Sub-Class Split (P1-Driven)

- **for**: M20.4 owner (third of three P1-driven sub-class
  splits)
- **from**: M18.7.1 P1 (commit 26d2157 +
  `reports/m18_7_1_p1_m20_4_handoff.md`) +
  M20.4 P0-4 (commit 93611ad) +
  M20.4 P0-5 (commit 9e91ad1)
- **date**: 2026-06-09
- **read time**: ~5 min

## TL;DR

1. **M20.4 tie-breaker now weights the
   `addressee_target_match` observable by sub-class.**
   "Addressed to assistant" claims
   (`addressed_to_assistant == True`) engage at **conf > 0.95**
   (P0-6 raised bar). "Not addressed" claims engage at
   the v1 **conf > 0.9** default. Reaction claims are
   unchanged at **conf > 0.7**.
2. **The change is data-driven.** P1 (bqxsmofri, real
   LLM on the held-out fixture) surfaced
   `recall_on_addressed = 0.0` and a high-band
   overconfidence drift at the 0.80-0.90 conf=0.85 bin
   (gap 0.85, the largest single-bin gap on the run).
   The 0.95 bar is calibrated against the actionable
   0.90-1.00 conf=0.95 bin (gap 0.283, the M20.4
   "engaged" signal).
3. **The change is tie-breaker-only.** Producer admit
   (P0-4: 0.7 / 0.4), write-path filter (P0-5: 0.9),
   settler, M20.4.1, and the reaction observable are
   unchanged.
4. **Per-sub-class diagnostic counters** are added
   (additive over the v1 aggregate). The aggregate
   `tie_breaker_engaged_total`,
   `tie_breaker_rejected_total`, and
   `tie_breaker_rejected_by_reason` are preserved for
   back-compat.

## What P0-6 changes (1 paragraph)

`m20_4_attribution.py` adds the frozen constant

```python
M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED: float = 0.95
```

and extends the per-field dispatch helper

```python
def _tie_breaker_min_for(
    kind: str,
    *,
    addressed_to_assistant: bool | None = None,
) -> float:
    if kind.strip().lower() == "addressee":
        if addressed_to_assistant is True:
            return M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED
    return M20_4_TIE_BREAKER_CONFIDENCE_MIN_BY_KIND.get(
        kind.strip().lower(),
        _M20_4_TIE_BREAKER_DEFAULT,
    )
```

`_tie_breaker_engaged` accepts an optional
`addressed_to_assistant` kwarg and passes it through.
`emit_m20_4_tie_breaker_feedback` reads
`addressed_to_assistant` from the hypothesis (only
when `kind == "addressee"`) and passes it through.
The M18.7 state surface is unchanged (M18.7 is the
upstream source; M20.4 is the consumer).

## P1 numbers driving P0-6

| metric | bqxsmofri value | P0-6 read |
|---|---|---|
| `recall_on_addressed` | **0.0** | LLM misses all 4 addressed cases; bad flips on false-positive "addressed" admits are the M20.4 risk. |
| `precision_on_not_addressed` | **1.0** | "Not addressed" claims are reliable; the v1 0.9 bar is appropriate. |
| addressee conf=0.85 bin | gap 0.85 | Below the 0.95 P0-6 bar; the tie-breaker now rejects "addressed" emits at this conf. P0-4 already rejects (0.85 > 0.7 producer admit admits this bin; the 0.9 write filter at P0-5 also drops it; the 0.95 P0-6 tie-breaker is the third layer of defense). |
| addressee conf=0.95 bin | gap 0.283 | Above the 0.95 P0-6 bar; tie-breaker engages. This is the bqxsmofri M20.4 actionable signal. |
| addressee conf=0.50-0.60 bin | gap 0.6 | Below the 0.95 P0-6 bar; rejected. P0-4 (0.7 producer admit) and P0-5 (0.9 write filter) also drop this bin. |
| addressee conf=0.20-0.30 bin | acc 1.0 | Below the 0.95 P0-6 bar; rejected. The LLM expressed low confidence and was right. P0-4 rejects at the producer (0.3 < 0.7). |

**The 0.95 value is calibrated against the bqxsmofri
0.90-1.00 conf=0.95 bin.** At conf=0.95, the 0.50-0.60
band (P0-4 producer + P0-5 write + P0-6 tie-breaker all
reject), the 0.80-0.90 band (P0-5 + P0-6 reject), and
the 0.95 boundary itself (P0-6 rejects at the strict
`>` inequality) are all dropped. The 0.96-1.00 band
is the only zone where the P0-6 tie-breaker engages.

The 0.95 value is **directional, not definitive**: a
future M18.7.1 stability rerun with n=30+ on the
addressee axis may surface a tighter value.

## What P0-6 does NOT change

| layer | frozen? | reason |
|---|---|---|
| Producer admit rule (P0-4: 0.7 / 0.4) | unchanged | P0-4 raised the producer bar for "addressed" admits. P0-6 raises the tie-breaker bar. P0-4 + P0-6 are two boundaries of the same drift signature at two different layers (producer vs. tie-breaker). |
| Write-path filter (P0-5: 0.9) | unchanged | P0-5 raised the write-path bar for "addressed" admits. P0-6 raises the tie-breaker bar. P0-5 + P0-6 are two boundaries at two different layers (write path vs. tie-breaker). |
| Settler (M20.1 LLM judge) | unchanged | The judge's job is "is the hypothesis consistent with the inbound turn", not "is the hypothesis correct". The judge's accuracy is independent of the tie-breaker. P0-6 drops engagement, not admits. |
| "Not addressed" sub-class tie-breaker | unchanged | P1 precision 1.0 on the bqxsmofri fixture. P0-6 explicitly does NOT raise the bar for this sub-class; the v1 0.9 default is preserved. |
| `reaction_attribution_match` tie-breaker | unchanged | The reaction axis is independent. P0-6 only touches `kind == "addressee"` with `addressed_to_assistant=True`. The v1 0.7 reaction threshold is preserved. |
| Tie-breaker for unknown kinds | unchanged | The v1 0.85 default is preserved. |
| Strict-inequality style (`>`) | unchanged | P0-6 uses `> 0.95` (strict), consistent with the M20.4 v1 / P0-3 / P0-4 / P0-5 style. The 0.95 boundary itself rejects under strict `>`. |
| M20.4.1 (same-turn gate) | unchanged | M20.4.1 is in P3 kill-switch (`M20_4_1_OVERRIDE_ENABLED = False`, audit-only). P1 surfaces a separate threshold decision for M20.4.1 (`conf > 0.85` is the M20.4.1 trigger; P1 data shows 2 wrong at conf > 0.85). M20.4.1 is a separate M20.4.x milestone; P0-6 does not touch it. |
| M18.7 state surface | unchanged | M18.7 is the upstream source. M20.4 reads the existing `state["m18_7_attribution_hypotheses"]` surface. P0-6 only changes how M20.4 interprets the `confidence` + `addressed_to_assistant` fields at tie-breaker engagement time. |

## Diagnostic surface (P0-6 additive)

P0-6 preserves the v1 aggregate counters for back-compat
and adds the following per-sub-class counters:

| key | meaning |
|---|---|
| `tie_breaker_engaged_addressee_directed_total` | # of tie-breaker engagements with `kind == "addressee"` AND `addressed_to_assistant == True` |
| `tie_breaker_engaged_addressee_not_directed_total` | # of tie-breaker engagements with `kind == "addressee"` AND `addressed_to_assistant == False` |
| `tie_breaker_rejected_confidence_low_addressee_directed_total` | # of tie-breaker rejections on confidence (P0-6 raise) with `kind == "addressee"` AND `addressed_to_assistant == True` |
| `tie_breaker_rejected_confidence_low_addressee_not_directed_total` | # of tie-breaker rejections on confidence (v1 default) with `kind == "addressee"` AND `addressed_to_assistant == False` |

The v1 `tie_breaker_engaged_total`,
`tie_breaker_rejected_total`, and
`tie_breaker_rejected_by_reason` are unchanged in
semantics. Existing diagnostic readers do not need
to migrate.

The P0-6 per-sub-class counters are the M20.4 owner-facing
signal for the tie-breaker sub-class split. If the
`..._addressee_directed_total` counters saturate (e.g.,
the LLM emits very few `addressed_to_assistant=True`
cases that pass the 0.95 bar), the bar may need to be
revised or the upstream P0-4 producer admit may need
to be tightened.

## Test surface (P0-6)

11 new tests in `tests/test_m20_4_tie_breaker.py`:

| test | pins |
|---|---|
| `test_p0_6_tie_breaker_addressee_directed_constant_is_frozen` | The 0.95 / 0.9 split is the v1 → P0-6 contract. |
| `test_p0_6_tie_breaker_min_for_addressee_subclass_dispatch` | `_tie_breaker_min_for` dispatches by sub-class for `kind == "addressee"`. |
| `test_p0_6_tie_breaker_addressee_directed_engages_at_0_96` | The "addressed" sub-class engages at conf=0.96 (just above 0.95). |
| `test_p0_6_tie_breaker_addressee_directed_rejects_at_0_91` | The "addressed" sub-class rejects at conf=0.91 (would have engaged under v1 0.9). |
| `test_p0_6_tie_breaker_addressee_directed_rejects_at_0_95_boundary` | The 0.95 boundary is strict `>`; at conf=0.95 the tie-breaker rejects. |
| `test_p0_6_tie_breaker_addressee_not_directed_keeps_v1_threshold` | The "not addressed" sub-class keeps the v1 0.9 default; at conf=0.91 it engages. |
| `test_p0_6_tie_breaker_reaction_observable_unaffected_by_subclass_flag` | The reaction kind ignores the `addressed_to_assistant` flag (the flag is only meaningful for the addressee kind). |
| `test_p0_6_tie_breaker_diagnostics_addressee_directed_engaged_counter` | The new `tie_breaker_engaged_addressee_directed_total` counter is bumped on engagement. |
| `test_p0_6_tie_breaker_diagnostics_addressee_not_directed_engaged_counter` | The new `tie_breaker_engaged_addressee_not_directed_total` counter is bumped on engagement. |
| `test_p0_6_tie_breaker_diagnostics_addressee_directed_low_confidence_reject_counter` | The new `tie_breaker_rejected_confidence_low_addressee_directed_total` counter is bumped on confidence rejection. |
| `test_p0_6_tie_breaker_mixed_batch_subclass_split` | Mixed batch: 1 addressed-engage + 1 addressed-reject + 1 not-addressed-engage. |

Plus 1 updated test fixture (`_commitment()` in
`tests/test_m20_4_tie_breaker.py`): the default
`confidence` for `addressee_target_match` was raised
from 0.91 to 0.96 so the v1-pinned "engages" tests
continue to engage (the existing tests still use
`addressed_to_assistant=True` as the default).

Plus 1 updated test
(`test_tie_breaker_per_field_addressee_threshold_engages_at_0_91`):
this test now uses `addressed_to_assistant=False` to
pin the "not addressed" sub-class engagement at
conf=0.91 (the v1 0.9 default still engages). The
"addressed" sub-class rejection at conf=0.91 is pinned
by the new
`test_p0_6_tie_breaker_addressee_directed_rejects_at_0_91`.

Cross-M18.7.1 regression: 342/342 pass (was 331/331
in P0-5; +11 new P0-6 tests).

## M20.4-relevant reads from the P1 handoff doc

The P1 handoff doc
(`reports/m18_7_1_p1_m20_4_handoff.md`) lists 4
M20.4-relevant reads. P0-6 addresses read 1 directly
and is consistent with reads 2, 3, 4:

| read | P0-6 stance |
|---|---|
| 1. Addressee precision/recall split is the structural story | **Addressed**: sub-class tie-breaker split. "Not addressed" claims (precision 1.0) engage at the v1 0.9; "addressed" claims (recall 0.0) engage at the P0-6 0.95. |
| 2. Reaction joint "all decidable" accuracy is the M20.4-honest signal | Out of scope: this is about the reaction axis, not addressee. P0-6 does not change the reaction tie-breaker; the v1 0.7 is preserved. |
| 3. High-band overconfidence drift is the actionable signal | **Consistent**: P0-6's 0.95 threshold sits strictly above the 0.80-0.90 conf=0.85 bin (the bqxsmofri high-band overconfidence drift zone) and at the 0.90-1.00 conf=0.95 bin (gap 0.283, M20.4 actionable). The 0.95 bar is the third of three defense layers (P0-4 producer + P0-5 write + P0-6 tie-breaker). |
| 4. Low-conf emits are unstable across runs | **Consistent**: P0-6 rejects the 0.50-0.60 band and the 0.80-0.90 band (P1's bqxsmofri drift zone). The 0.20-0.30 band (very low conf, LLM was right) is also rejected at the 0.95 bar — but the LLM's low-confidence "right" cases were already filtered by P0-4's 0.7 producer admit. |

## P0-4 + P0-5 + P0-6 — combined M20.4 sub-class surface

The three P0-x sub-class splits are three layers of
defense for the bqxsmofri `recall_on_addressed = 0.0`
drift signature:

| layer | constant | threshold | role |
|---|---|---|---|
| Producer admit | `M20_4_PRODUCER_ADMIT_CONFIDENCE_MIN_ADDRESSEE_DIRECTED` | 0.7 | Filter at admit time: rejects the 0.50-0.60 band before the settler runs. |
| Write-path filter | `M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE` | 0.9 (strict `<=`) | Filter at write time: rejects the 0.80-0.90 band before the persistent graph is written. |
| Tie-breaker | `M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED` | 0.95 (strict `>`) | Filter at engagement time: rejects the 0.95 boundary itself; only the 0.96-1.00 band engages. |

The "not addressed" sub-class (P1 precision 1.0) and
the reaction observable (independent axis) are
unchanged at all three layers.

## Open questions for M20.4 owner

1. **Is the 0.95 tie-breaker threshold right?** P1
   data is a single fresh replay (bqxsmofri). The
   bqxsmofri fixture has 0 `tp_addressed` cases; the
   0.95 bar is calibrated against the 0.90-1.00
   conf=0.95 bin (gap 0.283), which has 1 wrong out
   of 3 emits. A 5-run stability rerun on a fixture
   with at least 4-6 `addressed_to_assistant=True`
   emits at conf > 0.95 would tighten the
   recommendation.

2. **Should the M20.4.1 same-turn gate also split by
   sub-class?** P0-6 is tie-breaker-only. M20.4.1
   is in P3 kill-switch (`M20_4_1_OVERRIDE_ENABLED =
   False`, audit-only) and engages on `conf > 0.85`.
   A future M20.4.x milestone may raise M20.4.1's
   `conf > 0.85` to `conf > 0.95` for the
   "addressed" sub-class. This is a separate
   decision; P0-6 does not do it.

3. **What about the 5-run v2 stability on P0-4 +
   P0-5 + P0-6 producer + write + tie-breaker?**
   P0-4 + P0-5 + P0-6 ship with the 0.7 / 0.9 / 0.95
   values from bqxsmofri alone. A 5-run stability
   rerun with the P0-4 + P0-5 + P0-6 producer + write
   + tie-breaker is a future P0-7 task. If the 5-run
   data shows the 0.95 threshold is too lax (e.g.,
   the 0.95 bin's 1/3 wrong rate recurs in later
   runs), P0-6 may need to be raised to 0.97 or 0.98.

4. **Should the 5-run stability also surface
   per-sub-class admit / write / engage rates?**
   The current 5-run v2 stability report
   (`reports/m18_7_1_v2_stability_summary.md`) has
   aggregate accuracy / ECE / Brier. P0-7 should
   extend the report with per-sub-class
   (`addressed_to_assistant=True` vs.
   `addressed_to_assistant=False`) admit rates,
   write-skip counts, and tie-breaker engagement
   rates.

## CAVEAT (frozen, binding)

**M20.4 surfaces candidates. M20.4 owner sets the
threshold.** The 0.95 value is a P1-data-driven
recommendation, not a binding threshold. M20.4
owner can revise
`M20_4_TIE_BREAKER_CONFIDENCE_MIN_ADDRESSEE_DIRECTED`
up or down with a documented decision. The change
log lives in M20.4's own change log (P0-6 in
`reports/m20_4_p0_6_tie_breaker_subclass.md` and
follow-ups in subsequent M20.4 docs).

## Pointers

- P1 implementation report:
  `reports/m18_7_1_p1_precision_recall_split.md`
- P1 M20.4 handoff: `reports/m18_7_1_p1_m20_4_handoff.md`
- P0-4 implementation report:
  `reports/m20_4_p0_4_subclass_admit.md` (P0-6 sibling)
- P0-5 implementation report:
  `reports/m20_4_p0_5_write_path_filter.md` (P0-6 sibling)
- P0-6 implementation: `segmentum/dialogue/runtime/m20_4_attribution.py`
  (commit e81b447; ~80 lines added: constant, helper
  extension, tie-breaker sub-class dispatch, diagnostic
  counters)
- P0-6 tests: `tests/test_m20_4_tie_breaker.py` (+11 new tests,
  1 updated fixture, 1 updated test)
- Memory: `project_m18_7_1_p1_landed.md` (P1 status,
  references the P0-4 / P0-5 / P0-6 follow-up chain)
