# M20.4 P0-5 — Write-Path Addressee-Directed Filter (P1-Driven)

- **for**: M20.4 owner (write-path safety net for the P1 handoff)
- **from**: M20.4 P0-4 (commit 93611ad) + P1 (commit 26d2157)
- **date**: 2026-06-09
- **read time**: ~5 min

## TL;DR

1. **M20.4 write path now gates "addressed" rows on
   confidence.** Rows with
   `addressed_to_assistant == True` and `confidence <= 0.9`
   are silently dropped (no graph write, no audit event)
   and bump a new diagnostic counter
   `write_path_skip_addressee_directed_low_confidence_total`.
   The v1 admit path is preserved for `confidence > 0.9`.
2. **The change is data-driven.** P1 surfaced
   `recall_on_addressed = 0.0` on the bqxsmofri fresh
   verify, with high-band overconfidence drift at the
   0.50-0.60 conf bin (gap 0.6) and the 0.80-0.90 conf=0.85
   bin (gap 0.85 — the largest single-bin gap on the run).
   P0-4 (0.7 producer admit) and P0-5 (0.9 write filter)
   are calibrated against this signature.
3. **The change is additive.** Producer admit, settler,
   tie-breaker, and the "not addressed" sub-class
   (P1 precision 1.0) are unchanged. The reaction
   observable is unchanged. The v1 aggregate
   `producer_admit_total` / `producer_reject_low_confidence_total`
   are preserved.
4. **The 0.9 threshold is strict `>` for admit and
   strict `<=` for skip.** This matches the M20.4 v1
   tie-breaker style and makes the 0.9 boundary
   exclusively the P0-5 test surface (the v1 tests
   that previously pinned 0.9 now default to 0.95
   in the `_commitment()` fixture).

## What P0-5 changes (1 paragraph)

`m20_4_attribution.py` adds the frozen constant

```python
M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE: float = 0.9
```

and the reason code

```python
REASON_GRAPH_SKIP_ADDRESSEE_DIRECTED_LOW_CONFIDENCE: str = (
    "m20_4_addressee_graph_skip_addressee_directed_low_confidence"
)
```

and the pure helper

```python
def _should_skip_addressee_directed_write(
    *, confidence: float
) -> bool:
    return (
        not isinstance(confidence, (int, float))
        or bool(confidence != confidence)  # NaN
        or float(confidence)
        <= M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE
    )
```

`write_addressee_graph_microadjust` now gates on
`addressed_to_assistant` (in addition to the existing
`commitment.observable == "addressee_target_match"`
branch). When the sub-class is "addressed" and the
confidence is at or below 0.9, the function returns
None (no graph write, no audit event) and bumps the
new diagnostic counter.

## P1 numbers driving P0-5

| metric | bqxsmofri value | P0-5 read |
|---|---|---|
| `recall_on_addressed` | **0.0** | LLM misses ALL 4 addressed cases; the bqxsmofri fixture has 0 tp_addressed and 0 fp_not_addressed. |
| addressee conf=0.50-0.60 bin | gap 0.6 | Below the 0.9 P0-5 threshold; write skipped. P0-4 (0.7 producer admit) already rejects this band. |
| addressee conf=0.80-0.90 bin | gap 0.85 | Below the 0.9 P0-5 threshold; write skipped. P0-4 admits this band (0.85 > 0.7). This is the bqxsmofri actionable zone — P0-5 is the safety net. |
| addressee conf=0.90-1.00 bin | gap 0.283 | Above the 0.9 P0-5 threshold; write proceeds. P0-4 also admits (0.95 > 0.7). |
| addressee conf=0.20-0.30 bin | acc 1.0 | Below the 0.9 P0-5 threshold; write skipped. P0-4 (0.7) admits this band (0.3 < 0.7, rejected by P0-4, so no producer admit anyway). |

**The 0.9 value is calibrated against the bqxsmofri
0.80-0.90 drift zone.** At conf=0.9, the 0.50-0.60
band (P0-4 already rejects) and the 0.80-0.90 band
(the P0-5 target zone) are filtered. The 0.90-1.00
band (gap 0.283, M20.4 actionable) is admitted. The
0.9 value is **directional, not definitive**: a
future M18.7.1 stability rerun with n=30+ on the
addressee axis may surface a tighter value.

## What P0-5 does NOT change

| layer | frozen? | reason |
|---|---|---|
| Producer admit rule (P0-4: 0.7 / 0.4) | unchanged | P0-4 raised the producer bar for "addressed" admits. P0-5 raises the write bar. The two thresholds are at different boundaries of the same drift signature. |
| Settler (M20.1 LLM judge) | unchanged | The judge's job is "is the hypothesis consistent with the inbound turn", not "is the hypothesis correct". The judge's accuracy is independent of the write filter. P0-5 drops writes, not admits. |
| Tie-breaker (per-field 0.9 / 0.7) | unchanged | The tie-breaker flips M18.5's decision on `microadjust + confirmed + high + structural empty + conf > 0.9 (addressee)`. P0-5 is at the write boundary; the tie-breaker sees fewer false "addressed" admits from the producer, but the engagement rule itself is unchanged. The tie-breaker's 0.9 boundary and P0-5's 0.9 boundary are the same value but at different layers. |
| "Not addressed" sub-class admit | unchanged | P1 precision 1.0 on the bqxsmofri fixture. P0-5 explicitly does NOT filter this sub-class; the v1 0.4 producer admit and the v1 write path are preserved. |
| `reaction_attribution_match` write | unchanged | The reaction axis is independent. P0-5 only touches `observable == "addressee_target_match" AND addressed_to_assistant == True`. |
| M18.7 state surface | unchanged | M18.7 is the upstream source. M20.4 reads the existing `state["m18_7_attribution_hypotheses"]` surface. P0-5 only changes how M20.4 interprets the `confidence` field at write time. |
| M18.7.1 calibration | unchanged | M18.7.1 v2 + P0 stability data is the input; P0-5 is the write-path consumer. M18.7.1 P0 surfaces `recall_on_addressed = 0.0`; P0-5 is the corresponding write-path response. |

## Diagnostic surface (P0-5 additive)

P0-5 preserves the v1 aggregate counters for back-compat
and adds the following per-sub-class counter:

| key | meaning |
|---|---|
| `write_path_skip_addressee_directed_low_confidence_total` | # of writes skipped at the P0-5 filter (additive over the v1 write path) |

The v1 `producer_admit_total` and
`producer_reject_low_confidence_total` are unchanged
in semantics. The P0-4 per-sub-class
`producer_admit_addressee_directed_total` and
`producer_reject_low_confidence_addressee_directed_total`
are unchanged. Existing diagnostic readers do not
need to migrate.

The P0-5 counter is the M20.4 owner-facing signal
for the addressee_graph safety net. If the counter
saturates (e.g., > 50% of admitted "addressed"
hypotheses are dropped at the write), the 0.9
threshold may need revision or the upstream
producer (P0-4) may need to be tightened further.

## Test surface (P0-5)

12 new tests in `tests/test_m20_4_write_path.py`:

| test | pins |
|---|---|
| `test_p0_5_skip_threshold_constant_is_frozen` | The 0.9 boundary is the v1 → P0-5 contract. |
| `test_p0_5_skip_helper_returns_true_at_or_below_threshold` | The skip helper is strict `<=`. |
| `test_p0_5_skip_helper_returns_false_strictly_above_threshold` | The skip helper returns False strictly above 0.9. |
| `test_p0_5_skip_helper_handles_invalid_inputs` | NaN / non-numeric are treated as "skip" (conservative). |
| `test_p0_5_addressee_directed_write_skipped_at_0_9_boundary` | At conf=0.9, the write returns None and the counter is bumped. |
| `test_p0_5_addressee_directed_write_skipped_below_0_9` | At conf=0.5, the write is skipped. |
| `test_p0_5_addressee_directed_write_proceeds_strictly_above_0_9` | At conf=0.95, the v1 admit path is preserved. |
| `test_p0_5_reaction_observable_unaffected_by_filter` | The filter does NOT touch the reaction observable. |
| `test_p0_5_addressee_not_directed_unaffected_by_filter` | The filter does NOT touch the "not addressed" sub-class. |
| `test_p0_5_skip_counter_accumulates_across_calls` | The counter accumulates. |
| `test_p0_5_skip_reason_code_exists` | The reason code is string-stable. |
| `test_p0_5_mixed_batch_skips_directed_low_confidence_writes_only` | Mixed batch: skip / admit / admit-by-subclass. |

Plus 1 updated test fixture (`_commitment()` in
`tests/test_m20_4_write_path.py`): the default
`hypothesis_confidence` was raised from 0.9 to 0.95
so the v1-pinned tests (which use `addressed_to_assistant=True`)
continue to exercise the v1 admit path. The 0.9
boundary is now exclusively pinned by the new P0-5
tests. This is the same pattern as the P0-4 update
to `test_producer_filters_hypotheses_by_threshold`.

Cross-M18.7.1 regression: 331/331 pass (was 319/319
in P0-4; +12 new P0-5 tests).

## M20.4-relevant reads from the P1 handoff doc

The P1 handoff doc
(`reports/m18_7_1_p1_m20_4_handoff.md`) lists 4
M20.4-relevant reads. P0-5 addresses read 3 directly
and is consistent with reads 1, 2, 4:

| read | P0-5 stance |
|---|---|
| 1. Addressee precision/recall split is the structural story | **Consistent**: P0-5 explicitly filters the "addressed" sub-class (P1 recall 0.0) and preserves the "not addressed" sub-class (P1 precision 1.0). |
| 2. Reaction joint "all decidable" accuracy is the M20.4-honest signal | Out of scope: this is about the reaction axis, not addressee. P0-5 does not change the reaction write path. |
| 3. High-band overconfidence drift is the actionable signal | **Addressed**: P0-5's 0.9 threshold sits at the boundary of the 0.80-0.90 conf=0.85 bin (gap 0.85, the bqxsmofri high-band overconfidence drift zone) and the 0.90-1.00 conf=0.95 bin (gap 0.283, M20.4 actionable). The 0.80-0.90 band is filtered; the 0.90-1.00 band is admitted. |
| 4. Low-conf emits are unstable across runs | **Consistent**: P0-5 rejects the 0.80-0.90 band (which includes the 0.85 bin's overconfidence drift starting point). The 0.50-0.60 and 0.20-0.30 bands are filtered by both P0-4 (producer) and P0-5 (write). |

## Open questions for M20.4 owner

1. **Is the 0.9 write threshold right?** P1 data is a
   single fresh replay (bqxsmofri). The bqxsmofri
   fixture has 0 tp_addressed and 0 fp_not_addressed
   (LLM never emitted `addressed_to_assistant=True`
   in the 12-turn fixture); P0-5 is a safety net for
   future fixtures / LLM behavior changes where
   "addressed" emits do appear. A 5-run stability
   rerun on a fixture with at least 4-6
   `addressed_to_assistant=True` emits would tighten
   the 0.9 recommendation.

2. **Should the write path also filter the "not
   addressed" sub-class?** P0-5 is asymmetric: only
   the "addressed" sub-class is filtered. P1
   `precision_on_not_addressed = 1.0` is the
   structural reason. A future M20.4 milestone may
   add a similar filter for "not addressed" at a
   different boundary (e.g., 0.95) if the precision
   number degrades in future replays.

3. **Should the tie-breaker also gate on
   `addressed_to_assistant`?** P0-5 is write-only.
   The tie-breaker engages on `conf > 0.9 (addressee)`
   and flips M18.5's `no_reply` / `clarify_addressee`
   to `reply_to_current_speaker`. The current
   tie-breaker is symmetric (same conf for both
   sub-classes). A future M20.4 milestone may
   raise the tie-breaker for the "addressed"
   sub-class to 0.95 to prevent bad flips on
   false-positive "addressed" admits.

4. **What about the 5-run v2 stability on P0-5?**
   P0-5 ships with the 0.9 value from bqxsmofri
   alone. A 5-run stability rerun with the P0-5
   write path is a future P2 task. If the 5-run
   data shows the 0.9 threshold is too lax (e.g.,
   the 0.80-0.90 band recurs in later runs),
   P0-5 may need to be raised to 0.95.

## CAVEAT (frozen, binding)

**M20.4 surfaces candidates. M20.4 owner sets the
threshold.** The 0.9 value is a P1-data-driven
recommendation, not a binding threshold. M20.4
owner can revise
`M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE`
up or down with a documented decision. The change
log lives in M20.4's own change log (P0-5 in
`reports/m20_4_p0_5_write_path_filter.md` and
follow-ups in subsequent M20.4 docs).

## Pointers

- P1 implementation report:
  `reports/m18_7_1_p1_precision_recall_split.md`
- P1 M20.4 handoff: `reports/m18_7_1_p1_m20_4_handoff.md`
- P0-4 implementation report:
  `reports/m20_4_p0_4_subclass_admit.md` (P0-5 sibling)
- P0-5 implementation: `segmentum/dialogue/runtime/m20_4_attribution.py`
  (commit 9e91ad1; ~140 lines added: constant, reason
  code, helper, write-path filter, diagnostic counter)
- P0-5 tests: `tests/test_m20_4_write_path.py` (+12 new tests,
  1 updated test fixture)
- Memory: `project_m18_7_1_p1_landed.md` (P1 status,
  references this P0-5 follow-up)
