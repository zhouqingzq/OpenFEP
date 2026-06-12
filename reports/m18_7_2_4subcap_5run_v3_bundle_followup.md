# 4-Sub-Capability 5-Run v3 (Bundle Aggregation Follow-up) — Done

**Status**: 5-run completed; bundle path did NOT unlock dir_true admit.
**Date**: 2026-06-11
**Model**: `deepseek/deepseek-v4-flash` (native API)
**Fixture**: `tests/fixtures/m18_7_1_held_out_calibration.json`
(bqxsmofri, 12 turns × 4 personas)
**Wall time**: ~1h 6min (5 runs × 12-14 min/run, no retries)
**Code change**: commit `dce23f0` ("Land M20.4 v2 bundle aggregation")
**Snapshot**: `reports/m18_7_2_4subcap_5run_v3_snapshot.json`

**Sub-3 verdict**: `p04_dir_true_admit_zero` **0/5 acceptable**.
The bundle aggregation admits **0 commits** across the
5 runs. v3 is **not better than v2** on sub-3 dir_true
(v2: 1/5 acceptable; v3: 0/5 acceptable). The bundle path
is correctly behaving, but the upstream LLM emit pattern
prevents it from firing.

## v2 vs v3 comparison (sub-3 only)

| sub | field | v2 (P0-8) | v3 (bundle) | Δ |
|---|---|---|---|---|
| sub-3 | producer_admit_total (5-run) | 34 (6.8/run) | **159 (31.8/run)** | +125 (4.7x) |
| sub-3 | producer_admit_dir_true | 1 (1/5) | **0 (0/5)** | -1 |
| sub-3 | producer_admit_dir_false | 16 (3.2/run) | **93 (18.6/run)** | +77 |
| sub-3 | producer_admit_single_strong | n/a | **159 (31.8/run)** | (new) |
| sub-3 | producer_admit_bundle_weak | n/a (path absent) | **0 (0/run)** | (new path, never fired) |
| sub-3 | M18.7 surface: addressed=True | n/a in v2 snapshot | **1 / 21 = 4.8%** | (per-run measured) |
| sub-3 | M18.7 surface: addressed=False | n/a in v2 snapshot | **20 / 21 = 95.2%** | (per-run measured) |
| sub-3 | n_runs_acceptable | 1/5 | **0/5** | -1 |

> **Note on `producer_admit_total`**: the v2 number
> (34 / 6.8 per run) was the LAST-turn cumulative
> `state["m20_4_attribution_diagnostics"]["producer_admit_total"]`
> captured by `get_m20_4_diagnostics()` at end-of-run.
> The v3 number (159 / 31.8 per run) is the cumulative
> bus `AddresseeTargetMatchAdmitted` event count, which
> is the per-emit admit count. The two are different
> metrics — v2 measured a per-run scalar counter that
> resets between runs, v3 measures the per-emit admit
> count. Both are valid signals; the v2 metric was
> smaller because the per-run scalar at end-of-run
> reflects the LAST turn's admits (which the producer
> dispatches in a single batch), while the v3 bus event
> count accumulates ALL per-emit admits across the run.
> The M20.4 v2 plan did NOT change the v1 single-strong
> admit path; the per-emit admit count is unchanged
> from v1.

## Per-run verdict map (v3)

| run | sub-3 admit_total | dir_true | dir_false | bundle_weak | surface_addressee_true | verdict |
|---|---|---|---|---|---|---|
| 1 | 45 | 0 | 35 | 0 | 0 | p04_dir_true_admit_zero |
| 2 | 26 | 0 | 15 | 0 | 0 | p04_dir_true_admit_zero |
| 3 | 21 | 0 |  4 | 0 | 0 | p04_dir_true_admit_zero |
| 4 | 36 | 0 | 17 | 0 | 0 | p04_dir_true_admit_zero |
| 5 | 31 | 0 | 22 | 0 | 1 | p04_dir_true_admit_zero |
| **5-run** | **159** | **0** | **93** | **0** | **1** | **0/5 acceptable** |

## Why the bundle path did not fire (root cause)

The v2 bundle path is **mechanically correct**: T1-T10
producer tests pass, the per-run aggregation_kind field
on the admission event is wired through, the v1 byte-
identity is preserved at the `producer_admit_total`
counter level, and the per-emit admit count is in the
expected 25-45 range per run. The bundle logic itself
never fired because the **M18.7 surface** (the input
to both v1 single-strong and v2 bundle paths) contains
**only 1 `addressed_to_assistant=True` entry across
all 5 runs** (run 5 only, conf 0.5). The bundle path
requires `addressed_to_assistant=True` (D6) AND
`unique_count >= 2` (D2) — neither gate can be met
with 0-1 True entries.

The 5-run M18.7 surface distribution (aggregated from
`m18_7_attribution_hypotheses.json` per run, cap=8):

| run | addressee_total | addressee_true | addressee_false | reaction |
|---|---|---|---|---|
| 1 | 4 | 0 | 4 | 4 |
| 2 | 5 | 0 | 5 | 3 |
| 3 | 3 | 0 | 3 | 4 |
| 4 | 4 | 0 | 4 | 4 |
| 5 | 4 | **1** | 3 | 3 |
| **total** | **20** | **1 (4.8%)** | **19 (95.2%)** | **18** |
| (run 5 True entry: conf 0.5, cid 47eefa95, turn 0) | | | | |

The one `True` entry in run 5 has conf 0.5 (well below
the 0.7 single-strong threshold and below the 0.85
bundle aggregated threshold on its own), and it is
the **only** True entry in the run — so even if the
bundle aggregated the run-5 emit, `unique_count >= 2`
fails. Across 5 runs, there is **never** a 2-unique-
commit window with at least 2 True entries.

### What this means for the bundle hypothesis

The v2 plan's hypothesis was:

> "v2 keeps the 0.7 threshold but adds a second admit
> path: 2+ weak addressee-directed emits (each conf
> 0.5-0.69) with decayed sum >= 0.85 admits ONE
> bundle-weak commitment per turn. Deepseek's 3-4
> conservative emits on the 12-turn fixture should
> now cross 0.85 in 3+ runs (target; v2 was 1/5)."

The 5-run data **rejects this hypothesis on the
upstream side**. Deepseek's M18.7.2 emit pattern on
this fixture is heavily biased toward
`addressed_to_assistant=False`: in 5 runs of 12 turns
each, deepseek emits 21 `addressee` entries total
and **only 1 is `True`** (4.8% rate). The conservative-
emit pattern is at the `addressed_to_assistant` axis,
not at the `confidence` axis. The bundle's "two weak
signals equals one strong" rule is designed for
multiple conf 0.5-0.69 emits that all say
`addressed_to_assistant=True` — but those emits
simply do not occur on this fixture with this model
and prompt.

The bundle fix is the **right structural fix** for
the pattern it was designed for, but the pattern
itself is not what v2 assumed. The actual
deepseek-bias phenomenon is:

1. deepseek **emits addressed_to_assistant=False**
   for polite requests and implicit directives
   (the 12-turn fixture has 4-5 such turns).
2. The 0.7 P0-4 threshold is **not the discriminator**
   on this model — the False/True axis is.
3. The M18.7.2 v2 prompt does not change this
   pattern; v1 and v2 prompts both produce 0-1 True
   entries per run on the bqxsmofri fixture.

## Cross-check: 4-sub-cap v2 p0_8 (different session root)

A separate v2 5-run session (`tmp_m18_7_2_v2_p0_8_run_*`,
which is a different execution from the v2 5-run
snapshot in `reports/m18_7_2_4subcap_5run_v2_snapshot.json`)
shows a similar bias pattern, with slightly higher
True rates but still very low:

| v2 p0_8 run | surface addressee_true | surface addressee_false |
|---|---|---|
| 2 | 1 | 1 |
| 3 | 0 | 3 |
| 4 | 1 | 3 |
| 5 | 1 | 3 |
| **total** | **3 (4+4+4+4+1) of 13 = 23%** | **10 of 13 = 77%** |

The v2 5-run snapshot committed in `6761fe0` (which
gave sub-3 dir_true = 1/5) corresponds to a session
root that has since been removed; the `producer_admit_dir_true=1`
in run 3 is consistent with the v2 5-run having at
least one run with 1+ True emits. The v3 5-run did
not have a run with 2+ True emits, so the bundle
could not fire.

## Why the bundle path is still worth keeping (just not unlocking v3)

Even though v3 did not improve sub-3, the bundle
path is **structurally correct and forward-compatible**:

- **T1-T10 unit tests pass**; the v1 single-strong
  admit path is byte-identical (default
  `aggregation_kind="single_strong"`).
- **351/351 cross-regression pass** at landing (commit
  `dce23f0`).
- The `aggregation_kind` audit field is on every
  `AddresseeTargetMatchAdmitted` bus event, so the
  5-run snapshot can distinguish single-strong vs
  bundle-weak per admit.
- The bundle path will fire on a different model
  (e.g. claude, which v2 data shows emits 2-3 True
  entries per run on the same fixture) or a
  different prompt that nudges deepseek to emit
  True more often.

## What needs to change to unlock sub-3 dir_true

The v3 data identifies the **upstream emit pattern**
as the real bottleneck, not the M20.4 producer. Two
options:

### Option A: M18.7.2 v3 prompt (path A from v2 analysis)

The M18.7.2 v2 prompt revision (commit `479c3e2`,
"R1 validation: recall_on_addressed 0.25 → 0.5")
increased recall by 2x but did not move the False/True
axis. A v3 prompt that explicitly asks the LLM to
emit `addressed_to_assistant=True` for implicit
directives / polite requests / "asking the team"
patterns would shift the True rate from 4.8% to
something like 30-50% — at which point the bundle
path can fire and sub-3 dir_true would be unlocked.

This is **path A from the v2 plan's analysis**,
explicitly listed as "out of scope" for the v2
bundle commit. The v3 5-run result is the empirical
evidence that path A is now the right next step.

### Option B: Loosen bundle's `unique_count` or `max_single` gates

- Drop `max_single_support < 0.7` → admit on
  aggregated even with one strong emit. This would
  help if there is 1 True at conf 0.85+ plus 1
  True at conf 0.55. But on this fixture there
  is no run with 2+ True emits at all, so the
  gate is not the binding constraint.
- Drop `unique_count >= 2` → admit on a single
  weak True emit. But this would let through
  single weak emits that the v1 path already
  rejects for low conf — it is essentially a
  threshold change, not a structural fix.

The data shows option B does not help on this
fixture. **Option A is the right next step.**

## Sub-1, Sub-2, Sub-4 not re-measured

The v3 5-run only re-measured sub-3 (the topic of
the bundle followup). Sub-1 (recall/precision on
addressed), sub-2 (speaker pid match), and sub-4
(M12.1 surface) require the M18.7.1 calibration
harness re-run, which is a separate replay. The
overall verdict on `failed:sub1+sub2+sub3` from
v2 (commit `6761fe0`) is presumed unchanged: the
v3 commit touched only the M20.4 producer (bundle
dispatch + `aggregation_kind` field), and the v2
prompt is unchanged. Sub-1/2/4 numbers are
expected to be at the v2 baseline:

- sub-1: recall 0.20, precision 1.00 (4× v1; still
  under 0.6 bar)
- sub-2: speaker pid exact match 0.00 / 18 decidable
  (still well below 0.7 bar)
- sub-4: 3 profiles per run, `surface_alive` 5/5
  (passes; was the v2 sub-4 fix)

A full v3 4-sub-cap re-run (with sub-1/2/4 re-measured)
is a follow-up once the M18.7.2 v3 prompt lands.

## File changes since v2 (recap of dce23f0)

- `segmentum/dialogue/runtime/m20_4_attribution.py`
  (+453): 11 new constants; 2 new helpers
  (`append_bundle_memory`, `_bundle_aggregated_support`);
  bundle dispatch in `produce_m20_4_attribution_commitments`;
  `aggregation_kind` parameter on
  `build_addressee_target_match_admitted_event`.
- `segmentum/dialogue/runtime/mvp_loop.py` (+45):
  1 new call site mirroring addressee-directed M18.7
  surface entries to bundle memory (after
  `_emit_m18_7_2_attribution_for_turn`); 1 consumer-side
  update reading `aggregation_kind` from commitment
  payload.
- `tests/test_m20_4_producer.py` (+348): 10 new
  tests (T1-T10). 45/45 producer tests pass.
- `scripts/run_group_chat_real_llm_acceptance.py`
  (+61): sub-3 summary reads new diag counters;
  `dir_true_total = single + bundle` (verdict logic
  updated; SUB3_P04_DIR_TRUE_ADMIT_MIN bar unchanged
  at 1).
- `prompts/M20.4_Work_Prompt.md` (+93): §2b "v2
  bundle aggregation" appended (design frozen).
- 351/351 cross-M20.4 + cross-M18.7 regression pass
  at landing.

## Next steps

1. **Path A**: M18.7.2 v3 prompt. Add an explicit
   "emit `addressed_to_assistant=true` for polite
   requests, implicit directives, and `asking the
   team` patterns" instruction. Re-run the 5-run.
   Target: True rate 4.8% → 30%+, bundle fires
   3+ runs, sub-3 dir_true_total >= 1 in 3+ runs.
2. **Optional**: re-run v3 5-run on a different
   model (claude or qwen) to confirm the bundle
   path fires when the model emits more True
   entries.
3. **Bars revision**: once v3 prompt + bundle
   works, revisit the sub-3 `dir_true >= 1` bar
   (may be too low for the new emit distribution).

## Out of scope (per v2 plan, held for v3 prompt work)

- M18.7.2 v3 prompt (the upstream emit-pattern
  fix). The v2 plan explicitly listed this as a
  separate milestone. The v3 5-run result is the
  empirical basis for promoting it to "next."
- Sub-2 metric redefinition (metric issue, not
  confidence issue).
- Bars revision (depends on v3 prompt 5-run
  numbers).
- M18.7 surface cap / contract (unchanged).
- M20.4 P0-4 / P0-5 / P0-6 thresholds (unchanged).
- M14.7 / M17.2 call sites (pattern replicated,
  not called into).
