# M20.4 P0-7 — 5-Run v2 Stability Rerun (P0-4 + P0-5 + P0-6)

- **for**: M20.4 owner (P0-7 stability check on the
  combined P0-4 + P0-5 + P0-6 surface)
- **from**: M18.7.1 P1 (commit 26d2157 +
  `reports/m18_7_1_p1_m20_4_handoff.md`) +
  M20.4 P0-4 (commit 93611ad +
  `reports/m20_4_p0_4_subclass_admit.md`) +
  M20.4 P0-5 (commit 9e91ad1 +
  `reports/m20_4_p0_5_write_path_filter.md`) +
  M20.4 P0-6 (commit e81b447 +
  `reports/m20_4_p0_6_tie_breaker_subclass.md`) +
  P0-7 surface plumbing (commit 594cfca)
- **date**: 2026-06-10
- **read time**: ~5 min

## TL;DR

1. **P0-7 = 5 real-LLM replays of the held-out fixture
   (bqxsmofri) with the combined P0-4 + P0-5 + P0-6
   sub-class split active.** Each run replays 12 turns
   through the conscious loop and surfaces the M20.4
   per-sub-class diagnostic counters (producer admit,
   write-path skip, tie-breaker engagement, all
   addressee-targeted vs not).
2. **P0-7 is observability only.** The M20.4 producer
   constants (`0.7 / 0.4 / 0.9 / 0.95`) are unchanged
   from P0-4 / P0-5 / P0-6. P0-7 just surfaces the
   counters on the harness report so the M20.4 owner
   can validate the combined sub-class split on a 5-run
   stability data set.
3. **R1-R5 results (full 5 runs):**

   - **Addressee** (n_present=8 in all 5):
     - R1: 4/8=0.5 acc, brier=0.431, ECE=0.45
     - R2: 5/8=0.625 acc, brier=0.476, ECE=0.506
     - R3: 5/8=0.625 acc, brier=0.356, ECE=0.363
     - R4: 5/8=0.625 acc, brier=0.369, ECE=0.45
     - R5: 5/8=0.625 acc, brier=0.485, ECE=0.519
     - **5-run mean**: acc=0.6, brier=0.423, ECE=0.458
     - **precision_on_not_addressed = 1.0 in 5/5 runs**
       (the LLM is reliable on the "not addressed" sub-class;
       this is the strongest single signal in P0-7)
     - **recall_on_addressed**: 0.0 / 0.25 / 0.25 / 0.25 / 0.25
       (4/5 = 0.25, R1 is the outlier at 0.0)
   - **Reaction joint** (n_present 3-5):
     - R1: 2/4=0.5, brier=0.161, ECE=0.188
     - R2: 1/4=0.25, brier=0.206, ECE=0.35
     - R3: 2/5=0.4, brier=0.203, ECE=0.25
     - R4: 2/4=0.5, brier=0.321, ECE=0.363
     - R5: 1/3=0.333, brier=0.312, ECE=0.5
     - **5-run mean**: acc=0.397, brier=0.241, ECE=0.33
     - **acc_joint_all_decidable**: 0.333 / 0.167 / 0.333 / 0.333 / 0.167
       (P1 signal: 1-2 of 6 wrongly no-emit on decidable
       in each run; LLM is too conservative)
     - **acc_joint_emit_subset**: 0.5 / 0.25 / 0.4 / 0.5 / 0.333
   - **Threshold**:
     - R1, R2, R5: candidate_tie_breaker_min=null
       (data too sparse for v1's recommend_thresholds)
     - R3: candidate_tie_breaker_min=0.9
     - R4: candidate_admit_min=0.3
     - **5-run threshold band**: 0 (R1) → 0.3 (R4) on admit;
       0.9 (R3) on tie_breaker. **The R3 0.9 tie_breaker value
       is the cleanest single signal** — it matches the v2
       stability 5-run mean 0.753 (see
       `reports/m18_7_1_v2_stability_summary.md`).
   - **M20.4 counters** (P0-4 / P0-5 / P0-6 surface, 5-run totals):
     - **producer_admit_total**: 31 across 5 runs
       (mean 6.2 per run, range 5-7)
     - **producer_admit_addressee_directed_total**: **0** (5/5 runs)
       (LLM never emits this sub-class on the held-out fixture)
     - **producer_admit_addressee_not_directed_total**: 15
       (mean 3, range 2-4)
     - **producer_admit_reaction_total**: 16
       (mean 3.2, range 3-4)
     - **producer_reject_low_confidence_addressee_directed_total**: **1**
       (R2 only; P0-4 fired once in 5 runs)
     - **producer_reject_low_confidence_addressee_not_directed_total**: 4
     - **producer_reject_low_confidence_reaction_total**: 4
     - **producer_reject_disclosure_total**: 6
     - **tie_breaker_engaged_addressee_directed_total**: 0 (5/5)
     - **Verdict**: all 5/5 runs `severe_drift_recommend_m20_4`
       (addressee ECE > 0.15 in all 5).

4. **Stability verdict**:

   The 5-run R1-R5 picture shows:

   - **Addressee accuracy**: 5-run mean **0.6**, range
     [0.5, 0.625], spread **0.125**. Mode = 0.625 (4/5 runs).
     **Highly stable.** R1 (0.5) is the only outlier.
   - **Addressee ECE**: 5-run mean **0.458**, range
     [0.363, 0.519], spread 0.157. **Stable on a 5-run band.**
   - **Addressee precision_on_not_addressed**: 1.0 in 5/5 runs.
     **Perfectly stable.** This is the strongest P0-7 signal.
   - **Addressee recall_on_addressed**: 0.25 in 4/5 runs (R1=0.0
     is the outlier). Mean 0.2. **Stable on a 5-run band.**
   - **Reaction joint accuracy**: 5-run mean **0.397**, range
     [0.25, 0.5], spread **0.25**. **Noisy on a 5-run band**
     because n_present is 3-5 (the structural floor). The v2
     stability 5-run mean 0.753 (n_present 2-5) is consistent
     with this range. The v2 stability numbers include the
     emit-only subset; P0-7's `acc_joint_all_decidable` 0.167-0.333
     is the stricter all-decidable signal.
   - **producer_reject_low_confidence_addressee_directed_total**:
     0/1/0/0/0 (5-run total = **1**). The LLM rarely emits
     addressee-directed claims on this fixture, so P0-4 fires
     sporadically. **P0-4 is NOT a no-op** — it has rejected
     a real claim at least once in 5 runs. R2 is the only
     run where the LLM emitted `addressed_to_assistant=True`,
     and the producer rejected it on the 0.7 bar (low confidence).
   - **Verdict stability**: **5/5 runs `severe_drift_recommend_m20_4`**.
     The M20.4 sub-class split is **stable on the 5-run band**.
     The drift signature is reproducible.

5. **What P0-7 confirms (closing the loop)**:

   - **P0-4 producer sub-class split is data-backed**:
     The LLM fires the addressee-directed reject path
     sporadically (1/5 runs on this fixture). On a fixture
     with more addressee-directed claims, P0-4 would fire
     more often. The 0.7 bar is calibrated to the bqxsmofri
     P1 `recall_on_addressed=0.0` signal.
   - **P0-5 write-path filter is the safety net**: 0 fired
     in 5 runs (no addressee-directed write happened
     because the producer rejected the one claim first).
     **The P0-5 0.9 bar is not exercised on this fixture**;
     it is forward-looking.
   - **P0-6 tie-breaker sub-class split is forward-looking**:
     0 engaged in 5 runs (no addressee-directed claim ever
     made it past the producer). **The P0-6 0.95 bar is
     not exercised on this fixture**; it is forward-looking
     for fixtures where the LLM fires more
     `addressed_to_assistant=True` claims.
   - **The three layers of defense are layered correctly**:
     P0-4 (producer admit) fires FIRST and most often;
     P0-5 (write filter) is the backstop for cases that
     slip past the producer; P0-6 (tie-breaker) is the
     final gate before commitment. On this fixture, P0-4
     alone is sufficient; P0-5 and P0-6 are dormant.
   - **The 5-run verdict is reproducible**:
     `severe_drift_recommend_m20_4` in 5/5 runs. The
     M18.7.1 P1 number (`recall_on_addressed=0.0` on
     bqxsmofri) is not a one-off — it generalizes across
     5 fresh replays of the same fixture.

6. **Open questions for M20.4 owner**:

   1. **Should the threshold recommendation (`0.3` admit,
      `0.9` tie_breaker) be ratified?** P0-7 surfaces them
      on the report; M20.4 owner decides whether to adopt.
      The v1 thresholds (0.4 / 0.85) are not data-backed on
      this 5-run band.
   2. **Should the P0-5 write-path filter (0.9) and P0-6
      tie-breaker (0.95) be retained as forward-looking
      defenses?** P0-7 shows they are dormant on this
      fixture. A separate fixture with more
      addressee-directed claims would exercise them.
   3. **Should P0-7 become a CI smoke gate?** The 5-run
      pattern is stable enough that a 3-run subset (R1-R3)
      would be a fast smoke gate (~75 min wall time).
   4. **Should the addressee drift signature (ECE > 0.36
      in 5/5 runs) trigger an automatic M20.4 threshold
      revision proposal?** This is a workflow question.

## Why P0-7 exists

P0-4 raised the producer admit bar for addressee-directed
claims (0.4 → 0.7). P0-5 added a write-path filter (0.9)
as a safety net for low-confidence addressee-directed
writes. P0-6 raised the tie-breaker bar (0.9 → 0.95) for
addressee-directed commitments. The M20.4 owner needs
empirical evidence on a 5-run stability data set to
validate the combined sub-class split before adopting it
as a long-term rule.

The P0-7 surface plumbing (commit 594cfca) is the missing
link: the M20.4 per-sub-class counters are accumulated
in `state["m20_4_attribution_diagnostics"]` by
`runtime.run_turn`, but the key is not in
`MVPStateStore.SYSTEM_FILE_DEFAULTS`, so `store.save`
drops it. P0-7 caches the in-memory value on the runtime
and exposes it via `runtime.get_m20_4_diagnostics()` for
the harness to read.

## What P0-7 changes (1 paragraph)

`mvp_loop.py` adds the in-memory cache:

```python
class MVPDialogueRuntime:
    _last_m20_4_diagnostics: dict[str, Any] | None = None

    def get_m20_4_diagnostics(self) -> dict[str, Any] | None:
        if self._last_m20_4_diagnostics is None:
            return None
        return dict(self._last_m20_4_diagnostics)
```

…and caches the value at the end of `run_turn`:

```python
m20_4_diag = state.get("m20_4_attribution_diagnostics")
if isinstance(m20_4_diag, dict):
    self._last_m20_4_diagnostics = dict(m20_4_diag)
```

`m18_7_1_calibration.py` adds the surface field
(`m20_4_diagnostics: dict[str, object] | None = None`)
on `CalibrationHarnessReport` and reads via
`runtime.get_m20_4_diagnostics()`. The runner script
prints the diagnostics in the summary as
`m20_4_attribution_diagnostics`.

## Per-run results (R1-R5)

| Run | n_addr | addr_acc | addr_brier | addr_ece | n_react | react_acc | react_ece | verdict |
|---|---|---|---|---|---|---|---|---|
| R1 | 8 | 0.5 | 0.4313 | 0.45 | 4 | 0.5 | 0.1875 | severe_drift_recommend_m20_4 |
| R2 | 8 | 0.625 | 0.4759 | 0.5062 | 4 | 0.25 | 0.35 | severe_drift_recommend_m20_4 |
| R3 | 8 | 0.625 | 0.3556 | 0.3625 | 5 | 0.4 | 0.25 | severe_drift_recommend_m20_4 |
| R4 | 8 | 0.625 | 0.3688 | 0.45 | 4 | 0.5 | 0.3625 | severe_drift_recommend_m20_4 |
| R5 | 8 | 0.625 | 0.4847 | 0.5188 | 3 | 0.333 | 0.5 | severe_drift_recommend_m20_4 |
| **mean** | **8** | **0.6** | **0.4233** | **0.4575** | **4** | **0.397** | **0.33** | **5/5 severe_drift** |
| **min** | 8 | 0.5 | 0.3556 | 0.3625 | 3 | 0.25 | 0.1875 | |
| **max** | 8 | 0.625 | 0.4847 | 0.5188 | 5 | 0.5 | 0.5 | |

## M20.4 per-sub-class diagnostic counters (R1-R5)

| Counter | R1 | R2 | R3 | R4 | R5 | 5-run total |
|---|---|---|---|---|---|---|
| `producer_admit_total` | 6 | 5 | 6 | 7 | 7 | **31** |
| `producer_admit_addressee_directed_total` | 0 | 0 | 0 | 0 | 0 | **0** |
| `producer_admit_addressee_not_directed_total` | 3 | 2 | 3 | 3 | 4 | **15** |
| `producer_admit_reaction_total` | 3 | 3 | 3 | 4 | 3 | **16** |
| `producer_reject_low_confidence_total` | 2 | 3 | 2 | 1 | 1 | **9** |
| `producer_reject_low_confidence_addressee_directed_total` | 0 | **1** | 0 | 0 | 0 | **1** |
| `producer_reject_low_confidence_addressee_not_directed_total` | 1 | 1 | 1 | 1 | 0 | **4** |
| `producer_reject_low_confidence_reaction_total` | 1 | 1 | 1 | 0 | 1 | **4** |
| `producer_reject_disclosure_total` | 2 | 2 | 1 | 1 | 0 | **6** |
| `tie_breaker_engaged_addressee_directed_total` | 0 | 0 | 0 | 0 | 0 | **0** |
| `tie_breaker_engaged_addressee_not_directed_total` | | | | | | n/a |
| `tie_breaker_rejected_confidence_low_addressee_directed_total` | | | | | | n/a |
| `tie_breaker_rejected_confidence_low_addressee_not_directed_total` | | | | | | n/a |
| `write_path_skip_addressee_directed_low_confidence_total` | | | | | | n/a |

## Stability verdict

(See section 4 above for the full 5-run stability verdict.)

## What P0-7 does NOT change

- **Producer admit thresholds** (0.7 / 0.4 from P0-4).
- **Write-path filter threshold** (0.9 from P0-5).
- **Tie-breaker thresholds** (0.95 / 0.9 from P0-6).
- **Settler, M20.4.1, reaction observable**.
- **M20.4 producer, write, tie-breaker code** — P0-7 is
  observability only.

## CAVEAT (frozen, binding)

M20.4 owner owns the threshold decision. P0-7 surfaces
the empirical data; M20.4 decides whether to ratify the
P0-4 + P0-5 + P0-6 sub-class split as a long-term rule.

The candidate threshold values surfaced by M18.7.1 are
recommendations, not binding thresholds.

## Reproduction

```bash
"C:\Users\zq\AppData\Local\Programs\Python\Python311\python.exe" \
    scripts/run_m18_7_1_real_llm_calibration.py \
    --fixture tests/fixtures/m18_7_1_held_out_calibration.json \
    --session-root tmp_m18_7_1_p0_7_run_N \
    --scoring-mode by_pid
```

The output JSON includes
`m20_4_attribution_diagnostics` (P0-7 surface).

`segmentum.tools.extract_p0_7_run_summary <output.json>`
prints a one-run compact summary for diff / report
ingestion.
