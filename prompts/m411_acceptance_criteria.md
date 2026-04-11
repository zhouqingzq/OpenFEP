# M4.11 Acceptance Criteria - Natural Rollout Phenomenology

## Gate 1: Long-horizon free rollout without curated corpora

- A reproducible rollout harness runs the agent inside the existing M3/M4 open-world sim for at least N ticks (N defined by the harness, not trivially small) on a fixed seed.
- The harness does not read `data/m47_corpus.json` or any curated-label corpus as its evidence source, and does not inject labelled ground truth for recall grading.
- Memory encoding and consolidation run on the M4.10 dynamical path; the rollout audit shows `encoding_source="dynamics"` as the dominant source and `consolidation_source="dynamics"` on semantic memories.
- Rollout outputs (per-tick encoding events, consolidation events, recall events, self-relevance tags) are persisted under `artifacts/` alongside the seed.

## Gate 2: Serial position effect emerges naturally and collapses under negative control

- From the free rollout, the harness derives operational "study lists" (contiguous novel-event runs delimited by consolidation boundaries) without any curated list injection.
- The free-recall distribution over list position shows measurable primacy and recency bumps relative to middle positions on the default run.
- A paired negative-control run on the same seed with salience weights shuffled or zeroed produces a curve that is significantly flatter than the default run.
- Both curves and the primacy / recency scores are saved as acceptance artifacts.

## Gate 3: Retention curve is logarithmic, not linear, and the gap collapses under negative control

- For every encoded episode, retention vs. lag is tracked on the default rollout.
- A logarithmic decay (`a - b*log(1+t)`) fits the observed retention curve measurably better than a linear decay on the default run.
- Under the paired negative-control run, the logarithmic advantage collapses or the curve becomes degenerate, showing the shape was driven by salience-weighted dynamics rather than trivial bookkeeping.
- Fit statistics for both models, on both default and negative-control runs, are saved.

## Gate 4: Schema intrusions appear representationally, and collapse under negative control

- After M4.10 dynamical consolidation has produced semantic centroids with non-trivial support, free-recall probes on schema-structured clusters produce a nonzero, measurable rate of intrusions on the default run.
- Intrusions are identified as recall items that land near a semantic centroid in representation space despite never having been encoded as episodes - not by substring or keyword matching.
- Under the paired negative-control run, intrusion rate collapses toward zero (or cluster formation itself degenerates, which also satisfies the gate).
- Both intrusion rates and the identification criterion are saved.

## Gate 5: Identity continuity under perturbation beats a null baseline, and collapses under negative control

- Self-related memories are identified via the existing self-narrative / identity layer, not via a keyword filter.
- A strong perturbation drawn from existing perturbation hooks is applied mid-rollout.
- On the default run, retention of self-related memories is significantly higher than a matched null baseline (non-self-related memories with comparable encoding strength and age).
- Under the paired negative-control run, the retention gap between self-related and null-baseline memories collapses.
- Both retention curves and the significance statistic are saved.

## Gate 6: Negative controls are first-class and required

- Every effect in Gates 2-5 ships with a paired negative-control rollout using shuffled or zeroed salience weights on the same seed.
- The acceptance report presents default vs. negative-control side by side per effect.
- Any effect that does not show the required default-vs-control gap is reported as NOT PASS for that effect, regardless of absolute numbers.
- Negative-control artifacts are persisted under `artifacts/` alongside default rollout artifacts.

## Gate 7: Three-layer acceptance taxonomy is implemented and visible

- The acceptance machinery emits three explicit fields on any memory-milestone report: `structural_pass`, `behavioral_pass`, `phenomenological_pass`.
- A milestone is marked `ACCEPT` only when all three are true. Otherwise the report must declare the split and must not advertise itself as `ACCEPT`.
- The M4 milestone status table and `README.md` are updated to state:
  - M4.5-M4.7 currently meet only layer (a) (structural self-consistency).
  - M4.8 targets layer (b) (behavioral causation via ablation contrast).
  - M4.10 targets layer (b) upstream of recall (dynamical encoding / consolidation).
  - M4.11 targets layer (c) (phenomenological fit via the four effects above).
- M4.11's own report uses the three-layer split honestly: it claims layer (c) on the strength of Gates 1-6 and inherits layers (a) and (b) from their respective predecessor milestones, citing them by milestone id.

## Gate 8: Honesty / fail-closed audit is retained as a safety net, not the sole grader

- The honesty / fail-closed audit framework still runs on the M4.11 evidence and its findings are included in the report.
- The M4.11 acceptance report states explicitly that the honesty audit is an upper safety net, and that the primary grader is the four-effect phenomenological evidence with negative controls.
- A pass on the honesty audit alone does NOT satisfy M4.11; failing the honesty audit does block M4.11.

## Not in scope

- No acceptance based on `data/m47_corpus.json` or any curated-corpus label matching.
- No acceptance of an effect without its paired negative-control run.
- No `ACCEPT` stamp on a memory milestone from layer (a) alone.
- No schema-intrusion claim based on substring / keyword matching.
- No claim of human-parameter parity; qualitative direction with working negative controls is sufficient.

## 2026-04-11 Status Alignment

- M4.11 acceptance remains `NOT_ISSUED` until all eight gates pass.
- The three-layer taxonomy becomes the standard for all memory-milestone reports from M4.11 onward, and is applied retroactively as a disclosure update (not as a re-acceptance) to M4.5-M4.10.
