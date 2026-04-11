# M4.11 Work Prompt - Natural Rollout Phenomenology Instead of Curated Corpora

## Goal

Replace the curated-corpus validation path with a long-horizon free rollout in the existing open-world sim, and verify that four classical human-memory effects emerge naturally from the dynamical memory system built in M4.10 - without any curated event feed. In parallel, revise the milestone acceptance standard itself so that "artifact matches evaluator truth" is no longer the primary pass condition for memory milestones.

## Context

Until now, the strongest memory-evidence run has been `data/m47_corpus.json` - a curated, labelled corpus. That path can only prove that the framework is internally self-consistent: the evaluator knows what the agent should recall because the same corpus defined both. It cannot show that memory does anything human-like.

M4.10 moves encoding and consolidation to dynamics. M4.11 is where we stop grading the memory system against a cheat sheet and start grading it against **phenomenology**. The agent runs a long free rollout in the M3/M4 open-world sim, nobody tells it what matters, and we measure whether its memory exhibits:

1. **Serial position effect** - free-recall distribution over a studied sequence shows primacy and recency bumps.
2. **Retention / forgetting curve** - episode retention vs. lag follows an Ebbinghaus-like logarithmic decay, not a linear one.
3. **Schema intrusion** - after semantic consolidation, episodic recall starts mixing in schema-typical items that were never presented (DRM-style intrusions at the free-rollout level, not in a harness).
4. **Identity continuity under perturbation** - after a strong perturbation, self-related memories are retained at a significantly higher rate than a null baseline. This is the real test of the M4.5 self-continuity claim.

Each effect must pass a **negative control**: the same rollout with salience weights shuffled or zeroed must flatten the effect. If the effect survives the shuffle, it was not driven by the memory dynamics and does not count.

Separately, the milestone acceptance standard is revised. Acceptance is split into three layers:

- **(a) Structural self-consistency** - current practice: artifacts match evaluator truth. Kept but demoted.
- **(b) Behavioral causation** - M4.8 ablation-contrast: memory on/off changes behavior on the default path.
- **(c) Phenomenological fit** - M4.11: the four classical effects appear in a free rollout with negative controls.

An M-level memory milestone only gets a full `ACCEPT` stamp when all three layers pass. Otherwise the status must be disclosed split as `structural_pass`, `behavioral_pass`, `phenomenological_pass` with the missing layers explicit. M4.5-M4.7 are retroactively annotated as currently meeting only layer (a); M4.8 targets (b); M4.11 targets (c).

## Implementation Plan

### 1. Long-horizon free rollout harness

- Add a harness that runs the agent inside the existing M3/M4 open-world sim for a configurable N ticks (default large, e.g. 20k-50k) with:
  - no curated event injection,
  - no labelled ground truth over recall targets,
  - standard attention / homeostasis / FEP loop active,
  - memory encoding and consolidation on the M4.10 dynamical path.
- The harness records, per tick: episode encoding events (with `encoding_source`, `encoding_strength`, FEP signals, budget bookkeeping), consolidation events, recall events, and self-relevance tags.
- The harness must be reproducible from a seed and must not consult `data/m47_corpus.json` or any equivalent curated label file for grading.

#### 1a. Hard harness obligations (anti–quiet-boost)

These are **acceptance-bearing**: the M4.11 harness must actively check them and fail closed if evidence is missing. They prevent silently reusing the same “helped” paths as curated or default-path audits.

1. **Free rollout: replay only when the live loop spontaneously triggers it**  
   Verify that **replay / reconsolidation** (second-pass updates to semantic centroids, residual stats, salience deltas, etc.) occurs **only** through mechanisms that actually run during the long free rollout—e.g. sleep/consolidation phases and the in-loop dynamical replay selection tied to real ticks—not via test-only calls (`constrained_replay` invoked from a harness hook after the fact), extra post-rollout “sweep” passes, or schedules that do not exist in production. **Evidence:** per-tick (or per-sleep-episode) logs that correlate replay events with the genuine consolidation path for that seed; replay counts and timestamps must not be explainable by out-of-band invocations alone.

2. **Tick-level shared encoding budget**  
   Verify that **attention / encoding budget is one shared pool per tick** (or per agent decision cycle as defined in the sim), so all encoding candidates that compete in the same tick are scored under **shared** remaining budget—analogous to M4.10’s `EncodingDynamics.score_many` competition—not each receiving an independent full budget (a quiet boost that hides competition). **Evidence:** per tick, log `requested_budget`, `attention_budget` (or equivalent), per-candidate `attention_budget_granted` / `attention_budget_denied`, and show high-salience / high-PE events winning **at the expense of** other candidates **within the same tick**.

### 2. Serial position effect probe

- Inside the free rollout, define "study lists" operationally as contiguous runs of novel events delimited by rest / consolidation boundaries (no curated list injection).
- At designated recall probes, collect free recall and compute the serial position curve over list position.
- Compute a primacy score (mean retention in first K positions vs. middle) and a recency score (last K vs. middle).
- Negative control: re-run the same seed with salience weights shuffled; serial position curve must flatten toward uniform.

### 3. Retention / forgetting curve probe

- For every encoded episode, track retention as a function of lag (in ticks or wall-time-equivalent), measured as whether the episode is still recallable and/or its reconstruction error against its semantic centroid.
- Fit both a logarithmic (`a - b*log(1+t)`) and a linear decay to the observed curve and compare goodness-of-fit.
- Acceptance: logarithmic fit is measurably better than linear on the default run, and the gap collapses under the negative control.

### 4. Schema intrusion probe

- After the dynamical consolidation of M4.10 has produced real semantic centroids, run free-recall probes on clusters that have strong schema structure.
- Count recall outputs that are schema-typical but were never encoded as episodes (intrusions), vs. veridical recalls.
- Acceptance: a nonzero, measurable intrusion rate appears on the default run; under the negative control (salience shuffled, so semantic clusters do not form meaningfully), the intrusion rate collapses toward zero.
- Intrusions must be identified representationally (via closeness to centroid of an unencoded schema-typical vector), not by substring matching.

### 5. Identity continuity under perturbation probe

- Define "self-related memories" as episodes whose encoding context has high self-relevance (from the existing self-narrative / identity layer - not from a keyword filter).
- Apply a strong perturbation mid-rollout (e.g. homeostatic shock, identity challenge, environment shift) drawn from the existing perturbation hooks.
- Measure retention of self-related memories vs. a matched null baseline of non-self-related memories with similar encoding strength and age.
- Acceptance: self-related retention is significantly higher than the null baseline on the default run, and the gap collapses under the negative control (salience weights shuffled so that self-relevance no longer biases encoding / consolidation).

### 6. Negative controls, treated as first-class

- Every effect above ships with a paired negative-control run using shuffled or zeroed salience weights on the same seed.
- Acceptance compares default vs. negative control on a per-effect basis; an effect counts only if it is present in default AND absent (or significantly reduced) in the negative control.
- Negative-control rollout outputs must be saved alongside default rollout outputs in `artifacts/`.

### 7. Revised acceptance framework (structural / behavioral / phenomenological)

- Update the acceptance machinery so that any memory milestone report exposes three explicit fields: `structural_pass`, `behavioral_pass`, `phenomenological_pass`.
- A milestone is `ACCEPT` only if all three are true. Otherwise the report must show which layers passed and which did not, and must not present itself as `ACCEPT`.
- Apply the new taxonomy retroactively to M4.5, M4.6, M4.7, M4.8, M4.9, M4.10, and M4.11 in the milestone status table.
- `README.md` and the M4 milestone status table must state explicitly:
  - M4.5-M4.7 currently meet only layer (a) (structural self-consistency).
  - M4.8 targets layer (b) (behavioral causation via ablation contrast).
  - M4.10 targets layer (b) upstream of recall (dynamical encoding / consolidation).
  - M4.11 targets layer (c) (phenomenological fit via the four effects above).
- The honesty / fail-closed audit framework is retained as an upper safety net. It is no longer the sole pass criterion for memory milestones.

## Constraints

- Do NOT satisfy M4.11 by calling replay or consolidation helpers from outside the live tick loop in a way that **simulates** spontaneity; the harness must prove **spontaneous** replay under free rollout (see **§1a.1**).
- Do NOT give each encoding event an isolated full budget per tick; competition must be observable at **tick granularity** (see **§1a.2**).
- Do NOT use `data/m47_corpus.json` or any other curated-label corpus as the primary evidence source for M4.11.
- Do NOT grade recall by matching against evaluator-provided ground truth that was itself injected into the rollout.
- Do NOT accept an effect without its paired negative-control run.
- Do NOT claim `ACCEPT` on a memory milestone by passing layer (a) alone; the three-layer split must be visible in the report.
- Do NOT identify schema intrusions by string / keyword matching; intrusions must be representational.

## Out of Scope

- Human-subject comparison at quantitative parity. M4.11 only requires that the four effects appear in the right qualitative direction with the right negative-control behavior. Fitting specific human parameters is future work.
- Replacing the honesty / fail-closed audit framework. M4.11 reclassifies it as a safety net, not the primary grader, and does not remove it.

## Acceptance Criteria

See `prompts/m411_acceptance_criteria.md`.
