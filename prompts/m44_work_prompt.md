# M4.4 Agent Work Prompt — Cross-Task Parameter Consistency

## Project Context

You are working on the Segmentum project — a cognitive-style simulation framework.
The core engine lives in `segmentum/m4_cognitive_style.py` (CognitiveStyleParameters, 8 parameters).

**M4.3 delivered:**
- Single-task behavioral fit on real external data:
  - Confidence Database: 12,000 heldout trials, 192 subjects, heldout_likelihood = -0.405 (beats random -0.693, stimulus_only -0.437, statistical_logistic -0.415)
  - Iowa Gambling Task: 1,800 trials, 18 subjects, deck_match_rate = 0.309 (beats random 0.247, frequency_matching 0.261)
- Coordinate descent fitting (`m43_modeling.py`): separate fits for each task
- Independent baselines (`m43_baselines.py`): random, stimulus_only, statistical_logistic, human_match_ceiling (confidence); random, frequency_matching, human_behavior (IGT)
- Parameter sensitivity: 7/8 parameters active (resource_pressure_sensitivity inert)
- Subject-level train/validation/heldout split with leakage checks
- Failure analysis with real trial examples

**M4.3 known limitations (MUST inform M4.4 design):**

1. **IGT deck_match_rate is only 0.309 vs random 0.247** — This is an architectural ceiling, not a tuning problem. The agent simulates its own IGT experience (from `IGT_DECK_PROTOCOL`), so its internal state diverges from the human's after the first mismatch. Per-trial deck matching under path divergence has a very low theoretical ceiling.

2. **Massive cross-task parameter divergence in M4.3 single-task fits:**

   | Parameter | Confidence | IGT | Gap |
   |-----------|-----------|-----|-----|
   | uncertainty_sensitivity | 0.00 | 0.65 | 0.65 |
   | error_aversion | 0.30 | 0.70 | 0.40 |
   | virtual_prediction_error_gain | 0.68 | 1.00 | 0.32 |
   | confidence_gain | 1.00 | 0.78 | 0.22 |
   | exploration_bias | 0.55 | 0.35 | 0.20 |
   | update_rigidity | 0.65 | 0.45 | 0.20 |
   | attention_selectivity | 0.32 | 0.20 | 0.12 |
   | resource_pressure_sensitivity | 0.75 | 0.75 | 0.00 |

3. **`_score_action_candidates` is a shared linear scoring function** — Confidence and IGT go through the same decision core (`m4_benchmarks.py:566-633`). The function is well-suited for trial-independent perceptual decisions (confidence) but structurally limited for sequential reinforcement learning (IGT). The IGT `action_space()` compensates by computing rich features (value estimates, loss tracking, habit), but the single-step softmax decision boundary remains.

**M4.4 goal:** Determine which parameters are genuinely cross-task stable ("trait-like") versus which are task-specific artifacts of architectural mismatch. This requires honest methodology that does not force false consistency and does not hide inconsistency.

---

## What M4.4 Must Do

### Core Question

Given that the same person has the same cognitive style across tasks, can we find parameter settings that work *acceptably well* on both tasks simultaneously? And if not, can we identify *which* parameters are portable and which are task-bound?

### Key Methodological Insight

A naive approach — "find parameters that maximize joint score across tasks" — will fail or mislead because:
- The IGT fitting ceiling is low (path divergence problem), so joint optimization will be dominated by Confidence task gradients
- A parameter set that "works on both tasks" might just be one that ignores the IGT signal
- Declaring consistency failure because IGT-optimal differs from Confidence-optimal conflates modeling limitation with genuine cognitive inconsistency

**The correct approach is stratified:**
1. Fit a shared parameter set on both tasks jointly
2. Compare it against task-specific fits (from M4.3)
3. Measure the **degradation** on each task when using shared vs task-specific parameters
4. Classify each parameter as: **stable** (shared ≈ task-specific), **task-sensitive** (shared ≠ task-specific, for identifiable reasons), or **indeterminate** (insufficient signal to classify)

---

## Strict Prohibitions

### 1. No Fake Consistency
- Do NOT force parameters to be equal across tasks and claim "consistency achieved."
- Do NOT cherry-pick metrics to make consistency look better than it is.
- If the shared fit degrades Confidence performance by >5% relative to task-specific, report this honestly even if IGT improves.

### 2. No Fake External Validation
- Same rules as M4.3: `external_validation: false` on everything this repo produces.

### 3. No Synthetic Data Claims
- All cross-task results must come from real external bundle data.
- Smoke tests are for CI only — never in acceptance path.

### 4. No Fake Tests
- Tests must be falsifiable. Do not write tests that pass regardless of actual consistency.
- If consistency is poor, tests should detect and report this, not hide it.

### 5. No Architecture Rewrites
- Do NOT rewrite `_score_action_candidates` or `BenchmarkAdapter` protocol.
- Do NOT add new parameters to `CognitiveStyleParameters`.
- You may add task-specific metric computation and new analysis code.
- You may add an IGT aggregate metric layer (see below) as long as it doesn't change the adapter interface.

### 6. No Scope Bloat
- M4.4 is cross-task parameter consistency, not controlled environment transfer (M4.5), not longitudinal stability (M4.6).
- Do not implement grid worlds, multi-agent interactions, or temporal stability tests.

---

## Required Tasks

### T1: IGT Aggregate Metric Layer

Before doing cross-task comparison, fix the IGT evaluation lens. `deck_match_rate` is a bad primary metric for cross-task work due to the path divergence problem.

**Create `segmentum/m44_igt_aggregate.py`:**
1. Define aggregate IGT behavioral metrics that are robust to path divergence:
   - `advantageous_learning_curve`: Does the agent develop a preference for C/D decks over time? Measure by phase (trials 1-20, 21-40, 41-60, 61-80, 81-100). Compare agent curve to human curve via L1 distance on phase-wise advantageous rates.
   - `post_loss_switching_pattern`: After a loss, does the agent switch decks at a rate similar to humans? Compare agent and human post-loss switch rates.
   - `deck_distribution_alignment`: Does the agent's overall deck selection distribution match the human's? Measure via L1 distance on 4-deck frequency vectors.
   - `exploration_exploitation_transition`: Does the agent shift from exploratory (high entropy) to exploitative (low entropy) choices across phases, similar to humans?
2. Define a composite `igt_behavioral_similarity` score that averages these (each normalized to [0,1]).
3. This layer does NOT replace `deck_match_rate` — it supplements it. Both are reported.

### T2: Joint Parameter Fitting

**Add to `segmentum/m44_cross_task.py`:**
1. **Joint objective function:** Weighted combination of Confidence and IGT objectives.
   - Use the M4.3 scoring functions (`_score_confidence_metrics`, `_score_igt_metrics`) as sub-objectives
   - Add `igt_behavioral_similarity` from T1 as a third sub-objective for IGT
   - Weight: `joint_objective = w_conf * confidence_obj + w_igt * igt_obj + w_igt_agg * igt_agg_obj`
   - Default weights: `w_conf = 1.0, w_igt = 0.5, w_igt_agg = 0.8` (downweight raw IGT due to known ceiling)
   - Report sensitivity to these weights (see T4)
2. **Joint coordinate descent:**
   - Same algorithm as M4.3 `_coordinate_descent_fit_confidence` but optimizing joint objective
   - Train on union of Confidence training trials + IGT training trials
   - Validate on union of Confidence validation trials + IGT validation trials
   - Search over all 7 active parameters (exclude resource_pressure_sensitivity unless it becomes active in joint setting)
3. **Output:** `joint_fitted_parameters` — one parameter vector intended to work on both tasks

### T3: Degradation Analysis

**The core deliverable of M4.4.**

**Add to `segmentum/m44_cross_task.py`:**
1. Evaluate three parameter sets on both tasks:
   - `confidence_specific`: M4.3 Confidence-optimal parameters
   - `igt_specific`: M4.3 IGT-optimal parameters
   - `joint`: T2 joint-fitted parameters
2. For each parameter set × each task, compute full metrics on heldout data.
3. Compute degradation matrix:
   ```
   confidence_degradation = (joint_confidence_metric - confidence_specific_metric) / |confidence_specific_metric|
   igt_degradation = (joint_igt_metric - igt_specific_metric) / |igt_specific_metric|
   ```
4. Compute cross-application matrix (how does Confidence-optimal perform on IGT, and vice versa).
5. Report with honest interpretation:
   - If `confidence_degradation > 0.05`: "Joint fitting meaningfully degrades Confidence performance"
   - If `igt_degradation > 0.10`: "Joint fitting meaningfully degrades IGT performance"  (higher threshold because IGT has more noise)
   - If both < thresholds: "Shared parameters achieve acceptable cross-task fit"

### T4: Parameter Stability Classification

**Add to `segmentum/m44_cross_task.py`:**
1. For each of the 8 parameters, compute:
   - `confidence_specific_value`: from M4.3
   - `igt_specific_value`: from M4.3
   - `joint_value`: from T2
   - `gap = |confidence_specific - igt_specific|`
   - `joint_shift_conf = |joint - confidence_specific|`
   - `joint_shift_igt = |joint - igt_specific|`
2. Run leave-one-parameter-out ablation on the joint fit:
   - For each parameter, fix it at its joint value and re-optimize the other 7 for each task independently
   - If both tasks tolerate the fixed value (degradation < 5%), classify as **stable**
   - If one task degrades significantly, classify as **task-sensitive**
   - If the parameter is inert on both tasks (from M4.3 sensitivity), classify as **inert**
3. Expected result based on M4.3 data:
   - `resource_pressure_sensitivity`: **inert** (confirmed in M4.3)
   - `uncertainty_sensitivity`: likely **task-sensitive** (gap = 0.65)
   - `error_aversion`: likely **task-sensitive** (gap = 0.40)
   - Others: to be determined empirically
4. Record classification with evidence: `{"parameter": "...", "classification": "stable|task_sensitive|inert|indeterminate", "evidence": {...}}`

### T5: Weight Sensitivity Check

**Add to `segmentum/m44_cross_task.py`:**
1. Re-run T2 joint fitting with 3 different weight vectors:
   - Default: `w_conf=1.0, w_igt=0.5, w_igt_agg=0.8`
   - IGT-heavy: `w_conf=0.5, w_igt=1.0, w_igt_agg=1.0`
   - Confidence-heavy: `w_conf=1.0, w_igt=0.2, w_igt_agg=0.3`
2. Compare resulting joint parameter vectors. If they're nearly identical (all parameter differences < 0.1), the joint fit is robust to weighting. If they diverge, report which parameters are weight-sensitive.
3. This is diagnostic, not blocking — but it reveals whether the "cross-task consistency" finding is an artifact of how we weighted the tasks.

### T6: Honest Architecture Assessment

**Include in acceptance report as a structured finding:**
1. Quantify the IGT ceiling: what is the best possible deck_match_rate achievable by ANY parameter setting? (Run a broader parameter sweep if needed.)
2. Compare this ceiling to random (0.247) and to the aggregate metrics.
3. State whether IGT per-trial matching is a viable evaluation lens for future milestones (M4.5, M4.6) or whether aggregate metrics should replace it.
4. If the architecture assessment concludes that `_score_action_candidates` is fundamentally limited for sequential tasks, record this as a **finding** (not a gate failure) with a recommendation for M4.5.

### T7: Acceptance Reporting

**Create `segmentum/m44_audit.py`:**
1. Collect all artifacts from T1-T6.
2. Evaluate all gates (see acceptance criteria).
3. Generate `reports/m44_acceptance_report.json` with same structure as M4.3.
4. Generate `reports/m44_acceptance_summary.md` human-readable summary.
5. Blocked path when external bundle is missing (same pattern as M4.3).
6. Do NOT overwrite M4.3 artifacts — output to `artifacts/m44_*.json`.

### T8: Tests

**Create `tests/test_m44_cross_task.py`:**
1. Smoke tests (always run): joint fitting on smoke data, degradation computation correctness, parameter classification logic
2. External bundle tests (`@unittest.skipUnless`): full joint fit on real data, degradation thresholds, classification against M4.3 anchors
3. Blocked path test: mock missing bundle → BLOCKED status

**Create `tests/test_m44_igt_aggregate.py`:**
1. Test aggregate metrics on known synthetic IGT sequences with predictable outcomes
2. Test that learning_curve_distance = 0 when agent curve == human curve
3. Test that deck_distribution_alignment = 0 when distributions match

**Create `tests/test_m44_acceptance.py`:**
1. Same pattern as M4.3 acceptance tests: gate shapes, payload-derived truth, blocked path, official output protection

---

## File Structure

```
segmentum/m44_igt_aggregate.py     — IGT aggregate behavioral metrics (CREATE)
segmentum/m44_cross_task.py        — Joint fitting, degradation, classification (CREATE)
segmentum/m44_audit.py             — Acceptance artifact generation (CREATE)
tests/test_m44_cross_task.py       — Cross-task fit tests (CREATE)
tests/test_m44_igt_aggregate.py    — IGT aggregate metric tests (CREATE)
tests/test_m44_acceptance.py       — Acceptance gate tests (CREATE)
artifacts/m44_joint_fit.json       — Joint parameter fitting results
artifacts/m44_degradation.json     — Degradation analysis
artifacts/m44_parameter_stability.json — Parameter classification
artifacts/m44_weight_sensitivity.json  — Weight sensitivity check
artifacts/m44_igt_aggregate.json   — IGT aggregate metric results
artifacts/m44_architecture_assessment.json — Architecture findings
reports/m44_acceptance_report.json — Acceptance report
reports/m44_acceptance_summary.md  — Human-readable summary
```

---

## What M4.4 Inherits from M4.3 (Use Directly)

- `m43_modeling.py`: `run_fitted_confidence_agent`, `run_fitted_igt_agent`, `_coordinate_descent_fit_confidence`, `_coordinate_descent_fit_igt`, `_score_confidence_metrics`, `_score_igt_metrics`, `_simulate_confidence_trials`, `_simulate_igt_trials`
- `m43_baselines.py`: All baseline implementations (unchanged)
- `m43_audit.py`: Gate evaluation patterns (reference for M4.4 gates)
- `m4_benchmarks.py`: Adapters, data loading, leakage checks (unchanged)
- `m4_cognitive_style.py`: `CognitiveStyleParameters` (unchanged)
- `external_benchmark_registry/`: Real data (unchanged)

## What M4.4 Does NOT Inherit

- M4.3 fitted parameter values — these are inputs to M4.4 analysis, not starting points for M4.4 fitting
- M4.3 scoring function weights — M4.4 may need different weights for the joint objective
- M4.3 claim that "7/8 parameters are active" — re-verify in joint context

---

## Labeling Rules

Same as M4.3:

| Data source | `claim_envelope` | `external_validation` | `source_type` |
|-------------|------------------|-----------------------|---------------|
| external_benchmark_registry | `benchmark_eval` | `false` | `external_bundle` |
| Same-codebase synthetic | `synthetic_diagnostic` | `false` | `synthetic_protocol` |

---

## Regression Requirements

Before M4.4 acceptance, verify:
- All M4.3 tests still pass (`tests/test_m43_*.py`)
- All M4.2 tests still pass (`tests/test_m42_*.py`)
- All M4.1 tests still pass (`tests/test_m41_*.py`)
- No modifications to `CognitiveStyleParameters` interface
- No modifications to `BenchmarkAdapter` protocol
- No modifications to `_score_action_candidates` signature or behavior

---

## Key Principle

**M4.4 is a diagnostic milestone, not a triumph milestone.** The most valuable outcome is an honest parameter stability map that tells M4.5 which parameters to trust across contexts and which to treat as task-specific. A finding that "only 3 of 8 parameters are cross-task stable" is a perfectly valid M4.4 result — as long as the evidence is real and the methodology is sound.
