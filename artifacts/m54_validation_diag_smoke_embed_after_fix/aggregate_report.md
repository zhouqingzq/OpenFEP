# M5.4 Validation Aggregate Report

- Users: 3 (tested: 3, skipped no strategy: 0)
- Required users: 3
- Agent state: users with metric 3, skipped 0
- Topic split: {'users_with_topic_strategy_row': 3, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 3, 'users_topic_split_valid_for_hard_gate': 3}
- Metric version: m54_v5_formal_evidence (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Classifier evidence tier: repo_fixture_smoke
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: fail
- Hard pass: False
- Formal blockers: ['classifier_fixture_only', 'metric_hard_pass_failed', 'partner_gate_failed', 'topic_gate_failed']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace gate: True
- Agent-state differentiation gate: True
- Behavioral majority baseline gate: True
- Surface ablation gate: True
- Diagnostic trace rows: 179

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | False |
| semantic_similarity_vs_baseline_c_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
| formal_acceptance_eligible | False |
| semantic_embedding_gate | True |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | True |
| baseline_c_leave_one_out_population_average | True |
| diagnostic_trace_gate | True |
| agent_state_differentiation_gate | True |
| behavioral_majority_baseline_gate | True |
| surface_ablation_gate | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 2 / 1 / 0
- User delta median: 0.0084; IQR: 0.0087
- Pair-count distribution: {'2': 2, '5': 1, '8': 1, '10': 1, '17': 1, '19': 2, '20': 1, '25': 1, '26': 2}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0380 | 2 | 1 | 0.0288 | 0.0307 |
| `random` | -0.0011 | 1 | 2 | -0.0086 | 0.0202 |
| `temporal` | -0.0123 | 2 | 1 | 0.0160 | 0.0869 |
| `topic` | 0.0239 | 2 | 1 | 0.0341 | 0.0427 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): True (text_similarity_low)

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 179 | 0.4804 | 0.7151 | 0.0000 | 0.0000 | 0.0000 | 0.0106 | 0.1428 | 0.0076 |
| `baseline_c` | 179 | 0.4972 | 0.7039 | 0.0000 | 0.0000 | 0.0000 | -0.0366 | 0.1428 | 0.0076 |
| `baseline_b_best` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 179 | {'generic': 1.0} | {'unknown': 1.0} |
| `baseline_c` | 179 | {'generic': 1.0} | {'unknown': 1.0} |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {}

## Debug Readiness Gate

- Passed: False
- Checks: {'train_default_l2_positive': False, 'train_wrong_user_l2_positive': False, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4700 | 0.4579 | 0.0121 | 0.2500 | False |
| `behavioral_similarity_strategy` | 0.0257 | 0.0311 | -0.0054 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| `stylistic_similarity` | 0.8687 | 0.8827 | -0.0140 | 1.0000 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4700 | 0.5012 | -0.0312 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.0257 | 0.0311 | -0.0054 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| `stylistic_similarity` | 0.8687 | 0.8613 | 0.0073 | 0.2500 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 1.0000 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.4700, baseline_a=0.4579, p(vs_a)=0.2500, baseline_c=0.5012, p(vs_c)=1.0000
- `agent_state_similarity`: personality=1.0000, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.0257, baseline_a=0.0311, baseline_c=0.0311
- `behavioral_similarity_action11`: personality=0.0000, baseline_a=0.0000, baseline_c=0.0000
- `stylistic_similarity`: personality=0.8687, baseline_a=0.8827, baseline_c=0.8613
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
