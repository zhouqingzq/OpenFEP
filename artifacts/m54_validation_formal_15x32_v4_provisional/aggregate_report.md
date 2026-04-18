# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 15
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v4_stop_bleed (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Classifier evidence tier: llm_generated_provisional
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: partial
- Hard pass: False
- Formal blockers: ['behavioral_majority_baseline_matches_or_beats_personality', 'classifier_provisional_llm_labels', 'surface_ablation_failed']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: True
- Topic strategy hard pass: True
- Formal Baseline C gate: True
- Diagnostic trace gate: True
- Agent-state differentiation gate: True
- Behavioral majority baseline gate: False
- Surface ablation gate: False
- Diagnostic trace rows: 1884

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | True |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | True |
| formal_acceptance_eligible | False |
| semantic_embedding_gate | True |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | True |
| topic_gate | True |
| reproducibility_gate | True |
| baseline_c_leave_one_out_population_average | True |
| diagnostic_trace_gate | True |
| agent_state_differentiation_gate | True |
| behavioral_majority_baseline_gate | False |
| surface_ablation_gate | False |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 13 / 2 / 0
- User delta median: 0.0661; IQR: 0.0718
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0620 | 12 | 3 | 0.0612 | 0.0713 |
| `random` | 0.0668 | 14 | 1 | 0.0518 | 0.0409 |
| `temporal` | 0.0486 | 10 | 5 | 0.0647 | 0.1052 |
| `topic` | 0.0763 | 14 | 1 | 0.0536 | 0.0588 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1884 | 0.3721 | 0.5515 | 0.1173 | 0.5681 | 0.0000 | 0.0559 | 0.1412 | 0.0889 |
| `baseline_c` | 1884 | 0.1709 | 0.2341 | 0.0483 | 0.4504 | 0.0016 | -0.1362 | 0.4176 | 0.3934 |
| `baseline_b_best` | 1884 | 0.5165 | 0.5732 | 0.0000 | 0.4022 | 0.0005 | -0.0511 | 0.0210 | 0.0070 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 1884 | {'connector': 0.977707, 'focus': 0.728769, 'generic_focus': 0.022293} | {'direct_advisory': 0.001062, 'exploratory_questioning': 0.050955, 'guarded_short': 0.932059, 'warm_supportive': 0.015924} |
| `baseline_c` | 1884 | {'connector': 1.0} | {'guarded_short': 1.0} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.4655 | 0.0837 | 0.7764 | 0.7383 |
| `no_surface_profile` | 60 | 0.3893 | 0.0075 | 0.8934 | 0.6165 |
| `surface_only_default_agent` | 60 | 0.4569 | 0.0750 | 0.4114 | 0.7227 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.865258, 'train_default_l2': 0.629039, 'train_full_cosine': 0.997475, 'train_full_l2': 0.05642, 'train_wrong_user_cosine': 0.985388, 'train_wrong_user_l2': 0.198568}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4452 | 0.3818 | 0.0634 | 0.0042 | True |
| `behavioral_similarity_strategy` | 0.2221 | 0.0632 | 0.1589 | 0.0049 | True |
| `behavioral_similarity_action11` | 0.1107 | 0.0068 | 0.1039 | 0.0104 | True |
| `stylistic_similarity` | 0.9115 | 0.8661 | 0.0454 | 0.0000 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9975 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4452 | 0.5485 | -0.1032 | 1.0000 | False |
| `behavioral_similarity_strategy` | 0.2221 | 0.8879 | -0.6658 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1107 | 0.6998 | -0.5891 | 1.0000 | False |
| `stylistic_similarity` | 0.9115 | 0.7113 | 0.2002 | 0.0000 | True |
| `personality_similarity` | 1.0000 | 1.0000 | -0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9975 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.4452, baseline_a=0.3818, p(vs_a)=0.0042, baseline_c=0.5485, p(vs_c)=1.0000
- `agent_state_similarity`: personality=0.9975, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.2221, baseline_a=0.0632, baseline_c=0.8879
- `behavioral_similarity_action11`: personality=0.1107, baseline_a=0.0068, baseline_c=0.6998
- `stylistic_similarity`: personality=0.9115, baseline_a=0.8661, baseline_c=0.7113
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | True | True | False | True |
| `random` | True | True | False | True |
| `temporal` | True | True | False | True |
| `topic` | True | True | False | True |
