# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 15
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Semantic embedding gate: False
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: partial
- Hard pass: False
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: True
- Formal Baseline C gate: True
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
| semantic_embedding_gate | False |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | True |
| reproducibility_gate | True |
| baseline_c_full_population_implant | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 12 / 3 / 0
- User delta median: 0.0057; IQR: 0.0173
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0366 | 9 | 6 | 0.0023 | 0.0119 |
| `random` | 0.0530 | 9 | 6 | 0.0141 | 0.0252 |
| `temporal` | 0.0113 | 11 | 3 | 0.0091 | 0.0249 |
| `topic` | 0.0547 | 10 | 5 | 0.0148 | 0.0362 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): True (text_similarity_low,semantic_mean_low)

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1884 | 0.4475 | 0.6231 | 0.1592 | 0.4304 | 0.0021 | 0.0104 | 0.0683 | 0.0219 |
| `baseline_c` | 1884 | 0.2925 | 0.3689 | 0.0961 | 0.0345 | 0.0000 | 0.0449 | 0.1471 | 0.1272 |
| `baseline_b_best` | 1884 | 0.4835 | 0.5870 | 0.0000 | 0.0195 | 0.0000 | 0.0238 | 0.0218 | 0.0028 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 1884 | {'connector': 0.977707, 'focus': 0.935244, 'generic_focus': 0.022293} | {'direct_advisory': 0.266454, 'exploratory_questioning': 0.191083, 'guarded_short': 0.010616, 'warm_supportive': 0.531847} |
| `baseline_c` | 1884 | {'connector': 1.0} | {'direct_advisory': 0.721868, 'exploratory_questioning': 0.065287, 'warm_supportive': 0.212845} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.0954 | 0.0437 | 0.8457 | 0.6414 |
| `no_surface_profile` | 60 | 0.0519 | 0.0002 | 0.8719 | 0.3894 |
| `surface_only_default_agent` | 60 | 0.0900 | 0.0383 | 0.5060 | 0.6334 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.977899, 'train_default_l2': 0.213684, 'train_full_cosine': 0.977766, 'train_full_l2': 0.215356, 'train_wrong_user_cosine': 0.981327, 'train_wrong_user_l2': 0.199213}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0906 | 0.0517 | 0.0389 | 0.0051 | True |
| `behavioral_similarity_strategy` | 0.4031 | 0.4101 | -0.0070 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1033 | 0.1202 | -0.0169 | 1.0000 | False |
| `stylistic_similarity` | 0.9067 | 0.8661 | 0.0405 | 0.0000 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9778 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0906 | 0.0074 | 0.0832 | 0.0000 | True |
| `behavioral_similarity_strategy` | 0.4031 | 0.6256 | -0.2225 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1033 | 0.3293 | -0.2260 | 1.0000 | False |
| `stylistic_similarity` | 0.9067 | 0.8014 | 0.1053 | 0.0001 | True |
| `personality_similarity` | 1.0000 | 1.0000 | -0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9778 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0906, baseline_a=0.0517, p(vs_a)=0.0051, baseline_c=0.0074, p(vs_c)=0.0000
- `agent_state_similarity`: personality=0.9778, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.4031, baseline_a=0.4101, baseline_c=0.6256
- `behavioral_similarity_action11`: personality=0.1033, baseline_a=0.1202, baseline_c=0.3293
- `stylistic_similarity`: personality=0.9067, baseline_a=0.8661, baseline_c=0.8014
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | True | True | False | True |
| `temporal` | True | True | False | True |
| `topic` | True | True | False | True |

## Direction Auto-Escalation

- Applied: True
- Requested max users: 10
- Pilot required users: 15
- Rerun user count: 15
- Note: Earlier 10x32 outputs without this field should be treated as pre-final-patch or pilot-escalation stale artifacts.
