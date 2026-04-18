# M5.4 Stop-Bleed Notice

This historical artifact is fail-closed under m54_v4_stop_bleed. Repo-tracked classifier fixtures are smoke-only and do not support formal acceptance. See aggregate_report.json / m54_acceptance.json for machine-readable blockers.

# M5.4 Validation Aggregate Report

- Users: 5 (tested: 5, skipped no strategy: 0)
- Required users: 5
- Agent state: users with metric 5, skipped 0
- Topic split: {'users_with_topic_strategy_row': 5, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 5, 'users_topic_split_valid_for_hard_gate': 5}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Semantic embedding gate: False
- Statistical gate: True
- Formal acceptance eligible: False
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: fail
- Hard pass: False
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace rows: 237

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
| formal_acceptance_eligible | False |
| semantic_embedding_gate | False |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | True |
| baseline_c_full_population_implant | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 4 / 1 / 0
- User delta median: 0.0048; IQR: 0.0091
- Pair-count distribution: {'0': 2, '1': 4, '2': 2, '3': 1, '6': 1, '7': 1, '8': 1, '11': 2, '14': 2, '16': 2, '56': 1, '67': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0529 | 2 | 1 | 0.0000 | 0.0364 |
| `random` | -0.0135 | 0 | 4 | -0.0191 | 0.0143 |
| `temporal` | 0.1159 | 4 | 1 | 0.0750 | 0.0542 |
| `topic` | -0.0135 | 0 | 4 | -0.0191 | 0.0143 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): True (text_similarity_low)

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 237 | 0.5063 | 0.8987 | 0.1941 | 0.4301 | 0.0000 | 0.0126 | 0.0551 | 0.0259 |
| `baseline_c` | 237 | 0.5654 | 0.8734 | 0.1814 | 0.0216 | 0.0000 | 0.0547 | 0.0299 | 0.0034 |
| `baseline_b_best` | 237 | 0.2405 | 0.8523 | 0.0000 | 0.0232 | 0.0000 | 0.0270 | 0.1371 | 0.0066 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 237 | {'connector': 0.966245, 'focus': 0.915612, 'generic_focus': 0.033755} | {'direct_advisory': 0.025316, 'guarded_short': 0.042194, 'warm_supportive': 0.932489} |
| `baseline_c` | 237 | {'connector': 1.0} | {'warm_supportive': 1.0} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 20 | 0.1062 | 0.0738 | 0.8233 | 0.5629 |
| `no_surface_profile` | 20 | 0.0324 | 0.0000 | 0.8797 | 0.3080 |
| `surface_only_default_agent` | 20 | 0.1207 | 0.0883 | 0.4373 | 0.5420 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.929382, 'train_default_l2': 0.4365, 'train_full_cosine': 0.928951, 'train_full_l2': 0.438591, 'train_wrong_user_cosine': 0.922267, 'train_wrong_user_l2': 0.428164}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0679 | 0.0324 | 0.0354 | 0.0938 | False |
| `behavioral_similarity_strategy` | 0.6108 | 0.5919 | 0.0189 | 0.4375 | False |
| `behavioral_similarity_action11` | 0.0960 | 0.0000 | 0.0960 | 0.5000 | False |
| `stylistic_similarity` | 0.8993 | 0.8732 | 0.0261 | 0.0312 | True |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9290 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.0679 | 0.0059 | 0.0620 | 0.0312 | True |
| `behavioral_similarity_strategy` | 0.6108 | 0.6820 | -0.0712 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0960 | 0.4880 | -0.3920 | 1.0000 | False |
| `stylistic_similarity` | 0.8993 | 0.8300 | 0.0693 | 0.0625 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9290 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.0679, baseline_a=0.0324, p(vs_a)=0.0938, baseline_c=0.0059, p(vs_c)=0.0312
- `agent_state_similarity`: personality=0.9290, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.6108, baseline_a=0.5919, baseline_c=0.6820
- `behavioral_similarity_action11`: personality=0.0960, baseline_a=0.0000, baseline_c=0.4880
- `stylistic_similarity`: personality=0.8993, baseline_a=0.8732, baseline_c=0.8300
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | False | False | True |
| `random` | False | False | False | True |
| `temporal` | False | False | False | True |
| `topic` | False | False | False | True |
