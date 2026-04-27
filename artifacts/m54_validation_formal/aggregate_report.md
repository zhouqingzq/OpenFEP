# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 15
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v10_discriminative_diagnostics (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Classifier evidence tier: llm_generated_provisional
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: False
- Partial acceptance eligible: True
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: partial
- Hard pass: False
- Formal blockers: ['behavioral_majority_baseline_matches_or_beats_personality', 'classifier_provisional_llm_labels', 'metric_hard_pass_failed', 'partner_gate_failed', 'topic_gate_failed']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace gate: True
- Agent-state differentiation gate: True
- Behavioral majority baseline gate: False
- Surface ablation gate: True
- Diagnostic trace rows: 1890
- Ablation trace rows: 1890

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | True |
| semantic_similarity_vs_baseline_a_significant_better | True |
| semantic_similarity_vs_baseline_c_significant_better | True |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
| formal_acceptance_eligible | False |
| partial_acceptance_eligible | True |
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
| behavioral_majority_baseline_gate | False |
| surface_ablation_gate | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 12 / 3 / 0
- User delta median: 0.0359; IQR: 0.0501
- Pair-count distribution: {'1': 4, '2': 4, '3': 3, '4': 2, '5': 1, '6': 5, '7': 2, '10': 2, '14': 4, '16': 4, '17': 1, '19': 2, '20': 1, '24': 2, '25': 1, '27': 2, '28': 1, '29': 1, '31': 1, '32': 1, '35': 1, '36': 2, '43': 2, '53': 1, '56': 1, '76': 2, '82': 1, '93': 2, '108': 1, '170': 2, '180': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0573 | 9 | 6 | 0.0539 | 0.0944 |
| `random` | 0.0812 | 12 | 3 | 0.0575 | 0.0809 |
| `temporal` | 0.0531 | 12 | 3 | 0.0240 | 0.0611 |
| `topic` | 0.0812 | 12 | 3 | 0.0575 | 0.0809 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1890 | 0.3952 | 0.5667 | 0.1423 | 0.6493 | 0.0000 | 0.0242 | 0.1141 | 0.0479 |
| `baseline_c` | 1890 | 0.0825 | 0.1243 | 0.0206 | 0.4039 | 0.0000 | 0.0994 | 0.4736 | 0.4671 |
| `baseline_b_best` | 1890 | 0.5693 | 0.6709 | 0.0000 | 0.5415 | 0.0000 | -0.0741 | 0.0484 | 0.0139 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 1890 | {'action_phrase': 0.002116, 'anchor': 0.000529, 'connector': 0.902646, 'focus': 0.814815, 'generic_focus': 0.040212, 'policy_detail': 0.978307} | {'direct_advisory': 0.019048, 'exploratory_questioning': 0.102116, 'guarded_short': 0.361376, 'warm_supportive': 0.51746} |
| `baseline_c` | 1890 | {'connector': 1.0} | {'guarded_short': 0.990476, 'warm_supportive': 0.009524} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.3846 | 0.0145 | 0.6370 | 0.7866 |
| `no_surface_profile` | 60 | 0.3774 | 0.0073 | 0.9499 | 0.6679 |
| `surface_only_default_agent` | 60 | 0.3874 | 0.0174 | 0.5050 | 0.8019 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- Personality similarity is diagnostic-only legacy cosine and is not acceptance evidence.
- State distance means: {'train_default_cosine': 0.931235, 'train_default_l2': 0.415421, 'train_full_cosine': 0.984661, 'train_full_l2': 0.116136, 'train_wrong_user_cosine': 0.949159, 'train_wrong_user_l2': 0.333378}

## Diagnostic Metrics

| Metric | direction | personality | baseline A | baseline C | acceptance evidence |
| --- | --- | --- | --- | --- | --- |
| `personality_similarity` | diagnostic_only | 1.0000 | 1.0000 | 1.0000 | False |
| `personality_trait_mae` | lower_is_better | 0.0001 | 0.0001 | 0.0000 | False |
| `personality_trait_l2` | lower_is_better | 0.0001 | 0.0001 | 0.0000 | False |

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4383 | 0.3701 | 0.0682 | 0.0004 | True |
| `behavioral_similarity_strategy` | 0.4101 | 0.4340 | -0.0239 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1662 | 0.1436 | 0.0226 | 0.1206 | False |
| `behavioral_fingerprint_similarity` | 0.7153 | 0.6954 | 0.0199 | 0.0151 | True |
| `stylistic_similarity` | 0.8657 | 0.8625 | 0.0032 | 0.1651 | False |
| `agent_state_similarity` | 0.9847 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4383 | 0.2729 | 0.1654 | 0.0000 | True |
| `behavioral_similarity_strategy` | 0.4101 | 0.3750 | 0.0352 | 0.3193 | False |
| `behavioral_similarity_action11` | 0.1662 | 0.2485 | -0.0823 | 1.0000 | False |
| `behavioral_fingerprint_similarity` | 0.7153 | 0.7165 | -0.0012 | 1.0000 | False |
| `stylistic_similarity` | 0.8657 | 0.8036 | 0.0621 | 0.0001 | True |
| `agent_state_similarity` | 0.9847 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.4383, baseline_a=0.3701, p(vs_a)=0.0004, baseline_c=0.2729, p(vs_c)=0.0000
- `behavioral_similarity_strategy`: personality=0.4101, baseline_a=0.4340, p(vs_a)=1.0000, baseline_c=0.3750, p(vs_c)=0.3193
- `agent_state_similarity`: personality=0.9847, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_action11`: personality=0.1662, baseline_a=0.1436, baseline_c=0.2485
- `behavioral_fingerprint_similarity`: personality=0.7153, baseline_a=0.6954, baseline_c=0.7165
- `stylistic_similarity`: personality=0.8657, baseline_a=0.8625, baseline_c=0.8036

## Split Weakness Summary

| Strategy | monitored | weakness | semantic vs C sig | behavioral vs C diff | behavioral p | fingerprint vs C diff | fingerprint p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `random` | True | True | True | -0.0146 | 1.0000 | -0.0068 | 1.0000 |
| `temporal` | True | True | True | 0.0527 | 0.2660 | -0.0219 | 1.0000 |
| `partner` | False | True | True | 0.1170 | 0.1047 | 0.0307 | 0.6606 |
| `topic` | False | True | True | -0.0146 | 1.0000 | -0.0068 | 1.0000 |

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | True | False | True |
| `random` | False | True | False | True |
| `temporal` | False | True | False | True |
| `topic` | False | True | False | True |

## Per-strategy gate evidence

| Strategy | comparison behavioral vs C sig | formal hard pass | classifier-gated hard-pass block |
| --- | --- | --- | --- |
| `partner` | False | False | False |
| `random` | False | False | False |
| `temporal` | False | False | False |
| `topic` | False | False | False |
