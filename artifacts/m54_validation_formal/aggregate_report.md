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
- Formal blockers: ['classifier_provisional_llm_labels', 'metric_hard_pass_failed', 'partner_gate_failed', 'topic_gate_failed']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace gate: True
- Agent-state differentiation gate: True
- Behavioral majority baseline gate: True
- Surface ablation gate: True
- Diagnostic trace rows: 1884
- Ablation trace rows: 1884

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | False |
| behavioral_hard_metric_required | True |
| semantic_similarity_vs_baseline_a_significant_better | True |
| semantic_similarity_vs_baseline_c_significant_better | True |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | True |
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
| behavioral_majority_baseline_gate | True |
| surface_ablation_gate | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 14 / 1 / 0
- User delta median: 0.0386; IQR: 0.0526
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0583 | 14 | 1 | 0.0323 | 0.0768 |
| `random` | 0.0433 | 11 | 4 | 0.0217 | 0.0630 |
| `temporal` | 0.0411 | 14 | 1 | 0.0374 | 0.0381 |
| `topic` | 0.0509 | 12 | 3 | 0.0315 | 0.0547 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1884 | 0.4007 | 0.5674 | 0.1247 | 0.6792 | 0.0016 | 0.0299 | 0.1368 | 0.0960 |
| `baseline_c` | 1884 | 0.3450 | 0.4050 | 0.1173 | 0.3824 | 0.0000 | 0.1387 | 0.0743 | 0.0436 |
| `baseline_b_best` | 1884 | 0.5520 | 0.6290 | 0.0000 | 0.5385 | 0.0000 | -0.0658 | 0.0381 | 0.0308 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 1884 | {'action_phrase': 0.0138, 'anchor': 0.005839, 'connector': 0.965499, 'focus': 0.72293, 'generic_focus': 0.011146, 'policy_detail': 1.136412} | {'direct_advisory': 0.016454, 'exploratory_questioning': 0.019639, 'guarded_short': 0.457537, 'warm_supportive': 0.506369} |
| `baseline_c` | 1884 | {'connector': 1.0} | {'guarded_short': 0.42569, 'warm_supportive': 0.57431} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.4050 | 0.0243 | 0.6392 | 0.8104 |
| `no_surface_profile` | 60 | 0.3929 | 0.0123 | 0.9210 | 0.6863 |
| `surface_only_default_agent` | 60 | 0.4043 | 0.0236 | 0.5008 | 0.8098 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- Personality similarity is diagnostic-only legacy cosine and is not acceptance evidence.
- State distance means: {'train_default_cosine': 0.944713, 'train_default_l2': 0.372433, 'train_full_cosine': 0.995034, 'train_full_l2': 0.076561, 'train_wrong_user_cosine': 0.964163, 'train_wrong_user_l2': 0.275078}

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
| `semantic_similarity` | 0.4291 | 0.3806 | 0.0484 | 0.0001 | True |
| `behavioral_similarity_strategy` | 0.4094 | 0.4456 | -0.0362 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.1783 | 0.1513 | 0.0270 | 0.2315 | False |
| `behavioral_fingerprint_similarity` | 0.7592 | 0.7113 | 0.0479 | 0.0008 | True |
| `stylistic_similarity` | 0.8667 | 0.8661 | 0.0006 | 0.4890 | False |
| `agent_state_similarity` | 0.9950 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4291 | 0.2481 | 0.1810 | 0.0000 | True |
| `behavioral_similarity_strategy` | 0.4094 | 0.2970 | 0.1124 | 0.0319 | True |
| `behavioral_similarity_action11` | 0.1783 | 0.1193 | 0.0590 | 0.0042 | True |
| `behavioral_fingerprint_similarity` | 0.7592 | 0.7593 | -0.0002 | 1.0000 | False |
| `stylistic_similarity` | 0.8667 | 0.7862 | 0.0805 | 0.0000 | True |
| `agent_state_similarity` | 0.9950 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.4291, baseline_a=0.3806, p(vs_a)=0.0001, baseline_c=0.2481, p(vs_c)=0.0000
- `behavioral_similarity_strategy`: personality=0.4094, baseline_a=0.4456, p(vs_a)=1.0000, baseline_c=0.2970, p(vs_c)=0.0319
- `agent_state_similarity`: personality=0.9950, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_action11`: personality=0.1783, baseline_a=0.1513, baseline_c=0.1193
- `behavioral_fingerprint_similarity`: personality=0.7592, baseline_a=0.7113, baseline_c=0.7593
- `stylistic_similarity`: personality=0.8667, baseline_a=0.8661, baseline_c=0.7862

## Split Weakness Summary

| Strategy | monitored | weakness | semantic vs C sig | behavioral vs C diff | behavioral p | fingerprint vs C diff | fingerprint p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `random` | True | True | True | 0.1495 | 0.0503 | 0.0152 | 0.8486 |
| `temporal` | True | True | True | -0.0343 | 1.0000 | -0.0615 | 1.0000 |
| `partner` | False | True | True | 0.2069 | 0.0207 | 0.0434 | 0.2622 |
| `topic` | False | True | True | 0.1274 | 0.0422 | 0.0022 | 0.9465 |

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | False | True | True | True |
| `random` | False | True | False | True |
| `temporal` | False | True | False | True |
| `topic` | False | True | True | True |

## Per-strategy gate evidence

| Strategy | comparison behavioral vs C sig | formal hard pass | classifier-gated hard-pass block |
| --- | --- | --- | --- |
| `partner` | True | False | True |
| `random` | False | False | False |
| `temporal` | False | False | False |
| `topic` | True | False | True |
