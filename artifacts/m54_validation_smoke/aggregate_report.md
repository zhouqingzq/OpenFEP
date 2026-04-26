# M5.4 Validation Aggregate Report

- Users: 15 (tested: 15, skipped no strategy: 0)
- Required users: 15
- Agent state: users with metric 15, skipped 0
- Topic split: {'users_with_topic_strategy_row': 15, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 15, 'users_topic_split_valid_for_hard_gate': 15}
- Metric version: m54_v9_behavioral_policy_lift (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: False
- Classifier evidence tier: repo_fixture_smoke
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: False
- Partial acceptance eligible: True
- Behavioral hard metric degraded (soft-only): True
- Overall conclusion: partial
- Hard pass: False
- Formal blockers: ['classifier_fixture_only']
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: True
- Topic strategy hard pass: True
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
| behavioral_hard_metric_required | False |
| semantic_similarity_vs_baseline_a_significant_better | True |
| semantic_similarity_vs_baseline_c_significant_better | True |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | True |
| formal_acceptance_eligible | False |
| partial_acceptance_eligible | True |
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
| behavioral_majority_baseline_gate | True |
| surface_ablation_gate | True |

## Semantic Delta Diagnostics

- Users positive/negative/zero: 13 / 2 / 0
- User delta median: 0.0254; IQR: 0.0377
- Pair-count distribution: {'1': 1, '2': 7, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '10': 4, '11': 4, '13': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 3, '22': 1, '25': 2, '26': 2, '30': 1, '33': 1, '36': 3, '40': 1, '47': 1, '48': 1, '53': 1, '56': 1, '58': 1, '61': 1, '67': 2, '71': 3, '74': 1, '85': 2, '92': 1, '212': 1}

| Strategy | mean P-A delta | positive | negative | median | IQR |
| --- | --- | --- | --- | --- | --- |
| `partner` | 0.0495 | 12 | 3 | 0.0241 | 0.0639 |
| `random` | 0.0369 | 11 | 4 | 0.0251 | 0.0669 |
| `temporal` | 0.0285 | 11 | 4 | 0.0302 | 0.0538 |
| `topic` | 0.0422 | 12 | 3 | 0.0251 | 0.0555 |

## Baseline Audit Diagnostics

- Wrong-user masked warning: False
- Baseline C too-close warning: False ()
- Baseline C too-weak warning (diagnostic-only): False ()

| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_a` | 1884 | 0.4660 | 0.6927 | 0.1539 | 0.6888 | 0.0016 | 0.0182 | 0.0682 | 0.0480 |
| `baseline_c` | 1884 | 0.1534 | 0.7176 | 0.0504 | 0.4483 | 0.0000 | 0.1152 | 0.3629 | 0.0280 |
| `baseline_b_best` | 1884 | 0.6024 | 0.7378 | 0.0000 | 0.3854 | 0.0000 | -0.0782 | 0.0244 | 0.0160 |

## Profile Expression Diagnostics

| Surface | rows | expression source rates | rhetorical move rates |
| --- | --- | --- | --- |
| `personality` | 1884 | {'action_phrase': 0.0138, 'anchor': 0.004777, 'connector': 0.965499, 'focus': 0.866242, 'generic_focus': 0.011677, 'policy_detail': 1.281316} | {'direct_advisory': 0.085987, 'exploratory_questioning': 0.016985, 'guarded_short': 0.031847, 'warm_supportive': 0.86518} |
| `baseline_c` | 1884 | {'connector': 1.0} | {'warm_supportive': 1.0} |

## Ablation Diagnostics

| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |
| --- | --- | --- | --- | --- | --- |
| `no_policy_trait_bias` | 60 | 0.4053 | 0.0246 | 0.6916 | 0.8148 |
| `no_surface_profile` | 60 | 0.3897 | 0.0090 | 0.9369 | 0.7027 |
| `surface_only_default_agent` | 60 | 0.4043 | 0.0236 | 0.5275 | 0.8160 |

## State Saturation Diagnostics

- Personality similarity diagnostic-only saturation warning: True
- State distance means: {'train_default_cosine': 0.952409, 'train_default_l2': 0.353787, 'train_full_cosine': 0.998752, 'train_full_l2': 0.047235, 'train_wrong_user_cosine': 0.97716, 'train_wrong_user_l2': 0.233577}

## Debug Readiness Gate

- Passed: True
- Checks: {'train_default_l2_positive': True, 'train_wrong_user_l2_positive': True, 'wrong_user_masked_warning_false': True, 'no_surface_not_better_than_full': True}

## Comparisons vs baseline A (directional)

| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4199 | 0.3806 | 0.0393 | 0.0006 | True |
| `behavioral_similarity_strategy` | 0.4559 | 0.5058 | -0.0498 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0645 | 0.0437 | 0.0209 | 0.1869 | False |
| `stylistic_similarity` | 0.8586 | 0.8661 | -0.0075 | 1.0000 | False |
| `personality_similarity` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9988 | — | — | — | — |

## Comparisons vs baseline C (directional)

| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |
| --- | --- | --- | --- | --- | --- |
| `semantic_similarity` | 0.4199 | 0.2498 | 0.1701 | 0.0000 | True |
| `behavioral_similarity_strategy` | 0.4559 | 0.4760 | -0.0200 | 1.0000 | False |
| `behavioral_similarity_action11` | 0.0645 | 0.3839 | -0.3194 | 1.0000 | False |
| `stylistic_similarity` | 0.8586 | 0.7874 | 0.0712 | 0.0000 | True |
| `personality_similarity` | 1.0000 | 1.0000 | -0.0000 | 1.0000 | False |
| `agent_state_similarity` | 0.9988 | — | — | — | — |

## Hard metric rows (summary)
- `semantic_similarity`: personality=0.4199, baseline_a=0.3806, p(vs_a)=0.0006, baseline_c=0.2498, p(vs_c)=0.0000
- `agent_state_similarity`: personality=0.9988, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
- `behavioral_similarity_strategy`: personality=0.4559, baseline_a=0.5058, baseline_c=0.4760
- `behavioral_similarity_action11`: personality=0.0645, baseline_a=0.0437, baseline_c=0.3839
- `stylistic_similarity`: personality=0.8586, baseline_a=0.8661, baseline_c=0.7874
- `personality_similarity`: personality=1.0000, baseline_a=1.0000, baseline_c=1.0000

## Per-strategy hard pass

| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |
| --- | --- | --- | --- | --- |
| `partner` | True | True | False | True |
| `random` | True | True | False | True |
| `temporal` | True | True | False | True |
| `topic` | True | True | False | True |

## Per-strategy gate evidence

| Strategy | comparison behavioral vs C sig | formal hard pass | classifier-gated hard-pass block |
| --- | --- | --- | --- |
| `partner` | False | True | False |
| `random` | False | True | False |
| `temporal` | False | True | False |
| `topic` | False | True | False |

## Direction Auto-Escalation

- Applied: True
- Requested max users: 5
- Pilot required users: 15
- Rerun user count: 15
- Note: Earlier 10x32 outputs without this field should be treated as pre-final-patch or pilot-escalation stale artifacts.
