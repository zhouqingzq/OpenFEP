# M5.4 Stop-Bleed Notice

This historical artifact is fail-closed under m54_v4_stop_bleed. Repo-tracked classifier fixtures are smoke-only and do not support formal acceptance. See aggregate_report.json / m54_acceptance.json for machine-readable blockers.

# M5.4 Validation Aggregate Report

- Users: 3 (tested: 3, skipped no strategy: 0)
- Required users: 3
- Agent state: users with metric 3, skipped 0
- Topic split: {'users_with_topic_strategy_row': 3, 'users_topic_split_not_applicable': 0, 'users_topic_split_applicable': 3, 'users_topic_split_valid_for_hard_gate': 3}
- Metric version: m54_v3 (generated_action_direct_real_reply_classifier)
- Classifier 3-class gate: True
- Semantic embedding gate: True
- Statistical gate: True
- Formal acceptance eligible: True
- Behavioral hard metric degraded (soft-only): False
- Overall conclusion: fail
- Hard pass: False
- Pilot gate: True
- Split gate: True
- Partner strategy hard pass: False
- Topic strategy hard pass: False
- Formal Baseline C gate: True
- Diagnostic trace rows: 179

## Acceptance (hard metrics)

| Check | Result |
| --- | --- |
| classifier_3class_gate_passed | True |
| behavioral_hard_metric_required | True |
| semantic_similarity_vs_baseline_a_significant_better | False |
| behavioral_similarity_strategy_vs_baseline_c_significant_better | False |
| semantic_wilcoxon_valid | True |
| behavioral_wilcoxon_valid | True |
| agent_state_similarity_mean_ge_0.80 | True |
| metric_hard_pass | False |
| formal_acceptance_eligible | True |
| semantic_embedding_gate | True |
| statistical_gate | True |
| pilot_gate | True |
| split_gate_all_required_strategies | True |
| partner_gate | False |
| topic_gate | False |
| reproducibility_gate | True |
| baseline_c_full_population_implant | True |

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
- `behavioral_similarity_strategy`: personality=0.0257, baseline_a=0.0311, p(vs_a)=1.0000, baseline_c=0.0311, p(vs_c)=1.0000
- `agent_state_similarity`: personality=1.0000, baseline_a=—, p(vs_a)=—, baseline_c=—, p(vs_c)=—

## Soft Metrics
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
