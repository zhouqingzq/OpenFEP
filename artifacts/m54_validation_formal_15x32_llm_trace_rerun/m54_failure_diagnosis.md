# M5.4 Failure Diagnosis

## Artifact Sanity

- Passed: True
- Missing files: []
- Diagnostic trace rows: 1884
- Ablation trace rows: 1884
- Stale ablation-vs-Baseline-C warning: False

## Baseline C

- Baseline C win rate: 0.733015
- Baseline C win turns: 1381
- Interpretation: {'population_surface_win_rate': 1.0, 'short_reply_template_win_rate': 1.0, 'population_averaged_state_like_win_rate': 0.0}
- Surface metric suspect: True (all_baseline_c_wins_are_short_population_surface)
- Filtered semantic deltas: {'all_rows': {'rows': 1884, 'mean_personality_vs_c_delta': -0.136185, 'median_personality_vs_c_delta': -0.111059, 'personality_win_rate': 0.265393}, 'exclude_baseline_c_short_replies': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}, 'exclude_population_surface': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}, 'length_bucket_matched_to_real': {'rows': 1399, 'mean_personality_vs_c_delta': -0.164367, 'median_personality_vs_c_delta': -0.133118, 'personality_win_rate': 0.225876}, 'exclude_short_or_population_surface': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}}

Worst user-strategy deltas:
| user_uid | strategy | turns | mean_delta | median_delta | personality_win_rate |
| --- | --- | --- | --- | --- | --- |
| 2637171 | random | 2 | -0.426234 | -0.426234 | 0.0 |
| 2637171 | topic | 2 | -0.426234 | -0.426234 | 0.0 |
| 2637101 | temporal | 10 | -0.255082 | -0.162241 | 0.1 |
| 2637134 | temporal | 25 | -0.244263 | -0.260796 | 0.12 |
| 2637470 | partner | 212 | -0.220559 | -0.175552 | 0.179245 |
| 2637171 | temporal | 8 | -0.197185 | -0.245732 | 0.25 |
| 2637379 | partner | 10 | -0.175161 | -0.146589 | 0.4 |
| 2637470 | random | 85 | -0.17469 | -0.144048 | 0.235294 |
| 2637470 | topic | 85 | -0.17469 | -0.144048 | 0.235294 |
| 2637354 | temporal | 25 | -0.171153 | -0.214226 | 0.28 |

## Ablations

- Lift means: {'trait_policy_lift_full_minus_no_policy': 0.000458, 'surface_lift_full_minus_no_surface': 0.033876, 'surface_anchor_standalone_lift_vs_baseline_c': -0.136661}
- Lift medians: {'trait_policy_lift_full_minus_no_policy': 0.0, 'surface_lift_full_minus_no_surface': 0.038344, 'surface_anchor_standalone_lift_vs_baseline_c': -0.117437}
- Beating-full clusters: {'no_policy_trait_bias': {'real=escape|template=agree:1|surface=temporal:train|len=medium': 45, 'real=escape|template=agree:1|surface=random:train|len=medium': 43, 'real=escape|template=agree:2|surface=partner:train|len=medium': 40, 'real=escape|template=agree:2|surface=random:train|len=medium': 35, 'real=escape|template=agree:1|surface=topic:train|len=medium': 35, 'real=escape|template=agree:0|surface=random:train|len=medium': 31, 'real=escape|template=agree:2|surface=topic:train|len=medium': 30, 'real=escape|template=agree:0|surface=partner:train|len=medium': 28, 'real=escape|template=agree:0|surface=topic:train|len=medium': 28, 'real=escape|template=agree:1|surface=partner:train|len=medium': 27, 'real=escape|template=agree:2|surface=temporal:train|len=medium': 26, 'real=escape|template=agree:0|surface=temporal:train|len=medium': 26, 'real=escape|template=agree:1|surface=random:train|len=short': 21, 'real=escape|template=agree:0|surface=random:train|len=short': 20, 'real=escape|template=minimal_response:1|surface=partner:train|len=short': 20, 'real=escape|template=agree:0|surface=topic:train|len=short': 19}, 'surface_only_default_agent': {'real=escape|template=agree:2|surface=partner:train|len=medium': 53, 'real=escape|template=agree:0|surface=random:train|len=medium': 43, 'real=escape|template=agree:0|surface=partner:train|len=medium': 40, 'real=escape|template=agree:0|surface=topic:train|len=medium': 35, 'real=escape|template=agree:1|surface=partner:train|len=medium': 33, 'real=escape|template=agree:1|surface=topic:train|len=medium': 31, 'real=escape|template=agree:1|surface=random:train|len=medium': 29, 'real=escape|template=agree:0|surface=temporal:train|len=medium': 28, 'real=escape|template=agree:1|surface=temporal:train|len=medium': 26, 'real=escape|template=agree:2|surface=temporal:train|len=medium': 23, 'real=escape|template=elaborate:2|surface=partner:train|len=medium': 21, 'real=escape|template=elaborate:0|surface=partner:train|len=medium': 19, 'real=escape|template=elaborate:1|surface=partner:train|len=medium': 19, 'real=escape|template=agree:0|surface=random:train|len=short': 17, 'real=escape|template=share_opinion:2|surface=random:train|len=medium': 17, 'real=escape|template=agree:1|surface=partner:train|len=short': 16}}

## Behavior

- Diagnosis tags: ['classifier_definition_issue', 'state_modeling_issue']
- Real strategy distribution: {'escape': 1616, 'explore': 80, 'exploit': 188}
- Real majority coverage: 0.857749
- Classifier cue override / cue feature / without-cue F1: 0.0 / 0.646667 / 0.845776
- Balanced recall: {'personality': 0.324893, 'baseline_c': 0.333333, 'train_majority': 0.333333}
- Majority vs personality: {'majority_behavioral_strategy_mean': 0.887857, 'personality_behavioral_strategy_mean': 0.222103, 'majority_minus_personality': 0.665754}
- State collapse: {'evaluated': 60, 'mean_strategy_majority_coverage': 0.886691, 'policy_dominant_match_rate': 1.0}
- Label examples: {'escape': [{'user_uid': 2637101, 'strategy': 'random', 'real_action': 'minimal_response', 'real_text': '4萬鑽2500元', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'minimal_response', 'real_text': '匯這帳號', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'minimal_response', 'real_text': '沒辦法啦', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'minimal_response', 'real_text': '這是官方給的', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}], 'exploit': [{'user_uid': 2637215, 'strategy': 'random', 'real_action': 'joke', 'real_text': '抒發情緒壓力也不錯', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'empathize', 'real_text': '我能體會', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'empathize', 'real_text': '妳自己也需要人關心', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'elaborate', 'real_text': '還關心別人', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}], 'explore': [{'user_uid': 2637215, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '怎麼了？發生什麼事', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '妳是？', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '怎麼了啊？', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637215, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '還在探喔？', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}]}
