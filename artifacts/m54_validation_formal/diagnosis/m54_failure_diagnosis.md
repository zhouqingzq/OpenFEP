# M5.4 Failure Diagnosis

## Artifact Sanity

- Passed: True
- Missing files: []
- Diagnostic trace rows: 1884
- Ablation trace rows: 1884
- Stale ablation-vs-Baseline-C warning: False

## Baseline C

- Baseline C win rate: 0.206476
- Baseline C win turns: 389
- Interpretation: {'population_surface_win_rate': 1.0, 'short_reply_template_win_rate': 0.0, 'population_averaged_state_like_win_rate': 0.0}
- Surface metric suspect: True (filtered_personality_delta_positive)
- Filtered semantic deltas: {'all_rows': {'rows': 1884, 'mean_personality_vs_c_delta': 0.138689, 'median_personality_vs_c_delta': 0.128479, 'personality_win_rate': 0.793524}, 'exclude_baseline_c_short_replies': {'rows': 1884, 'mean_personality_vs_c_delta': 0.138689, 'median_personality_vs_c_delta': 0.128479, 'personality_win_rate': 0.793524}, 'exclude_population_surface': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}, 'length_bucket_matched_to_real': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}, 'exclude_short_or_population_surface': {'rows': 0, 'mean_personality_vs_c_delta': 0.0, 'median_personality_vs_c_delta': 0.0, 'personality_win_rate': 0.0}}

Worst user-strategy deltas:
| user_uid | strategy | turns | mean_delta | median_delta | personality_win_rate |
| --- | --- | --- | --- | --- | --- |
| 2637379 | partner | 10 | 0.01504 | 0.015478 | 0.5 |
| 2637101 | random | 19 | 0.043465 | -0.009425 | 0.473684 |
| 2637101 | topic | 19 | 0.043465 | -0.009425 | 0.473684 |
| 2637134 | temporal | 25 | 0.079408 | 0.079115 | 0.72 |
| 2637101 | partner | 20 | 0.080499 | 0.006816 | 0.5 |
| 2637379 | random | 11 | 0.08087 | 0.061805 | 0.727273 |
| 2637379 | topic | 11 | 0.08087 | 0.061805 | 0.727273 |
| 2637134 | partner | 17 | 0.083252 | 0.084042 | 0.705882 |
| 2637532 | temporal | 8 | 0.083283 | 0.144447 | 0.75 |
| 2637532 | random | 71 | 0.093829 | 0.103808 | 0.704225 |

## Ablations

- Lift means: {'trait_policy_lift_full_minus_no_policy': 0.013975, 'surface_lift_full_minus_no_surface': 0.015379, 'surface_anchor_standalone_lift_vs_baseline_c': 0.122969}
- Lift medians: {'trait_policy_lift_full_minus_no_policy': 0.010322, 'surface_lift_full_minus_no_surface': 0.015854, 'surface_anchor_standalone_lift_vs_baseline_c': 0.109654}
- Beating-full clusters: {'no_policy_trait_bias': {'real=escape|template=agree:0|surface=partner:train:no_policy_anchor_only|len=medium': 27, 'real=escape|template=agree:2|surface=partner:train:no_policy_anchor_only|len=medium': 26, 'real=escape|template=agree:2|surface=random:train:no_policy_anchor_only|len=medium': 23, 'real=escape|template=agree:0|surface=random:train:no_policy_anchor_only|len=medium': 23, 'real=escape|template=agree:2|surface=topic:train:no_policy_anchor_only|len=medium': 22, 'real=escape|template=agree:0|surface=topic:train:no_policy_anchor_only|len=medium': 22, 'real=escape|template=agree:1|surface=random:train:no_policy_anchor_only|len=medium': 21, 'real=escape|template=agree:1|surface=partner:train:no_policy_anchor_only|len=medium': 20, 'real=exploit|template=agree:0|surface=random:train:no_policy_anchor_only|len=medium': 19, 'real=escape|template=agree:1|surface=topic:train:no_policy_anchor_only|len=medium': 19, 'real=exploit|template=agree:1|surface=topic:train:no_policy_anchor_only|len=medium': 18, 'real=exploit|template=agree:0|surface=partner:train:no_policy_anchor_only|len=medium': 16, 'real=exploit|template=agree:0|surface=topic:train:no_policy_anchor_only|len=medium': 16, 'real=escape|template=agree:1|surface=temporal:train:no_policy_anchor_only|len=medium': 16, 'real=exploit|template=agree:2|surface=random:train:no_policy_anchor_only|len=medium': 16, 'real=exploit|template=agree:1|surface=random:train:no_policy_anchor_only|len=medium': 15}, 'surface_only_default_agent': {'real=escape|template=agree:2|surface=partner:train:surface_only_anchor_only|len=medium': 24, 'real=escape|template=agree:0|surface=random:train:surface_only_anchor_only|len=medium': 20, 'real=escape|template=agree:0|surface=topic:train:surface_only_anchor_only|len=medium': 18, 'real=escape|template=agree:0|surface=partner:train:surface_only_anchor_only|len=medium': 17, 'real=exploit|template=agree:0|surface=random:train:surface_only_anchor_only|len=medium': 16, 'real=exploit|template=agree:0|surface=topic:train:surface_only_anchor_only|len=medium': 16, 'real=escape|template=elaborate:1|surface=partner:train:surface_only_anchor_only|len=medium': 15, 'real=escape|template=agree:2|surface=random:train:surface_only_anchor_only|len=medium': 14, 'real=exploit|template=agree:1|surface=partner:train:surface_only_anchor_only|len=medium': 13, 'real=escape|template=agree:0|surface=random:train:surface_only_anchor_only|len=short': 13, 'real=escape|template=agree:1|surface=random:train:surface_only_anchor_only|len=medium': 13, 'real=escape|template=agree:0|surface=topic:train:surface_only_anchor_only|len=short': 13, 'real=escape|template=elaborate:2|surface=partner:train:surface_only_anchor_only|len=medium': 13, 'real=escape|template=elaborate:0|surface=partner:train:surface_only_anchor_only|len=medium': 12, 'real=escape|template=agree:1|surface=temporal:train:surface_only_anchor_only|len=medium': 11, 'real=escape|template=agree:1|surface=topic:train:surface_only_anchor_only|len=medium': 11}}

## Behavior

- Diagnosis tags: []
- Real strategy distribution: {'exploit': 648, 'escape': 1114, 'explore': 122}
- Real majority coverage: 0.591295
- Classifier cue override / cue feature / without-cue F1: 0.0 / 0.26 / 0.704732
- Balanced recall: {'personality': 0.355465, 'baseline_c': 0.310959, 'train_majority': 0.352915}
- Majority vs personality: {'majority_behavioral_strategy_mean': 0.363177, 'personality_behavioral_strategy_mean': 0.409377, 'majority_minus_personality': -0.0462}
- State collapse: {'evaluated': 60, 'mean_strategy_majority_coverage': 0.628867, 'policy_dominant_match_rate': 0.916667}
- Label examples: {'escape': [{'user_uid': 2637101, 'strategy': 'random', 'real_action': 'minimal_response', 'real_text': '沒辦法啦', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'disengage', 'real_text': '5570161', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'exploit'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'disengage', 'real_text': '感謝', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'exploit'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'disengage', 'real_text': '我要睡一下了', 'personality_strategy': 'explore', 'baseline_c_strategy': 'escape'}], 'exploit': [{'user_uid': 2637101, 'strategy': 'random', 'real_action': 'elaborate', 'real_text': '4萬鑽2500元', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'agree', 'real_text': '匯這帳號', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'elaborate', 'real_text': '這是官方給的', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'random', 'real_action': 'agree', 'real_text': '好', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'exploit'}], 'explore': [{'user_uid': 2637101, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '鑽入這號嗎', 'personality_strategy': 'explore', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'partner', 'real_action': 'ask_question', 'real_text': '鑽入這號嗎', 'personality_strategy': 'exploit', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637101, 'strategy': 'topic', 'real_action': 'ask_question', 'real_text': '鑽入這號嗎', 'personality_strategy': 'explore', 'baseline_c_strategy': 'escape'}, {'user_uid': 2637134, 'strategy': 'random', 'real_action': 'ask_question', 'real_text': '合唱入圍有二個，那還有邀請卡嗎', 'personality_strategy': 'explore', 'baseline_c_strategy': 'exploit'}]}
