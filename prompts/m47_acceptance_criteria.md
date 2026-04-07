# M4.7 验收标准 — 动态调节、认知风格集成与行为验证

## 前置依赖

M4.5 + M4.6 必须 PASS。

## Gate 列表

### G1: state_vector_dynamics [BLOCKING]
- AgentStateVector 包含 active_goals, recent_mood_baseline, recent_dominant_tags, identity_active_themes, threat_level, reward_context_active, social_context_active, last_updated
- 滑动窗口 N=20-50 条记忆更新
- threat_level 随负面输入增多而上升
- identity_active_themes 反映近期高频的自我叙事/角色连续性主题
- 证据：序列输入后的 state vector 快照

### G2: salience_dynamic_regulation [BLOCKING]
- 高 threat → w_arousal 增大系数 = 1 + threat_level × 0.5
- reward_context → w_novelty 增大系数 = 1.3
- goal_match → w_relevance 增大系数 = 1 + overlap × 0.5
- identity/self match → `effective_w_relevance` 或 `relevance_self` 增大，且贡献可打印
- 调节前后 salience 差异可数值验证（固定输入，变 state vector）
- 证据：3 种状态下的对比计算

### G3: cognitive_style_memory_integration [BLOCKING]
- 5 个认知参数对记忆行为的影响各有独立数值测试
- 影响方向和幅度符合设计文档
- 至少 1 个交互用例体现认知风格对 identity-relevant memory 稳定性的间接影响
- 证据：参数 0.0 vs 1.0 时的输出差异

### G4: behavioral_scenario_A [BLOCKING]
- 高 error_aversion (0.8) vs 低 (0.2) 在威胁记忆 salience 上 Cohen's d > 0.5
- 样本量 ≥ 10 条/组
- 证据：两组 salience 分布 + 效应量

### G5: behavioral_scenario_B [BLOCKING]
- 构造 ≥ 5 条 semantic_tags 高重合的记忆
- 检索时 interference_risk=true 至少发生 1 次
- 高 attention_selectivity (0.8) 组干扰率 < 低 (0.2) 组
- 证据：干扰率对比

### G6: behavioral_scenario_C [BLOCKING]
- 200+ cycle 运行
- ≥ 1 条完整 short→mid→long 或 short→long 升级路径
- 固化后 short 层 < 80% max_episodes
- procedural long trace_strength > 同龄 episodic long trace_strength
- 证据：固化报告 + 层级分布

### G7: long_term_subtypes [BLOCKING]
- procedural λ_trace < episodic λ_trace（long 层）
- semantic 检索干扰可观测
- episodic 高 arousal 直接 long，abstractness 随 cycle 上升
- semantic skeleton 具备较强可迁移性，且易受同结构条目干扰
- identity-relevant semantic / autobiographical cluster 不被短期噪声轻易稀释
- 证据：子类型行为对比表

### Gx: identity_continuity_retention [BLOCKING]
- 在低 arousal / 低 novelty 条件下，高 identity relevance 记忆仍可进入 mid 或 long
- 长周期后仍可被检索
- 能支持 self-related recall 或 identity narrative continuity
- 提供与高 novelty、低 identity relevance 噪声组的对照比较
- 证据：场景 D 的保留率、检索结果与对照统计

### G8: integration_interface [BLOCKING]
- 完整 cycle 流程：encode → store → retrieve → reconsolidate → consolidate
- AgentStateVector 自动更新
- 不破坏 SegmentAgent
- 证据：50 cycle 运行日志

### G9: regression [BLOCKING]
- M4.1-M4.4 + M4.5 + M4.6 全部通过

### G10: report_honesty [BLOCKING]
- 所有 gate 有非空 evidence
- 无伪造通过

## 效应量阈值

| 指标 | 阈值 |
|------|------|
| 威胁学习 Cohen's d | > 0.5 |
| 干扰率差异（高 vs 低 selectivity）| 方向正确即可 |
| salience 动态调节差异 | > 0.05 |
| 固化后 short 层占比 | < 80% |

## 回归

M4.1, M4.2, M4.3, M4.4, M4.5, M4.6 全部测试通过。
