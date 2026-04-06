# M4.7 Work Prompt — 仿人类记忆系统：动态调节、认知风格集成与行为验证

## 前置依赖

M4.5（数据模型 + 编码）和 M4.6（检索 + 固化）必须通过。

## 目标

1. 实现当前状态向量对记忆编码/检索的动态调节
2. 将认知风格参数（CognitiveStyleParameters）与记忆系统双向集成
3. 在真实行为任务上验证新记忆系统的功能正确性和行为可区分性
4. 为 M4.8（开放世界接口）扫清最后的记忆架构障碍

## 实现要求

### 1. 当前状态向量

在 `segmentum/memory_state.py` 中实现：

```python
@dataclass
class AgentStateVector:
    active_goals: list[str]
    recent_mood_baseline: str
    recent_dominant_tags: list[str]
    threat_level: float           # 0.0 ~ 1.0
    reward_context_active: bool
    social_context_active: bool
    last_updated: int             # cycle
```

更新规则：
- 每次新记忆写入 short 层后，从最近 N 条记忆（N=20-50）的滑动窗口统计 `recent_dominant_tags`、`recent_mood_baseline`、`threat_level`
- `active_goals` 由外部系统或测试设定
- 离线固化开始前更新一次全量快照

### 2. Salience 权重动态调节

`AgentStateVector` 对 `SalienceConfig` 权重的影响：

```python
effective_w_arousal = w_arousal × (1 + threat_level × 0.5)
effective_w_novelty = w_novelty × (1 + learning_mode_weight)
effective_w_relevance = w_relevance × (1 + goal_match_boost)
```

- `learning_mode_weight`：当 `reward_context_active=true` 时为 0.3，否则为 0.0
- `goal_match_boost`：新记忆的 semantic_tags 与 active_goals 关键词的重合度 × 0.5

所有调节规则显式、可打印、可回溯。

### 3. 认知风格参数与记忆系统的集成

关键集成点——认知风格参数影响记忆行为：

#### 3a. `uncertainty_sensitivity` → 编码时的 novelty 放大

高 uncertainty_sensitivity 的 agent 对预测误差更敏感 → novelty 评分放大：
```python
effective_novelty = novelty × (1 + uncertainty_sensitivity × 0.3)
```

#### 3b. `error_aversion` → 负面记忆的 salience 放大

高 error_aversion 的 agent 更重视负面经历 → 负 valence 记忆的 arousal 放大：
```python
if valence < 0:
    effective_arousal = arousal × (1 + error_aversion × 0.25)
```

#### 3c. `update_rigidity` → 再巩固强度调节

高 update_rigidity 的 agent 记忆更难被改写：
```python
effective_boost_access = boost_access × (1 - update_rigidity × 0.3)
# content 改写概率也受 update_rigidity 抑制
```

#### 3d. `attention_selectivity` → 编码注意力分配

高 attention_selectivity 的 agent 只记住最强信号：
```python
# 仅 top-k 最强信号获得高 encoding_attention
# k 随 attention_selectivity 降低
effective_encoding_attention = raw_attention × (1 if is_top_signal else (1 - attention_selectivity × 0.5))
```

#### 3e. `exploration_bias` → 检索时的新奇偏好

高 exploration_bias 的 agent 检索时偏好低 retrieval_count 的记忆：
```python
novelty_retrieval_bonus = exploration_bias × 0.2 × (1 / (1 + retrieval_count))
```

### 4. 行为验证任务

设计 3 个行为验证场景，证明记忆系统在不同认知风格下产生可区分的行为差异：

#### 场景 A: 威胁学习序列

给 agent 一系列包含"安全→危险→安全"模式的 episode。测量：
- 高 error_aversion agent 是否更强烈记住威胁 episode（salience 更高）
- 低 error_aversion agent 是否正常编码但不特别放大

验收：两组 agent 在威胁记忆的平均 salience 上差异 Cohen's d > 0.5

#### 场景 B: 记忆干扰与混淆

构造多个高相似度的 semantic 记忆（如同结构不同细节的事件）。测量：
- 检索时是否发生候选竞争
- 高 attention_selectivity agent 是否更少发生干扰（因为编码更聚焦）

验收：interference_risk 发生率在低 selectivity 组 > 高 selectivity 组

#### 场景 C: 长期固化与遗忘

运行 200+ cycle 的编码→衰减→固化循环。测量：
- 高 salience 记忆是否从 short 升级到 mid/long
- 低 salience 记忆是否被清除
- procedural 类型记忆 trace_strength 衰减是否最慢
- 固化后的 inferred 记忆是否有 support_count 支持

验收：
- ≥1 条 short→long 升级路径完整可追踪
- 固化后 short 层容量 < 80% max
- procedural long 的 trace_strength > 同龄 episodic long 的 trace_strength

### 5. 长期记忆三子类型的行为差异

验证 long 层内三种子类型的衰减和检索特性符合设计：

- **procedural**：trace_strength 衰减最慢，不受语义标签干扰，但动作序列可能生疏
- **语义骨架**：trace_strength 慢衰减，但**会被同结构条目干扰**
- **情景快照**：高 arousal 直接写入，abstractness 随时间上升

### 6. MemoryStore 完整集成

将 `MemoryStore` 集成到 agent 主循环中（或提供明确的集成接口），使得：
- agent 每个 cycle 的感知输入经过 `encode_memory()` → `MemoryStore.add()`
- 决策时通过 `retrieve()` 获取相关记忆
- 每 N cycle 执行 `run_consolidation_cycle()`
- `AgentStateVector` 在每次编码后更新

提供 `MemoryAwareAgent` 接口或 mixin，不破坏现有 `SegmentAgent`。

## 验收标准

### G1: state_vector_dynamics [BLOCKING]
- AgentStateVector 从最近 N 条记忆更新
- threat_level 随高 arousal + 负 valence 记忆增多而上升
- recent_dominant_tags 反映近期高频标签
- 证据：20 条记忆序列后的 state vector 快照

### G2: salience_dynamic_regulation [BLOCKING]
- 高 threat_level → effective_w_arousal 增大（可数值验证）
- reward_context_active → effective_w_novelty 增大
- goal_match → effective_w_relevance 增大
- 调节前后 salience 差异 > 0.05（给定相同输入）
- 证据：3 种状态下的 salience 计算对比

### G3: cognitive_style_memory_integration [BLOCKING]
- uncertainty_sensitivity 影响 novelty 放大（可数值验证）
- error_aversion 影响负面记忆 arousal 放大
- update_rigidity 影响再巩固强度
- attention_selectivity 影响编码注意力分配
- exploration_bias 影响检索新奇偏好
- 证据：5 个参数各 1 个数值测试用例

### G4: behavioral_scenario_A_threat_learning [BLOCKING]
- 高 vs 低 error_aversion agent 在威胁记忆 salience 上 Cohen's d > 0.5
- 证据：两组 ≥10 条记忆的 salience 分布

### G5: behavioral_scenario_B_interference [BLOCKING]
- 高相似度 semantic 记忆检索时 interference_risk=true 发生
- 高 attention_selectivity agent 干扰率低于低 selectivity agent
- 证据：两组 agent 的干扰率对比

### G6: behavioral_scenario_C_consolidation [BLOCKING]
- 200+ cycle 运行后，≥1 条 short→long 升级路径
- 固化后 short 层容量 < 80% max
- procedural long 的 trace_strength > 同龄 episodic long
- 证据：固化周期报告 + 层级分布统计

### G7: long_term_subtypes [BLOCKING]
- procedural trace_strength 衰减率 < episodic（同 elapsed_cycles）
- semantic 记忆在高相似度条目存在时发生检索干扰
- episodic 高 arousal 直接 store_level=long，abstractness 随 cycle 上升
- 证据：三种子类型的衰减/检索行为对比

### G8: integration_interface [BLOCKING]
- MemoryAwareAgent 或等价接口可用
- 每 cycle 编码 → 检索 → 再巩固的完整流程可运行
- AgentStateVector 每次编码后更新
- 不破坏现有 SegmentAgent API
- 证据：50 cycle 的集成运行日志

### G9: regression [BLOCKING]
- M4.1, M4.2, M4.3, M4.4, M4.5, M4.6 全部测试通过

### G10: report_honesty [BLOCKING]
- 所有 gate 有非空 evidence
- 无伪造通过

## 回归要求

M4.1-M4.4 + M4.5 + M4.6 全部测试通过。
