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
    identity_active_themes: list[str]
    threat_level: float           # 0.0 ~ 1.0
    reward_context_active: bool
    social_context_active: bool
    last_updated: int             # cycle
```

更新规则：
- 每次新记忆写入 short 层后，从最近 N 条记忆（N=20-50）的滑动窗口统计 `recent_dominant_tags`、`recent_mood_baseline`、`threat_level`
- 同时追踪近期与自我叙事、角色连续性、长期承诺相关的高频主题，更新 `identity_active_themes`
- `active_goals` 由外部系统或测试设定
- 离线固化开始前更新一次全量快照

### 2. Salience 权重动态调节

`AgentStateVector` 对 `SalienceConfig` 权重的影响：

```python
effective_w_arousal = w_arousal × (1 + threat_level × 0.5)
effective_w_novelty = w_novelty × (1 + learning_mode_weight)
effective_w_relevance = w_relevance × (1 + goal_match_boost)
effective_relevance_self = relevance_self × (1 + identity_match_boost)
```

- `learning_mode_weight`：当 `reward_context_active=true` 时为 0.3，否则为 0.0
- `goal_match_boost`：新记忆的 semantic_tags 与 active_goals 关键词的重合度 × 0.5
- `identity_match_boost`：新记忆与 `identity_active_themes` 或长期自我叙事关键词的重合度 × 0.5

所有调节规则显式、可打印、可回溯。
当新记忆与 `identity_active_themes` 或长期自我叙事高度相关时，允许通过提高 `effective_w_relevance` 或直接提高 `relevance_self` 来提升保留优先级，并且必须打印 identity/self 贡献。

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

补充约束：
- 认知风格参数不得只影响 threat / novelty / retrieval 偏好，也应允许通过 attention allocation 或 update rigidity 间接影响 identity-relevant memory 的稳定性。

### 4. 行为验证任务

设计 5 个行为验证场景，证明记忆系统在不同认知风格下产生可区分的行为差异：

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

#### 场景 D: 自我连续性与低显著性保留

构造两类输入并运行长期固化：
- 低 arousal、低 novelty、低即时 reward，但高 identity relevance 的事件，如长期承诺、角色变化、稳定关系、自我叙述节点
- 高 novelty、但低 identity relevance 的噪声事件

测量：
- 长期固化后，高 identity relevance 组的 mid/long 保留率是否高于对照噪声组
- 检索时这些记忆是否更容易支持 self-related recall / identity narrative continuity

验收：
- 高 identity relevance 组的 long/mid 保留率显著高于对照噪声组
- 至少 1 条 identity-relevant 记忆在较长周期后仍可被检索并影响 self-related recall
- 必须提供正例/反例对照，验证两类输入不会被系统打成同一种“重要”

#### 场景 E: 结构相似导致的自然误归因

构造：
- 多条高结构相似、但低层细节不同的 episodic 记忆
- 随时间运行细节衰减，使 weak anchors / 细节槽位更易丢失
- 检索时只提供部分线索，使候选竞争与受约束重构同时发生

测量：
- 是否出现“非随机、结构性”的误补全或误归因
- 错误是否更多落在 weak anchors / 细节槽位，而不是 locked / strong anchors
- 错误是否可由 `candidates`、`competing_interpretations`、`source_trace` / `reconstruction_trace` 回溯解释

验收：
- 至少 1 个错误案例不是随机噪声注入造成，而是结构相似 + 细节衰减 + 重构竞争自然涌现
- 误归因主要发生在 weak anchors / 细节槽位，不得主要破坏 locked / strong anchors
- 错误必须可由检索分数、候选竞争和重构 trace 解释

### 5. 长期记忆三子类型的行为差异

验证 long 层内三种子类型的衰减和检索特性符合设计：

- **procedural**：trace_strength 衰减最慢，不受语义标签干扰，但动作序列可能生疏
- **语义骨架**：trace_strength 慢衰减，具备较强可迁移性，但**会被同结构条目干扰**
- **情景快照**：高 arousal 直接写入，abstractness 随时间上升
- **identity-relevant semantic / autobiographical cluster**：不应被短期高新奇噪声轻易稀释

### 6. MemoryStore 完整集成

将 `MemoryStore` 集成到 agent 主循环中（或提供明确的集成接口），使得：
- agent 每个 cycle 的感知输入经过 `encode_memory()` → `MemoryStore.add()`
- 决策时通过 `retrieve()` 获取相关记忆
- 每 N cycle 执行 `run_consolidation_cycle()`
- `AgentStateVector` 在每次编码后更新

提供 `MemoryAwareAgent` 接口或 mixin，不破坏现有 `SegmentAgent`。

### 7. Implementation Anti-patterns / 禁止退化行为

- 不得把 `relevance_self` 简化为“是否包含第一人称”或“是否与当前 agent 有关”。
- 不得把 `retrieve()` 退化成 top-1 条目直返。
- 不得把 forgetting 默认实现成只靠 cleanup 删除。
- 不得在冲突出现时默认直接覆盖旧 episodic。
- 不得让 `source_confidence` 与 `reality_confidence` 永远同步变化。
- 不得把所有锚点默认设为 `weak`。
- 不得把 procedural 只实现成普通文本摘要。
- 不得用随机噪声注入来伪装“人类式记忆错误”。

### 8. 失败示例说明

以下情况即使部分场景跑通，也应视为实现跑偏：
- identity-relevant 低 arousal 事件与高 novelty 噪声长期保留效果没有差异。
- `recall_hypothesis` 实际只是 `entry.content` 原样回传，缺少 trace 与竞争信息。
- semantic skeleton 没有 support 链，或被当成另一条无来源的普通 semantic 记忆。
- 所有 mid/long 遗忘最终都靠删除完成。
- procedural 没有显式步骤结构。
- 所谓“人类式错误”只能靠随机注噪触发，无法通过候选竞争与重构 trace 解释。

## 验收标准

### G1: state_vector_dynamics [BLOCKING]
- AgentStateVector 从最近 N 条记忆更新
- threat_level 随高 arousal + 负 valence 记忆增多而上升
- recent_dominant_tags 反映近期高频标签
- `identity_active_themes` 或等价追踪机制反映近期高频的自我叙事/角色连续性主题
- 证据：20 条记忆序列后的 state vector 快照

### G2: salience_dynamic_regulation [BLOCKING]
- 高 threat_level → effective_w_arousal 增大（可数值验证）
- reward_context_active → effective_w_novelty 增大
- goal_match → effective_w_relevance 增大
- identity/self match → `effective_w_relevance` 或 `relevance_self` 增大，且贡献可打印
- 调节前后 salience 差异 > 0.05（给定相同输入）
- 证据：3 种状态下的 salience 计算对比
- 至少 1 组正例/反例验证：高 identity continuity 低 arousal 事件与高 novelty 低 self relevance 噪声，不会被打成同一种重要性

### G3: cognitive_style_memory_integration [BLOCKING]
- uncertainty_sensitivity 影响 novelty 放大（可数值验证）
- error_aversion 影响负面记忆 arousal 放大
- update_rigidity 影响再巩固强度
- attention_selectivity 影响编码注意力分配
- exploration_bias 影响检索新奇偏好
- 认知风格至少有 1 个交互用例影响 identity-relevant memory 的稳定性
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
- semantic skeleton 表现出较强可迁移性，且易受同结构条目干扰
- identity-relevant semantic / autobiographical cluster 不会被短期噪声轻易稀释
- 证据：三种子类型的衰减/检索行为对比

### Gx: identity_continuity_retention [BLOCKING]
- 在低 arousal / 低 novelty 条件下，高 identity relevance 记忆仍可进入 mid 或 long
- 长周期后仍可被检索
- 能支持 self-related recall 或 identity narrative continuity
- 提供与高 novelty、低 identity relevance 噪声组的对照比较
- 证据：场景 D 的保留率、检索结果与对照统计
- 反例测试必须显示：低 self relevance 噪声不会挤掉 identity-relevant 记忆

### Gy: behavioral_scenario_E_natural_misattribution [BLOCKING]
- 结构相似 + 细节衰减 + 部分线索检索可自然产生至少 1 个可解释的误归因案例
- 误归因主要落在 weak anchors / 细节槽位，而不是 locked / strong anchors
- 错误可由 `candidates`、`competing_interpretations`、`source_trace` / `reconstruction_trace` 回溯解释
- 明确验证该错误不是通过随机噪声注入伪造出来的
- 证据：至少 1 个完整错误案例的检索分数、竞争结果、重构 trace

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

M4.5–M4.7 增补修订 Prompt
Supplementary Anti-Degeneration Addendum for Codex

以下补充约束用于防止实现过程把“仿人类记忆系统”退化为“带解释字段的普通记忆库 / 检索器 / 打分器”。
这些约束与现有 M4.5、M4.6、M4.7 共同生效。若与局部实现便利性冲突，以本增补约束为准。

1. relevance 的五子项必须先独立生成，再聚合
目的

防止把 relevance 做成“表面可解释、实质黑箱”的伪透明实现。

强制约束

relevance_goal / relevance_threat / relevance_self / relevance_social / relevance_reward 必须是先独立计算得到的子项分数，然后再通过显式聚合函数生成总 relevance。
禁止以下退化实现：

先计算一个总 relevance，再反推或伪造五个子项
五个子项共享同一个黑箱中间结果，只是换名输出
只输出总分，把子项当作日志装饰物
通过单一关键词启发式同时决定多个子项，而不区分证据来源
实现要求

必须提供等价于以下结构的实现或审计日志：

goal_score, goal_evidence = score_goal_relevance(raw_input, current_state)
threat_score, threat_evidence = score_threat_relevance(raw_input, current_state)
self_score, self_evidence = score_self_relevance(raw_input, current_state)
social_score, social_evidence = score_social_relevance(raw_input, current_state)
reward_score, reward_evidence = score_reward_relevance(raw_input, current_state)

relevance = aggregate_relevance(
    goal=goal_score,
    threat=threat_score,
    self=self_score,
    social=social_score,
    reward=reward_score,
    config=config,
)
审计要求

系统必须能打印并追踪：

每个子项的独立分数
每个子项的证据来源
聚合函数的权重与中间结果
最终 relevance 如何由五子项生成
验收要求

至少提供一个测试，证明以下两类情况不会被打成同一个分数来源：

高 threat、低 self 的记忆
高 self、低 threat 的记忆

也就是说，即使总 relevance 接近，其内部构成也必须可区分、可追踪、可解释。

2. 分层表征不能只做“分类分桶”，必须体现层间演化
目的

防止 episodic / semantic / procedural / inferred 只变成静态类型标签，而没有“从具体走向抽象”的长期演化主轴。

强制约束

系统必须把“分层表征”实现成可追踪的层间变换过程，而不仅是写入时分类。

尤其要体现以下主轴：

episodic -> semantic skeleton
episodic cluster -> inferred candidate pattern
repeated procedural execution -> stabilized procedural schema
identity-relevant episodic cluster -> autobiographical / identity-relevant semantic structure
新增字段 / 元数据要求

任何由已有记忆演化出的 semantic / inferred / autobiographical 结构，除现有字段外，必须额外记录至少以下信息：

abstraction_reason: 为什么发生抽象/压缩
predictive_use_cases: 该结构未来用于何种预测、决策或行动支持
lineage_type: 如 episodic_compression / pattern_extraction / identity_consolidation / procedural_stabilization

若已有 compression_metadata，可在其中扩展这些字段。

禁止退化行为

禁止以下实现：

只是把多条 episodic 拼成一句总结，就称之为 semantic skeleton
semantic / inferred 只保留抽象文本，不保留“为什么抽象”与“将如何使用”
只记录 support ids，不记录功能意义
让抽象条目成为普通文本备份，而不承担未来预测/迁移作用
验收要求

至少提供一条完整可追踪的演化链，展示：

起始 episodic 条目集合
它们如何形成 semantic skeleton 或 inferred pattern
舍弃了哪些细节
保留了哪些稳定结构
该结构如何影响后续 recall、prediction 或 action selection
3. 未通过验证门的 inferred 记忆，默认只能作为“候选解释”，不能充当事实补全源
目的

防止“推断需验证”只停留在命名层，实际却让低置信度推断污染 factual recall 和决策。

强制约束

凡是 memory_class == inferred 且未通过验证门的条目，默认属于候选解释层，而不是事实层。

默认权限限制

未通过验证门的 inferred 条目，默认必须满足以下约束：

不能用于改写或补全 episodic 的 locked / strong anchors
不能直接提升其他条目的 reality_confidence
不能在 recall 中作为 factual detail donor
不能在冲突解析中直接压过经验性 episodic 条目
可以作为：
competing interpretation
hypothesis candidate
low-confidence planning hint
待验证规律候选
检索与决策要求

对未验证 inferred 的使用，必须引入显式折扣，例如：

effective_inference_weight = base_weight * validation_discount

其中 validation_discount < 1.0，且默认显著低于经验性记忆。

审计要求

当 recall 或 decision 使用 inferred 条目时，必须明确标注其状态：

validated
partially_supported
unvalidated
contradicted
验收要求

必须有测试证明：

未验证 inferred 可以出现在 competing_interpretations
但不会直接改写 episodic 的 protected anchors
也不会被当成“事实来源”去提高别的记忆真实性
4. 再巩固必须区分“更新类型”，不能把所有检索后的变化混成同一种操作
目的

防止 reconsolidation 退化成：

只做缓存刷新，或者
任何成功检索都偷偷改写内容
强制约束

每次成功检索后，系统必须显式判定本次再巩固属于哪一种更新类型，并在报告中记录。

必须支持的四类更新类型
A. reinforcement_only

只增强提取性与痕迹强度，不改写内容结构。

适用场景：

高置信度重复提取
无新上下文冲突
无细节补全需求
B. contextual_rebinding

允许更新情绪/场景绑定，但不改写核心内容。

适用场景：

同一记忆在新语境下被再次召回
更新 mood_context、context_tags
不改变事实骨架
C. structural_reconstruction

允许对 weak detail、缺失槽位、低保护字段做受约束补全或重组。

适用场景：

abstractness 高
细节明显不足
存在合理辅助来源
D. conflict_marking

发现冲突，但暂不覆盖；优先降置信、积累反证、保留竞争解释。

适用场景：

factual / source / interpretive conflict
新证据不能直接推翻旧 episodic
禁止退化行为

禁止以下实现：

所有 recall 一律只做 accessibility++、trace_strength++
所有 recall 一律允许 content 改写
reconsolidation 类型只存在于注释里，不出现在运行日志或结果对象中
把冲突处理偷偷塞进 structural reconstruction 中而不单独标记
实现要求

ReconsolidationReport 或等价结构中，必须包含：

update_type
fields_strengthened
fields_rebound
fields_reconstructed
conflict_flags
confidence_delta
version_changed
验收要求

至少提供 4 个测试样例，各触发上述一种更新类型，并证明它们的行为边界不同。

5. “遗忘”必须至少实现三条不同路径，而不是主要靠 cleanup 删除
目的

防止遗忘退化成单纯的存储清理，而无法体现“功能性遗忘、抽象吸收、置信漂移”的设计目标。

强制约束

系统必须把遗忘实现成多路径机制，至少覆盖以下三种：

A. 提取失败型遗忘

表现为：

trace_strength 仍大致存在
accessibility 显著下降
recall 难以成功触发或只能低置信度触发

这对应“记忆还在，但一时想不起来”。

B. 抽象吸收型遗忘

表现为：

具体 episodic 的细节价值下降
其结构被 semantic skeleton 或更高层结构吸收
原条目不一定立刻删除，可转为辅助补全源、休眠源或低可达源

这对应“具体细节忘了，但留下了规律/印象/骨架”。

C. 置信漂移型遗忘

表现为：

内容可能还能被召回
但 source_confidence 和/或 reality_confidence 随时间、冲突、重构而下降
系统对“这是真的吗 / 我从哪知道的”越来越不确定

这对应“记得有这么回事，但不太确定细节、来源或真假”。

禁止退化行为

禁止以下实现：

把 forgetting 主要实现成 trace_strength 降到阈值后删除
mid/long 只做 accessibility 下降，不体现抽象吸收或置信漂移
semantic skeleton 形成后直接删光源 episodic，导致 recall 无补全来源
置信变化只在冲突时出现，不随时间/重构出现漂移
验收要求

必须提供一个长期运行测试，证明同一系统中三类遗忘路径都实际发生过，并且各自有可追踪示例。

6. 动态调节不能只改 salience 权重，还必须影响结构性决策
目的

防止 M4.7 的动态调节退化成“状态驱动的加权打分器”，而没有真正改变系统对“什么重要、如何保存、如何更新”的判断方式。

强制约束

AgentStateVector、identity themes、active goals、threat/reward/social context、以及 cognitive style parameters，不仅可以调节 salience 权重，还必须能够影响至少以下四类结构性决策中的若干项：

A. 编码分类倾向

动态状态可以影响更倾向编码为：

episodic
semantic
procedural
inferred

例如：

高 threat / high arousal 情境更容易保留 episodic snapshot
重复任务与技能场景更容易形成 procedural structure
identity-active 场景更容易被纳入 autobiographical / self-relevant semantic path
B. 锚点强度分配

动态状态可以影响 anchor_strengths 的初始分配。
例如：

threat 高时，agent / action / outcome 更容易被强化
identity-active 时，关系角色、承诺节点、关键生涯事件的锚点更强
高 uncertainty sensitivity 时，与异常点相关的 anchor 更容易被锁定
C. 固化 / 晋升阈值

动态状态可以影响 short->mid->long 的晋升门槛。
例如：

identity-relevant 低 arousal 事件在 identity-active 阶段应更容易被保留
短期高 novelty 但低长期价值的噪声，不应仅因一时 salience 高就稳定晋升
D. 再巩固保守度

动态状态和认知风格可以影响内容改写难度。
例如：

高 update_rigidity 更难发生 structural reconstruction
高 error_aversion 面对负面记忆时更可能强化而非改写
高 identity match 的记忆更应保守更新，不应轻易被新噪声重写
禁止退化行为

禁止把动态调节仅实现为：

w_arousal *= x
w_relevance *= y
effective_novelty *= z

而不影响任何分类、锚点、固化、更新边界。

验收要求

至少提供一个测试，证明同样的输入在两种不同状态向量下，不仅 salience 不同，而且至少一个结构性结果不同，例如：

memory_class 不同
anchor_strength 分配不同
store_level 晋升结果不同
reconsolidation update_type 不同
统一补充验收原则

除各条单独要求外，整体实现还必须满足以下总原则：

1. 不能只“看起来更复杂”

新增字段、日志、报告、trace，必须真正影响编码、检索、固化、遗忘、再巩固或冲突处理，而不是仅供展示。

2. 不能把“人类式记忆”偷换成“高级可解释数据库”

若系统最终表现为：

分类存储
可解释打分
top-k 检索
偶尔压缩
cleanup 删除

但没有层间演化、候选竞争、受约束重构、多路径遗忘、验证门权限边界、状态依赖结构决策，则视为未满足本增补约束。

3. 所有新增机制都必须可追踪、可审查、可单测

每个新增约束至少应对应：

1 个单元测试或集成测试
1 个可读的 debug / audit 输出
1 个失败示例说明其防止了哪种退化
Codex 执行要求

请基于现有 M4.5 / M4.6 / M4.7 实现与 prompt，完成以下工作：

检查当前代码设计中是否存在上述 6 类退化风险
对数据结构、编码、检索、固化、再巩固、遗忘、动态调节逻辑进行必要修改
为每项新增约束补充测试与审计输出
在最终报告中逐条说明：
如何满足本增补约束
原先可能的退化点是什么
现在如何避免
仍然存在哪些未完全解决的边界问题

如果某项约束暂时无法完全实现，不得用模糊表述假装完成，必须明确标注：

未完成项
当前替代方案
风险残留点
后续建议