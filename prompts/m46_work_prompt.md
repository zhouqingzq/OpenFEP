# M4.6 Work Prompt — 仿人类记忆系统：检索、再巩固与离线固化

## 前置依赖

M4.5（数据模型 + 编码 + 衰减）必须通过。

## 目标

实现记忆系统的检索层（多线索激活 + 候选竞争 + 重构）、再巩固机制、和完整的离线固化流水线。这是记忆系统从"能存"到"能用"的关键跃迁。检索层的目标不是从库中找到一条最匹配记录，而是通过多线索激活、候选竞争和受约束重构，生成一个带置信度和来源痕迹的回忆假设。

## 实现要求

### 1. 多线索激活检索

在 `segmentum/memory_retrieval.py` 中实现：

```python
def retrieve(
    query: RetrievalQuery,
    store: MemoryStore,
    current_mood: str | None = None,
    k: int = 5,
) -> RetrievalResult:
    """
    对每条非休眠记忆计算 retrieval_score：
      score = w1 × tag_overlap(query.semantic_tags, entry.semantic_tags)
            + w2 × context_overlap(query.context_tags, entry.context_tags)
            + w3 × mood_match(current_mood, entry.mood_context)
            + w4 × entry.accessibility
            + w5 × recency_bonus(entry.last_accessed, current_cycle)
    
    返回 top-k 候选 + 激活分数，并生成 recall_hypothesis。
    """
```

`RetrievalQuery` 支持：
- semantic_tags：语义标签匹配
- context_tags：场景标签匹配
- content_keywords：内容关键词（纯文本匹配，不引入 embedding）
- state_vector：向量相似度（兼容现有 cosine 检索）

检索默认权重：w1=0.35, w2=0.15, w3=0.15, w4=0.20, w5=0.15

```python
@dataclass
class RecallArtifact:
    content: str
    primary_entry_id: str
    auxiliary_entry_ids: list[str]
    confidence: float
    source_trace: list[str]
    reconstructed_fields: list[str]
    protected_fields: list[str]
    competing_interpretations: list[str] | None = None
    procedure_step_outline: list[str] | None = None
```

```python
def build_recall_artifact(
    primary: MemoryEntry,
    auxiliaries: list[MemoryEntry],
    competition: CompetitionResult,
    query: RetrievalQuery,
) -> RecallArtifact:
    """
    默认以 primary 作为主干，以 0-2 条辅助来源补全 weak / 缺失细节，
    生成新的 recall artifact。
    RecallArtifact 不是 MemoryEntry 的别名，也不是 top-1 entry 的直接回传。
    默认输出必须包含 source_trace、protected_fields、reconstructed_fields；
    当 primary 为 procedural 时，默认还必须输出 `procedure_step_outline`。
    """
```

`RetrievalResult` 必须同时包含：
- `candidates`: 候选列表及其分数
- `recall_hypothesis`: `RecallArtifact` 类型的最终回忆假设，不等于原始 entry 的直接回传
- `recall_confidence`: 对该回忆假设的置信度
- `source_trace` 或 `reconstruction_trace`: 指明主干来源、辅助来源、受保护字段与补全字段

默认行为约束：
- `RecallArtifact` 不能只是 `MemoryEntry` 的别名、浅拷贝或换名返回。
- `RecallArtifact.content` 默认应由主干条目 + 辅助来源 + 查询线索共同重建生成，而不是原样回传任一单条 `entry.content`。
- 若检索目标为 procedural，回忆产物默认应输出步骤骨架，而不是只有一段叙事句子。
- 若实现需要暴露原始条目文本，必须通过独立 debug/view API；`retrieve()` 默认不承担“返回原条目”的职责。

### 2. 候选竞争

```python
def compete_candidates(
    candidates: list[ScoredCandidate],
    dominance_threshold: float = 0.15,
) -> CompetitionResult:
    """
    - 第一名优势 > threshold → 直接作为主干，confidence=high
    - 多个候选分数接近（差距 < threshold）→
      输出 confidence=low，记录 competing_ids / competing_interpretations
    - 低置信度回忆本身是有效产物，不是异常
    """
```

竞争结果包含：
- `primary`: 主干记忆
- `competitors`: 分数接近的竞争者
- `confidence`: 检索置信度
- `interference_risk`: 是否存在干扰风险
- `competing_interpretations`: 并列解释或候选叙述

### 3. 重构机制

```python
def maybe_reconstruct(
    primary: MemoryEntry,
    candidates: list[MemoryEntry],
    store: MemoryStore,
    config: ReconstructionConfig,
) -> ReconstructionResult:
    """
    触发条件（任一组满足）：
    A: abstractness > 0.7 AND len(content) < 50
    B: abstractness > 0.7 AND memory_class == semantic
    C: reality_confidence < 0.4 AND retrieval_count > 0
    
    重构约束：
    - 最多从 1-2 条相似记忆借用细节
    - episodic 的 locked anchors 默认不得被改写；strong anchors 默认不得在无支持证据时被改写
    - episodic 的 weak anchors 可由相似条目补全
    - semantic / inferred 记忆允许更高自由度的受约束重构
    - procedural 默认不得在无明确支持证据时改写核心动作序列
    - 重构后 reality_confidence 下调
    - source_type 更新为 reconstruction
    - content_hash 变化时 version += 1
    """
```

重构来源优先级：
1. 同 derived_from 链中的旧版本
2. 高 semantic_tags 重合度的条目
3. 同 mood_context 的条目
4. 同 context_tags 的条目

`ReconstructionResult` 或等价结构必须记录：
- 主干条目 id
- 借用来源 id 列表
- 被补全的字段
- 被保护且未改写的锚点字段
- 最终 `reconstruction_trace`

### 4. 再巩固（Reconsolidation）

每次成功检索后触发：

```python
def reconsolidate(
    entry: MemoryEntry,
    current_mood: str | None,
    current_context_tags: list[str] | None,
) -> ReconsolidationReport:
    """
    - accessibility 显著回升 (+boost_access)
    - trace_strength 小幅回升 (+boost_trace)
    - mood_context 可能更新为当前情绪
    - abstractness 小幅上升（每次回忆丢失一点细节）
    - content 若被改写 → content_hash 变化 → version += 1
    - retrieval_count += 1
    - last_accessed = current_cycle
    - 若出现与旧记忆冲突的新信息，优先降低 reality_confidence、增加 counterevidence_count、生成 competing_interpretations
    """
```

再巩固的默认目标是“强化提取并受约束更新”，不是在冲突出现时立即覆写旧 episodic。若新信息与旧条目冲突，默认行为应为：
- 优先降低 `reality_confidence`
- 提高 `counterevidence_count`
- 生成 `competing_interpretations`
- 必要时并列保留候选或保留旧版本为 `is_dormant=true` 的影子记录

### 5. 离线固化流水线

在 `segmentum/memory_consolidation.py` 中实现四阶段流水线：

#### 阶段 1: 记忆升级

```python
def consolidate_upgrade(store: MemoryStore, current_cycle: int) -> UpgradeReport:
    """
    对 short/mid 层条目计算 consolidation_priority：
      priority = c1×salience + c2×retrieval_count_norm + c3×pattern_support - c4×redundancy
    
    short → mid: priority > threshold 或被多次提取
    mid → long: 多次经验支持 + 稳定提取历史，或已抽象为语义骨架
    """
```

#### 阶段 2: 模式提取

```python
def compress_episodic_cluster_to_semantic_skeleton(
    entries: list[MemoryEntry],
) -> MemoryEntry:
    """
    将多条 episodic 压缩为 semantic skeleton。
    semantic skeleton 不是“另一条普通 semantic 记忆”，而是 cluster compression 的产物；
    产物必须在 `compression_metadata` 中保留：
    - support_entry_ids
    - discarded_detail_types
    - stable_structure
    默认不得立即删除源 episodic；源条目仍可作为后续 recall 的补全来源。
    """

def extract_patterns(store: MemoryStore) -> list[MemoryEntry]:
    """
    扫描 short/mid 中的多条相关记忆，寻找共享结构。
    当 ≥ minimum_support 条记忆共享 semantic_tags 组合时：
    → 生成 inferred 类型的 MemoryEntry，或调用 `compress_episodic_cluster_to_semantic_skeleton()` 将多条 episodic 压缩为可迁移的 semantic skeleton
    → support_count = 共享记忆数
    → reality_confidence 按 support/(support+counter+smoothing) 计算
    → 保留 support 链接到原始经历，不能只留下一个脱离来源的抽象结果
    """
```

#### 阶段 3: 受约束重组

```python
def constrained_replay(
    store: MemoryStore,
    rng: random.Random,
    batch_size: int = 32,
) -> list[MemoryEntry]:
    """
    加权采样偏好：近期高 salience + 未解决问题 + 高 arousal + 高频提取。
    在相关主题/情绪/情境范围内重组。
    产物标记为 inferred + inference，reality_confidence 初始较低。
    必须通过验证门（阶段 4）才能升级。
    """
```

#### 阶段 4: 清理

```python
def consolidation_cleanup(store: MemoryStore, current_cycle: int) -> CleanupReport:
    """
    - 降低低价值条目 accessibility
    - short 层低价值残余且 trace_strength < threshold → 清除
    - mid/long 层优先走 accessibility 下降、abstractness 上升、source/reality confidence 漂移、被 semantic skeleton 吸收或休眠
    - long 层极少使用 → is_dormant = true
    """
```

### 6. 冲突类型判定与差异化处理

```python
class ConflictType(str, Enum):
    FACTUAL = "factual"
    SOURCE = "source"
    INTERPRETIVE = "interpretive"


def resolve_conflict(
    existing: MemoryEntry,
    incoming: MemoryEntry | RecallArtifact,
    conflict_type: ConflictType,
) -> ConflictResolution:
    """
    factual:
        主要下调 reality_confidence，并增加 counterevidence_count。
    source:
        主要下调 source_confidence，并记录来源不一致或来源可疑的 trace。
    interpretive:
        主要保留 competing_interpretations；
        默认不直接打掉 episodic 底层事件，也不直接覆盖 locked / strong anchors。
    """
```

默认行为约束：
- 事实冲突：内容层面互相排斥，默认优先打击 `reality_confidence`。
- 来源冲突：内容相近但来源不一致或来源可疑，默认优先打击 `source_confidence`。
- 解释冲突：底层事件可并存但归因/规律冲突，默认主要生成 `competing_interpretations`，而不是直接否定 episodic 本体。

### 7. 推断验证门

```python
def validate_inference(entry: MemoryEntry) -> ValidationResult:
    """
    inference_write_score =
        d1 × replay_persistence
        + d2 × support_count
        + d3 × cross_context_consistency
        + d4 × predictive_gain
        - d5 × contradiction_penalty
    
    通过 → 允许升级至 long，reality_confidence 提升
    未通过 → 留在 mid，等待更多证据
    """
```

冲突处理补充要求：
- 当新信息与旧条目冲突时，默认不直接覆盖原 episodic。
- 优先降低 `reality_confidence`、增加 `counterevidence_count`、生成 `competing_interpretations`，必要时保留并列候选。

### 8. 与现有 sleep_consolidator 的集成

新的 `memory_consolidation.py` 作为 `sleep_consolidator.py` 的替代方案。在 MemoryStore 层面提供：

```python
class MemoryStore:
    def run_consolidation_cycle(self, current_cycle: int, rng: random.Random) -> ConsolidationReport:
        """执行完整的四阶段固化。"""
```

通过桥接层确保旧 `LongTermMemory.replay_during_sleep()` 仍可工作。

### 9. Implementation Anti-patterns / 禁止退化行为

- 不得把 `retrieve()` 实现为 top-1 `MemoryEntry` 直接返回，只是换个名字叫 `recall_hypothesis`。
- 不得把 `relevance_self` 简化为“是否包含第一人称”或“是否与当前 agent 有关”。
- 不得把 forgetting 默认实现成只靠 cleanup 删除。
- 不得在冲突出现时默认直接覆盖旧 episodic。
- 不得把 `source_confidence` 与 `reality_confidence` 永远绑成同步变化。
- 不得把所有锚点默认设为 `weak`。
- 不得把 procedural 只实现成普通文本摘要。
- 不得用随机噪声注入来伪装“人类式记忆错误”。

### 10. 失败示例说明

以下情况即使部分测试通过，也应视为实现跑偏：
- `recall_hypothesis` 实际只是 `entry.content` 原样回传。
- semantic skeleton 没有 support 链、没有被舍弃细节类型、没有稳定结构记录。
- 所有 mid/long 遗忘最终都靠删除完成，没有吸收、休眠或不确定性漂移。
- conflict 处理没有 `competing_interpretations`，只做覆盖或只做统一降分。
- procedural recall 没有显式步骤骨架，只有普通叙事句子。

## 验收标准

### G1: retrieval_multi_cue [BLOCKING]
- 检索结果按 retrieval_score 排序
- tag_overlap 贡献可独立验证
- mood_match 实现情绪一致性效应（负面情绪 → 更容易召回负 valence 记忆）
- accessibility 低的记忆检索分数低（即使标签完全匹配）
- is_dormant 记忆在普通检索中不出现
- 检索最终输出不只是 entry，而是 `recall_hypothesis`
- `recall_hypothesis` 的类型是显式 `RecallArtifact` 或等价结构，而不是 `MemoryEntry`
- `recall_hypothesis` 能指出主干来源和辅助来源
- 至少 1 个用例证明：最终 recall 内容不等于任一单条 `entry.content` 的原样返回
- procedural 检索至少 1 个用例输出步骤骨架，而不是普通叙事文本
- 证据：≥5 个检索场景的分数分解

### G2: candidate_competition [BLOCKING]
- 分数悬殊时直接返回主干，confidence=high
- 分数接近时返回多个候选，confidence=low，interference_risk=true
- 低置信度回忆视为有效输出而不是异常
- `competing_interpretations` 或等价结构可追踪并有 evidence
- 证据：构造两组测试（悬殊 vs 接近），验证竞争行为

### G3: reconstruction_mechanism [BLOCKING]
- 三组触发条件均有覆盖测试
- 重构后 reality_confidence 下降
- 重构后 source_type = reconstruction
- 重构后 content_hash 变化 → version 递增
- 最多借用 1-2 条来源（不超出约束）
- 测试 locked anchors 在默认路径下不会被改写
- 测试 strong anchors 不会被无证据改写
- 测试 weak anchors 可以被补全
- semantic / inferred 条目允许比 episodic 更高的重构自由度
- reconstruction_trace 能记录主干来源、借用来源、补全字段和受保护字段
- 至少 1 个反例测试证明：实现不能把所有锚点默认都做成 weak
- 证据：重构前后的字段对比

### G4: reconsolidation [BLOCKING]
- 检索后 accessibility 回升
- 检索后 trace_strength 小幅回升
- retrieval_count 递增
- abstractness 小幅上升
- content 被改写时 version 递增
- procedural 核心动作序列在无明确支持证据时不得被改写
- 冲突场景下默认不直接覆盖旧 episodic，而是降低 reality_confidence 或保留 competing interpretations
- 至少 3 个单元测试分别覆盖事实冲突、来源冲突、解释冲突，并验证其主要影响字段不同
- 事实冲突主要打击 `reality_confidence`
- 来源冲突主要打击 `source_confidence`
- 解释冲突主要生成 `competing_interpretations`，不直接打掉底层 episodic
- 证据：再巩固前后字段变化记录

### G5: offline_consolidation_pipeline [BLOCKING]
- 四阶段依次执行
- 阶段 1：至少 1 条 short→mid 升级（给定足够高 salience 的条目）
- 阶段 2：至少 1 条 inferred 或 semantic skeleton 记忆被提取（给定重复模式）
- 阶段 3：重组产物标记 inferred + inference
- 阶段 4：低价值条目被清除或休眠
- 至少 1 个用例验证多条 episodic 被压缩为 semantic skeleton，且 skeleton 保留 support 溯源
- 至少 1 个用例验证：多条 episodic → 1 条 semantic skeleton，且 skeleton 的 `abstractness` 高于源条目
- semantic skeleton 必须保留 `compression_metadata.support_entry_ids`、`discarded_detail_types`、`stable_structure`
- 源 episodic 在生成 skeleton 后不能立即删除，且仍可作为 recall 补全来源
- 证据：ConsolidationReport 包含每阶段统计

### G6: inference_validation_gate [BLOCKING]
- 高 support_count + 跨情境一致性 → 通过验证门 → 可升级至 long
- 低 support_count 或高 contradiction → 未通过 → 留在 mid
- 证据：构造通过和未通过各 1 个用例

### G7: legacy_integration [BLOCKING]
- 新固化流水线可通过 MemoryStore 调用
- 旧 LongTermMemory.replay_during_sleep() 仍可工作
- M4.1-M4.4 + M4.5 全部测试通过

### G8: report_honesty [BLOCKING]
- 所有 gate 有非空 evidence
- 无伪造通过

## 回归要求

M4.1, M4.2, M4.3, M4.4, M4.5 所有测试通过。

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
## M4.10 Supersession Note

Sections in this document that describe keyword-table salience, template-string semantic skeletons, inferred-pattern text, or text-only replay are superseded by M4.10. After M4.10, acceptance follows dynamic encoding, attention-budget competition, centroid/residual semantic consolidation, and replay re-encoding.
