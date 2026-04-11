# M4.5 Work Prompt — 仿人类记忆系统：数据模型与编码机制

## 背景

当前 `memory.py` 中的 `Episode` + `LongTermMemory` 架构存在以下瓶颈：
- 记忆无分类（episodic/semantic/procedural/inferred 不区分）
- 只有 active/archived 两级，无层级固化
- 显著性评分不透明，无法审查"为什么保留这条"
- 无衰减双变量（trace_strength / accessibility）
- 无 source_confidence / reality_confidence
- 语义提取硬编码 7 种 event type，无法扩展

本里程碑基于"仿人类记忆系统设计文档 v3.1"，实现记忆系统的数据层和编码层。

## 目标

建立统一记忆库的数据模型，实现编码流水线（感知输入 → salience 计算 → 分类 → 写入），实现衰减与紧急固化通道。该数据模型应服务于“分层表征 + 检索重构”，而不是把记忆等同为静态事实条目；存储单元仅为后续回忆重建提供材料，不等于最终回忆本身。

## 实现要求

### 1. 统一记忆条目数据模型 `MemoryEntry`

新建 `segmentum/memory_model.py`，定义 `MemoryEntry` dataclass，包含以下核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | UUID |
| `content` | str | 记忆内容摘要 |
| `content_hash` | str | content 的哈希，用于判定实质变化 |
| `memory_class` | enum | `episodic` / `semantic` / `procedural` / `inferred` |
| `store_level` | enum | `short` / `mid` / `long` |
| `source_type` | enum | `experience` / `rehearsal` / `inference` / `hearsay` / `reconstruction` |
| `created_at` | int | 首次建立 cycle |
| `last_accessed` | int | 上次检索 cycle |
| `valence` | float | 情绪方向 -1.0 ~ +1.0 |
| `arousal` | float | 唤醒强度 0.0 ~ 1.0 |
| `encoding_attention` | float | 编码时注意力 0.0 ~ 1.0 |
| `novelty` | float | 新奇性/预测误差 0.0 ~ 1.0 |
| `relevance_goal` | float | 与当前目标的相关性 0.0 ~ 1.0 |
| `relevance_threat` | float | 与威胁/风险的相关性 0.0 ~ 1.0 |
| `relevance_self` | float | 与自我连续性/身份叙事的相关性 0.0 ~ 1.0 |
| `relevance_social` | float | 与社会关系/互动的相关性 0.0 ~ 1.0 |
| `relevance_reward` | float | 与奖励/损失的相关性 0.0 ~ 1.0 |
| `relevance` | float | 由多个可审查分项聚合得到的综合相关性 0.0 ~ 1.0 |
| `salience` | float | 显式公式计算的综合显著性 |
| `trace_strength` | float | 痕迹强度（慢变量）|
| `accessibility` | float | 可提取性（快变量）|
| `abstractness` | float | 0.0 具体 ~ 1.0 抽象 |
| `source_confidence` | float | 来源置信度 0.0 ~ 1.0，可随回忆/冲突/时间漂移 |
| `reality_confidence` | float | 内容真实性置信度 0.0 ~ 1.0，冲突时优先下调而非直接覆写 |
| `semantic_tags` | list[str] | 语义标签 |
| `context_tags` | list[str] | 场景标签 |
| `anchor_slots` | dict[str, str \| None] | 关键锚点槽位，建议至少支持 `time/place/agents/action/outcome` |
| `anchor_strengths` | dict[str, str] | 锚点强度，值为 `locked` / `strong` / `weak` |
| `procedure_steps` | list[str] | procedural 条目的显式步骤结构；非 procedural 可为空 |
| `step_confidence` | list[float] | 与 `procedure_steps` 对齐的步骤置信度 |
| `execution_contexts` | list[str] | procedural 适用或习得的执行场景 |
| `mood_context` | str | 编码时情绪语境 |
| `retrieval_count` | int | 被检索次数 |
| `support_count` | int | 支持证据次数 |
| `counterevidence_count` | int | 冲突证据次数，用于累积反证而不是触发立即覆盖 |
| `competing_interpretations` | list[str] \| None | 并存解释；用于保留竞争而不是立刻覆盖 |
| `compression_metadata` | dict[str, object] \| None | semantic skeleton 或压缩产物的 support 链、稳定结构、被舍弃细节类型 |
| `derived_from` | list[str] | 派生来源 id 列表 |
| `version` | int | 仅 content_hash 变化时递增 |
| `is_dormant` | bool | 是否休眠 |

必须实现 `to_dict()` / `from_dict()` 序列化，与现有 Episode payload 格式桥接。

补充约束：
- `relevance` 必须由 `relevance_goal` / `relevance_threat` / `relevance_self` / `relevance_social` / `relevance_reward` 聚合得到，且每项贡献可打印、可追踪、可验证。
- `relevance_self` 只用于表示与“自我连续性”有关的保留价值，至少覆盖以下一类：长期角色连续性、重要关系连续性、长期承诺/责任、关键自我叙事节点、`我是谁 / 我经历了什么 / 我与谁绑定` 相关内容。
- 以下因素不得自动等价为高 `relevance_self`：只是第一人称表述、只是当前任务相关、只是高情绪强度、只是和 agent 本体发生过一次接触。
- `episodic` 条目必须支持关键锚点槽位的显式表示，供后续 reconstruction 使用；如实现不采用完整 dict 结构，也必须保留等价的显式锚点表示。
- `episodic` 条目的默认锚点规则必须是结构化约束而不是空白占位：`agents` / `action` / `outcome` 至少两项默认为 `strong` 或其中一项为 `locked`；`time` / `place` 默认可为 `weak`，但若事件本身以时空定位为核心则必须升级；在未显式标注时，不允许所有锚点都默认为 `weak`。
- `procedural` 条目必须有可序列化步骤结构；`content` 只可作为摘要，不得充当 procedural 的全部表示。
- `counterevidence_count`、`source_confidence`、`reality_confidence` 的语义必须支持“冲突先积累、先降置信、先保留竞争解释”，而不是将新证据直接视为覆盖旧记忆的理由。
- `source_confidence` 与 `reality_confidence` 必须允许共同变化，也必须允许分离变化；实现不得将两者永久绑定为同步升降。

### 2. Salience 计算模块

新建 `segmentum/memory_encoding.py`，实现：

```python
def compute_salience(
    arousal: float,
    encoding_attention: float,
    novelty: float,
    relevance: float,
    config: SalienceConfig | None = None,
) -> float:
    """显式加权公式，每一项贡献可打印可回溯。"""
```

默认权重配置：
- w_arousal = 0.30
- w_attention = 0.20
- w_novelty = 0.20
- w_relevance = 0.30

`SalienceConfig` 存储权重，可被外部状态向量调节（M4.7 实现调节逻辑，本里程碑只暴露接口）。

补充要求：
- `relevance` 不允许作为黑箱单值直接输入后“吞没”内部结构；`compute_salience()` 的审计报告必须同时打印 relevance 总值与 5 个 relevance 子项。
- 必须能输出类似 `relevance = agg(goal, threat, self, social, reward)` 的可验证聚合过程，而不是只输出一个总 relevance 分数。

### 3. 编码流水线

在 `memory_encoding.py` 中实现：

```python
def encode_memory(
    raw_input: dict,        # 感知输入或内部事件
    current_state: dict,    # 当前 agent 状态（用于计算 relevance）
    config: SalienceConfig,
) -> MemoryEntry:
    """
    感知输入 → 计算 arousal/attention/novelty/relevance 及其子项
    → salience → 分类 memory_class/source_type
    → 生成 content 摘要、标签与关键锚点
    → 返回 MemoryEntry（store_level=short）
    """
```

分类规则：
- `episodic`：带具体时间/地点/角色的经历
- `semantic`：从多次经历中抽象出的规律
- `procedural`：动作序列/操作模式
- `inferred`：推断/重组产物

编码约束：
- 编码时必须显式计算 `relevance_goal` / `relevance_threat` / `relevance_self` / `relevance_social` / `relevance_reward`，再聚合为 `relevance`。
- `relevance_self` 的默认评分过程必须可审计；建议提供等价于 `score_self_relevance(raw_input, current_state) -> (score, evidence)` 的实现或日志，明确列出角色连续性、关系连续性、承诺责任、自我叙事节点等子证据。
- 若缺少明确的 identity/self 连续性证据，`relevance_self` 默认不得仅因第一人称、当前任务关联或高 arousal 而被拉高。
- 与长期角色、关系、承诺、自传连续性相关的输入，即便 `arousal` / `novelty` 较低，也可因较高 `relevance_self` 获得较高保留优先级，不得被系统性视为低价值噪声。
- `episodic` 条目必须写出关键锚点槽位及其强弱；后续 reconstruction 默认以这些锚点为边界。
- `procedural` 编码时默认同时填充 `procedure_steps`、`step_confidence`、`execution_contexts`；若只能产出自然语言摘要，应视为编码不完整而不是退化为普通文本条目。

初始 source_confidence / reality_confidence 赋值规则：

| source_type | source_confidence | reality_confidence |
|-------------|------------------|--------------------|
| experience | 0.9 | 0.85 |
| rehearsal | 0.8 | 0.8 |
| hearsay | 0.7 | 0.5 |
| inference | 0.9 | 0.35 |
| reconstruction | 0.4 | 0.5 |

默认行为约束：
- 初始赋值只提供起点，不代表后续更新必须同步；后续冲突、验证、重构、遗忘可单独影响 `source_confidence` 或 `reality_confidence`。
- 必须支持并测试四种组合：source 高 / reality 高，source 高 / reality 低，source 低 / reality 高，source 低 / reality 低。

### 4. 紧急固化通道

```python
EMERGENCY_AROUSAL_THRESHOLD = 0.9
EMERGENCY_SALIENCE_THRESHOLD = 0.92

# 编码时：若超过阈值，直接 store_level = long
```

### 5. 双变量衰减

在 `segmentum/memory_decay.py` 中实现：

```python
def decay_trace_strength(entry: MemoryEntry, elapsed_cycles: int) -> float:
    """trace_strength(t) = trace_strength₀ × e^(-λ_trace × Δt)"""

def decay_accessibility(entry: MemoryEntry, elapsed_cycles: int) -> float:
    """accessibility(t) = accessibility₀ × e^(-λ_access × Δt)"""
```

不同层级的 λ 参数：

| 层级 | λ_trace | λ_access |
|------|---------|----------|
| short | 0.010 | 0.050 |
| mid | 0.002 | 0.020 |
| long | 0.0002 | 0.005 |

`procedural` 类型在 long 层的 λ_trace 额外乘 0.1。

遗忘/衰减语义补充：
- 遗忘不只等于删除；除 `trace_strength` / `accessibility` 下降外，还可表现为 `abstractness` 上升、`source_confidence` 漂移下降、条目被抽象吸收到 semantic skeleton、进入 `is_dormant=true`。
- `MemoryStore.cleanup_short()` 只允许清除低价值 short 层残余，不得作为 mid/long 层记忆的默认遗忘机制。
- 对 mid/long 层，默认优先走“更难访问、更抽象、更不确定、被吸收、休眠”的路径，而不是直接删除。

### 6. 统一记忆库 `MemoryStore`

新建 `segmentum/memory_store.py`，实现：

```python
class MemoryStore:
    entries: list[MemoryEntry]

    def add(self, entry: MemoryEntry) -> str: ...
    def get(self, entry_id: str) -> MemoryEntry | None: ...
    def query_by_tags(self, tags: list[str], k: int = 5) -> list[MemoryEntry]: ...
    def apply_decay(self, current_cycle: int) -> DecayReport: ...
    def mark_dormant(self, entry_id: str) -> None: ...
    def cleanup_short(self, threshold: float = 0.05) -> int: ...
    def to_dict(self) -> dict: ...
    def from_dict(cls, payload: dict) -> MemoryStore: ...
```

`cleanup_short()` 的实现与报告必须区分“真正删除的低价值 short 残余”和“发生多路径遗忘但仍被保留的条目”，避免把 forgetting 退化成单一路径的 cleanup。

### 7. 与现有 LongTermMemory 的桥接

不破坏 `LongTermMemory` 的外部 API。在 `MemoryStore` 中实现：
- `from_legacy_episodes(episodes: list[dict]) -> MemoryStore`：将现有 episode payload 转换为 MemoryEntry
- `to_legacy_episodes() -> list[dict]`：反向转换

确保 M4.1-M4.4 的所有测试仍然通过。

### 8. Implementation Anti-patterns / 禁止退化行为

- 不得把 `retrieve()` 实现为 top-1 `MemoryEntry` 直接返回，只是换个名字叫 `recall_hypothesis`。
- 不得把 `relevance_self` 简化为“是否包含第一人称”或“是否与当前 agent 有关”。
- 不得把 forgetting 默认实现成只靠 `cleanup_short()` 删除。
- 不得让 `source_confidence` 与 `reality_confidence` 永远同步变化。
- 不得把所有锚点默认设为 `weak`。
- 不得把 `procedural` 只实现成普通文本摘要。
- 不得在冲突出现时默认直接覆盖旧 episodic。
- 不得用随机噪声注入来伪装“人类式记忆错误”。

### 9. 失败示例说明

以下情况即使部分测试通过，也应视为实现跑偏：
- `recall_hypothesis` 实际只是 `entry.content` 原样回传。
- 低 `relevance_self` 的高新奇噪声与高 identity continuity 的低 arousal 事件，被系统打成同一种“重要”。
- episodic 条目存在完整 `anchor_slots`，但 `anchor_strengths` 全部默认为 `weak`。
- procedural 条目没有显式 `procedure_steps`，只有一段自然语言摘要。
- mid/long 遗忘几乎都通过 cleanup 删除完成，而非抽象化、休眠、置信漂移或被骨架吸收。
- `source_confidence` 与 `reality_confidence` 在所有测试中始终同向同幅变化。

## 验收标准

### Gate 1: 数据模型完整性
- `MemoryEntry` 包含所有核心字段
- `to_dict()` / `from_dict()` 往返无损
- `content_hash` 在 content 不变时稳定，变化时递增 version
- procedural 相关字段、`competing_interpretations`、`compression_metadata` 可往返保留
- episodic 默认锚点中不允许出现“所有关键槽位均为 weak”的空壳实现

### Gate 2: Salience 可审计性
- 对任意 MemoryEntry，可打印 `arousal=X, attention=Y, novelty=Z, relevance=W → salience=公式展开`
- 可打印 `relevance_goal / relevance_threat / relevance_self / relevance_social / relevance_reward` 五个分项
- `relevance` 的聚合过程可验证，而不是黑箱单值
- 权重来自 SalienceConfig，不硬编码在公式体内
- 单元测试：固定输入 → 固定输出，可复现
- `relevance_self` 的审计输出必须能指出它为何高或为何低，而不是只给分数

### Gate 3: 编码流水线功能
- 给定感知输入，产出带完整字段的 MemoryEntry
- memory_class 分类逻辑覆盖四种类型
- source_confidence / reality_confidence 按 source_type 正确初始化
- 高 arousal 输入触发紧急固化（store_level=long）
- 至少 1 个正例测试：低 arousal、低 novelty、但高 identity continuity 的事件获得高于噪声基线的保留优先级
- 至少 1 个反例测试：高新奇但与自我连续性无关的噪声事件，不得自动获得与上例相同类型的“重要”
- 至少 1 个反例测试证明：第一人称表述或当前任务相关，不足以单独抬高 `relevance_self`
- procedural 条目编码后必须含显式 `procedure_steps`，且其结构可与普通 episodic 文本区分
- 四种 `source_confidence` / `reality_confidence` 组合均有测试，且明确禁止实现把两者做成总是同步升降

### Gate 4: 双变量衰减正确性
- trace_strength 衰减慢于 accessibility（同层级、同 elapsed）
- short 层衰减快于 mid 快于 long
- procedural + long 的 trace_strength 衰减最慢
- 衰减后 trace_strength=0 的 short 条目被清除
- 衰减后双低的 long 条目标记 is_dormant
- 至少 1 个测试覆盖“非删除式遗忘”，如 `abstractness` 上升或 `source_confidence` 漂移
- 清理报告能区分删除、休眠、抽象化吸收或置信漂移等不同路径

### Gate 5: 桥接兼容性
- `from_legacy_episodes()` 能转换现有 episode payload
- 转换后的 MemoryEntry 保留原始 timestamp、action、outcome 等关键信息
- M4.1-M4.4 全部测试通过（回归）

### Gate 6: 存储层级转移
- MemoryEntry 的 store_level 可从 short→mid→long 升级
- 升级条件明确：高 salience + 高 retrieval_count，或紧急固化
- 降级/清除条件明确：低 trace_strength
- 构造低 arousal、低 novelty、但高 `relevance_self` 的输入，验证其不会因总显著性偏低而被立即视为低价值噪声
- identity/self relevance 可影响保留优先级，至少能支撑 short 或 mid 层保留
- 至少 1 个反例测试验证：低 self relevance 的短期噪声不会挤掉 identity-relevant 记忆
- 至少 1 个测试验证：高 identity continuity 事件与高 novelty 噪声在长期保留结果上分化，而不是被同一“重要性”通道吞并

### Gate 7: 报告诚实性
- 所有 gate 有非空 evidence
- acceptance_state 准确反映实际状态
- 不伪造通过

## 回归要求

M4.1, M4.2, M4.3, M4.4 所有现有测试必须通过。

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
## 2026-04-09 Hotfix Addendum

The M4.5 identity-retention hotfix freezes the promotion rule used by the default code path.

- Token normalization: trim, lowercase, split on `[^a-z0-9_]+`.
- `entry_tokens = semantic_tags + context_tags + content`
- `identity_tokens = state_vector.identity_active_themes`
- `identity_match_ratio = |shared(entry_tokens, identity_tokens)| / min(|entry_tokens|, |identity_tokens|)` when both sides are non-empty, else `0.0`.
- This hotfix does not add synonym expansion, embeddings, or learned theme anchors.

Promotion rule:

- `identity_link_strength = clamp(0.6 * relevance_self + 0.4 * identity_match_ratio)`
- `identity_link_active = identity_link_strength >= identity_priority_threshold`, with the state-backed path preferred and an entry-level fallback used only when no current identity state is available.
- The self-relevance multiplier applies only on the `short -> mid` branch.
- Ordering is fixed: compute the base score first, apply `novelty_noise_penalty`, then apply the multiplier.
- `boosted_short_to_mid = min(score_cap, base_short_to_mid * (1 + alpha * identity_link_strength))`
- `alpha = 0.35`
- `score_cap = 0.95`
- `mid -> long` promotion must use the unboosted base score.

Required audit fields:

- `identity_match_ratio`
- `identity_link_strength`
- `identity_link_active`
- `self_relevance_multiplier`
- `base_short_to_mid_score`
- `boosted_short_to_mid_score`
- `score_cap_applied`
## M4.10 Supersession Note

Sections in this document that describe keyword-table salience, template-string semantic skeletons, inferred-pattern text, or text-only replay are superseded by M4.10. After M4.10, acceptance follows dynamic encoding, attention-budget competition, centroid/residual semantic consolidation, and replay re-encoding.
