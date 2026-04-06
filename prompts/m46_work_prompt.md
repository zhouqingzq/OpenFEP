# M4.6 Work Prompt — 仿人类记忆系统：检索、再巩固与离线固化

## 前置依赖

M4.5（数据模型 + 编码 + 衰减）必须通过。

## 目标

实现记忆系统的检索层（多线索激活 + 候选竞争 + 重构）、再巩固机制、和完整的离线固化流水线。这是记忆系统从"能存"到"能用"的关键跃迁。

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
    
    返回 top-k 候选 + 激活分数。
    """
```

`RetrievalQuery` 支持：
- semantic_tags：语义标签匹配
- context_tags：场景标签匹配
- content_keywords：内容关键词（纯文本匹配，不引入 embedding）
- state_vector：向量相似度（兼容现有 cosine 检索）

检索默认权重：w1=0.35, w2=0.15, w3=0.15, w4=0.20, w5=0.15

### 2. 候选竞争

```python
def compete_candidates(
    candidates: list[ScoredCandidate],
    dominance_threshold: float = 0.15,
) -> CompetitionResult:
    """
    - 第一名优势 > threshold → 直接作为主干，confidence=high
    - 多个候选分数接近（差距 < threshold）→ 
      输出 confidence=low，记录 competing_ids（可能混淆来源）
    """
```

竞争结果包含：
- `primary`: 主干记忆
- `competitors`: 分数接近的竞争者
- `confidence`: 检索置信度
- `interference_risk`: 是否存在干扰风险

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
    """
```

覆写式再巩固为默认。当 content 差异超过阈值时，保留旧版本为 `is_dormant=true` 的影子记录。

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
def extract_patterns(store: MemoryStore) -> list[MemoryEntry]:
    """
    扫描 short/mid 中的多条相关记忆，寻找共享结构。
    当 ≥ minimum_support 条记忆共享 semantic_tags 组合时：
    → 生成 inferred 类型的 MemoryEntry
    → support_count = 共享记忆数
    → reality_confidence 按 support/(support+counter+smoothing) 计算
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
    - short 层 trace_strength < threshold → 清除
    - mid 层 trace_strength 极低 → 清除或休眠
    - long 层极少使用 → is_dormant = true
    """
```

### 6. 推断验证门

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

### 7. 与现有 sleep_consolidator 的集成

新的 `memory_consolidation.py` 作为 `sleep_consolidator.py` 的替代方案。在 MemoryStore 层面提供：

```python
class MemoryStore:
    def run_consolidation_cycle(self, current_cycle: int, rng: random.Random) -> ConsolidationReport:
        """执行完整的四阶段固化。"""
```

通过桥接层确保旧 `LongTermMemory.replay_during_sleep()` 仍可工作。

## 验收标准

### G1: retrieval_multi_cue [BLOCKING]
- 检索结果按 retrieval_score 排序
- tag_overlap 贡献可独立验证
- mood_match 实现情绪一致性效应（负面情绪 → 更容易召回负 valence 记忆）
- accessibility 低的记忆检索分数低（即使标签完全匹配）
- is_dormant 记忆在普通检索中不出现
- 证据：≥5 个检索场景的分数分解

### G2: candidate_competition [BLOCKING]
- 分数悬殊时直接返回主干，confidence=high
- 分数接近时返回多个候选，confidence=low，interference_risk=true
- 证据：构造两组测试（悬殊 vs 接近），验证竞争行为

### G3: reconstruction_mechanism [BLOCKING]
- 三组触发条件均有覆盖测试
- 重构后 reality_confidence 下降
- 重构后 source_type = reconstruction
- 重构后 content_hash 变化 → version 递增
- 最多借用 1-2 条来源（不超出约束）
- 证据：重构前后的字段对比

### G4: reconsolidation [BLOCKING]
- 检索后 accessibility 回升
- 检索后 trace_strength 小幅回升
- retrieval_count 递增
- abstractness 小幅上升
- content 被改写时 version 递增
- 覆写式为默认（旧版本不保留，除非差异超阈值）
- 证据：再巩固前后字段变化记录

### G5: offline_consolidation_pipeline [BLOCKING]
- 四阶段依次执行
- 阶段 1：至少 1 条 short→mid 升级（给定足够高 salience 的条目）
- 阶段 2：至少 1 条 inferred 记忆被提取（给定重复模式）
- 阶段 3：重组产物标记 inferred + inference
- 阶段 4：低价值条目被清除或休眠
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
