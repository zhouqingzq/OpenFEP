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

建立统一记忆库的数据模型，实现编码流水线（感知输入 → salience 计算 → 分类 → 写入），实现衰减与紧急固化通道。

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
| `relevance` | float | 综合相关性 0.0 ~ 1.0 |
| `salience` | float | 显式公式计算的综合显著性 |
| `trace_strength` | float | 痕迹强度（慢变量）|
| `accessibility` | float | 可提取性（快变量）|
| `abstractness` | float | 0.0 具体 ~ 1.0 抽象 |
| `source_confidence` | float | 来源置信度 0.0 ~ 1.0 |
| `reality_confidence` | float | 内容真实性置信度 0.0 ~ 1.0 |
| `semantic_tags` | list[str] | 语义标签 |
| `context_tags` | list[str] | 场景标签 |
| `mood_context` | str | 编码时情绪语境 |
| `retrieval_count` | int | 被检索次数 |
| `support_count` | int | 支持证据次数 |
| `counterevidence_count` | int | 冲突证据次数 |
| `derived_from` | list[str] | 派生来源 id 列表 |
| `version` | int | 仅 content_hash 变化时递增 |
| `is_dormant` | bool | 是否休眠 |

必须实现 `to_dict()` / `from_dict()` 序列化，与现有 Episode payload 格式桥接。

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

### 3. 编码流水线

在 `memory_encoding.py` 中实现：

```python
def encode_memory(
    raw_input: dict,        # 感知输入或内部事件
    current_state: dict,    # 当前 agent 状态（用于计算 relevance）
    config: SalienceConfig,
) -> MemoryEntry:
    """
    感知输入 → 计算 arousal/attention/novelty/relevance
    → salience → 分类 memory_class/source_type
    → 生成 content 摘要和标签
    → 返回 MemoryEntry（store_level=short）
    """
```

分类规则：
- `episodic`：带具体时间/地点/角色的经历
- `semantic`：从多次经历中抽象出的规律
- `procedural`：动作序列/操作模式
- `inferred`：推断/重组产物

初始 source_confidence / reality_confidence 赋值规则：

| source_type | source_confidence | reality_confidence |
|-------------|------------------|--------------------|
| experience | 0.9 | 0.85 |
| rehearsal | 0.8 | 0.8 |
| hearsay | 0.7 | 0.5 |
| inference | 0.9 | 0.35 |
| reconstruction | 0.4 | 0.5 |

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

### 7. 与现有 LongTermMemory 的桥接

不破坏 `LongTermMemory` 的外部 API。在 `MemoryStore` 中实现：
- `from_legacy_episodes(episodes: list[dict]) -> MemoryStore`：将现有 episode payload 转换为 MemoryEntry
- `to_legacy_episodes() -> list[dict]`：反向转换

确保 M4.1-M4.4 的所有测试仍然通过。

## 验收标准

### Gate 1: 数据模型完整性
- `MemoryEntry` 包含所有核心字段
- `to_dict()` / `from_dict()` 往返无损
- `content_hash` 在 content 不变时稳定，变化时递增 version

### Gate 2: Salience 可审计性
- 对任意 MemoryEntry，可打印 `arousal=X, attention=Y, novelty=Z, relevance=W → salience=公式展开`
- 权重来自 SalienceConfig，不硬编码在公式体内
- 单元测试：固定输入 → 固定输出，可复现

### Gate 3: 编码流水线功能
- 给定感知输入，产出带完整字段的 MemoryEntry
- memory_class 分类逻辑覆盖四种类型
- source_confidence / reality_confidence 按 source_type 正确初始化
- 高 arousal 输入触发紧急固化（store_level=long）

### Gate 4: 双变量衰减正确性
- trace_strength 衰减慢于 accessibility（同层级、同 elapsed）
- short 层衰减快于 mid 快于 long
- procedural + long 的 trace_strength 衰减最慢
- 衰减后 trace_strength=0 的 short 条目被清除
- 衰减后双低的 long 条目标记 is_dormant

### Gate 5: 桥接兼容性
- `from_legacy_episodes()` 能转换现有 episode payload
- 转换后的 MemoryEntry 保留原始 timestamp、action、outcome 等关键信息
- M4.1-M4.4 全部测试通过（回归）

### Gate 6: 存储层级转移
- MemoryEntry 的 store_level 可从 short→mid→long 升级
- 升级条件明确：高 salience + 高 retrieval_count，或紧急固化
- 降级/清除条件明确：低 trace_strength

### Gate 7: 报告诚实性
- 所有 gate 有非空 evidence
- acceptance_state 准确反映实际状态
- 不伪造通过

## 回归要求

M4.1, M4.2, M4.3, M4.4 所有现有测试必须通过。
