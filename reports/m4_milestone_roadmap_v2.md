# M4 里程碑路线图 v2 — 含仿人类记忆系统

## 编号变更

| 新编号 | 原编号 | 名称 | 状态 |
|--------|--------|------|------|
| M4.1 | M4.1 | 认知风格参数空间 | PASS |
| M4.2 | M4.2 | 单任务行为区分 | PASS |
| M4.3 | M4.3 | 真实 IGT 拟合与验收 | PASS |
| M4.4 | M4.4 | 跨任务稳定性检验 | PASS |
| **M4.5** | 原 M4.5a | 记忆数据模型与编码机制 | PENDING |
| **M4.6** | 原 M4.5b | 检索、再巩固与离线固化 | PENDING |
| **M4.7** | 原 M4.5c | 动态调节、认知风格集成与行为验证 | PENDING |
| **M4.8** | 原 M4.5 | 开放世界接口接入与行为投射 | PENDING |
| **M4.9** | 原 M4.6 | 纵向稳定性验证 | PENDING |

## 依赖链

```
M4.4 (PASS)
└─ M4.5 (原 M4.5a: 数据模型 + 编码 + 衰减)
   └─ M4.6 (原 M4.5b: 检索 + 再巩固 + 固化)
      └─ M4.7 (原 M4.5c: 动态调节 + 认知风格集成 + 行为验证)
         └─ M4.8 (开放世界接口, 原 M4.5)
            └─ M4.9 (纵向稳定性, 原 M4.6)
```

## 为什么将记忆系统重排为 M4.5-M4.7

### 瓶颈分析

当前 `memory.py` 中的 `LongTermMemory` 存在以下问题，直接阻碍当前 M4.8（原 M4.5）的实现：

1. **无注意力预算概念** — M4.8（原 M4.5）要求 "compute or attention-budget constraints"，但当前编码机制对所有输入等权处理
2. **无信息来源可靠性** — M4.8（原 M4.5）要求 "partially reliable cue sources"，但当前无 source_confidence
3. **语义提取硬编码** — 只有 7 种 event type（关键词匹配），grid-world 新场景无法被覆盖
4. **记忆不参与认知风格表达** — M4.8（原 M4.5）要求 "different parameter settings yield separable strategies"，但参数只影响决策打分，不影响记忆编码/检索
5. **1024 episode 硬上限** — M4.9（原 M4.6）要求 500+ cycle 纵向运行，当前无层级固化，早期关键记忆会被挤出

### 收益

完成 M4.5/M4.6/M4.7 后：
- M4.8（原 M4.5）可直接使用 `source_confidence` 建模 cue 可靠性
- M4.8 可直接使用 `AgentStateVector` 实现 attention-budget
- M4.8 中 cognitive style 通过记忆行为（而非仅决策权重）影响 grid-world 策略
- M4.9（原 M4.6）的纵向运行有层级固化保障，不会溢出

## 各里程碑核心交付物

### M4.5 — 数据模型与编码
- `segmentum/memory_model.py`：MemoryEntry（25+ 核心字段）
- `segmentum/memory_encoding.py`：salience 计算 + 编码流水线
- `segmentum/memory_decay.py`：双变量衰减
- `segmentum/memory_store.py`：统一记忆库 + 桥接

### M4.6 — 检索与固化
- `segmentum/memory_retrieval.py`：多线索激活 + 候选竞争 + 重构
- `segmentum/memory_consolidation.py`：四阶段离线固化
- 推断验证门

### M4.7 — 集成与验证
- `segmentum/memory_state.py`：AgentStateVector + 动态调节
- 认知风格 × 记忆 5 个集成点
- 3 个行为验证场景
- MemoryAwareAgent 接口

### M4.8 — 开放世界接口（原 M4.5 + 记忆增强）
- MiniGrid/grid-world 环境
- 认知参数 → grid-world 行为映射
- source_confidence 驱动的 cue 可靠性
- attention-budget 约束

### M4.9 — 纵向稳定性（原 M4.6 + 记忆稳定性）
- 500+ cycle × 多 seed × 多 profile
- ICC, clustering purity, Cohen's d, recovery similarity
- 记忆层级分布稳定性
