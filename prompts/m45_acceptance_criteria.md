# M4.5 验收标准 — 记忆数据模型与编码机制

## 验收状态定义

- **PASS**: 所有 blocking gate 通过
- **FAIL**: 任一 blocking gate 未通过
- **BLOCKED**: 依赖项（M4.4）未通过

## Gate 列表

### G1: data_model_integrity [BLOCKING]
- MemoryEntry 包含文档定义的全部核心字段（≥25 个）
- to_dict() → from_dict() 往返后所有字段值相等
- content_hash 在 content 不变时保持稳定
- version 仅在 content_hash 变化时递增
- 证据：单元测试输出，字段数统计

### G2: salience_auditability [BLOCKING]
- compute_salience() 接受 4 个输入 + SalienceConfig
- 输出 = w_arousal×arousal + w_attention×attention + w_novelty×novelty + w_relevance×relevance
- 默认权重：0.30, 0.20, 0.20, 0.30
- 可打印每项贡献值的审计字符串
- 证据：3 组不同输入的精确数值验证

### G3: encoding_pipeline [BLOCKING]
- encode_memory() 从原始输入产出完整 MemoryEntry
- memory_class 四种分类均有覆盖（测试用例各至少 1 个）
- source_confidence / reality_confidence 按 source_type 表初始化
- 紧急固化：arousal > 0.9 → store_level=long
- 紧急固化：salience > 0.92 → store_level=long
- 证据：编码测试用例 + 紧急通道触发用例

### G4: dual_decay_correctness [BLOCKING]
- trace_strength 指数衰减，λ 按层级递减
- accessibility 指数衰减，λ > λ_trace（同层级）
- procedural + long 的 λ_trace 为所有组合中最小
- short 层 trace_strength < threshold → 条目清除
- long 层 trace_strength 极低 + accessibility 极低 → is_dormant=true
- 证据：衰减曲线数值验证（≥3 层级 × 2 变量 × 3 时间点）

### G5: legacy_bridge [BLOCKING]
- from_legacy_episodes() 能转换现有 Episode.to_dict() 产出的 payload
- 转换后保留 timestamp, action, outcome, prediction_error, total_surprise
- to_legacy_episodes() 反向转换后可被现有 LongTermMemory 消费
- 证据：桥接往返测试 + M4.1-M4.4 回归通过

### G6: store_level_transitions [BLOCKING]
- short → mid 升级条件可配置、有默认值
- mid → long 升级条件可配置、有默认值
- 升级后 trace_strength 和 accessibility 按新层级重新计算衰减系数
- 证据：升级场景测试用例

### G7: report_honesty [BLOCKING]
- 所有 gate 有非空 evidence 字段
- acceptance_state 与实际 gate 结果一致
- 无伪造通过

## 数值阈值

| 参数 | 值 |
|------|-----|
| EMERGENCY_AROUSAL_THRESHOLD | 0.9 |
| EMERGENCY_SALIENCE_THRESHOLD | 0.92 |
| SHORT_CLEANUP_TRACE_THRESHOLD | 0.05 |
| LONG_DORMANT_TRACE_THRESHOLD | 0.02 |
| LONG_DORMANT_ACCESS_THRESHOLD | 0.01 |
| DEFAULT_w_arousal | 0.30 |
| DEFAULT_w_attention | 0.20 |
| DEFAULT_w_novelty | 0.20 |
| DEFAULT_w_relevance | 0.30 |

## 回归

M4.1, M4.2, M4.3, M4.4 全部现有测试必须通过。
