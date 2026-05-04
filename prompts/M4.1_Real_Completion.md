# Prompt: 真正完成 M4.1

## 目标

把认知变量操作化成统一接口，而不是把 `M4.1` 做成一个混杂了 benchmark、
fit、baseline、外部验证口号的总包。

## 这轮工作应该完成什么

1. 固化参数接口：
   - `CognitiveStyleParameters` 的 schema、默认值、roundtrip
   - 八维参数的名称、语义、范围、决策路径说明

2. 固化观测接口：
   - `observable_metrics_registry()`
   - `observable_parameter_contracts()`
   - 每个参数至少两个可执行间接指标

3. 固化日志接口：
   - `DecisionLogRecord`
   - `audit_decision_log()`
   - 参数快照、候选动作、证据、置信度、更新字段齐全

4. 提供最小可执行内部场景：
   - 仅用于证明接口不是空壳
   - 允许是 toy internal simulator
   - 不得包装成 benchmark environment 或 human-data validation

5. 让 `m41_audit.py` 只验证接口层 gates：
   - schema completeness
   - trial variability
   - observability
   - intervention sensitivity
   - log completeness
   - stress behavior
   - report structure

## 明确不要做的事

- 不要把 benchmark adapter、bundle、registry 塞进 `M4.1`
- 不要把 blind classification、parameter recovery、falsification 当成 `M4.1` 必选验收
- 不要把任何 same-framework synthetic 结果说成 external validation
- 不要为了“显得更完整”把 `M4.2/M4.3` 的内容偷渡回 `M4.1`

## 输出要求

- 文档口径必须把 `M4.1` 明确写成 interface layer
- 报告里不得暗示已经完成 benchmark environment 或 behavioral fit
- 如果保留 synthetic sidecar 模块，必须明确它们不是 `M4.1` acceptance evidence
