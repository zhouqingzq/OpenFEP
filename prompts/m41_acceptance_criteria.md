# M4.1 验收标准

> Cognitive Variable Operationalization and Unified Interfaces
>
> 验收原则：`M4.1` 只验证接口层是否成立，不验证 benchmark 环境、任务拟合或人类数据结论。

---

## 目标

把“有限能量约束下固化的先验偏好结构”翻译为三套统一接口：

- 参数接口：`CognitiveStyleParameters`
- 观测接口：parameter-to-observable contracts
- 日志接口：`DecisionLogRecord`

这些接口要能被后续 benchmark 与开放环境工作复用。

## 必须通过的 Gates

### G1 Schema Completeness

- [ ] 八维参数 schema 稳定、支持 roundtrip
- [ ] `DecisionLogRecord` 必需字段齐全
- [ ] 内部试次日志的 `parameter_snapshot` 完备率为 100%

### G2 Trial Variability

- [ ] 同参数不同 seed 会产生不同轨迹
- [ ] 同参数同 seed 会产生相同轨迹
- [ ] 改参数会改变轨迹

### G3 Observability

- [ ] 每个参数至少绑定两个可执行观测指标
- [ ] 所有指标 evaluator 可执行
- [ ] 稀疏日志会正确触发 `insufficient_data`

### G4 Intervention Sensitivity

- [ ] 对每个参数做干预时，目标观测指标按预期方向变化
- [ ] 效应量达到当前 registry 设定的最低阈值

### G5 Log Completeness

- [ ] 日志 invalid rate ≤ 0.05
- [ ] 关键字段都能被审计

### G6 Stress Behavior

- [ ] 压力模式下能观察到低成本/恢复导向行为上升

### R1 Report Structure

- [ ] report 中每个 gate 都有 `passed` 和非空 `evidence`
- [ ] `status` 与 blocking gates 一致

## 明确不属于 M4.1 的内容

- benchmark adapter 与 bundle 环境搭建
- external bundle provenance
- benchmark replay / leakage 检查
- 任务级 behavioral fit
- baseline comparison
- blind classification
- parameter recovery
- falsification

## 口径要求

- `M4.1` 可以是最小可执行内部模拟器，但不能被表述成真实人类验证。
- 如果某项结论依赖 benchmark task，它应进入 `M4.2` 或更后面。
- 如果某项结论依赖 held-out fit、human-alignment 或 baseline，对应 `M4.3` 或更后面。
