# M4.1 验收标准

> Cognitive Variable Operationalization and Unified Interfaces
>
> 验收原则：`M4.1` 只验证接口层是否成立，不验证 benchmark 环境、任务拟合或人类数据结论。

---

## 目标

把"有限能量约束下固化的先验偏好结构"翻译为三套统一接口：

- 参数接口：`CognitiveStyleParameters`
- 观测接口：parameter-to-observable contracts
- 日志接口：`DecisionLogRecord`

这些接口要能被后续 benchmark 与开放环境工作复用。

## 必须通过的 Gates

### G1 Schema Completeness

- [x] 八维参数 schema 稳定、支持 roundtrip
- [x] `DecisionLogRecord` 必需字段齐全
- [x] 内部试次日志的 `parameter_snapshot` 完备率为 100%

### G2 Trial Variability

- [x] 同参数不同 seed 会产生不同轨迹
- [x] 同参数同 seed 会产生相同轨迹
- [x] 改参数会改变轨迹

### G3 Observability

- [x] 每个参数至少绑定两个可执行观测指标
- [x] 所有指标 evaluator 可执行
- [x] 稀疏日志会正确触发 `insufficient_data`

### G4 Intervention Sensitivity

- [x] 对每个参数做干预时，目标观测指标按预期方向变化
- [x] 效应量达到当前 registry 设定的最低阈值

### G5 Log Completeness

- [x] 日志 invalid rate ≤ 0.05
- [x] 关键字段都能被审计

### G6 Stress Behavior

- [x] 压力模式下能观察到低成本/恢复导向行为上升

### R1 Report Structure

- [x] report 中每个 gate 都有 `passed` 和非空 `evidence`
- [x] `status` 与 blocking gates 一致

## M4.1 正式交付文件

仅以下文件构成 M4.1 验收证据：

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `segmentum/m41_explanations.py`
- `tests/test_m41_cognitive_parameters.py`
- `tests/test_m41_observables.py`
- `tests/test_m41_decision_logging.py`
- `tests/test_m41_acceptance.py`
- `reports/m41_acceptance_report.json`

## 明确不属于 M4.1 的内容

以下内容虽挂 `m41_` 前缀，但属于合成诊断 sidecar，不计入 M4.1 验收：

- `m41_inference.py` — 玩具参数恢复引擎 → M4.3 预研
- `m41_blind_classifier.py` — 跨生成器合成分类器 → M4.3 sidecar
- `m41_baselines.py` — 同框架 baseline → M4.3 baseline ladder
- `m41_falsification.py` — 内部灵敏度检查 → M4.3 falsification
- `m41_identifiability.py` — 同框架可恢复性分析 → M4.3 identifiability
- `m41_external_generator.py` — 第二合成生成器（不是外部数据）
- `m41_external_dataset.py` — holdout 数据加载器
- `m41_external_observables.py` — 备用观测计算
- `m41_external_validation.py` — task eval 转发
- `m41_external_task_eval.py` — 属于 M4.2 范围
- `data/m41_external/` — 合成 holdout 数据（不是人类数据）
- 所有 `artifacts/m41_*.json`（blind classification、baseline、falsification 等）

## 口径要求

- `M4.1` 可以是最小可执行内部模拟器，但不能被表述成真实人类验证。
- 如果某项结论依赖 benchmark task，它应进入 `M4.2` 或更后面。
- 如果某项结论依赖 held-out fit、human-alignment 或 baseline，对应 `M4.3` 或更后面。
- 任何 sidecar artifact 不得标记 `external_validation: true`。同框架跨生成器结果应标记为 `cross_generator_synthetic`。
