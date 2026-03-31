# M4.1 差距评估与最小改造方案

日期：2026-03-31

结论：按当前这版验收标准，`M4.1 不可验收`。建议状态为 `BLOCK`。

## 一、评估范围

本评估按以下四条验收标准审查仓库当前实现：

1. 参数族完整性
2. 可观测性
3. 可区分性
4. 日志完整性

本评估不沿用仓库内置 `reports/m41_acceptance_report.json` 的 `PASS/ACCEPT` 结论，因为仓库当前 M4.1 规格与本次验收口径不一致。

## 二、总评

当前仓库中的 M4.1 实现更接近“旧口径/另一口径”的参数与日志原型，具备以下优点：

- 已有版本化参数 schema
- 已有版本化 decision log schema
- 已有 toy-style action scoring bridge
- 已有基础 ablation / stress artifact

但与本次验收要求相比，存在四类关键缺口：

- 核心参数数目不足，且缺失 `virtual_prediction_error_gain`
- 参数到行为的可观测映射不够，无法满足“每参数至少两个间接指标”
- 没有真正的盲测分类实验与准确率统计
- 决策日志字段不完整，无法满足完整性检查

## 三、逐条验收判断

### 1. 参数族完整性

判定：`不通过`

#### 现状证据

- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L16) 定义了 `CognitiveStyleParameters`
- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L18) 到 [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L24) 仅包含 7 个参数：
  - `uncertainty_sensitivity`
  - `error_aversion`
  - `exploration_bias`
  - `attention_selectivity`
  - `confidence_gain`
  - `update_rigidity`
  - `resource_pressure_sensitivity`
- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L50) 的 schema `required` 中也只有以上 7 项
- 全仓未发现 `virtual_prediction_error_gain`

#### 与标准的差距

- 缺少第 8 个核心参数 `virtual_prediction_error_gain`
- 当前实现未提供“每个参数的明确物理含义文档”，只有简短 observable 描述
- 当前仓库中的 [reports/m41_milestone_spec.md](/E:/workspace/segments/reports/m41_milestone_spec.md#L11) 仍是另一套参数口径，包含 `source_precision_gain` 与 `source_authority_weighting`

#### 最小改造项

1. 在 `CognitiveStyleParameters` 中新增 `virtual_prediction_error_gain`
2. 同步更新参数 schema、artifact schema、测试和默认配置
3. 新增一份参数文档，至少覆盖：
   - 名称
   - 值域
   - 默认值
   - 物理含义
   - 对决策过程的影响路径
   - 与可观测指标的关系
4. 明确当前 M4.1 的正式参数口径，避免与 source-trust 版本混用

#### 优先级

`P0`

### 2. 可观测性

判定：`不通过`

#### 现状证据

- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L198) 的 `observable_parameter_contracts()` 为每个参数只提供 1 条 observable 描述
- [artifacts/m41_behavior_mapping.json](/E:/workspace/segments/artifacts/m41_behavior_mapping.json) 中仅覆盖少量 pattern，且不是“每参数至少两个指标”的结构
- 当前测试仅检查 `confidence_gain` 是否出现在 contract 中，见 [tests/test_m41_cognitive_parameters.py](/E:/workspace/segments/tests/test_m41_cognitive_parameters.py#L30)

#### 与标准的差距

- 不满足“每个参数至少两个可从行为日志直接计算的观测指标”
- 不满足“映射不能是一对一恒等映射，必须通过行为间接推断”
- 未覆盖新增参数 `virtual_prediction_error_gain`

#### 建议的参数-指标最小集合

- `uncertainty_sensitivity`
  - 高不确定情境下 internal confidence 下降斜率
  - 高不确定情境下 commit 相对 inspect/scan 的回避比例
- `error_aversion`
  - 高 expected-error 候选被拒绝的比例
  - 错误后风险动作切换到保守动作的概率
- `exploration_bias`
  - 未知选项选择比例
  - 连续重复选择长度
- `attention_selectivity`
  - 高证据候选与干扰候选的价值差平均值
  - 注意力集中后选择与主证据一致的比例
- `confidence_gain`
  - 证据分离度提升时 confidence 增长斜率
  - 高证据条件下 commit 率提升幅度
- `update_rigidity`
  - 相同预测误差下的平均更新量
  - 错误反馈后策略切换滞后长度
- `resource_pressure_sensitivity`
  - 高压力条件下低成本动作比例
  - 资源下降时 rest/conserve/recover 触发阈值
- `virtual_prediction_error_gain`
  - 虚拟误差信号与直接反馈冲突时的决策偏移比例
  - 仅由 counterfactual / imagined loss 驱动的回避倾向

#### 最小改造项

1. 定义统一的“observable metrics registry”
2. 为每个参数绑定至少 2 个间接指标
3. 每个指标补齐：
   - 计算公式
   - 依赖日志字段
   - 指标方向性
   - 解释边界
4. 增加自动校验测试，确保每个参数都绑定不少于 2 个指标

#### 优先级

`P0`

### 3. 可区分性

判定：`不通过`

#### 现状证据

- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L470) 有 `parameter_identifiability_probe()`
- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L493) 仅返回三个布尔型 `identifiable` 判断
- [tests/test_m41_cognitive_parameters.py](/E:/workspace/segments/tests/test_m41_cognitive_parameters.py#L32) 只断言 `all(probe["identifiable"].values())`

#### 与标准的差距

- 没有至少三组 profile 的正式盲测实验设计
- 没有“分析者不知道参数配置、只看行为日志进行分类”的流程
- 没有分类准确率统计
- 没有达到或证明 `>= 80%` 的准确率

#### 建议的最小盲测实验设计

至少构造三组 agent profile：

- `高探索低谨慎`
- `低探索高谨慎`
- `中等均衡`

实验流程建议：

1. 在 toy world 中为每组 profile 生成多条独立日志轨迹
2. 隐藏真实参数标签，仅保留行为日志
3. 由分析脚本只基于行为指标进行分类
4. 输出：
   - overall accuracy
   - per-class precision / recall
   - confusion matrix
   - seeds / sample count

最低通过线：

- 总准确率 `>= 0.80`

#### 最小改造项

1. 新增 `profile_registry`
2. 新增 `blind_classification_experiment`
3. 输出 artifact，例如：
   - `artifacts/m41_blind_classification.json`
   - `reports/m41_blind_classification_summary.md`
4. 新增测试断言 accuracy 门槛

#### 优先级

`P0`

### 4. 日志完整性

判定：`不通过`

#### 现状证据

- [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L114) 到 [segmentum/m4_cognitive_style.py](/E:/workspace/segments/segmentum/m4_cognitive_style.py#L125) 的 `DecisionLogRecord` 当前字段为：
  - `schema_version`
  - `tick`
  - `seed`
  - `task_context`
  - `observation_evidence`
  - `candidate_actions`
  - `resource_state`
  - `internal_confidence`
  - `selected_action`
  - `prediction_error`
  - `update_magnitude`
- [artifacts/m41_cognitive_trace.json](/E:/workspace/segments/artifacts/m41_cognitive_trace.json) 的实际日志也仅包含上述字段

#### 与标准的差距

你要求每条决策记录必须包含：

- 时间戳
- 当前感知输入摘要
- 预测误差向量
- 注意力分配结果
- 资源状态
- 候选行动及其预期价值
- 实际选择
- 结果反馈
- 模型更新量

当前缺失或不充分项包括：

- `timestamp` 缺失
- “当前感知输入摘要”没有单独结构化字段
- “预测误差向量”当前只有单个 `prediction_error` 标量
- “注意力分配结果”缺失
- “结果反馈”缺失
- “候选行动及其预期价值”虽有 `candidate_actions`，但建议显式规范 `expected_value`
- “模型更新量”只有 `update_magnitude`，语义较粗

此外，当前没有“不合格率 <= 5%”的自动审计。

#### 最小改造项

建议把 decision log 扩展为至少包含：

- `timestamp`
- `percept_summary`
- `prediction_error_vector`
- `attention_allocation`
- `resource_state`
- `candidate_actions`
- `selected_action`
- `result_feedback`
- `model_update`

建议 `candidate_actions` 内至少包含：

- `action`
- `expected_value`
- `expected_confidence`
- `expected_prediction_error`
- `resource_cost`

建议新增日志审计器，输出：

- 总记录数
- 合格记录数
- 不合格记录数
- 不合格率
- 各字段缺失统计

#### 优先级

`P0`

## 四、建议的最小实现路线

### Phase 1：先补齐 schema 与日志

目标：让“参数族完整性”和“日志完整性”先达标。

改造项：

1. 扩展 `CognitiveStyleParameters` 到 8 参数
2. 重构 `DecisionLogRecord` 为完整日志 schema
3. 补齐 artifact 生成逻辑
4. 补一份正式参数文档

建议新增或修改文件：

- `segmentum/m4_cognitive_style.py`
- `segmentum/m41_audit.py`
- `scripts/generate_m41_acceptance_artifacts.py`
- `artifacts/m41_cognitive_schema.json`
- `reports/m41_parameter_reference.md`

### Phase 2：补观测指标注册表

目标：让“可观测性”可自动校验。

改造项：

1. 增加 `observable metrics registry`
2. 为每个参数补两个以上行为指标
3. 新增指标计算脚本与单元测试

建议新增文件：

- `segmentum/m41_observables.py`
- `tests/test_m41_observables.py`

### Phase 3：补盲测分类实验

目标：让“可区分性”可量化验收。

改造项：

1. 增加三组及以上 canonical profiles
2. 生成隐藏标签日志
3. 用行为指标进行 profile 分类
4. 输出 accuracy 与 confusion matrix

建议新增文件：

- `segmentum/m41_blind_classification.py`
- `tests/test_m41_blind_classification.py`
- `artifacts/m41_blind_classification.json`

## 五、建议新增测试清单

- `tests/test_m41_parameter_schema.py`
  - 断言 8 个参数全部存在
  - 断言默认值和值域齐全

- `tests/test_m41_decision_log_completeness.py`
  - 断言每条日志具备必填字段
  - 断言日志不合格率 <= 5%

- `tests/test_m41_observables.py`
  - 断言每个参数至少绑定 2 个间接指标
  - 断言指标都能从日志计算

- `tests/test_m41_blind_classification.py`
  - 断言至少 3 组 profile
  - 断言 blind accuracy >= 0.80

## 六、建议调整验收口径

如果项目决定采用你这版标准，建议同步做两件事：

1. 更新 [reports/m41_milestone_spec.md](/E:/workspace/segments/reports/m41_milestone_spec.md)
   - 移除或下放旧的 source-trust 参数口径
   - 明确采用 8 参数口径

2. 更新 [reports/m41_acceptance_report.json](/E:/workspace/segments/reports/m41_acceptance_report.json)
   - 当前报告中的 `PASS` 不能代表本次标准下已通过
   - 应新增对应四条标准的 gate

## 七、最终建议

当前建议：`不验收`

阻塞项：

- 缺失 `virtual_prediction_error_gain`
- 每参数少于 2 个间接观测指标
- 无盲测分类准确率证据
- 决策日志字段不完整

最小达标顺序建议：

1. 先补 8 参数与完整日志
2. 再补 observables registry
3. 最后补 blind classification 实验

在不大改整体架构的前提下，这是一条相对稳妥、返工最少的达标路径。
