# M4.4 验收标准

> Cross-Task Parameter Consistency
>
> 验收原则：M4.4 验证的是参数化 agent 在两个真实外部任务上的**跨任务参数一致性**。
> 核心产出是诚实的参数稳定性分类图——哪些参数是跨任务可移植的 trait，哪些是任务绑定的 artifact。
> 这是一个**诊断性里程碑**，而不是胜利里程碑。"只有 3/8 参数跨任务稳定"是完全合法的通过结果，
> 只要证据真实、方法论健全。

---

## 目标

用 M4.3 的单任务拟合结果作为锚点，在 Confidence Database 和 Iowa Gambling Task 上做
跨任务联合拟合，回答：

1. 是否存在一组参数能在两个任务上同时达到可接受的表现？
2. 联合拟合相比任务专用拟合，各任务的性能退化有多大？
3. 8 个参数中，哪些是跨任务稳定的（trait-like），哪些是任务敏感的？
4. 结论是否对联合目标函数的权重选择鲁棒？

## M4.4 从 M4.3 继承什么

稳定环境（直接使用）：

- `m43_modeling.py` — 单任务拟合函数、scoring 函数、模拟函数
- `m43_baselines.py` — 独立 baseline 实现
- `m4_benchmarks.py` — Adapters、数据加载、泄漏检测
- `m4_cognitive_style.py` — CognitiveStyleParameters（不修改）
- `external_benchmark_registry/` — 真实外部数据
- M4.3 fitted parameters — 作为输入/锚点，不作为默认值

M4.4 不继承以下为已完成工作：

- M4.3 的 "7/8 active" 结论（需在联合语境中重新验证）
- M4.3 的 `deck_match_rate` 作为 IGT 唯一评估指标（需补充聚合指标）

## 核心约束

### 必须使用外部 bundle

- 所有拟合和评估必须通过 `external_benchmark_registry/` 的真实数据
- 没有外部 bundle 时，验收必须诚实阻塞（blocked）
- Smoke 测试只用于 CI，不出现在验收路径

### 必须诚实标记

- `claim_envelope: "benchmark_eval"` — 真实外部数据结果
- `claim_envelope: "smoke_only"` — smoke fixture 结果
- `external_validation: false` — 本仓库产出的所有东西
- 不允许强行声称一致性：如果联合拟合退化显著，必须如实报告

### 不允许修改核心接口

- 不修改 `CognitiveStyleParameters` 参数数量或接口
- 不修改 `BenchmarkAdapter` 协议
- 不修改 `_score_action_candidates` 签名或行为
- 可以新增分析代码和聚合指标层

---

## 已知的架构背景（影响 gate 设计）

M4.3 审查发现 IGT `deck_match_rate` 只有 0.309（random 0.247），原因是 agent 模拟自身
IGT 体验导致路径发散，per-trial 匹配的理论天花板很低。

**因此 M4.4 的 IGT 评估使用双轨制：**
- `deck_match_rate`：保留报告，但不作为跨任务一致性的 blocking gate
- `igt_behavioral_similarity`（聚合指标）：学习曲线、损失后切换、牌组分布、探索-利用转换。
  这些聚合指标不受路径发散影响，作为 IGT 侧的主要评估维度

---

## 必须通过的 Gates

### G1 Joint Fit Exists

- [ ] 在真实数据上成功运行联合参数拟合
- [ ] 输出一组 `joint_fitted_parameters`
- [ ] 联合目标函数同时包含 Confidence 和 IGT 子目标
- [ ] trial_count 和 subject_count 满足 M4.3 同等要求（Confidence ≥ 1000 trials, IGT ≥ 3 subjects）
- [ ] 标记为 `claim_envelope: "benchmark_eval"`

### G2 Degradation Bounded

- [ ] 联合参数在 Confidence 上的 heldout_likelihood 退化 ≤ 10% relative to M4.3 task-specific
- [ ] 联合参数在 IGT 上的 igt_behavioral_similarity 退化 ≤ 20% relative to M4.3 task-specific
- [ ] 退化矩阵完整报告（3 parameter sets × 2 tasks × 全指标）
- [ ] 如果退化超标，诚实报告并标记为 finding，不伪造通过
- [ ] 注意：G2 通过不代表 consistency 好——只代表联合拟合没有灾难性退化

### G3 Parameter Stability Map

- [ ] 8 个参数中每个都有 classification：`stable` / `task_sensitive` / `inert` / `indeterminate`
- [ ] 每个 classification 有量化 evidence（gap, degradation, ablation 结果）
- [ ] 至少 2 个参数被分类为 `stable`（如果 0-1 个是 stable，整体架构可能有问题）
- [ ] 至少 1 个参数被分类为 `task_sensitive`（如果 0 个是 task_sensitive，说明分类粒度不足）
- [ ] `resource_pressure_sensitivity` 的分类与 M4.3 inert 结论一致

### G4 Cross-Application Matrix

- [ ] 报告 Confidence-specific 参数在 IGT 上的表现
- [ ] 报告 IGT-specific 参数在 Confidence 上的表现
- [ ] 报告 joint 参数在两个任务上的表现
- [ ] 总共 3×2 = 6 个单元格，每个有完整指标

### G5 Weight Sensitivity

- [ ] 至少 3 种权重向量的联合拟合结果
- [ ] 报告参数向量在不同权重下的最大分量差异
- [ ] 如果差异 > 0.15 on any parameter，标记该参数为 weight-sensitive
- [ ] 结论不被权重选择主导（或诚实报告被主导的情况）

### G6 IGT Aggregate Metrics

- [ ] 实现并报告 igt_behavioral_similarity 及其子指标
- [ ] 子指标至少包含：learning_curve_distance, post_loss_switch_gap, deck_distribution_l1, exploration_exploitation_entropy_gap
- [ ] 在真实 IGT 数据上计算（非 smoke）
- [ ] 报告 agent vs human 在每个子指标上的具体数值

### G7 Architecture Assessment

- [ ] 量化 IGT per-trial matching ceiling（最广参数扫描下的 best deck_match_rate）
- [ ] 对比 ceiling 与 aggregate metrics 的信息量
- [ ] 包含对 M4.8 的结构性建议（是否应改用 aggregate 作为 IGT 主指标）
- [ ] 标记为 finding 而非 gate failure

### G8 Honest Failure Analysis

- [ ] 如果联合拟合退化超标，识别具体是哪些参数的妥协导致的
- [ ] 如果 task_sensitive 参数 > 5 个，分析是否因为 IGT 模型太弱导致虚假的 task sensitivity
- [ ] 用真实 trial 的具体例子说明退化模式
- [ ] 区分三种退化来源：参数妥协、IGT 架构限制、数据不足

### G9 Non-Circular Scoring

- [ ] 联合拟合使用 training data，评估使用 heldout data
- [ ] 不使用 M4.3 的 heldout data 做 M4.4 的 training（M4.3 heldout 仍然是 M4.4 heldout）
- [ ] Subject 不跨 split 泄漏
- [ ] degradation 计算基于 heldout 指标

### G10 Regression

- [ ] M4.1 所有测试仍然通过
- [ ] M4.2 所有测试仍然通过
- [ ] M4.3 所有测试仍然通过
- [ ] 不修改 CognitiveStyleParameters 接口
- [ ] 不修改 BenchmarkAdapter 协议

### R1 Report Honesty

- [ ] report 中每个 gate 都有 `passed` 和非空 `evidence`
- [ ] `status` 与 blocking gates 一致
- [ ] headline_metrics 包含 joint_degradation, stable_parameter_count, task_sensitive_count
- [ ] `recommendation` 只在所有 G-gates 通过时为 ACCEPT
- [ ] 如果 M4.4 的结论是 "大部分参数是 task-sensitive"，这不是 failure——前提是证据充分

---

## Gate 阻塞性分级

| Gate | Blocking | 说明 |
|------|----------|------|
| G1 Joint Fit Exists | YES | 没有联合拟合就没有 M4.4 |
| G2 Degradation Bounded | YES | 联合拟合必须不比 random 差；但阈值宽松 |
| G3 Parameter Stability Map | YES | 核心交付物，必须存在且有量化证据 |
| G4 Cross-Application Matrix | YES | 核心分析，不可缺少 |
| G5 Weight Sensitivity | NO | 诊断性——报告了即可，不要求权重无关 |
| G6 IGT Aggregate Metrics | YES | 没有这个，IGT 侧的一致性评估不可信 |
| G7 Architecture Assessment | NO | 诊断性 finding，不阻塞验收 |
| G8 Honest Failure Analysis | YES | 必须有失败分析 |
| G9 Non-Circular Scoring | YES | 方法论基线 |
| G10 Regression | YES | 不破坏已有里程碑 |
| R1 Report Honesty | YES | 报告结构完整性 |

---

## M4.4 正式交付文件

- `segmentum/m44_igt_aggregate.py` — IGT 聚合行为指标
- `segmentum/m44_cross_task.py` — 联合拟合、退化分析、参数分类
- `segmentum/m44_audit.py` — 验收 artifact 生成
- `tests/test_m44_cross_task.py` — 联合拟合测试
- `tests/test_m44_igt_aggregate.py` — IGT 聚合指标测试
- `tests/test_m44_acceptance.py` — 验收 gate 测试
- `artifacts/m44_joint_fit.json` — 联合参数拟合结果
- `artifacts/m44_degradation.json` — 退化分析
- `artifacts/m44_parameter_stability.json` — 参数稳定性分类
- `artifacts/m44_weight_sensitivity.json` — 权重敏感性检查
- `artifacts/m44_igt_aggregate.json` — IGT 聚合指标结果
- `artifacts/m44_architecture_assessment.json` — 架构评估 findings
- `reports/m44_acceptance_report.json` — 验收报告
- `reports/m44_acceptance_summary.md` — 人类可读摘要

---

## 明确不属于 M4.4 的内容

- 受控环境迁移（M4.8）
- 纵向稳定性（M4.9）
- Grid world 或工具环境
- 修改 CognitiveStyleParameters 参数数量
- 修改 BenchmarkAdapter 协议
- 重写 `_score_action_candidates`
- 用合成数据充当 benchmark 结果
- 群体层面聚类（per-subject parameter fitting 是 out of scope）

---

## M4.4 通过后对下游的影响

M4.4 的参数稳定性分类将直接影响 M4.8 和 M4.9 的设计：

- **stable 参数**：M4.8 可以信任这些参数在新环境中的行为预测
- **task_sensitive 参数**：M4.8 需要为每个新环境重新拟合这些参数，或者接受它们在新环境中的表现可能退化
- **inert 参数**：M4.8/M4.9 可以考虑冻结或移除
- **architecture assessment**：如果 M4.4 确认 IGT per-trial matching 是死胡同，M4.8 应该从一开始就用 aggregate 指标

## 验收心态

M4.4 的正确心态是**科学诊断**，不是工程交付：

- "只有 3 个参数是 stable" = 合法结果（如果证据充分）
- "联合拟合在 Confidence 上退化 8%" = 合法结果（如果诚实报告）
- "IGT 侧的一致性信号被路径发散噪声淹没" = 合法 finding
- "我们无法区分真正的 task sensitivity 和建模不足" = 合法的 indeterminate 分类

不合法的：
- "所有参数完美一致" 但没有退化数据支撑
- 降低 IGT baseline 来让联合拟合看起来更好
- 忽略权重敏感性结果
- 把 M4.3 的 smoke test 结果包装成 M4.4 的 benchmark 结果
