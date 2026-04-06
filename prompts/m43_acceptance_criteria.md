# M4.3 验收标准

> Single-Task Behavioral Fit and Initial Falsification
>
> 验收原则：`M4.3` 验证的是参数化 agent 在**单一真实外部任务**上能否产生非平凡行为拟合，不是跨任务一致性或迁移。

---

## 目标

用 M4.2 搭好的 benchmark 环境，在 Confidence Database 和 Iowa Gambling Task 上分别
做单任务行为拟合，回答：

1. CognitiveStyleParameters 是否能解释人类关键行为模式？
2. 参数化 agent 是否能打败 trivial baselines？
3. 哪些参数是活跃的（对真实数据产生可测量的行为差异），哪些是惰性的？

## M4.3 从 M4.2 继承什么

稳定环境（直接使用）：

- `benchmark_registry.py` — bundle 发现/验证
- `m4_benchmarks.py` — ConfidenceDatabaseAdapter, IowaGamblingTaskAdapter 完整管道
- `m42_audit.py` — 环境验收 artifact 生成
- `external_benchmark_registry/` — 真实外部数据（Confidence DB 825,344 trials, IGT 11,800 trials）

M4.3 不继承以下为已完成工作：

- M4.1 sidecar 的参数恢复、分类、falsification 结论（需在真实数据上重做）
- M4.2 的 `_score_action_candidates` 启发式得分（这是 scaffold 而不是行为模型）
- 当前 `m43_modeling.py` 的 smoke-only 结果（trial_count=4, subject_count=1）

## 核心约束

### 必须使用外部 bundle

- 所有拟合和评估必须通过 `external_benchmark_registry/` 的真实数据
- `allow_smoke_test=True` 只能出现在 smoke 测试中，不能出现在验收路径
- 没有外部 bundle 时，验收必须诚实阻塞（blocked），不允许用 smoke fixture 假装通过

### 必须诚实标记

- `claim_envelope: "benchmark_eval"` — 真实外部 bundle 数据结果
- `claim_envelope: "smoke_only"` — repo smoke fixture 结果
- `external_validation: false` — 本仓库生成的所有东西
- 不允许用合成数据充当 benchmark 结果

## 必须通过的 Gates

### F1 Confidence Database Fit

- [ ] 在真实 Confidence DB 上运行 agent（非 smoke fixture）
- [ ] trial_count ≥ 1000，subject_count ≥ 10
- [ ] 打败所有 lower baselines (random, stimulus-only)
- [ ] 报告与至少一个 competitive baseline 的对比

### F2 Iowa Gambling Task Fit

- [ ] 在真实 IGT 上运行 agent（protocol_mode: standard_100）
- [ ] 至少覆盖 3 个 subjects
- [ ] 打败所有 lower baselines (random, frequency-matching)
- [ ] 报告与 competitive baseline (human behavior pattern) 的对比

### F3 Baseline Ladder

- [ ] 每个任务有至少两层 baseline：lower (must beat) 和 competitive (report gap)
- [ ] lower baselines 必须被打败，否则 M4.3 失败
- [ ] competitive baseline 对比结果必须诚实报告，打不过就说打不过
- [ ] ceiling baseline (per-subject best-fit) 作为参考上限

### F4 Parameter Sensitivity on Real Data

- [ ] 对 8 个参数逐一做 ±1σ 扫描（sweep），在真实数据上测量行为差异
- [ ] 报告哪些参数是 active（产生可测量的 metric 变化），哪些是 inert
- [ ] 至少 4/8 参数必须是 active，否则参数空间有冗余
- [ ] 标记为 `claim_envelope: "benchmark_eval"`

### F5 Honest Failure Analysis

- [ ] 识别 agent 系统性失败的模式（specific stimulus types, confidence ranges, IGT phases）
- [ ] 用真实 trial 的具体例子说明失败模式
- [ ] 如果打不过 random baseline，必须诚实报告
- [ ] 包含 `failure_modes` section

### F6 Non-Circular Scoring

- [ ] 拟合和评估不能用相同数据（train/test split 或 CV）
- [ ] subject 不能跨 split 泄漏
- [ ] 评估指标不能依赖 agent 自身输出的内部分数

### F7 Regression

- [ ] M4.1 所有测试仍然通过
- [ ] M4.2 所有测试仍然通过
- [ ] 不修改 CognitiveStyleParameters 接口
- [ ] 不修改 BenchmarkAdapter 协议

### R1 Report Honesty

- [ ] report 中每个 gate 都有 `passed` 和非空 `evidence`
- [ ] `status` 与 blocking gates 一致
- [ ] headline_metrics 包含 trial_count, subject_count, claim_envelope, external_bundle
- [ ] `recommendation` 只在所有 F-gates 通过且 sample size 足够时为 ACCEPT

## M4.3 正式交付文件

- `segmentum/m43_modeling.py` — 拟合逻辑（需大幅修改：加外部 bundle 路径、IGT 拟合、真实 baseline ladder）
- `segmentum/m43_baselines.py` — 独立 baseline 实现
- `segmentum/m43_audit.py` — 验收 artifact 生成
- `tests/test_m43_single_task_fit.py` — 拟合测试（需改为外部 bundle 路径）
- `tests/test_m43_baselines.py` — Baseline 正确性测试
- `tests/test_m43_acceptance.py` — 验收 gate 测试
- `artifacts/m43_confidence_fit.json` — Confidence DB 拟合结果
- `artifacts/m43_igt_fit.json` — IGT 拟合结果
- `artifacts/m43_parameter_sensitivity.json` — 参数敏感性分析
- `artifacts/m43_failure_analysis.json` — 失败模式分析
- `artifacts/m43_baseline_comparison.json` — Baseline 对比
- `reports/m43_acceptance_report.json` — 验收报告

## 明确不属于 M4.3 的内容

- 跨任务参数一致性（M4.4）
- 受控环境迁移（M4.8）
- 纵向稳定性（M4.9）
- 修改 CognitiveStyleParameters 参数数量
- 修改 BenchmarkAdapter 协议
- 用合成数据充当 benchmark 结果
- 对 `_score_action_candidates` 启发式做架构级重写（可调参但不改结构）

## 已知的当前问题

1. **现有 `m43_modeling.py` 全部用 `allow_smoke_test=True`** — 所有拟合跑在 4 条 trial 上，
   subject_count=1。这是 smoke scaffold，不是 benchmark 结果。必须改为外部 bundle 路径。

2. **没有 IGT 拟合** — 现有代码只做 Confidence DB，完全没有 IGT 单任务拟合。

3. **Grid search 是暴力穷举** — 81 点网格搜索，没有从数据中学习。对 4 条 trial 有效，
   对 825,344 条 trial 不可接受（太慢）。需要更高效的拟合策略。

4. **Baselines 是手调公式** — `run_signal_detection_baseline` 用固定 slope=3.6，
   不是真正的 SDT。需要独立实现的 baseline。

5. **Tests assert FAIL** — `test_m43_acceptance.py` 第 14 行 `assertEqual(report["status"], "FAIL")`。
   当验收通过时，这个测试会反而失败。需要改为条件断言或分 blocked/pass 两条路径。

6. **Test 覆写正式 artifact** — 与 M4.2 同样的问题，测试写入正式路径。
