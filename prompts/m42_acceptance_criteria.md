# M4.2 验收标准

> Cognitive Benchmark Environment Setup
>
> 验收原则：`M4.2` 验证的是 benchmark 环境是否建好并可复现，不是 benchmark 质量是否已经成立。

---

## 目标

基于 `M4.1` 的统一接口，搭建可运行、可追溯、可复现的认知 benchmark 环境。

## M4.2 从 M4.1 继承什么

稳定接口（可直接使用）：

- `CognitiveStyleParameters` — 八维参数 dataclass
- `DecisionLogRecord` — 决策日志 schema
- `observable_parameter_contracts()` — 参数→观测映射
- `compute_observable_metrics()` — 从日志计算指标
- `run_cognitive_style_trial()` — 最小模拟器（smoke test 用）

M4.2 不继承以下为已完成工作（它们是 sidecar 预研，必须在 benchmark 上重新验证）：

- `m41_inference.py` 的参数恢复结论
- `m41_blind_classifier.py` 的分类准确率
- `m41_baselines.py` 的 baseline 对比结论
- `m41_falsification.py` 的 falsification 结论
- `m41_identifiability.py` 的可识别性结论
- `data/m41_external/` 的"外部"数据结论

## 必须通过的 Gates

### P1 Bundle Provenance

- [ ] benchmark registry 能区分 repo smoke fixture 与 acceptance-grade external bundle
- [ ] manifest 校验通过
- [ ] `smoke_test_only`、`source_type`、`is_synthetic` 等字段语义一致

### P2 Claim Honesty

- [ ] smoke fixture 只能给出 smoke-only 级别口径
- [ ] 缺少 acceptance-grade bundle 时，report 必须诚实阻塞
- [ ] 不允许用 repo fixture 伪装正式 benchmark acceptance

### I1 Protocol Schema

- [ ] Confidence Database、IGT、bandit 的协议字段完整
- [ ] trace export 可验证
- [ ] task-level protocol metadata 清楚可追踪

### I2 Adapter Execution

- [ ] 各 benchmark adapter 可运行
- [ ] protocol mode、trial count、subject/session 信息能正确落入 artifacts

### I3 Determinism And Replay

- [ ] 同 seed 同输入可重放
- [ ] 不同 seed 会产生可区分结果
- [ ] reproducibility artifact 清楚记录 replay 证据

### I4 Leakage Checks

- [ ] subject/session 不跨 split 泄漏
- [ ] leakage report 可导出并进入 acceptance artifacts

### I5 Smoke Fixture Rejection

- [ ] acceptance path 不把 repo smoke fixture 当正式外部 bundle
- [ ] report 会显式说明 blocked / smoke-only / acceptance-grade

### R1 Report Honesty

- [ ] report 区分 blocked、smoke-only、pass、fail
- [ ] artifacts、provenance、selected source 都有记录

## M4.2 正式交付文件

- `segmentum/benchmark_registry.py`
- `segmentum/m4_benchmarks.py`
- `segmentum/m42_audit.py`
- `tests/test_m42_benchmark_adapter.py`
- `tests/test_m42_confidence_benchmark.py`
- `tests/test_m42_external_bundle_integration.py`
- `tests/test_m42_reproducibility.py`
- `tests/test_m42_acceptance.py`
- `reports/m42_acceptance_report.json`

## 明确不属于 M4.2 的内容

- benchmark-quality 结论
- strong human-alignment 结论
- baseline superiority
- latent parameter identifiability
- falsification
- 把 M4.1 sidecar 的合成结果当作 benchmark 证据

这些属于 `M4.3` 或更后面的里程碑。
