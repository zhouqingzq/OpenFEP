# M4.9 Work Prompt — 纵向稳定性验证（原 M4.6）

> **编号变更说明**：本里程碑原编号 M4.6，因将记忆系统重排为 M4.5 / M4.6 / M4.7 而后移至 M4.9。

## 前置依赖

M4.8（开放世界接口接入）必须通过。

## 目标

Implement a longitudinal stability milestone for cognitive-style differentiation.

## Requirements

- Run multiple seeds across multiple style profiles.
- Measure within-profile restart drift, between-profile divergence, and recovery-retention after corruption.
- **新增**：纵向运行中 MemoryStore 的记忆层级分布应保持稳定（不出现 short 层持续溢出或 long 层无限增长）。
- **新增**：验证记忆系统的再巩固机制在 500+ cycle 后不导致认知风格漂移（即记忆更新不改变 cognitive style 的行为表达方向）。
- Quantify acceptance with explicit thresholds:
  - style stability over at least 500 decision cycles using ICC, with above 0.7 counted as stable,
  - style reproducibility across 5 seed-separated runs using clustering purity, with at least 85 percent purity counted as strong evidence,
  - non-trivial style differentiation using a between-profile versus within-profile effect size target, with Cohen's d above 0.8 counted as strong evidence,
  - perturbation recovery using post-recovery cosine similarity, with above 0.7 counted as good recovery.
  - **新增**：memory layer distribution stability — 固化后各层占比的变异系数 < 0.3 across seeds.
- Produce divergence, recovery, stress, and acceptance artifacts.
- Regress M4.8, M4.7, M4.4, M4.3, and continuity/restart acceptance evidence.
