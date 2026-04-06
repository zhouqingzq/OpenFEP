# M4.8 Work Prompt — 开放世界接口接入与行为投射（原 M4.5）

> **编号变更说明**：本里程碑原编号 M4.5，因将记忆系统重排为 M4.5 / M4.6 / M4.7 而后移至 M4.8。

## 前置依赖

M4.7（记忆系统完整集成 + 行为验证）必须通过。新记忆系统为本里程碑提供：
- 注意力预算（AgentStateVector + encoding_attention）→ 支持 compute/attention-budget constraints
- 信息来源可靠性（source_confidence / reality_confidence）→ 支持 partially reliable cue sources
- 认知风格与记忆的双向集成 → 支持 parameter settings yield separable strategies
- 层级固化 + 动态衰减 → 支持长期运行不溢出

## 目标

Implement a controlled complex-environment milestone that maps benchmark-constrained cognitive parameters into survival-oriented behavior in MiniGrid or an equivalent partially observable grid world.

## Requirements

- Explicitly route cognitive parameters into action scoring for resource gathering, threat avoidance, and information search.
- **新增**：通过 MemoryStore 的 encode_memory() 实现 cue source 的 source_confidence 差异化建模，而非仅用参数旁路。
- **新增**：通过 AgentStateVector.threat_level 和 salience 动态调节实现 attention-budget constraints，而非硬编码 budget cap。
- Add partially reliable cue sources so `source_precision_gain` can be measured behaviorally.
- Add compute or attention-budget constraints to test graceful degradation under scarce processing resources.
- Produce mapping, transfer, ablation, stress, and acceptance artifacts.
- Enforce these acceptance rules:
  - different parameter settings must yield statistically separable behavioral strategies,
  - parameter rankings inferred from Confidence Database and IGT must align with controlled-environment behavior with Spearman rho above 0.6,
  - under tighter compute budgets the DOTTORE agent must degrade selectively rather than uniformly,
  - high and low `source_precision_gain` settings must diverge significantly under both helpful and misleading cues.
- Open-world tool environments are deferred to M5; M4.8 is the bridge milestone that validates cross-context style stability before real tools.
- **新增**：验证 MemoryStore 在 grid-world 中的行为：episodic 记忆记录探索历史，semantic 提取环境规律，procedural 形成导航技能。
- Regress M4.7, M4.4 and M3.6 acceptance.
