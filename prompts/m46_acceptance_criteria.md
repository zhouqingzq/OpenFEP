# M4.6 验收标准 — 检索、再巩固与离线固化

## 前置依赖

M4.5 必须 PASS。

## Gate 列表

### G1: retrieval_multi_cue [BLOCKING]
- retrieval_score = w1×tag_overlap + w2×context_overlap + w3×mood_match + w4×accessibility + w5×recency
- 默认权重 0.35/0.15/0.15/0.20/0.15
- 情绪一致性效应可观测（负面 mood → 高 valence 负记忆排名上升）
- accessibility=0.01 的记忆即使标签完全匹配也排名低于 accessibility=0.8 的部分匹配
- is_dormant=true 的记忆不出现在普通检索结果中
- RetrievalResult 最终输出包含 `recall_hypothesis`，而不是只返回 entry/top-k
- `recall_hypothesis` 可追踪主干来源与辅助来源
- 证据：≥5 个检索场景的分数分解表

### G2: candidate_competition [BLOCKING]
- dominance_threshold=0.15（可配置）
- 第一名与第二名差距 > threshold → confidence=high, interference_risk=false
- 差距 < threshold → confidence=low, interference_risk=true, competitors 非空
- 低置信度回忆被视为合法输出
- `competing_interpretations` 或等价结构非空且可审计
- 证据：2 组测试用例（悬殊 vs 接近）

### G3: reconstruction_mechanism [BLOCKING]
- 条件 A: abstractness>0.7 AND len(content)<50 → 触发
- 条件 B: abstractness>0.7 AND memory_class==semantic → 触发
- 条件 C: reality_confidence<0.4 AND retrieval_count>0 → 触发
- 重构后：reality_confidence 下降，source_type=reconstruction
- 借用来源 ≤ 2 条
- content_hash 变化 → version += 1
- episodic strong anchors 无支持证据时不得被改写
- episodic weak anchors 可被补全
- semantic / inferred 条目允许比 episodic 更高的重构自由度
- reconstruction_trace 记录主干来源、借用来源、补全字段与保护字段
- 证据：三组条件各 1 个测试用例

### G4: reconsolidation [BLOCKING]
- 每次成功检索后：accessibility += boost_access（建议 0.15-0.25）
- trace_strength += boost_trace（建议 0.02-0.05）
- retrieval_count += 1
- abstractness += 微量（建议 0.005-0.01）
- last_accessed 更新
- procedural 核心动作序列默认不允许无证据改写
- 冲突场景默认降低 reality_confidence、增加 counterevidence_count、生成 competing_interpretations，而不是直接覆盖旧 episodic
- 证据：再巩固前后数值对比

### G5: offline_consolidation_pipeline [BLOCKING]
- 四阶段完整执行
- 升级：≥1 条 short→mid（高 salience + 高 retrieval_count）
- 模式提取：≥1 条 inferred 或 semantic skeleton 被生成（给定 ≥ minimum_support 条共享结构记忆）
- 重组：产物 memory_class=inferred, source_type=inference
- 清理：低 trace_strength 的 short 条目被清除
- 至少 1 个多 episodic → semantic skeleton 用例，且 skeleton 保留 support 链
- ConsolidationReport 包含每阶段统计数字
- 证据：完整固化周期的 report

### G6: inference_validation_gate [BLOCKING]
- inference_write_score 公式显式，输入项可追踪
- 高 support + 低 contradiction → score > threshold → 可升级 long
- 低 support 或高 contradiction → score < threshold → 留 mid
- 证据：各 1 个通过/未通过用例

### G7: legacy_integration [BLOCKING]
- MemoryStore.run_consolidation_cycle() 可调用
- LongTermMemory.replay_during_sleep() 在桥接下仍返回合法 replay batch
- M4.1-M4.4 + M4.5 回归通过
- 证据：回归测试输出

### G8: report_honesty [BLOCKING]
- 所有 gate 有非空 evidence
- 无伪造通过

## 数值阈值

| 参数 | 默认值 |
|------|--------|
| retrieval w1 (tag_overlap) | 0.35 |
| retrieval w2 (context_overlap) | 0.15 |
| retrieval w3 (mood_match) | 0.15 |
| retrieval w4 (accessibility) | 0.20 |
| retrieval w5 (recency) | 0.15 |
| dominance_threshold | 0.15 |
| reconstruction_abstract_threshold | 0.70 |
| reconstruction_content_min_length | 50 |
| reconstruction_confidence_threshold | 0.40 |
| boost_access | 0.20 |
| boost_trace | 0.03 |
| abstractness_increment | 0.008 |
| minimum_support (pattern extraction) | 5 |
| smoothing (rule confidence) | 2.0 |

## 回归

M4.1, M4.2, M4.3, M4.4, M4.5 全部测试通过。

## 2026-04-09 Official Status Note

The current repository-wide official status for M4.6 is `INCOMPLETE / NOT_ISSUED`.
The present blocking gate is `legacy_integration=NOT_RUN` in the acceptance builder.
Any committed report claiming a pass state is superseded by the current builder output.
## M4.10 Supersession Note

Sections in this document that describe keyword-table salience, template-string semantic skeletons, inferred-pattern text, or text-only replay are superseded by M4.10. After M4.10, acceptance follows dynamic encoding, attention-budget competition, centroid/residual semantic consolidation, and replay re-encoding.
