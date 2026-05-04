# M9.0 Acceptance Report — Memory Dynamics Integration

**Milestone:** M9.0  
**Source prompt:** [prompts/M9.0_Work_Prompt.md](../prompts/M9.0_Work_Prompt.md)  
**Report date:** 2026-05-04  
**Verdict:** **ACCEPT** (MVP scope satisfied; see residual risks below)

## 摘要

本仓库在 M9.0 要求下，已将「记忆动态」接入主对话路径：统一证据结构进入生成边界、检索干扰经总线与认知状态、记忆写入路径上的价值—衰减审计、以及记忆干扰触发的 `SubjectState` 补丁提案与提交。自动化测试覆盖 M9 单测与相关回归套件，当前结论为 **通过验收（带已知边界）**。

## 对照 M9.0 完成标准

完成标准（原文意译）：记忆动态应能通过**事件**与 **patch proposal** 因果影响认知，并保留 source、confidence、value、cue、conflict 边界。

| 要求 | 状态 | 说明 |
|------|------|------|
| 事件路径 | **满足（MVP）** | `run_conversation` 在 `PathSelectionEvent` 之后发布 `MemoryRecallEvent`、`MemoryInterferenceEvent`；`update_cognitive_state` → `_derive_memory` 消费并合并 bus 载荷。 |
| Patch 路径 | **满足（收窄范围）** | 记忆干扰检测到后，经 `m9_state_patch_runtime` 产生 `StatePatchProposal`，通过置信门槛后 `StateCommitEvent` 入账本并小幅调整 `SubjectState`；事件写回总线。 |
| 边界 | **满足（MVP）** | 统一 `MemoryEvidence` + `interference_controls` 写入 `ResponseEvidenceContract`；cue 与 unknown 桶在 builder 中显式标注。 |

## 门禁（Gates）

| ID | 名称 | 结果 | 证据 |
|----|------|------|------|
| G1 | 统一记忆证据表面 | **PASS** | [`segmentum/memory_evidence.py`](../segmentum/memory_evidence.py)；[`build_response_evidence_contract`](../segmentum/memory_anchored.py) 使用 `unify_evidence`，合同字段 `unified_evidence` / `interference_controls`。 |
| G2 | 价值保留与衰减审计 | **PASS** | [`MemoryStore._audit_m9_retention_for_entry`](../segmentum/memory_store.py) 在 `add()` 后写入 `compression_metadata.m9_retention` 与历史列表。 |
| G3 | Cue 与未知立场 | **PASS** | Builder 在无 cue 时增加 `ltm_requires_explicit_cue`；与 `unify_evidence(..., current_cue=...)` 一致；单测 [`tests/test_m9_0_cue_recall.py`](../tests/test_m9_0_cue_recall.py)。 |
| G4 | 干扰与过主导反馈 | **PASS** | `derive_interference_feedback` + `apply_interference_to_evidence_contract` 接入对话证据合同；[`tests/test_m9_0_interference_feedback.py`](../tests/test_m9_0_interference_feedback.py)。 |
| G5 | StatePatch / Commit MVP | **PASS** | [`segmentum/m9_state_patch_runtime.py`](../segmentum/m9_state_patch_runtime.py)；[`tests/test_m9_0_state_patch.py`](../tests/test_m9_0_state_patch.py)；`SegmentAgent.state_patch_log`。 |
| G6 | 自动化测试 | **PASS** | 见下一节命令（在本提交基线上执行）。 |

## 建议执行的测试命令

在仓库根目录：

```text
python -m pytest tests/test_m9_0_memory_evidence.py tests/test_m9_0_cue_recall.py tests/test_m9_0_retention_decay.py tests/test_m9_0_interference_feedback.py tests/test_m9_0_state_patch.py tests/test_m8_9_response_evidence_contract.py tests/test_m63_cognitive_state.py -q
python -m pytest tests/test_m6x_cognitive_event_bus_loop.py tests/test_m65_meta_control_guidance.py tests/test_m66_prompt_capsule_upgrade.py tests/test_m64_cognitive_paths.py tests/test_m62_turn_trace.py -q
```

（Windows PowerShell 下请勿依赖 `tests/test_m9_0_*.py` 通配；应列出文件或使用 `-k m9_0`。）

## 已知边界与后续硬化（非本报告否决项）

1. **StatePatch 触发源**：当前主要绑定「记忆干扰」；FEP、SelfThought、Outcome 等其它提案源未全部接入同一套策略。  
2. **M9 保留审计挂载点**：仅在 `MemoryStore.add()`；`upsert_legacy_episode`、纯检索路径等未统一打同一审计。  
3. **SubjectState 与 `derive_subject_state`**：后续决策若重算 subject，可能削弱本拍 patch 的数值残留；若需强一致，应在 derive 链路中显式合并 patch 账本。  
4. **StatePatchProposal 与 CognitiveLoop 同一 tick 消费**：提案在循环后由对话驱动应用，而非 reducer 内统一消费；若架构上要求「仅 loop 提交」，可再收敛。

## 关键实现索引

| 主题 | 路径 |
|------|------|
| 总线载荷构建 | [`segmentum/m9_bus_integration.py`](../segmentum/m9_bus_integration.py) |
| 对话内发布与证据合同 | [`segmentum/dialogue/conversation_loop.py`](../segmentum/dialogue/conversation_loop.py) |
| 认知状态合并 bus 记忆事件 | [`segmentum/cognitive_state.py`](../segmentum/cognitive_state.py) |
| 证据合同与 prompt 渲染 | [`segmentum/memory_anchored.py`](../segmentum/memory_anchored.py) |
| 存储层保留/衰减日志 | [`segmentum/memory_store.py`](../segmentum/memory_store.py) |
| SubjectState 补丁运行时 | [`segmentum/m9_state_patch_runtime.py`](../segmentum/m9_state_patch_runtime.py) |
| Agent 侧账本 | [`segmentum/agent.py`](../segmentum/agent.py)（`state_patch_log`） |

## 结论

在 **M9.0 Work Prompt** 定义的 MVP 语义下，本仓库状态记为 **ACCEPT**：主路径已具备「事件 + 证据 + 干扰控制 + 存储审计 + 可追踪 patch/commit」。上述「已知边界」列为 M9.x 或 M10 的硬化 backlog，不作为当前里程碑否决条件。
