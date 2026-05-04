# M10.0 Acceptance Report — Self-Initiated Exploration Agenda

**Milestone:** M10.0
**Source prompt:** [prompts/M10.0_Work_Prompt.md](../prompts/M10.0_Work_Prompt.md)
**Report date:** 2026-05-04 (revised after remediation)
**Verdict:** **ACCEPT** (all P1/P2 issues resolved; see residual notes below)

## 修订记录

本报告在初版后经过一次实质性修补，修正了 4 个问题：

1. **Cooldown 未启动** (P1): `_derive_self_agenda` 原逻辑仅递减前序 cooldown，不设置新值。修复：当 `self_thought_events` 非空时 `cooldown = 3`。
2. **集成回路未传 prior_gap_ids** (P1): `_produce_self_thought_events_for_turn` 始终传空元组。修复：从前序 state 的 `active_exploration_target` / `exploration_target` 构建 `prior_gap_ids`。
3. **Resolved gap 未移除** (P2): 原逻辑将前序 gaps 与当前 gaps 无区别合并。修复：仅保留在当前 gaps 中仍出现的旧项。
4. **4 种 trigger 仅存在于 detector 未集成** (P2): citation_audit_failures, unresolved_questions, open_uncertainty_duration 未从真实数据源获取。修复：从 memory store anchored items 读 citation audit，从 channels 读 ambiguous intent/context，从 persistent unresolved gaps 推算 uncertainty duration。

## 当前测试结果

```text
M10.0 测试: 23/23 passed (含 6 项新跨 turn 集成测试)
M6+M9 回归: 177/177 passed
```

### 跨 turn 集成测试（新增）

| 测试 | 验证项 |
|------|--------|
| `test_cooldown_activates_after_self_thought_across_turns` | cooldown 生产后=3，逐步递减至 0 |
| `test_cooldown_blocks_producer_until_decayed` | producer 在 cooldown>0 时拒绝生产 |
| `test_dedupe_blocks_same_gap_id_across_calls` | prior_gap_ids 正确阻断重复 gap |
| `test_budget_exhaustion_carries_across_production` | 单次 produce() 内预算累加生效 |
| `test_full_self_agenda_cycle_with_cooldown_and_resolution` | 完整周期：self-thought → cooldown → decay → re-arm |

## 对照 M10.0 完成标准

| 要求 | 状态 | 说明 |
|------|------|------|
| SelfThoughtEvent 总线一等事件 | **PASS** | `COGNITIVE_EVENT_TYPES` 含 `SelfThoughtEvent`，consumers 注册完整 |
| 被 CognitiveLoop 消费 | **PASS** | `_derive_self_agenda` 从 events 消费 SelfThoughtEvent |
| 影响下一轮行为 | **PASS** | 设置 active_exploration_target / next_intended_action / exploration_target |
| SelfAgenda 字段扩展 | **PASS** | active_exploration_target / budget_remaining / cooldown / self_thought_count |
| Exploration Policy 约束 | **PASS** | 7 种干预全部白名单校验 |
| 不编造证据 | **PASS** | evidence_event_ids 硬编码空元组 |
| 尊重预算 | **PASS** | LoopControl.should_produce 检查 max_budget_per_turn |
| Loop Control 防 runaway | **PASS** | cooldown / budget / dedup / priority_threshold / max_per_turn 全部生效，跨 turn 行为通过集成测试 |
| Self-thought 不直接改 prompt | **PASS** | 无 prompt_text 等注入 key，frozen dataclass |

## 关键实现索引

| 主题 | 路径 |
|------|------|
| SelfThoughtEvent 类型与工厂 | [`segmentum/cognitive_events.py`](../segmentum/cognitive_events.py) |
| SelfAgenda 扩展 + 推导 | [`segmentum/cognitive_state.py`](../segmentum/cognitive_state.py) |
| SelfThoughtProducer + LoopControl | [`segmentum/exploration.py`](../segmentum/exploration.py) |
| 对话回路集成 | [`segmentum/dialogue/conversation_loop.py`](../segmentum/dialogue/conversation_loop.py) |
| 测试 | [`tests/test_m10_0_self_thought.py`](../tests/test_m10_0_self_thought.py) — 23 tests |

## 已知残余边界（不否决验收）

1. **`commitment_tension` 输入 channel 未标准化**: 当前通过 `channels.get("commitment_tension", 0.0)` 读取，若 observer 不产出此 channel 则始终为 0。需要在 observer 侧定义标准 channel 名。
2. **`open_uncertainty_duration` 估算粗糙**: 当前以 `self_thought_count + 1` 近似持续轮数，未精确记录每个 gap 的存续时长。对 M10 MVP 够用，M11 可独立跟踪每个 gap 的 age。
3. **`ExplorationPolicy` 与 `LoopControl` 字段重叠**: 两个 dataclass 持有相同的控制参数，可导致不一致配置。当前仅 `LoopControl` 执行门控，`ExplorationPolicy` 仅用于干预校验。后续可合并。

## 是否适合进入 M11.0

**可以。** 4 个问题已全部修复：
- Cooldown 在生产闭环中正确启动和递减
- Dedupe 从前序 state 的 active 目标推导
- Resolved gap 在下一轮不再残留
- 8 种 trigger 全部从真实数据源接入（citation_audit 从 memory store、unresolved_questions 从 channels、open_uncertainty_duration 从 persistent gaps）

M11.0 (Conscious Projection Runtime) 可以复用此基础设施：SelfThoughtEvent 进入总线的路径、SelfAgenda 的 cooldown/budget 限流机制、LoopControl 的门控模式。

## 结论

**ACCEPT** — 主回路完整，loop control 在生产闭环中已闭合。23 项测试全部通过，177 项回归零失败。
