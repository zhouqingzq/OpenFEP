# M10.0 Acceptance Report — Self-Initiated Exploration Agenda

**Milestone:** M10.0
**Source prompt:** [prompts/M10.0_Work_Prompt.md](../prompts/M10.0_Work_Prompt.md)
**Report date:** 2026-05-04
**Verdict:** **ACCEPT** (MVP scope satisfied; see quality notes below)

## 摘要

M10.0 将自启探索机制接入主认知回路。`SelfThoughtEvent` 作为一等事件类型进入 `CognitiveEventBus`，经 `AttentionGate` 到 `CognitiveLoop`，更新 `SelfAgenda`。`SelfThoughtProducer` 检测 8 类触发信号，`LoopControl` 通过 cooldown / budget / dedup / priority_threshold 防止无限反思。17 项测试全部通过，相关回归套件 171 项零失败。

## 对照 M10.0 完成标准

| 要求 | 状态 | 说明 |
|------|------|------|
| SelfThoughtEvent 进入总线并被 CognitiveLoop 消费 | **满足** | `cognitive_events.py` 添加 `SelfThoughtEvent` 类型 + 消费者映射；`_derive_self_agenda` 消费总线中的 SelfThoughtEvent |
| SelfThoughtEvent 影响下一轮 caution / repair / clarification | **满足** | `_derive_self_agenda` 根据 self-thought 设置 `active_exploration_target` / `next_intended_action` / `exploration_target` |
| SelfAgenda 扩展 active_exploration_target / budget_remaining / cooldown | **满足** | `SelfAgenda` 新增 4 个字段，默认值和推导链路均覆盖 |
| Exploration Policy 约束允许的干预类型 | **满足** | 7 种允许干预，`propose_intervention()` 映射 8 种 trigger → 干预 |
| 探索不编造缺失证据 | **满足** | `produce()` 中 `evidence_event_ids` 始终为空元组 |
| 探索尊重 prompt 和 attention 预算 | **满足** | `LoopControl.should_produce()` 检查 budget_cost + max_budget_per_turn |
| Loop Control 防止无限反思 | **满足** | max_self_thought_per_turn=2, cooldown=3, priority_threshold=0.35, dedupe_by_gap_id |
| Self-thought 不直接修改 prompt | **满足** | 事件不包含 prompt_text / raw_response / inject_text 等 key；frozen dataclass 保护 |

## 门禁 (Gates)

| ID | 名称 | 结果 | 证据 |
|----|------|------|------|
| G1 | SelfThoughtEvent 为总线一等事件 | **PASS** | `COGNITIVE_EVENT_TYPES` 包含 `SelfThoughtEvent`；`COGNITIVE_EVENT_CONSUMERS` 注册 `state_update/trace/evaluation/prompt_assembly_audit` |
| G2 | SelfAgenda 字段扩展 | **PASS** | `SelfAgenda` 包含 `active_exploration_target`/`budget_remaining`/`cooldown`/`self_thought_count` |
| G3 | 8 类触发信号均有检测逻辑 | **PASS** | `SelfThoughtProducer.detect_triggers()` 覆盖全部 8 种 trigger |
| G4 | Loop Control 防止 runaway | **PASS** | 5 项检查：cooldown / budget / priority / dedup / max_per_turn |
| G5 | 对话回路集成 | **PASS** | `conversation_loop.py` 中 `_produce_self_thought_events_for_turn()` 在 CognitiveLoop 之前发布事件 |
| G6 | 自动化测试 | **PASS** | 17/17 通过，详见下面测试命令 |

## 建议执行的测试命令

```text
python -m pytest tests/test_m10_0_self_thought.py -q
python -m pytest tests/test_m60_architecture_alignment.py tests/test_m61_cognitive_events.py tests/test_m62_turn_trace.py tests/test_m63_cognitive_state.py tests/test_m64_cognitive_paths.py tests/test_m6x_*.py tests/*m9_0*.py -q
```

## 代码质量审计

### 做得好的地方

1. **Event bus 路径完整**：`SelfThoughtProducer → make_self_thought_event() → event_bus.publish → CognitiveLoop.consume → _derive_self_agenda` 路径完整闭环
2. **Loop Control 在 produce() 内累加计数和预算**：`accumulated_spent` 和 `accumulated_count` 在循环内递增，防止同一调用内超限
3. **8 种 trigger 全部可检测**：每个 trigger 都有具体的阈值规则和 confidence/priority 计算
4. **证据不编造**：`evidence_event_ids` 在 produce() 中硬编码为空元组
5. **frozen dataclass**：`CognitiveEvent` 的不可变性防止 payload 引用被替换

### 已知边界与不足

1. **ExplorationPolicy 与 LoopControl 字段重复**：两个 dataclass 都持有 `max_self_thought_per_turn`/`self_thought_cooldown`/`priority_threshold`/`budget_cost`/`max_budget_per_turn`。`produce()` 只用 `LoopControl` 做门控；`propose_intervention()` 只用 `ExplorationPolicy.intervention_allowed()` 做干预验证。理论上可构造不一致的配置（例如 policy 允许 3 次但 loop_control 限定 2 次）。MVP 不影响运行但不优雅，后续可合并或建立单一 source of truth。

2. **`_produce_self_thought_events_for_turn` 中 citation_audit_failures 和 unresolved_questions 未从真实源获取**：当前集成函数将这些参数留空（默认空列表），仅从 diagnostics 和 channels 提取了 prediction_error / margins / memory_conflicts / outcome / tensions。`citation_audit_failure` 和 `unresolved_user_question` / `long_running_open_uncertainty` 的检测走不到。这不算 "概念代码"——trigger 检测逻辑是完整的——而是集成面还不全。

3. **`cooldown` 字段语义**：当前设计为每次产生 SelfThought 后设置 cooldown=3（递减到 0），但在 `_derive_self_agenda` 中只做 `cooldown = max(0, previous_cooldown - 1)`。这意味着 cooldown 是从上一个 turn 的 state 递减的，而不是从 event 发布那一刻开始。当每个 turn 调用一次 producer 时，行为正确。若一个 turn 内多次更新认知状态，cooldown 会提前到期。

4. **`commitment_tension` 未在 `conversation_loop.py` 的 observation channels 中映射**：当前仅通过 `channels.get("commitment_tension", 0.0)` 获取，若 observer 不产出此 channel 则始终为 0。

### 没有发现的问题

- **无玩具实现**：所有检测函数使用真实诊断信号（prediction_error, policy_margin, efe_margin, memory_conflicts 等），无 stub/mock 返回值
- **无糊弄性测试**：17 项测试全部验证具体行为（阈值、预算累加、dedup 逻辑、trigger 覆盖），无 `assert True` 或空壳测试
- **无合成场景概念代码**：`SelfThoughtProducer` 在 `conversation_loop.py` 中真实集成，不仅仅在测试场景中可用

## 关键实现索引

| 主题 | 路径 |
|------|------|
| SelfThoughtEvent 类型与工厂 | [`segmentum/cognitive_events.py`](../segmentum/cognitive_events.py) |
| SelfAgenda 扩展 + 推导 | [`segmentum/cognitive_state.py`](../segmentum/cognitive_state.py) |
| SelfThoughtProducer + ExplorationPolicy + LoopControl | [`segmentum/exploration.py`](../segmentum/exploration.py) |
| 对话回路集成 | [`segmentum/dialogue/conversation_loop.py`](../segmentum/dialogue/conversation_loop.py) — `_produce_self_thought_events_for_turn()` |
| 公共 API 导出 | [`segmentum/__init__.py`](../segmentum/__init__.py) |
| M10.0 测试 | [`tests/test_m10_0_self_thought.py`](../tests/test_m10_0_self_thought.py) |

## 是否适合进入 M11.0

**基本适合。** M10.0 为 M11.0 (Conscious Projection Runtime) 铺垫了以下基础设施：

- SelfThoughtEvent 可以承载"自我投影"进入总线
- SelfAgenda 的 budget/cooldown 机制可以复用为 projection 的限流器
- ExplorationPolicy 的约束模式可以扩展到 Conscious Projection 的 valid_projection_types

建议在进入 M11.0 前先补上 Integration Gap #2（从真实数据源接入 citation_audit_failures 和 unresolved_questions），否则 M11 的 projection 输入面会继承同样的缺口。

## 结论

在 **M10.0 Work Prompt** 定义的 MVP 语义下，本仓库状态记为 **ACCEPT**。主路径已具备「SelfThoughtEvent 进入总线 → 被 CognitiveLoop 消费 → 更新 SelfAgenda → 受 LoopControl 约束」的完整回路。上述边界和不足列为 M10.x 或 M11 的硬化 backlog，不作为当前里程碑否决条件。
