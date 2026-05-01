# M6 Roadmap: Resource-Bounded Cognitive Loop

## Vision

M6 的目标不是新增一个孤立的“自我问答模块”，而是把现有 Segmentum / OpenFEP 的对话运行时升级为一个可观察、可调节、可反馈的资源受限认知闭环。

核心方向从：

```text
Self-Cognition: 我是谁 / 我要做什么 / 我怎么做 / 我还缺什么
```

升级为：

```text
Resource-Bounded Cognitive Loop:
事件流 -> 注意力筛选 -> 认知状态场 -> 候选路径竞争 -> meta-control
-> prompt/action -> outcome -> memory update
```

M6 应该融入现有系统，而不是另起炉灶。现有的 `SegmentAgent`、`AttentionBottleneck`、`GlobalWorkspace`、`MetaCognitiveLayer`、`FEPPromptCapsule` 和 M5 dialogue runtime 已经提供了大部分底座。M6 的任务是让这些模块之间形成更清晰的内部事件流、状态整合层和闭环调控机制。

一句话定义：

```text
M6 = 将现有 FEP dialogue runtime 事件化、状态化、闭环化。
```

---

## Core Thesis

M6 采用的理论链条是：

```text
有限资源
-> 候选认知路径竞争
-> 当前自由能 / 未来 EFE / 能量 / 注意力 / 记忆 / 控制成本共同形成路径成本
-> 记忆保存低成本可复用路径
-> 抽象是跨场景降低未来自由能的压缩结构
-> 高阶观察者调节精度、成本权重、更新增益、探索温度和控制增益
```

Transformer 不被视为完整认知系统，而是外环可调度的语义器官：

```text
Transformer:
- 候选解释生成器
- 候选行动生成器
- 上下文压缩器
- 用户意图解释器
- 局部模拟器
- 自然语言生成器

Agent outer loop:
- 维护状态变量
- 计算路径成本
- 更新路径权重
- 管理记忆
- 管理资源预算
- 调整 meta-control 参数
- 决定最终给 Transformer 的 prompt guidance
```

M6 的重点不是模拟“人会想哪些句子”，而是模拟：

```text
有限资源智能体如何维护状态、竞争路径、复用记忆、识别缺口、调节控制参数。
```

---

## Local Architecture Fit

M6 必须贴合当前项目结构。它不应复制现有决策系统，而应在现有链路上加事件、状态和调控。

当前关键链路：

```text
DialogueObserver.observe
-> SegmentAgent.decision_cycle_from_dict
-> build_fep_prompt_capsule
-> ResponseGenerator.generate
-> classify_dialogue_outcome
-> SegmentAgent.integrate_outcome
```

M6 概念与现有模块的对应关系：

| M6 Concept | Existing Local Anchor | M6 Role |
|---|---|---|
| Observation input | `segmentum/dialogue/observer.py` | 产生对话观察和 channel values |
| Signal bus | `segmentum/io_bus.py` | 扩展为认知事件基础设施 |
| Attention gate | `segmentum/attention.py` | 复用 salience / bottleneck 思路筛选事件 |
| Global workspace | `segmentum/workspace.py` | 现有广播和 suppressed channel 机制继续保留 |
| Path competition | `segmentum/agent.py` | 复用 `ranked_options` / `InterventionScore` / `policy_score` |
| Prompt compression | `segmentum/dialogue/fep_prompt.py` | 扩展 `FEPPromptCapsule`，不要另建 prompt capsule |
| Meta-control | `segmentum/metacognitive.py` | 扩展为可输出 control guidance 的高阶调节层 |
| Outcome feedback | `segmentum/dialogue/outcome.py` + `integrate_outcome` | 结果反馈进入 memory / gap / meta-control |
| Trace output | `segmentum/tracing.py` | 输出 turn-level cognitive trace |

需要避免的割裂设计：

```text
不要新建一套平行的 cognition runtime 来替代 SegmentAgent。
不要重新实现 attention、decision_cycle、metacognition、prompt capsule。
不要让 message bus 变成没人消费的日志系统。
不要把原始事件流直接塞进 prompt。
```

推荐定位：

```text
MessageBus / CognitiveEventLayer = 内部事件可观测性基础设施
AttentionGate = 认知事件筛选器
CognitiveState = 从事件流派生出的状态场
PathCompetition = 现有 ranked_options 的认知路径视图
MetaControl = 对 prompt / policy / memory 的慢速调节信号
PromptAdapter = FEPPromptCapsule 的增强层
```

---

## Target Loop

M6 完成后的单轮对话目标流程：

```text
1. Observe
   DialogueObserver 读取用户输入、上下文、关系状态。

2. Publish ObservationEvent
   将观察结果、channel values、uncertainty、source 写入认知事件流。

3. Attention Select
   通过 salience / priority / ttl / source 筛选本轮高价值事件。

4. Update CognitiveState
   更新 task / memory / gap / meta-control 等 MVP 状态。

5. Decision Cycle
   复用 SegmentAgent.decision_cycle_from_dict 生成 ranked_options。

6. Path View
   将 ranked_options 映射为 CognitivePath view，而不是替换原策略系统。

7. Meta-Control
   根据低 margin、高冲突、高负载、失败 outcome 等调节 guidance。

8. Prompt Assembly
   扩展 FEPPromptCapsule，加入 selected path、gap、meta-control guidance。

9. Generate
   Transformer / rule generator 根据压缩后的 guidance 生成回复。

10. Outcome Feedback
    分类 outcome，写回 memory，并影响下一轮 gap / meta-control。
```

最终闭环：

```text
Cognitive events
-> Attention-selected events
-> CognitiveState MVP
-> Existing path competition
-> Meta-control guidance
-> Prompt/action
-> Outcome
-> Memory and next-turn control
```

---

## Affective Maintenance Addendum

M6 should include a lightweight affective maintenance layer, but it should not become a full emotion simulator or a second personality system.

Recommended placement:

```text
AffectiveStateMVP = a bounded subsection of CognitiveStateMVP
MetaControlGuidance = consumes affective state and emits compressed stance guidance
FEPPromptCapsule = carries affective summaries, not raw affective notes
TurnTrace / Dashboard = expose affective maintenance in readable audit form
```

MVP fields:

```text
mood_valence
arousal
social_safety
irritation
warmth
fatigue_pressure
repair_need
decay_rate
affective_notes
```

Inputs:

```text
emotional_tone
conflict_tension
hidden_intent with low certainty
stress / fatigue / energy pressure
prior outcome
social gap / repair result
bounded decay from previous affective state
```

Prompt rule:

```text
Use affective state only as compressed conversational guidance:
deescalate, preserve warmth, reduce intensity, repair gently.
Do not expose raw affective notes or confident claims about the user's emotions.
```

Evaluation rule:

```text
Closed-loop evaluation must show that prior outcome or repair changes next-turn
affective state, and that affective guidance changes the next prompt capsule
without changing core policy selection by default.
```

## Conscious Artifact Addendum

M6 can use Markdown artifacts as the first human-readable observability layer, but they must remain projections of state, not the source of truth.

Recommended structure:

```text
artifacts/conscious/
  personas/
    {persona_id}/
      profile.json
      Self-consciousness.md
      sessions/
        {session_id}/
          Conscious.md
          conscious_trace.jsonl
          turn_summaries/
            turn_0001.md
```

Roles:

```text
Self-consciousness.md = long-term persona-scoped self-prior
Conscious.md = session-scoped current conscious context
conscious_trace.jsonl = machine-readable evidence
Dashboard = current parameters and readable summaries
```

Update speeds:

```text
Conscious.md: fast, per turn, may be rewritten or kept as a rolling window
Self-consciousness.md: slow, consolidation-gated, cross-session
Trace JSONL: append-only evidence
Dashboard: view over current state
```

Persona isolation:

```text
persona_id must be stable
display_name is not identity
session_id is scoped under persona_id
no persona may share or mutate another persona's Self-consciousness.md
prompt assembly must resolve persona_id before reading conscious artifacts
```

Prompt rule:

```text
Use only compressed summaries from Self-consciousness.md and Conscious.md.
Never put full Markdown artifacts into the prompt.
Never let Markdown artifacts override diagnostics, memory stores, or ranked options.
```

## M6 Milestone Overview

| Milestone | Title | Core Deliverable |
|---|---|---|
| M6.0 | Local Architecture Alignment | M6 与现有代码结构的映射和边界确认 |
| M6.1 | Cognitive Event Layer | 在现有 bus 基础上增加认知事件层 |
| M6.2 | Turn Trace Integration | 每轮对话输出完整可解释 trace |
| M6.3 | Cognitive State MVP | 从事件流派生最小认知状态场 |
| M6.4 | Existing Path Competition Adapter | 将现有 ranked_options 映射为 cognitive path view |
| M6.5 | Meta-Control Guidance | 让高阶观察输出可用的调控 guidance |
| M6.6 | Prompt Capsule Upgrade | 扩展 FEPPromptCapsule，承载状态压缩结果 |
| M6.7 | Closed-Loop Evaluation | 验证 outcome 能影响下一轮状态、prompt 和记忆 |

---

## M6.0: Local Architecture Alignment

### Goal

明确 M6 不替换现有系统，而是增强现有 M5 dialogue runtime。先把架构边界写清楚，再动实现。

### Scope

1. 标出 M6 对应的现有模块。
2. 明确哪些概念复用现有实现，哪些需要新增轻量适配层。
3. 定义 MVP 事件、MVP 状态和 first integration points。
4. 确认 M6 不重新实现 `decision_cycle`、`AttentionBottleneck`、`MetaCognitiveLayer` 或 `FEPPromptCapsule`。

### Deliverables

```text
M6 architecture note
M6 local module map
M6 event/state MVP schema draft
M6 non-goals
```

### Acceptance Criteria

```text
1. 能清楚说明 M6 与 M5.3/M5.6 dialogue runtime 的关系。
2. 能指出每个 M6 概念落在哪些现有文件或适配层上。
3. 明确 MessageBus 是内部事件基础设施，不是意识本身。
4. 明确 CognitiveState 是事件整合后的状态场，不是 prompt 文本。
5. 明确 Transformer 是生成器/解释器/模拟器，不是完整认知系统。
```

---

## M6.1: Cognitive Event Layer

### Goal

在现有 `io_bus.py` 的基础上增加轻量认知事件层，让关键内部过程从“函数中间变量”变成可观察、可筛选、可追踪的事件。

### Scope

MVP 事件类型：

```text
ObservationEvent
MemoryActivationEvent
DecisionEvent
CandidatePathEvent
PathSelectionEvent
PromptAssemblyEvent
GenerationEvent
OutcomeEvent
```

事件基础字段：

```text
event_id
event_type
turn_id
cycle
session_id
source
timestamp
salience
priority
ttl
payload
```

### Local Integration Points

```text
DialogueObserver.observe
-> ObservationEvent

SegmentAgent.decision_cycle_from_dict
-> DecisionEvent / CandidatePathEvent / PathSelectionEvent

build_fep_prompt_capsule
-> PromptAssemblyEvent

ResponseGenerator.generate
-> GenerationEvent

classify_dialogue_outcome
-> OutcomeEvent
```

### Deliverables

```text
CognitiveEvent dataclass / schema
CognitiveEventBus or io_bus extension
event publish/consume/filter primitives
minimal tests for serialization and deterministic ordering
```

### Acceptance Criteria

```text
1. 事件可序列化为 JSON-safe dict。
2. 事件带 turn_id / cycle / session_id。
3. 可以按 event_type/source/salience 过滤。
4. 不破坏现有 PerceptionBus / ActionBus 行为。
5. 没有消费者时，事件层仍是无副作用旁路。
```

---

## M6.2: Turn Trace Integration

### Goal

让每轮对话产生可读的 cognitive trace，用来回答：

```text
系统这轮观察到了什么？
哪些事件被注意力选中？
候选行动是什么？
为什么选中这个 action？
哪些信号进入 prompt，哪些被省略？
生成是否遵循 selected path？
outcome 如何反馈？
```

### Scope

为每轮对话记录：

```text
observation channels
attention selected/suppressed channels
retrieved memory summary
ranked options
chosen action
policy score / EFE / margin
fep prompt capsule
generation diagnostics
outcome label
memory update signal
```

### Local Integration Points

```text
segmentum/dialogue/conversation_loop.py
segmentum/dialogue/fep_prompt.py
segmentum/tracing.py
```

### Deliverables

```text
TurnTrace schema
JSONL trace writer integration
test trace generation for scripted dialogue
```

### Acceptance Criteria

```text
1. run_conversation 可以产生 turn-level trace。
2. trace 中包含 observation -> decision -> prompt -> generation -> outcome 的关键字段。
3. trace 不泄漏完整 prompt 或敏感原始数据，除非显式启用 debug。
4. 现有 M5 dialogue tests 不回归。
```

---

## M6.3: Cognitive State MVP

### Goal

建立最小认知状态场。第一版不追求完整人格状态，而是先维护能推动路径选择闭环的状态。

### MVP State

```text
TaskState
MemoryState
GapState
MetaControlState
```

建议结构：

```text
TaskState:
- explicit_request
- inferred_need
- current_goal
- task_phase
- success_criteria
- urgency

MemoryState:
- activated_memories
- reusable_patterns
- memory_conflicts
- abstraction_candidates
- memory_helpfulness

GapState:
- epistemic_gaps
- contextual_gaps
- instrumental_gaps
- resource_gaps
- social_gaps
- blocking_gaps

MetaControlState:
- lambda_energy
- lambda_attention
- lambda_memory
- lambda_control
- beta_efe
- exploration_temperature
- control_gain
- memory_retrieval_gain
- abstraction_gain
```

### Local Integration Points

```text
conversation_loop consumes selected CognitiveEvents
agent diagnostics provide decision and memory fields
FEPPromptCapsule receives compressed state summary
```

### Deliverables

```text
CognitiveStateMVP dataclasses
state update from selected events
state to_dict/from_dict
unit tests for deterministic state update
```

### Acceptance Criteria

```text
1. CognitiveStateMVP 可从一轮事件流中更新。
2. 状态更新是 deterministic 的。
3. 状态可序列化、可 trace。
4. 第一版不引入完整 SelfState/UserState/WorldState，以免膨胀。
5. 状态影响 prompt guidance，但不立即重写 policy core。
```

---

## M6.4: Existing Path Competition Adapter

### Goal

把现有 `ranked_options` 解释为认知路径竞争结果，而不是重写一套路劲评分器。

### CognitivePath View

M6 可以先从现有 `InterventionScore` 派生：

```text
path_id
interpretation
proposed_action
expected_outcome
current_free_energy
expected_free_energy
energy_cost
attention_cost
memory_cost
control_cost
social_risk
long_term_value
total_cost
posterior_weight
source_action
source_policy_score
```

第一版字段映射可以是近似的：

```text
expected_free_energy <- InterventionScore.expected_free_energy
control_cost <- action ambiguity / repair / commitment tension proxy
memory_cost <- memory_bias / memory context proxy
social_risk <- risk / social assessment proxy
long_term_value <- goal_alignment / commitment compatibility proxy
total_cost <- derived from existing policy_score and cost components
posterior_weight <- softmax over derived total_cost or normalized policy_score
```

### Local Integration Points

```text
SegmentAgent.decision_cycle
DecisionDiagnostics.ranked_options
FEPPromptCapsule.top_alternatives
```

### Deliverables

```text
CognitivePath adapter
path posterior weight computation
path margin / uncertainty summary
tests using existing diagnostics fixtures
```

### Acceptance Criteria

```text
1. 不改变现有 action selection 行为。
2. 能从 diagnostics.ranked_options 生成 cognitive path view。
3. 能解释 chosen path 与 runner-up 的成本差异。
4. path view 可进入 trace 和 prompt capsule。
5. 为后续真正 cost formula 留出字段。
```

---

## M6.5: Meta-Control Guidance

### Goal

让高阶观察者不只是记录内部模式，而是输出对下一步生成/提示/记忆/策略有影响的调控 guidance。

### Triggers

```text
low policy margin
high EFE margin uncertainty
high conflict_tension
high hidden_intent with low observability
repeated outcome failure
prompt overload
memory conflict
identity or commitment tension
```

### Guidance Outputs

```text
increase_caution
ask_clarifying_question
lower_assertiveness
compress_context
reduce_memory_reliance
increase_control_gain
increase_exploration_temperature
prefer_repair_strategy
avoid_overinterpreting_hidden_intent
```

### Local Integration Points

```text
segmentum/metacognitive.py
segmentum/dialogue/fep_prompt.py
segmentum/dialogue/generator.py
```

### Deliverables

```text
MetaControlGuidance schema
guidance generation from CognitiveStateMVP and diagnostics
prompt capsule integration
tests for low-margin and high-conflict cases
```

### Acceptance Criteria

```text
1. Meta-control guidance is deterministic for the same trace.
2. Guidance first affects prompt conditioning, not core policy selection.
3. Low-confidence decisions produce lower assertion / clarification guidance.
4. High hidden_intent does not automatically cause paranoid interpretation.
5. Guidance appears in trace for auditability.
```

---

## M6.6: Prompt Capsule Upgrade

### Goal

扩展 `FEPPromptCapsule`，让 Transformer 看到的是压缩后的认知状态，而不是原始事件流。

### Additions

```text
selected_path_summary
path_competition_summary
active_gaps
meta_control_guidance
memory_use_guidance
omitted_signals
prompt_budget_summary
```

### Design Rule

Prompt capsule 只承载：

```text
selected path
top alternatives
decision uncertainty
active gaps
memory constraints
meta-control guidance
style/control hints
```

不要承载：

```text
raw event stream
full memory dump
full diagnostics object
unfiltered hidden-intent speculation
```

### Local Integration Points

```text
segmentum/dialogue/fep_prompt.py
segmentum/dialogue/generator.py
segmentum/dialogue/conversation_loop.py
```

### Deliverables

```text
FEPPromptCapsule v2 fields
backward-compatible to_dict
generator uses new guidance fields
tests for capsule fallback and normal path
```

### Acceptance Criteria

```text
1. Existing tests expecting old capsule fields continue to pass or are migrated cleanly.
2. New capsule fields are optional/backward-compatible where possible.
3. Prompt guidance changes when meta-control guidance changes.
4. Raw events are not directly inserted into prompts.
5. PromptAssemblyEvent records included and omitted signals.
```

---

## M6.7: Closed-Loop Evaluation

### Goal

证明 M6 不是日志增强，而是真正形成闭环：outcome 会影响下一轮状态、prompt guidance、memory update 或 meta-control。

### Evaluation Scenarios

至少覆盖：

```text
1. Low-margin ambiguity:
   用户问题含糊，候选路径 margin 很低。
   预期：系统降低断言强度，倾向澄清。

2. High-conflict dialogue:
   用户质疑或不满。
   预期：GapState 出现 social/contextual gap，MetaControl 提升 repair/control guidance。

3. Memory interference:
   激活记忆与当前输入冲突。
   预期：MemoryState 标记 conflict，prompt 不盲目复用旧记忆。

4. Prompt overload:
   事件和状态过多。
   预期：AttentionGate 丢弃低 salience 事件，PromptAssemblyEvent 记录 omitted_signals。

5. Outcome failure:
   上一轮生成没有遵循 selected path 或 outcome 负面。
   预期：下一轮 control_gain / repair guidance 上升。
```

### Deliverables

```text
M6 closed-loop scenario tests
trace-based evaluation report
regression suite covering M5 dialogue behavior
```

### Acceptance Criteria

```text
1. outcome event can influence next-turn CognitiveStateMVP.
2. CognitiveStateMVP can influence next-turn prompt capsule.
3. Meta-control guidance changes under low margin / high conflict / failure.
4. Existing M5.3 dialogue tests remain stable.
5. Trace can explain why the system selected a path and how feedback changed the next turn.
```

---

## Final M6 Acceptance Criteria

M6 完成时，系统应满足：

```text
1. 每轮对话产生可观察的内部认知事件流。
2. AttentionGate / salience filter 能选择哪些事件进入状态整合。
3. CognitiveStateMVP 能维护任务、记忆、缺口和 meta-control 状态。
4. 现有 ranked_options 能被解释为候选认知路径竞争。
5. 路径选择和 prompt guidance 能被 EFE、成本、margin、gap、memory 信号解释。
6. MetaControl 能根据负载、冲突、低置信度、失败结果输出调节 guidance。
7. FEPPromptCapsule 承载压缩后的认知状态，而不是原始事件流。
8. OutcomeEvent 能反馈到下一轮状态、memory 或 prompt guidance。
9. Debug trace 能回答“为什么系统这轮这样回答”。
10. 现有 M5 dialogue runtime 继续工作，不被新系统割裂或替代。
```

---

## Non-Goals

M6 暂不做：

```text
完整 SelfState / UserState / WorldState 九宫格实现
训练或微调 Transformer
替换 SegmentAgent.decision_cycle
替换 AttentionBottleneck / GlobalWorkspace
替换 MetaCognitiveLayer
把所有事件塞进 prompt
做完整人格意识模拟
```

这些可以留给 M7 或 M6 后续增强。M6 的重点是最小闭环。

---

## Recommended Implementation Order

建议分三阶段推进：

### Phase 1: Observability

```text
M6.0 -> M6.1 -> M6.2
```

目标：先让系统能看见自己的内部过程。

### Phase 2: State and Path View

```text
M6.3 -> M6.4
```

目标：把事件整合为状态，把现有策略结果解释为候选认知路径。

### Phase 3: Closed-Loop Control

```text
M6.5 -> M6.6 -> M6.7
```

目标：让 meta-control 影响 prompt，让 outcome 影响下一轮状态，形成真正闭环。

---

## Design Judgment

M6 的关键判断是：

```text
消息总线不是意识本身，而是内部事件可观测性基础设施。
CognitiveState 不是 prompt，而是事件整合后的状态场。
MetaControl 不是小人，而是调节成本权重、探索温度、控制增益和记忆依赖的慢变量系统。
Transformer 不是完整认知系统，而是外环调度的语义生成、压缩和模拟器。
```

因此，M6 的成功不在于新增多少模块，而在于让现有模块从线性链路变成闭环：

```text
观察 -> 决策 -> 生成 -> 结果
```

升级为：

```text
观察 -> 事件 -> 注意力 -> 状态 -> 路径竞争 -> meta-control
-> prompt/action -> 结果 -> 记忆/状态/控制参数更新
```

这就是 M6 的核心交付。
