## Goal

把当前对话系统逐步升级为“资源受限认知路径选择系统 + MessageBus + CognitiveLoop + MetaControl”的可测试实现。

## Non-goals

- 不模拟真正主观意识。
- 不一次性替换 SegmentAgent。
- 不把所有认知状态暴露给最终用户。
- 不让 prompt 成为唯一认知层。
- 不破坏现有通过测试。

## Stages

### Stage 1：事件总线最小闭环

目标：

- 让 CognitiveEventBus 从 append-only trace 变成可被 CognitiveLoop 消费的事件源。
- 增加 AttentionGate，对事件按 salience / priority / ttl / budget 筛选。
- run_conversation 中不再直接只把 turn_events 传给 updater，而是至少走一条 bus -> AttentionGate -> CognitiveLoop 的路径。
- 保留兼容旧的 turn_events 方式。

验收测试：

- test_message_bus_events_are_consumed_by_cognitive_loop
- test_attention_gate_filters_events_by_salience_priority_ttl_and_budget
- test_expired_events_are_not_consumed
- test_low_salience_events_remain_trace_only

### Stage 2：CognitiveStateMVP 扩展

目标：

- 在现有 CognitiveStateMVP 上增量添加 ResourceState、UserState、WorldState、CandidatePathState。
- GapState 从字符串列表升级为结构化 Gap 对象。
- 不破坏旧字段和 trace schema。
- prompt 仍只接收压缩后的 cognitive guidance。

验收测试：

- test_cognitive_loop_updates_resource_user_world_and_path_state
- test_gap_detector_returns_structured_blocking_soft_latent_gaps
- test_cognitive_state_backward_compatible_with_existing_fields

### Stage 3：候选路径竞争

目标：

- 保留现有 SegmentAgent decision_cycle。
- 在其输出上构建 CognitivePathCandidate。
- 使用 meta-control lambdas 计算 total_cost。
- 使用 effective_temperature 计算 posterior_weight。
- 输出 selection_margin / uncertainty / low_confidence_reason。
- 初期可以仍不改变最终 chosen action，但必须能解释和记录 alternative selection。

验收测试：

- test_candidate_paths_include_cost_components
- test_path_scoring_uses_meta_control_lambdas
- test_path_selection_uses_total_cost_temperature_and_margin
- test_low_margin_selection_sets_uncertainty

### Stage 4：MetaControl 最小因果权力

目标：

- MetaControlEvent 至少能影响以下 2 项：
  1. memory retrieval gain 或 retrieval k
  2. path scoring lambdas 或 effective_temperature
- 影响必须有限、可测、可回滚。
- prompt guidance 仍保留，但不再是唯一影响路径。

验收测试：

- test_meta_control_changes_memory_retrieval_gain
- test_meta_control_changes_path_scoring_lambdas
- test_memory_bias_overdominance_triggers_control_signal
- test_resource_overload_increases_effective_temperature_or_compresses_candidates

### Stage 5：PromptAdapter / PromptBuilder 收敛

目标：

- PromptBuilder 接收 compressed cognitive guidance。
- 不接收 raw events。
- 不输出自我独白。
- 可以表达：
  - 当前任务
  - 当前目标
  - selected path
  - missing gaps
  - uncertainty / assertiveness guidance
  - memory use constraints
  - generation style constraints

验收测试：

- test_prompt_builder_uses_compressed_cognitive_guidance_not_raw_events
- test_low_margin_selection_reduces_assertiveness
- test_prompt_does_not_include_raw_event_dump
- test_prompt_does_not_claim_consciousness

### Stage 6：记忆动力学升级

目标：

- successful path 可以编码为 reusable pattern。
- memory interference / memory overdominance 可以被检测。
- outcome-driven consolidation 可以影响 reusable path。
- memory conflict 时 MetaControl 降低 memory gain 或增加 caution。

验收测试：

- test_successful_path_encoded_as_reusable_pattern
- test_memory_interference_detected
- test_outcome_driven_consolidation_updates_reusable_pattern
- test_memory_conflict_reduces_memory_gain

## Implementation Status

This section records the current implementation state of the staged plan. It is
not a claim that Segmentum implements subjective consciousness; it only records
test-backed engineering milestones for the resource-bounded cognitive path
selection architecture.

| Stage | Status | Primary Files | Acceptance Coverage |
| --- | --- | --- | --- |
| Stage 1: MessageBus minimal loop | PASS | `segmentum/cognitive_events.py`, `segmentum/cognition/attention_gate.py`, `segmentum/cognition/cognitive_loop.py`, `segmentum/dialogue/conversation_loop.py` | `tests/test_m6x_cognitive_event_bus_loop.py` |
| Stage 2: CognitiveStateMVP expansion | PASS | `segmentum/cognitive_state.py`, `segmentum/dialogue/turn_trace.py`, `scripts/generate_m63_acceptance_artifacts.py` | `tests/test_m6x_cognitive_state_stage2.py` |
| Stage 3: candidate path competition | PASS | `segmentum/cognitive_paths.py`, `segmentum/cognitive_state.py` | `tests/test_m6x_candidate_path_competition_stage3.py` |
| Stage 4: minimal MetaControl causal power | PASS | `segmentum/meta_control.py`, `segmentum/agent.py`, `segmentum/cognitive_state.py`, `segmentum/dialogue/conversation_loop.py` | `tests/test_m6x_meta_control_causal_stage4.py` |
| Stage 5: PromptAdapter / PromptBuilder convergence | PASS | `segmentum/dialogue/cognitive_guidance.py`, `segmentum/dialogue/runtime/prompts.py`, `segmentum/dialogue/generator.py` | `tests/test_m6x_prompt_builder_stage5.py` |
| Stage 6: memory dynamics upgrade | PASS | `segmentum/memory_dynamics.py`, `segmentum/memory.py`, `segmentum/agent.py`, `segmentum/cognitive_state.py`, `segmentum/dialogue/conversation_loop.py` | `tests/test_m6x_memory_dynamics_stage6.py` |

## Regression Groups

Focused Stage 1-6 acceptance:

```bash
pytest tests/test_m6x_cognitive_event_bus_loop.py tests/test_m6x_cognitive_state_stage2.py tests/test_m6x_candidate_path_competition_stage3.py tests/test_m6x_meta_control_causal_stage4.py tests/test_m6x_prompt_builder_stage5.py tests/test_m6x_memory_dynamics_stage6.py -q
```

M6 neighborhood regression:

```bash
pytest tests/test_m60_architecture_alignment.py tests/test_m61_cognitive_events.py tests/test_m62_acceptance.py tests/test_m62_turn_trace.py tests/test_m62_ui_inner_world.py tests/test_m63_acceptance.py tests/test_m63_cognitive_state.py tests/test_m64_cognitive_paths.py tests/test_m65_meta_control_guidance.py tests/test_m66_prompt_capsule_upgrade.py tests/test_m67_closed_loop_evaluation.py tests/test_m6x_cognitive_event_bus_loop.py tests/test_m6x_cognitive_state_stage2.py tests/test_m6x_candidate_path_competition_stage3.py tests/test_m6x_meta_control_causal_stage4.py tests/test_m6x_prompt_builder_stage5.py tests/test_m6x_memory_dynamics_stage6.py -q
```

Current full-suite note:

- `pytest -q` exceeded the local execution window twice during this pass
  (5 minutes and 10 minutes). No failure summary was emitted before timeout.
- Historical artifact/report files refreshed by that full-suite attempt were
  restored, because they were timestamp/provenance noise unrelated to the
  staged implementation.

## Remaining Work

- The implementation is still an incremental engineering approximation, not a
  full cognitive architecture replacement.
- Candidate path selection is still recorded as an explainable competition
  layer; it does not yet replace every policy decision surface.
- MetaControl has bounded causal effects for memory retrieval and path scoring,
  but broader causal authority should remain opt-in and test-gated.
- Memory dynamics now support reusable path patterns and interference controls,
  but long-term consolidation policy still needs production-scale evaluation.
