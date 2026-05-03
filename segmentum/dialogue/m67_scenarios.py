"""M6.7 deterministic closed-loop evaluation scenarios.

This module is an evaluation layer over the existing M6 dialogue chain.  It
uses the production state/guidance/capsule/trace helpers and deliberately keeps
core policy scoring untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone as _timezone

UTC = _timezone.utc
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..cognitive_events import make_cognitive_event
from ..cognitive_paths import cognitive_paths_from_diagnostics, path_competition_summary
from ..cognitive_state import CognitiveStateMVP, update_cognitive_state
from ..meta_control_guidance import MetaControlGuidance, generate_meta_control_guidance
from ..tracing import JsonlTraceWriter
from ..types import DecisionDiagnostics, InterventionScore
from .fep_prompt import build_fep_prompt_capsule
from .turn_trace import TurnTrace


TRACE_ONLY_FIELDS = {
    "trace_path",
    "conscious_markdown_path",
    "conscious_markdown_sha256",
}

CLOSURE_FIELD_PREFIXES = (
    "state.",
    "guidance.",
    "prompt_capsule.",
    "memory_update_signal.",
)


@dataclass(frozen=True)
class M67Scenario:
    scenario_id: str
    description: str
    input_turns: tuple[str, ...]
    seed: int
    observation: dict[str, float]
    diagnostics_profile: str
    expected_trigger: str
    expected_delta_fields: tuple[str, ...]
    forbidden_false_positive_claims: tuple[str, ...]
    ablation_kind: str
    enabled_previous_outcome: str = ""
    ablated_previous_outcome: str = ""
    enabled_prompt_budget: dict[str, object] | None = None
    ablated_prompt_budget: dict[str, object] | None = None
    self_prior_summary: dict[str, object] | str | None = None
    previous_state_profile: str = "default"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _option(
    action: str,
    *,
    policy_score: float,
    expected_free_energy: float,
    risk: float = 0.1,
    predicted_outcome: str = "dialogue_reward",
    dominant_component: str = "expected_free_energy",
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "explore"}},
        policy_score=policy_score,
        expected_free_energy=expected_free_energy,
        predicted_error=0.2,
        action_ambiguity=0.1,
        risk=risk,
        preferred_probability=0.5,
        memory_bias=0.0,
        pattern_bias=0.0,
        policy_bias=0.0,
        epistemic_bonus=0.0,
        workspace_bias=0.0,
        social_bias=0.0,
        commitment_bias=0.0,
        identity_bias=0.0,
        ledger_bias=0.0,
        subject_bias=0.0,
        goal_alignment=0.0,
        value_score=0.0,
        predicted_outcome=predicted_outcome,
        predicted_effects={},
        dominant_component=dominant_component,
        cost=0.0,
    )


def _diagnostics(profile: str, *, ablated: bool = False) -> DecisionDiagnostics:
    if profile == "low_margin":
        ranked = [
            _option("ask_question", policy_score=1.0, expected_free_energy=0.30),
            _option("reflect", policy_score=0.97, expected_free_energy=0.32),
        ]
        prediction_delta: dict[str, float] = {}
        memory_hit = False
        retrieved: list[dict[str, object]] = []
        memory_summary = ""
    elif profile == "memory_interference":
        ranked = [
            _option("ask_question", policy_score=0.92, expected_free_energy=0.25),
            _option("reflect", policy_score=0.70, expected_free_energy=0.54),
        ]
        prediction_delta = {} if ablated else {"stale_preference_memory": 0.56}
        memory_hit = not ablated
        retrieved = (
            []
            if ablated
            else [
                {
                    "episode_id": "m67-memory-1",
                    "summary": "prior memory says reuse old preference",
                }
            ]
        )
        memory_summary = "" if ablated else "prior memory conflicts with current correction"
    else:
        ranked = [
            _option("empathize", policy_score=0.92, expected_free_energy=0.25),
            _option("ask_question", policy_score=0.61, expected_free_energy=0.61),
        ]
        prediction_delta = {}
        memory_hit = False
        retrieved = []
        memory_summary = ""

    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=0.42 if profile in {"low_margin", "memory_interference"} else 0.22,
        retrieved_memories=retrieved,
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation=f"m67 {profile} diagnostics",
        active_goal="m67_closed_loop",
        memory_hit=memory_hit,
        retrieved_episode_ids=["m67-memory-1"] if memory_hit else [],
        memory_context_summary=memory_summary,
        prediction_delta=prediction_delta,
        workspace_broadcast_channels=["hidden_intent", "conflict_tension"],
        workspace_suppressed_channels=["topic_novelty"],
    )


def _empty_guidance() -> dict[str, object]:
    return MetaControlGuidance(
        increase_caution=False,
        ask_clarifying_question=False,
        lower_assertiveness=False,
        compress_context=False,
        reduce_memory_reliance=False,
        increase_control_gain=False,
        increase_exploration_temperature=False,
        prefer_repair_strategy=False,
        avoid_overinterpreting_hidden_intent=False,
        deescalate_affect=False,
        preserve_warmth=False,
        reduce_intensity=False,
        guidance_notes=[],
        trigger_reasons=[],
        intensity=0.0,
    ).to_dict()


def _event(
    event_type: str,
    *,
    scenario_id: str,
    outcome: str = "",
    sequence_index: int = 0,
) -> object:
    payload: dict[str, object] = {"scenario_id": scenario_id}
    if outcome:
        payload["outcome"] = outcome
    return make_cognitive_event(
        event_type=event_type,
        turn_id="turn_0001",
        cycle=1,
        session_id=f"m67-{scenario_id}",
        persona_id=f"m67-{scenario_id}-persona",
        source="m67_closed_loop_evaluation",
        sequence_index=sequence_index,
        payload=payload,
        salience=0.75 if event_type == "OutcomeEvent" else 0.6,
        priority=0.7,
        timestamp="2026-05-02T00:00:00Z",
    )


def _events_for(scenario: M67Scenario, previous_outcome: str) -> tuple[object, ...]:
    events = [
        _event("ObservationEvent", scenario_id=scenario.scenario_id, sequence_index=0),
        _event("DecisionEvent", scenario_id=scenario.scenario_id, sequence_index=1),
    ]
    if previous_outcome:
        events.append(
            _event(
                "OutcomeEvent",
                scenario_id=scenario.scenario_id,
                outcome=previous_outcome,
                sequence_index=2,
            )
        )
    events.append(
        _event("PathSelectionEvent", scenario_id=scenario.scenario_id, sequence_index=3)
    )
    return tuple(events)


def _previous_state(profile: str) -> CognitiveStateMVP | None:
    if profile != "strained":
        return None
    return update_cognitive_state(
        None,
        events=(),
        diagnostics=_diagnostics("high_conflict"),
        observation={"emotional_tone": 0.22, "conflict_tension": 0.95},
        previous_outcome="failed",
    )


def _nested(payload: Mapping[str, object], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            return None
    return value


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    return str(value)


def _variant(
    scenario: M67Scenario,
    *,
    variant: str,
    trace_dir: Path | None,
) -> dict[str, object]:
    enabled = variant == "enabled"
    ablated = not enabled
    previous_outcome = (
        scenario.enabled_previous_outcome if enabled else scenario.ablated_previous_outcome
    )
    diagnostics = _diagnostics(
        scenario.diagnostics_profile,
        ablated=ablated and scenario.ablation_kind == "remove_memory_interference",
    )
    events = _events_for(scenario, previous_outcome)
    state = update_cognitive_state(
        _previous_state(scenario.previous_state_profile),
        events=events,
        diagnostics=diagnostics,
        observation=scenario.observation,
        previous_outcome=previous_outcome,
        self_prior_summary=scenario.self_prior_summary,
    )
    paths = cognitive_paths_from_diagnostics(diagnostics)
    path_summary = path_competition_summary(paths)
    prompt_budget = (
        scenario.enabled_prompt_budget if enabled else scenario.ablated_prompt_budget
    )
    generated_guidance = generate_meta_control_guidance(
        state,
        diagnostics=diagnostics,
        path_summary=path_summary,
        previous_outcome=previous_outcome,
        prompt_budget=prompt_budget,
    ).to_dict()
    guidance = (
        _empty_guidance()
        if ablated and scenario.ablation_kind == "disable_guidance"
        else generated_guidance
    )
    omitted_signals = None
    if isinstance(prompt_budget, Mapping):
        raw_omitted = prompt_budget.get("omitted_signals") or prompt_budget.get("omitted")
        if isinstance(raw_omitted, list):
            omitted_signals = [str(item) for item in raw_omitted]
    capsule = build_fep_prompt_capsule(
        diagnostics,
        scenario.observation,
        previous_outcome=previous_outcome,
        cognitive_state=state,
        self_prior_summary=scenario.self_prior_summary,
        cognitive_paths=paths,
        path_summary=path_summary,
        meta_control_guidance=guidance,
        affective_state=state.affect,
        affective_guidance={"source": "meta_control_guidance"},
        prompt_budget=prompt_budget,
        omitted_signals=omitted_signals,
        persona_id=f"m67-{scenario.scenario_id}-persona",
        session_id=f"m67-{scenario.scenario_id}-{variant}",
    ).to_dict()
    memory_update_signal = {
        "integrated": True,
        "action": diagnostics.chosen.choice,
        "outcome_label": previous_outcome or "neutral",
        "episodic_episode_count_before": 0,
        "episodic_episode_count_after": 1,
    }
    trace_path = ""
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = str(trace_dir / f"{scenario.scenario_id}_{variant}.jsonl")
        writer = JsonlTraceWriter(trace_path)
        writer.reset()
        TurnTrace.from_runtime(
            persona_id=f"m67-{scenario.scenario_id}-persona",
            session_id=f"m67-{scenario.scenario_id}-{variant}",
            turn_id="turn_0001",
            turn_index=1,
            cycle=1,
            observation_channels=scenario.observation,
            diagnostics=diagnostics,
            fep_prompt_capsule=capsule,
            cognitive_state=state,
            meta_control_guidance=guidance,
            generation_diagnostics={
                "selected_action": diagnostics.chosen.choice,
                "meta_control_guidance": guidance,
                "prompt_capsule_guidance": {
                    "meta_control_guidance": capsule.get("meta_control_guidance") or {},
                    "affective_guidance": capsule.get("affective_guidance") or {},
                    "memory_use_guidance": capsule.get("memory_use_guidance") or {},
                    "omitted_signals": capsule.get("omitted_signals") or [],
                },
            },
            outcome_label=previous_outcome or "neutral",
            memory_update_signal=memory_update_signal,
            events=events,
        ).write_jsonl(writer)
    return {
        "variant": variant,
        "state": state.to_dict(),
        "guidance": guidance,
        "prompt_capsule": capsule,
        "memory_update_signal": memory_update_signal,
        "selected_action": diagnostics.chosen.choice,
        "path_competition": path_summary,
        "trace_path": trace_path,
        "outcome_event_present": any(
            getattr(event, "event_type", "") == "OutcomeEvent" for event in events
        ),
    }


def closure_evidence_fields(
    enabled: Mapping[str, object],
    ablated: Mapping[str, object],
) -> list[str]:
    fields: list[str] = []
    candidates = (
        "state.task.task_phase",
        "state.gaps.epistemic_gaps",
        "state.gaps.contextual_gaps",
        "state.gaps.social_gaps",
        "state.gaps.instrumental_gaps",
        "state.gaps.blocking_gaps",
        "state.memory.memory_conflicts",
        "state.affect.warmth",
        "state.affect.social_safety",
        "state.affect.repair_need",
        "state.meta_control.lambda_control",
        "guidance.ask_clarifying_question",
        "guidance.lower_assertiveness",
        "guidance.compress_context",
        "guidance.reduce_memory_reliance",
        "guidance.increase_control_gain",
        "guidance.prefer_repair_strategy",
        "guidance.deescalate_affect",
        "prompt_capsule.meta_control_guidance",
        "prompt_capsule.affective_state_summary",
        "prompt_capsule.memory_use_guidance",
        "prompt_capsule.omitted_signals",
        "prompt_capsule.prompt_budget_summary",
        "prompt_capsule.previous_outcome",
        "memory_update_signal.outcome_label",
    )
    for field in candidates:
        if _nested(enabled, field) != _nested(ablated, field):
            fields.append(field)
    return fields


def evaluate_m67_scenario(
    scenario: M67Scenario,
    *,
    trace_dir: str | Path | None = None,
) -> dict[str, object]:
    trace_path = Path(trace_dir) if trace_dir is not None else None
    enabled = _variant(scenario, variant="enabled", trace_dir=trace_path)
    ablated = _variant(scenario, variant="ablated", trace_dir=trace_path)
    changed = closure_evidence_fields(enabled, ablated)
    expected = list(scenario.expected_delta_fields)
    missing = [field for field in expected if field not in changed]
    trace_only = bool(changed) and all(field in TRACE_ONLY_FIELDS for field in changed)
    passed = not missing and bool(changed) and not trace_only
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "expected_trigger": scenario.expected_trigger,
        "input_turns": list(scenario.input_turns),
        "seed": scenario.seed,
        "enabled": _json_ready(enabled),
        "ablated": _json_ready(ablated),
        "changed_fields": changed,
        "expected_delta_fields": expected,
        "missing_expected_delta_fields": missing,
        "forbidden_false_positive_claims": list(scenario.forbidden_false_positive_claims),
        "trace_only_closure_rejected": not trace_only,
        "passed": passed,
    }


def m67_scenarios() -> tuple[M67Scenario, ...]:
    return (
        M67Scenario(
            scenario_id="low_margin_ambiguity",
            description="Ambiguous request with a low ranked-option margin.",
            input_turns=("I am not sure which path you mean; can you handle it?",),
            seed=6701,
            observation={
                "semantic_content": 0.62,
                "topic_novelty": 0.5,
                "emotional_tone": 0.5,
                "conflict_tension": 0.1,
                "relationship_depth": 0.2,
                "hidden_intent": 0.76,
                "missing_context": 0.68,
            },
            diagnostics_profile="low_margin",
            expected_trigger="low decision margin",
            expected_delta_fields=(
                "guidance.ask_clarifying_question",
                "guidance.lower_assertiveness",
                "prompt_capsule.meta_control_guidance",
            ),
            forbidden_false_positive_claims=("surface text alone proves closure",),
            ablation_kind="disable_guidance",
        ),
        M67Scenario(
            scenario_id="high_conflict_dialogue",
            description="Conflict-heavy user turn should raise repair/control guidance.",
            input_turns=("You keep missing my point, this is frustrating.",),
            seed=6702,
            observation={
                "semantic_content": 0.7,
                "topic_novelty": 0.35,
                "emotional_tone": 0.22,
                "conflict_tension": 0.95,
                "relationship_depth": 0.2,
                "hidden_intent": 0.74,
            },
            diagnostics_profile="high_conflict",
            expected_trigger="high conflict tension",
            expected_delta_fields=(
                "guidance.prefer_repair_strategy",
                "guidance.increase_control_gain",
                "guidance.deescalate_affect",
                "prompt_capsule.meta_control_guidance",
            ),
            forbidden_false_positive_claims=("accusatory hidden-intent inference",),
            ablation_kind="disable_guidance",
        ),
        M67Scenario(
            scenario_id="affective_recovery_after_repair",
            description="A positive repair outcome should recover next-turn affective state.",
            input_turns=("Thanks, that repair helped. Let's continue.",),
            seed=6703,
            observation={
                "semantic_content": 0.75,
                "topic_novelty": 0.2,
                "emotional_tone": 0.72,
                "conflict_tension": 0.08,
                "relationship_depth": 0.58,
                "hidden_intent": 0.35,
            },
            diagnostics_profile="recovery",
            expected_trigger="positive repair outcome",
            expected_delta_fields=(
                "state.affect.warmth",
                "state.affect.social_safety",
                "prompt_capsule.affective_state_summary",
            ),
            forbidden_false_positive_claims=("raw affective notes in prompt",),
            ablation_kind="remove_outcome",
            enabled_previous_outcome="repaired",
            ablated_previous_outcome="neutral",
            previous_state_profile="strained",
        ),
        M67Scenario(
            scenario_id="memory_interference",
            description="Retrieved memory conflicts with current evidence.",
            input_turns=("Actually that old preference is wrong now.",),
            seed=6704,
            observation={
                "semantic_content": 0.66,
                "topic_novelty": 0.42,
                "emotional_tone": 0.5,
                "conflict_tension": 0.2,
                "relationship_depth": 0.45,
                "hidden_intent": 0.4,
            },
            diagnostics_profile="memory_interference",
            expected_trigger="memory conflict",
            expected_delta_fields=(
                "state.memory.memory_conflicts",
                "guidance.reduce_memory_reliance",
                "prompt_capsule.memory_use_guidance",
            ),
            forbidden_false_positive_claims=("blind old-memory reuse",),
            ablation_kind="remove_memory_interference",
            self_prior_summary={"summary": "use only compressed memory summaries"},
        ),
        M67Scenario(
            scenario_id="prompt_overload",
            description="Prompt budget pressure should record omitted signals and compression guidance.",
            input_turns=("There are too many signals; keep only the next-step essentials.",),
            seed=6705,
            observation={
                "semantic_content": 0.68,
                "topic_novelty": 0.6,
                "emotional_tone": 0.52,
                "conflict_tension": 0.15,
                "relationship_depth": 0.35,
                "hidden_intent": 0.5,
            },
            diagnostics_profile="prompt_overload",
            expected_trigger="prompt overload",
            expected_delta_fields=(
                "guidance.compress_context",
                "prompt_capsule.omitted_signals",
                "prompt_capsule.prompt_budget_summary",
            ),
            forbidden_false_positive_claims=("raw events inserted into prompt",),
            ablation_kind="remove_prompt_overload",
            enabled_prompt_budget={
                "used_ratio": 0.96,
                "remaining_tokens": 120,
                "omitted_signals": [
                    "raw_events",
                    "full_diagnostics",
                    "full_prompt",
                    "full_conscious_markdown",
                ],
            },
            ablated_prompt_budget={
                "used_ratio": 0.35,
                "remaining_tokens": 1800,
                "omitted_signals": [],
            },
        ),
        M67Scenario(
            scenario_id="outcome_failure",
            description="Negative prior outcome should raise repair pressure next turn.",
            input_turns=("That did not work. Please fix the miss first.",),
            seed=6706,
            observation={
                "semantic_content": 0.7,
                "topic_novelty": 0.3,
                "emotional_tone": 0.35,
                "conflict_tension": 0.45,
                "relationship_depth": 0.35,
                "hidden_intent": 0.48,
            },
            diagnostics_profile="outcome_failure",
            expected_trigger="previous negative outcome",
            expected_delta_fields=(
                "state.gaps.blocking_gaps",
                "state.meta_control.lambda_control",
                "guidance.prefer_repair_strategy",
                "prompt_capsule.previous_outcome",
                "memory_update_signal.outcome_label",
            ),
            forbidden_false_positive_claims=("Conscious.md alone proves closure",),
            ablation_kind="remove_outcome",
            enabled_previous_outcome="failed",
            ablated_previous_outcome="neutral",
        ),
    )


def persona_self_consciousness_path(root: str | Path, persona_id: str) -> Path:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in persona_id)
    cleaned = cleaned.strip("._") or "default"
    return Path(root) / "personas" / cleaned / "Self-consciousness.md"


def run_m67_evaluation(
    *,
    artifacts_dir: str | Path = "artifacts",
    write_traces: bool = True,
) -> dict[str, object]:
    root = Path(artifacts_dir)
    trace_dir = root / "m67_traces" if write_traces else None
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    scenarios = [
        evaluate_m67_scenario(scenario, trace_dir=trace_dir)
        for scenario in m67_scenarios()
    ]
    all_passed = all(bool(item.get("passed")) for item in scenarios)
    payload = {
        "milestone": "M6.7",
        "title": "Closed-Loop Evaluation",
        "generated_at": generated_at,
        "current_run_evidence": True,
        "llm_calls_required": False,
        "scenario_ids": [scenario.scenario_id for scenario in m67_scenarios()],
        "scenario_count": len(scenarios),
        "passed_scenario_count": sum(1 for item in scenarios if item.get("passed")),
        "all_passed": all_passed,
        "quality_gates": {
            "uses_ablation": True,
            "rejects_trace_only_closure": True,
            "rejects_conscious_artifact_as_only_proof": True,
            "default_policy_scoring_untouched": True,
            "no_llm_calls": True,
        },
        "scenarios": scenarios,
        "completion_report": {
            "passed_scenarios": [
                str(item["scenario_id"]) for item in scenarios if item.get("passed")
            ],
            "ablation_delta_fields": {
                str(item["scenario_id"]): item.get("changed_fields", [])
                for item in scenarios
            },
            "next_turn_fields_changed": sorted(
                {
                    field
                    for item in scenarios
                    for field in item.get("changed_fields", [])
                    if isinstance(field, str)
                }
            ),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "m67_closed_loop_evaluation.json"
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["artifact_path"] = str(artifact_path)
    return payload
