from __future__ import annotations

import asyncio
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
from typing import cast

from .action_registry import build_governed_action_registry
from .action_schema import ActionSchema, ensure_action_schema
from .agent import SegmentAgent
from .environment import SimulatedWorld
from .evaluation import RunMetrics
from .fep import advance_state, infer_policy
from .governance import AuthorizationDecision, GovernanceController
from .homeostasis import HomeostasisScheduler, MaintenanceAgenda
from .interoception import ProcessInteroceptor
from .io_bus import ActionBus, ActionDispatchRecord, PerceptionBus, PerceptionPacket
from .llm import InnerSpeechEngine, build_inner_speech_engine
from .logging_utils import ConsciousnessLogger
from .memory import MemoryDecision
from .narrative_ingestion import NarrativeIngestionService
from .narrative_world import NarrativeWorld, NarrativeWorldConfig
from .persistence import SnapshotLoadError, atomic_write_json, load_snapshot, quarantine_snapshot
from .predictive_coding import PredictiveCodingHyperparameters
from .self_model import (
    CapabilityModel,
    ClassificationResult,
    RuntimeFailureEvent,
    SelfModel,
    build_default_self_model,
)
from .state import AgentState, PolicyTendency, TickInput
from .sleep_consolidator import build_sleep_llm_extractor
from .subject_state import SubjectState, apply_subject_state_to_maintenance_agenda, derive_subject_state
from .tracing import JsonlTraceWriter, derive_trace_path
from .types import DecisionDiagnostics, InterventionScore, SleepSummary


STATE_VERSION = "0.6"
SUPPORTED_STATE_VERSIONS = {STATE_VERSION, "0.5", "0.4", "0.3", "0.2", "0.1"}
RESTART_MEMORY_CONTINUITY_WINDOW = 24
RESTART_REBIND_MIN_CYCLE = 128
MATURE_CONTINUITY_MIN_CYCLE = 400


def format_state(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in values.items())


def format_update_summary(label: str, update) -> str:
    return (
        f"{label} raw={update.mean_abs_raw_error():.3f} "
        f"weighted={update.mean_abs_weighted_error():.3f} "
        f"propagated={update.mean_abs_propagated_error():.3f} "
        f"digested={not update.digestion_exceeded}"
    )


def format_action_scores(choice_ranking: list[InterventionScore]) -> str:
    return " | ".join(
        f"{option.choice} score={option.policy_score:.3f} "
        f"efe={option.expected_free_energy:.3f} "
        f"risk={option.risk:.3f} "
        f"mem={option.memory_bias:.3f} "
        f"pat={option.pattern_bias:.3f} "
        f"pol={option.policy_bias:.3f} "
        f"epi={option.epistemic_bonus:.3f} "
        f"ws={option.workspace_bias:.3f} "
        f"social={option.social_bias:.3f} "
        f"commit={option.commitment_bias:.3f} "
        f"id={option.identity_bias:.3f} "
        f"ledger={option.ledger_bias:.3f} "
        f"subj={option.subject_bias:.3f} "
        f"recon={option.reconciliation_bias:.3f} "
        f"verify={option.verification_bias:.3f}"
        for option in choice_ranking
    )


class SegmentRuntime:
    def __init__(
        self,
        agent: SegmentAgent | None = None,
        world: SimulatedWorld | None = None,
        metrics: RunMetrics | None = None,
        host_state: AgentState | None = None,
        interoceptor: ProcessInteroceptor | None = None,
        inner_speech_engine: InnerSpeechEngine | None = None,
        consciousness_logger: ConsciousnessLogger | None = None,
        self_model: SelfModel | None = None,
        perception_bus: PerceptionBus | None = None,
        action_bus: ActionBus | None = None,
        homeostasis_scheduler: HomeostasisScheduler | None = None,
        governance: GovernanceController | None = None,
        state_path: str | Path | None = None,
        trace_path: str | Path | None = None,
        state_load_status: str = "fresh",
    ) -> None:
        self.world = world or SimulatedWorld()
        self.agent = agent or SegmentAgent(rng=self.world.rng)
        self.agent.rng = self.world.rng
        if self.agent.sleep_llm_extractor is None:
            self.agent.sleep_llm_extractor = build_sleep_llm_extractor()
        self.metrics = metrics or RunMetrics()
        self.host_state = host_state or AgentState()
        self.interoceptor = interoceptor or ProcessInteroceptor()
        self.inner_speech_engine = inner_speech_engine or build_inner_speech_engine()
        self.consciousness_logger = consciousness_logger or ConsciousnessLogger()
        self.self_model = self_model or build_default_self_model()
        self.perception_bus = perception_bus or PerceptionBus()
        self.action_bus = action_bus or ActionBus()
        self.homeostasis_scheduler = homeostasis_scheduler or HomeostasisScheduler()
        workspace_root = Path(state_path).resolve().parent if state_path else Path.cwd()
        self.governance = governance or GovernanceController(workspace_root=workspace_root)
        for schema in build_governed_action_registry().get_all():
            if not self.agent.action_registry.contains(schema.name):
                self.agent.action_registry.register(schema, schema.cost_estimate)
        self.agent.self_model.capability_model = CapabilityModel(
            action_schemas=tuple(self.agent.action_registry.get_all()),
            api_limits=self.agent.self_model.capability_model.api_limits,
        )
        self.last_error_attribution: ClassificationResult | None = None
        self.state_path = Path(state_path) if state_path else None
        resolved_trace_path = Path(trace_path) if trace_path else derive_trace_path(self.state_path)
        self.trace_writer = (
            JsonlTraceWriter(resolved_trace_path) if resolved_trace_path else None
        )
        self.state_load_status = state_load_status
        self.narrative_ingestion_service = NarrativeIngestionService()
        self.last_continuity_report = self.agent.self_model.continuity_audit.to_dict()
        self.subject_state = derive_subject_state(
            self.agent,
            continuity_report=self.last_continuity_report,
            previous_state=getattr(self.agent, "subject_state", SubjectState()),
        )
        self.agent.subject_state = self.subject_state
        self.restart_policy_anchors: dict[str, object] = {}
        self.continuity_rebind_ticks_remaining = 0
        self.continuity_rebind_total_ticks = 0

    @classmethod
    def load_or_create(
        cls,
        state_path: str | Path | None = None,
        trace_path: str | Path | None = None,
        seed: int = 17,
        reset: bool = False,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        reset_predictive_precisions: bool = False,
        enable_restart_rebind: bool = False,
    ) -> SegmentRuntime:
        path = Path(state_path) if state_path else None
        resolved_trace_path = (
            Path(trace_path) if trace_path else derive_trace_path(path)
        )
        if reset and resolved_trace_path:
            JsonlTraceWriter(resolved_trace_path).reset()
        if not path or reset or not path.exists():
            world = SimulatedWorld(seed=seed)
            runtime = cls(
                agent=SegmentAgent(
                    rng=world.rng,
                    predictive_hyperparameters=predictive_hyperparameters,
                ),
                world=world,
                state_path=path,
                trace_path=resolved_trace_path,
                state_load_status="fresh" if not reset else "reset",
            )
            runtime.agent.self_model.record_restart_consistency(None, current_tick=runtime.agent.cycle)
            runtime.last_continuity_report = runtime.agent.self_model.continuity_audit.to_dict()
            runtime.subject_state = derive_subject_state(
                runtime.agent,
                continuity_report=runtime.last_continuity_report,
                previous_state=runtime.subject_state,
            )
            runtime.agent.subject_state = runtime.subject_state
            return runtime

        try:
            payload = load_snapshot(
                path,
                supported_versions=SUPPORTED_STATE_VERSIONS,
            )
        except SnapshotLoadError as exc:
            quarantine_snapshot(path, reason=exc.reason)
            if resolved_trace_path:
                JsonlTraceWriter(resolved_trace_path).reset()
            world = SimulatedWorld(seed=seed)
            runtime = cls(
                agent=SegmentAgent(rng=world.rng),
                world=world,
                state_path=path,
                trace_path=resolved_trace_path,
                state_load_status=f"recovered_from_{exc.reason}",
            )
            runtime.agent.self_model.record_restart_consistency(None, current_tick=runtime.agent.cycle)
            runtime.last_continuity_report = runtime.agent.self_model.continuity_audit.to_dict()
            runtime.subject_state = derive_subject_state(
                runtime.agent,
                continuity_report=runtime.last_continuity_report,
                previous_state=runtime.subject_state,
            )
            runtime.agent.subject_state = runtime.subject_state
            return runtime

        world = SimulatedWorld.from_dict(payload.get("world"))
        agent = SegmentAgent.from_dict(
            payload.get("agent"),
            rng=world.rng,
            predictive_hyperparameters=predictive_hyperparameters,
            reset_predictive_precisions=reset_predictive_precisions,
        )
        metrics = RunMetrics.from_dict(payload.get("metrics"))
        host_state = AgentState.from_dict(payload.get("host_state"))
        io_bus_payload = payload.get("io_bus")
        perception_bus = PerceptionBus.from_dict(
            dict(io_bus_payload.get("perception_bus", {}))
            if isinstance(io_bus_payload, dict)
            else None
        )
        action_bus = ActionBus.from_dict(
            dict(io_bus_payload.get("action_bus", {}))
            if isinstance(io_bus_payload, dict)
            else None
        )
        governance = GovernanceController.from_dict(
            payload.get("governance") if isinstance(payload.get("governance"), dict) else None,
            workspace_root=path.resolve().parent if path else Path.cwd(),
        )
        homeostasis_scheduler = HomeostasisScheduler.from_dict(
            payload.get("homeostasis")
            if isinstance(payload.get("homeostasis"), dict)
            else None
        )
        runtime = cls(
            agent=agent,
            world=world,
            metrics=metrics,
            host_state=host_state,
            perception_bus=perception_bus,
            action_bus=action_bus,
            homeostasis_scheduler=homeostasis_scheduler,
            governance=governance,
            state_path=path,
            trace_path=resolved_trace_path,
            state_load_status="restored",
        )
        should_enable_restart_rebind = False
        restart_anchors = payload.get("restart_anchors")
        if isinstance(restart_anchors, dict):
            should_enable_restart_rebind = bool(enable_restart_rebind)
            runtime.restart_policy_anchors = dict(restart_anchors)
            runtime.restart_policy_anchors["policy_rebind_enabled"] = should_enable_restart_rebind
            if should_enable_restart_rebind:
                runtime._activate_continuity_rebind()
            runtime.agent.self_model.apply_restart_anchors(runtime.restart_policy_anchors)
        if should_enable_restart_rebind:
            runtime.agent.long_term_memory.activate_restart_continuity_window(
                current_cycle=runtime.agent.cycle,
                duration=max(RESTART_MEMORY_CONTINUITY_WINDOW, runtime.continuity_rebind_total_ticks),
            )
        runtime.agent.self_model.record_restart_consistency(
            payload.get("m218") if isinstance(payload.get("m218"), dict) else None,
            current_tick=runtime.agent.cycle,
        )
        runtime.last_continuity_report = runtime.agent.self_model.continuity_audit.to_dict()
        runtime.subject_state = SubjectState.from_dict(
            payload.get("subject_state") if isinstance(payload.get("subject_state"), dict) else None
        )
        if not runtime.subject_state.core_identity_summary and not runtime.subject_state.subject_priority_stack:
            runtime.subject_state = derive_subject_state(
                runtime.agent,
                continuity_report=runtime.last_continuity_report,
                previous_state=runtime.agent.subject_state,
                restart_anchors=runtime.restart_policy_anchors,
            )
        runtime.agent.subject_state = runtime.subject_state
        return runtime

    def save_snapshot(self) -> None:
        if not self.state_path:
            return

        atomic_write_json(self.state_path, self._snapshot_payload())

    def execute_governed_action(
        self,
        action: ActionSchema | str | dict[str, object],
        *,
        predicted_effects: dict[str, float] | None = None,
        verbose: bool = False,
    ) -> dict[str, object]:
        schema = ensure_action_schema(action)
        cycle = max(1, self.agent.cycle)
        decision = self.governance.authorize(
            schema,
            predicted_effects=predicted_effects or {},
        )
        trace_record: dict[str, object] = {
            "event": "external_action",
            "cycle": cycle,
            "action": schema.to_dict(),
            "governance": decision.to_dict(),
        }
        if decision.status != "allowed":
            if self.trace_writer:
                self.trace_writer.append(trace_record)
            return {
                "action_name": schema.name,
                "status": decision.status,
                "governance": decision.to_dict(),
                "dispatch": None,
                "repair": None,
            }

        adapter = self.governance.adapters.get(schema.name)
        if adapter is None:
            missing_decision = AuthorizationDecision(
                action_name=schema.name,
                status="denied",
                reason="missing_adapter",
                capability=decision.capability,
                predicted_effects=decision.predicted_effects,
                budget_before=decision.budget_before,
                budget_after=decision.budget_before,
            )
            trace_record["governance"] = missing_decision.to_dict()
            if self.trace_writer:
                self.trace_writer.append(trace_record)
            return {
                "action_name": schema.name,
                "status": "denied",
                "governance": missing_decision.to_dict(),
                "dispatch": None,
                "repair": None,
            }

        try:
            dispatch = self.action_bus.dispatch_to_external_adapter(
                adapter,
                schema,
                cycle=cycle,
                source_type="governed_external",
                source_id=schema.name,
            )
            self.governance.commit(decision, success=dispatch.acknowledgment.success)
            trace_record["dispatch"] = dispatch.to_dict()
            if self.trace_writer:
                self.trace_writer.append(trace_record)
            return {
                "action_name": schema.name,
                "status": "allowed",
                "governance": decision.to_dict(),
                "dispatch": dispatch.to_dict(),
                "repair": None,
            }
        except Exception as exc:
            self.governance.record_failure(
                action_name=schema.name,
                reason=str(exc),
                cycle=cycle,
            )
            repair_schema = self.governance.repair_action(
                failed_action_name=schema.name,
                cycle=cycle,
            )
            repair_decision = self.governance.authorize(
                repair_schema,
                predicted_effects={"repair_signal": 1.0},
            )
            repair_payload: dict[str, object] = {
                "governance": repair_decision.to_dict(),
                "dispatch": None,
            }
            if repair_decision.status == "allowed":
                repair_adapter = self.governance.adapters.get(repair_schema.name)
                if repair_adapter is not None:
                    repair_dispatch = self.action_bus.dispatch_to_external_adapter(
                        repair_adapter,
                        repair_schema,
                        cycle=cycle,
                        source_type="governed_external_repair",
                        source_id=repair_schema.name,
                    )
                    self.governance.commit(
                        repair_decision,
                        success=repair_dispatch.acknowledgment.success,
                    )
                    repair_payload["dispatch"] = repair_dispatch.to_dict()
            trace_record["failure"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            trace_record["repair"] = repair_payload
            if self.trace_writer:
                self.trace_writer.append(trace_record)
            return {
                "action_name": schema.name,
                "status": "failed",
                "governance": decision.to_dict(),
                "dispatch": None,
                "repair": repair_payload,
            }

    def run(
        self,
        cycles: int | None,
        verbose: bool = True,
        host_telemetry: bool = False,
        tick_interval_seconds: float = 0.0,
    ) -> dict[str, object]:
        return asyncio.run(
            self.arun(
                cycles=cycles,
                verbose=verbose,
                host_telemetry=host_telemetry,
                tick_interval_seconds=tick_interval_seconds,
            )
        )

    async def arun(
        self,
        cycles: int | None,
        verbose: bool = True,
        host_telemetry: bool = False,
        tick_interval_seconds: float = 0.0,
    ) -> dict[str, object]:
        if verbose:
            print("Project Segmentum :: Segment Prototype v0.1")
            print("-" * 64)
            print(f"State load: {self.state_load_status}")

        cycle_index = 0
        termination_reason = "running"
        while cycles is None or cycle_index < cycles:
            started = asyncio.get_running_loop().time()
            try:
                cycle_result = await self.astep(
                    verbose=verbose,
                    host_telemetry=host_telemetry,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                termination_reason = f"exception:{type(exc).__name__}"
                self._record_error_event(
                    exc,
                    stage="runtime_step",
                    verbose=verbose,
                    cycle=self.agent.cycle,
                )
                self._warn(
                    f"runtime step failed at cycle {self.agent.cycle}: {exc}",
                    verbose=verbose,
                )
                self._save_snapshot_with_guard(verbose=verbose)
                break
            if not cycle_result["alive"]:
                termination_reason = "agent_depleted"
                break
            cycle_index += 1
            if tick_interval_seconds > 0 and (cycles is None or cycle_index < cycles):
                elapsed = asyncio.get_running_loop().time() - started
                await asyncio.sleep(max(0.0, tick_interval_seconds - elapsed))

        if termination_reason == "running":
            termination_reason = (
                "cycles_exhausted" if cycles is not None else "stopped"
            )
        self.metrics.termination_reason = termination_reason
        self._save_snapshot_with_guard(verbose=verbose)
        summary = self.metrics.summary()
        if verbose:
            print("-" * 64)
            print("Final strategic beliefs:", format_state(self.agent.strategic_layer.beliefs))
            print(
                "Final sensorimotor beliefs:",
                format_state(self.agent.world_model.beliefs),
            )
            print(
                "Final interoceptive beliefs:",
                format_state(self.agent.interoceptive_layer.belief_state.beliefs),
            )
            print(f"Semantic rules stored: {len(self.agent.semantic_memory)}")
            print(f"Sleep cycles stored: {len(self.agent.sleep_history)}")
            print(f"Long-term memory episodes: {len(self.agent.long_term_memory.episodes)}")
            print("Metrics:", json.dumps(summary, ensure_ascii=True, sort_keys=True))

        return summary

    def step(
        self,
        verbose: bool = True,
        host_telemetry: bool = False,
    ) -> dict[str, object]:
        return asyncio.run(
            self.astep(
                verbose=verbose,
                host_telemetry=host_telemetry,
            )
        )

    async def astep(
        self,
        verbose: bool = True,
        host_telemetry: bool = False,
    ) -> dict[str, object]:
        self.last_error_attribution = None
        self.agent.cycle += 1
        observation = self.world.observe()
        observation_packet = self.perception_bus.capture_simulated_world(
            observation,
            cycle=self.agent.cycle,
        )
        decision = self.agent.decision_cycle(observation)
        observed = decision["observed"]
        prediction = decision["prediction"]
        errors = decision["errors"]
        free_energy_before = float(decision["free_energy_before"])
        hierarchy = decision["hierarchy"]
        diagnostics = decision["diagnostics"]
        assert isinstance(diagnostics, DecisionDiagnostics)
        memory_hits = len(diagnostics.retrieved_memories)
        maintenance_agenda = self.homeostasis_scheduler.assess(
            cycle=self.agent.cycle,
            energy=self.agent.energy,
            stress=self.agent.stress,
            fatigue=self.agent.fatigue,
            temperature=self.agent.temperature,
            telemetry_error_count=self.metrics.telemetry_error_count,
            persistence_error_count=self.metrics.persistence_error_count,
            should_sleep=self.agent.should_sleep(),
        )
        maintenance_agenda = self._apply_subject_state_maintenance_priority(
            diagnostics,
            maintenance_agenda,
        )
        maintenance_agenda = self._apply_workspace_maintenance_priority(
            diagnostics,
            maintenance_agenda,
        )
        maintenance_agenda = self._apply_prediction_ledger_maintenance_priority(
            diagnostics,
            maintenance_agenda,
        )
        maintenance_agenda = self._apply_reconciliation_maintenance_priority(
            diagnostics,
            maintenance_agenda,
        )
        maintenance_agenda = self._apply_verification_maintenance_priority(
            diagnostics,
            maintenance_agenda,
        )
        original_choice_name = diagnostics.chosen.choice
        self._enforce_restart_commitment_continuity()
        self._apply_homeostatic_policy_landscape(diagnostics, maintenance_agenda)
        self._apply_continuity_rebind_prior(diagnostics)
        self._apply_mature_continuity_anchor(diagnostics)
        if maintenance_agenda.interrupt_action:
            self._apply_maintenance_interrupt(diagnostics, maintenance_agenda)
        choice = ActionSchema.from_dict(diagnostics.chosen.action_descriptor)
        expected_fe = diagnostics.chosen.expected_free_energy
        choice_cost = diagnostics.chosen.cost

        action_dispatch = self.action_bus.dispatch_to_simulated_world(
            self.world,
            choice,
            cycle=self.agent.cycle,
        )
        verification_action_update = self.agent.verification_loop.register_action_ack(
            tick=self.agent.cycle,
            action_name=choice.name,
            success=bool(action_dispatch.acknowledgment.success),
        )
        direct_feedback = dict(action_dispatch.feedback)
        self.agent.apply_action_feedback(direct_feedback)

        validation_observation = self.world.observe()
        validation_packet = self.perception_bus.capture_simulated_world(
            validation_observation,
            cycle=self.agent.cycle,
            source_id="simulated_world_validation",
        )
        _, _, _, free_energy_after, _ = self.agent.perceive(validation_observation)
        verification_validation = self.agent.verification_loop.process_observation(
            tick=self.agent.cycle,
            observation=asdict(validation_observation),
            ledger=self.agent.prediction_ledger,
            source="validation_observation",
            subject_state=self.subject_state,
        )
        memory_decision = self.agent.integrate_outcome(
            choice=choice,
            observed=observed,
            prediction=prediction,
            errors=errors,
            free_energy_before=free_energy_before,
            free_energy_after=free_energy_after,
        )

        sleep_summary = None
        if self.agent.should_sleep():
            sleep_summary = self.agent.sleep()
            self._enforce_restart_commitment_continuity()
        maintenance_effects = self.homeostasis_scheduler.apply_background_maintenance(
            self.agent
        )
        retired_count = self.agent.long_term_memory.retire_stale_episodes(
            current_cycle=self.agent.cycle,
        )
        rehearsal_batch = self.agent.long_term_memory.rehearsal_batch(
            current_cycle=self.agent.cycle,
        )
        continuity_audit = self.agent.self_model.update_continuity_audit(
            episodic_memory=list(self.agent.long_term_memory.episodes),
            archived_memory=list(self.agent.long_term_memory.archived_episodes),
            action_history=list(self.agent.action_history),
            rehearsal_queue=[
                str(payload.get("episode_id", ""))
                for payload in rehearsal_batch
                if payload.get("episode_id")
            ],
            current_tick=self.agent.cycle,
            slow_continuity_modifier=self.agent.slow_variable_learner.continuity_modifier(),
        )
        self.last_continuity_report = continuity_audit.to_dict()
        self.last_continuity_report["continuity_score"] = max(
            0.0,
            min(
                1.0,
                float(self.last_continuity_report.get("continuity_score", 1.0))
                + self.agent.reconciliation_engine.continuity_modifier(),
            ),
        )
        ledger_runtime_update = self.agent.prediction_ledger.record_runtime_discrepancies(
            tick=self.agent.cycle,
            diagnostics=diagnostics,
            errors=errors,
            maintenance_agenda=maintenance_agenda,
            continuity_score=float(self.last_continuity_report.get("continuity_score", 1.0)),
            subject_state=self.subject_state,
            memory_surprise=memory_decision.total_surprise,
        )
        reconciliation_runtime_update = self.agent.reconciliation_engine.observe_runtime(
            tick=self.agent.cycle,
            diagnostics=diagnostics,
            narrative=self.agent.self_model.identity_narrative,
            prediction_ledger=self.agent.prediction_ledger,
            verification_loop=self.agent.verification_loop,
            subject_state=self.subject_state,
            continuity_score=float(self.last_continuity_report.get("continuity_score", 1.0)),
            slow_biases=self.agent.slow_variable_learner.state.bias_payload(),
        )
        self.subject_state = derive_subject_state(
            self.agent,
            diagnostics=diagnostics,
            continuity_report=self.last_continuity_report,
            maintenance_agenda=maintenance_agenda,
            previous_state=self.subject_state,
            restart_anchors=self.restart_policy_anchors,
        )
        self.agent.subject_state = self.subject_state
        diagnostics.subject_state_summary = self.subject_state.summary_text()
        diagnostics.subject_status_flags = dict(self.subject_state.status_flags)
        diagnostics.subject_priority_stack = [
            item.to_dict() for item in self.subject_state.subject_priority_stack[:4]
        ]
        details = dict(diagnostics.structured_explanation)
        details["prediction_ledger"] = {
            **self.agent.prediction_ledger.explanation_payload(),
            "runtime_update": ledger_runtime_update.to_dict(),
        }
        details["reconciliation"] = {
            **self.agent.reconciliation_engine.explanation_payload(),
            "runtime_update": reconciliation_runtime_update,
        }
        details["verification"] = {
            **self.agent.verification_loop.explanation_payload(chosen_action=choice.name),
            "action_update": verification_action_update.to_dict(),
            "validation_update": verification_validation.to_dict(),
        }
        details["subject_state"] = self.subject_state.explanation_payload()
        diagnostics.structured_explanation = details
        diagnostics.ledger_summary = str(details["prediction_ledger"]["summary"])
        diagnostics.ledger_payload = dict(details["prediction_ledger"])
        diagnostics.reconciliation_summary = str(details["reconciliation"]["summary"])
        diagnostics.reconciliation_payload = dict(details["reconciliation"])
        diagnostics.verification_summary = str(details["verification"]["summary"])
        diagnostics.verification_payload = dict(details["verification"])
        diagnostics.explanation = (
            str(details["prediction_ledger"]["summary"])
            + " "
            + str(details["reconciliation"]["summary"])
            + " "
            + str(details["verification"].get("verification_motive") or details["verification"]["summary"])
            + " "
            + str(details["subject_state"]["summary"])
            + " "
            + diagnostics.explanation
        )
        if self.continuity_rebind_ticks_remaining > 0:
            self.continuity_rebind_ticks_remaining -= 1
        self.homeostasis_scheduler.note_interrupt(
            maintenance_agenda,
            previous_choice=original_choice_name,
            final_choice=choice.name,
        )

        alive = self.agent.energy > 0.01
        self.metrics.record_cycle(
            choice=choice.name,
            free_energy_before=free_energy_before,
            free_energy_after=free_energy_after,
            energy=self.agent.energy,
            stress=self.agent.stress,
            memory_hits=memory_hits,
            slept=sleep_summary is not None,
            alive=alive,
        )
        host_tick = None
        if host_telemetry:
            host_tick = await self._run_host_telemetry_with_guard(verbose=verbose)
        self._write_trace_with_guard(
            self._build_cycle_trace(
                observation_packet=observation_packet,
                validation_packet=validation_packet,
                observed=observed,
                prediction=prediction,
                errors=errors,
                hierarchy=hierarchy,
                diagnostics=diagnostics,
                choice=choice,
                action_dispatch=action_dispatch,
                expected_fe=expected_fe,
                choice_cost=choice_cost,
                free_energy_before=free_energy_before,
                free_energy_after=free_energy_after,
                memory_hits=memory_hits,
                memory_decision=memory_decision,
                sleep_summary=sleep_summary,
                maintenance_agenda=maintenance_agenda,
                maintenance_effects={
                    **maintenance_effects,
                    "retired_episodes": retired_count,
                },
                alive=alive,
                host_tick=host_tick,
            ),
            verbose=verbose,
        )
        self._save_snapshot_with_guard(verbose=verbose)

        if verbose:
            self._print_cycle(
                observed=observed,
                prediction=prediction,
                errors=errors,
                hierarchy=hierarchy,
                diagnostics=diagnostics,
                choice=choice,
                expected_fe=expected_fe,
                choice_cost=choice_cost,
                free_energy_before=free_energy_before,
                free_energy_after=free_energy_after,
                memory_hits=memory_hits,
                memory_decision=memory_decision,
                sleep_summary=sleep_summary,
                maintenance_agenda=maintenance_agenda,
                maintenance_effects=maintenance_effects,
                host_tick=host_tick,
            )

        return {
            "cycle": self.agent.cycle,
            "choice": choice.name,
            "choice_name": choice.name,
            "alive": alive,
            "free_energy_before": free_energy_before,
            "free_energy_after": free_energy_after,
            "host_strategy": (
                host_tick["policy"].chosen_strategy.value if host_tick else None
            ),
        }

    def run_host_telemetry_step(self) -> dict[str, object]:
        return asyncio.run(self.arun_host_telemetry_step())

    async def arun_host_telemetry_step(self) -> dict[str, object]:
        state_before = replace(self.host_state)
        reading = self.interoceptor.sample()
        input_packet = self.perception_bus.capture_interoception(
            reading,
            cycle=state_before.tick_count + 1,
        )
        tick_input = reading.to_tick_input()
        policy = infer_policy(state_before, tick_input)
        inner_speech = await self.inner_speech_engine.generate(
            state_before,
            tick_input,
            policy,
        )
        self.consciousness_logger.append(
            state_before,
            tick_input,
            policy,
            inner_speech,
        )
        self.host_state = advance_state(state_before, tick_input, policy)
        return {
            "state_before": state_before,
            "input_packet": input_packet,
            "tick_input": tick_input,
            "policy": policy,
            "inner_speech": inner_speech,
            "state_after": self.host_state,
        }

    def _body_state(self) -> dict[str, float]:
        return {
            "cycle": float(self.agent.cycle),
            "energy": self.agent.energy,
            "stress": self.agent.stress,
            "fatigue": self.agent.fatigue,
            "temperature": self.agent.temperature,
        }

    async def _run_host_telemetry_with_guard(
        self,
        verbose: bool,
    ) -> dict[str, object] | None:
        try:
            return await self.arun_host_telemetry_step()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics.telemetry_error_count += 1
            self._record_error_event(
                exc,
                stage="host_telemetry",
                verbose=verbose,
                cycle=self.agent.cycle,
            )
            self._warn(f"host telemetry step failed: {exc}", verbose=verbose)
            return None

    def _save_snapshot_with_guard(self, verbose: bool) -> None:
        try:
            self.save_snapshot()
        except Exception as exc:
            self.metrics.persistence_error_count += 1
            self._record_error_event(
                exc,
                stage="snapshot_persistence",
                verbose=verbose,
                cycle=self.agent.cycle,
            )
            self._warn(f"snapshot persistence failed: {exc}", verbose=verbose)

    def _write_trace_with_guard(
        self,
        record: dict[str, object],
        verbose: bool,
    ) -> None:
        if not self.trace_writer:
            return
        try:
            self.trace_writer.append(record)
        except Exception as exc:
            self.metrics.persistence_error_count += 1
            self._record_error_event(
                exc,
                stage="trace_persistence",
                verbose=verbose,
                cycle=self.agent.cycle,
            )
            self._warn(f"trace persistence failed: {exc}", verbose=verbose)

    def _warn(self, message: str, verbose: bool) -> None:
        if verbose:
            print(f"  warning     {message}")

    def _sync_self_model_resource_state(self) -> None:
        token_budget = max(1, self.self_model.body_schema.token_budget)
        internal_energy = max(0.0, min(1.0, self.host_state.internal_energy))
        self.self_model.resource_state.tokens_remaining = int(
            round(internal_energy * token_budget)
        )
        self.self_model.resource_state.cpu_budget = max(
            0.05,
            1.0 - max(0.0, self.host_state.prediction_error),
        )
        self.self_model.resource_state.memory_free = max(
            0.0,
            1024.0 * (1.0 - max(0.0, self.host_state.surprise_load)),
        )

    def _record_error_event(
        self,
        exc: Exception,
        *,
        stage: str,
        verbose: bool,
        cycle: int,
    ) -> ClassificationResult:
        self._sync_self_model_resource_state()
        if stage in {"runtime_step", "host_telemetry", "snapshot_persistence", "trace_persistence"}:
            error_event = RuntimeFailureEvent(
                name=type(exc).__name__,
                stage=stage,
                category=self._runtime_failure_category(exc, stage=stage),
                origin_hint=self._runtime_failure_origin(exc, stage=stage),
                details={
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
                resource_state=self.self_model.resource_state.snapshot(),
            )
            result = self.self_model.inspect_event(error_event)
        else:
            result = self.self_model.inspect_event(exc)
        self.last_error_attribution = result
        self._emit_self_model_warning(result, stage=stage, verbose=verbose)
        if self.trace_writer and stage != "trace_persistence":
            try:
                self.trace_writer.append(
                    {
                        "event": "error",
                        "stage": stage,
                        "cycle": cycle,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "self_model": {
                            "classification": result.classification,
                            "attribution": result.attribution,
                            "surprise_source": result.surprise_source,
                            "detected_threats": list(result.detected_threats),
                            "resource_state": dict(result.resource_state),
                            "evidence": dict(result.evidence),
                        },
                        "error_attribution": {
                            "classification": result.classification,
                            "attribution": result.attribution,
                            "surprise_source": result.surprise_source,
                            "detected_threats": list(result.detected_threats),
                            "evidence": dict(result.evidence),
                        },
                    }
                )
            except Exception:
                pass
        return result

    def _runtime_failure_origin(self, exc: Exception, *, stage: str) -> str:
        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        if any(token in name or token in message for token in ("token", "memory", "context", "budget", "oom")):
            return "self"
        if any(token in name or token in message for token in ("http", "timeout", "network", "external", "readonly", "dom")):
            return "world"
        if stage in {"snapshot_persistence", "trace_persistence"}:
            return "world"
        return "world"

    def _runtime_failure_category(self, exc: Exception, *, stage: str) -> str:
        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        if any(token in name or token in message for token in ("token", "context")):
            return "context_budget"
        if any(token in name or token in message for token in ("memory", "oom")):
            return "memory_budget"
        if "timeout" in name or "timeout" in message:
            return "timeout"
        if any(token in name or token in message for token in ("http", "network", "external")):
            return "external_failure"
        if stage in {"snapshot_persistence", "trace_persistence"}:
            return "tool_failure"
        return "environment_shift"

    def _emit_self_model_warning(
        self,
        result: ClassificationResult,
        *,
        stage: str,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        for line in result.to_log_string().splitlines():
            self._warn(f"{stage} {line}", verbose=verbose)

    def _snapshot_payload(self) -> dict[str, object]:
        return {
            "state_version": STATE_VERSION,
            "agent": self.agent.to_dict(),
            "world": self.world.to_dict(),
            "metrics": self.metrics.to_dict(),
            "host_state": self.host_state.to_dict(),
            "io_bus": {
                "perception_bus": self.perception_bus.to_dict(),
                "action_bus": self.action_bus.to_dict(),
            },
            "homeostasis": self.homeostasis_scheduler.to_dict(),
            "governance": self.governance.to_dict(),
            "subject_state": self.subject_state.to_dict(),
            "m218": dict(self.last_continuity_report),
            "restart_anchors": self.agent.self_model.build_restart_anchors(
                maintenance_agenda=(
                    self.homeostasis_scheduler.last_agenda.to_dict()
                    if self.homeostasis_scheduler.last_agenda is not None
                    else None
                ),
                memory_anchors=self.agent.long_term_memory.restart_anchor_payload(limit=16),
                recent_actions=list(self.agent.action_history[-32:]),
            ),
        }

    def _apply_maintenance_interrupt(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> None:
        if agenda.interrupt_action is None:
            return
        override = next(
            (
                option
                for option in diagnostics.ranked_options
                if option.choice == agenda.interrupt_action
            ),
            None,
        )
        if override is None:
            return
        previous_choice = diagnostics.chosen.choice
        diagnostics.chosen = override
        diagnostics.explanation = (
            f"{diagnostics.explanation} Homeostasis interrupt: switched from "
            f"{previous_choice} to {override.choice} because {agenda.interrupt_reason}."
        )
        details = dict(diagnostics.structured_explanation)
        details["homeostasis_interrupt"] = {
            "previous_choice": previous_choice,
            "override_choice": override.choice,
            "reason": agenda.interrupt_reason,
        }
        diagnostics.structured_explanation = details

    def _activate_continuity_rebind(self) -> None:
        policy_distribution = self.restart_policy_anchors.get("preferred_policy_distribution")
        window = 6
        if isinstance(policy_distribution, dict) and policy_distribution:
            window += 4
        commitment_snapshot = self.restart_policy_anchors.get("commitment_snapshot")
        if isinstance(commitment_snapshot, list) and commitment_snapshot:
            window = max(window, 12)
        self.continuity_rebind_ticks_remaining = window
        self.continuity_rebind_total_ticks = window

    def _enforce_restart_commitment_continuity(self) -> None:
        if self.continuity_rebind_ticks_remaining <= 0 or not self.restart_policy_anchors:
            return
        reference_commitments = [
            str(item)
            for item in self.restart_policy_anchors.get("commitment_snapshot", [])
            if str(item)
        ]
        if not reference_commitments:
            return
        self.agent.self_model.enforce_restart_commitment_continuity(
            reference_commitment_ids=reference_commitments,
            max_new_commitments=1,
        )

    def _resort_diagnostics(self, diagnostics: DecisionDiagnostics) -> None:
        diagnostics.ranked_options.sort(
            key=lambda option: (
                option.policy_score,
                -option.expected_free_energy,
                option.choice,
            ),
            reverse=True,
        )
        diagnostics.chosen = diagnostics.ranked_options[0]
        diagnostics.policy_scores = {
            option.choice: option.policy_score for option in diagnostics.ranked_options
        }

    def _apply_homeostatic_policy_landscape(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> None:
        if not agenda.protected_mode and not agenda.recovery_rebound_active:
            return
        safe_actions = {"rest", "hide", "exploit_shelter", "thermoregulate"}
        suppressed_actions = set(agenda.suppressed_actions)
        for option in diagnostics.ranked_options:
            if option.choice in safe_actions:
                option.policy_score += 0.22 + (agenda.policy_shift_strength * 0.70)
                if option.choice == agenda.recommended_action:
                    option.policy_score += 0.18 + (agenda.policy_shift_strength * 0.45)
            if option.choice in suppressed_actions:
                option.policy_score -= 0.20 + (agenda.policy_shift_strength * 0.85)
            if agenda.recovery_rebound_active and option.choice == "rest":
                option.policy_score += 0.22
            if agenda.recovery_rebound_active and option.choice == "hide":
                option.policy_score += 0.08
        self._resort_diagnostics(diagnostics)
        details = dict(diagnostics.structured_explanation)
        details["homeostatic_policy_landscape"] = {
            "protected_mode": agenda.protected_mode,
            "recovery_rebound_active": agenda.recovery_rebound_active,
            "policy_shift_strength": agenda.policy_shift_strength,
            "suppressed_actions": list(agenda.suppressed_actions),
            "recommended_action": agenda.recommended_action,
        }
        diagnostics.structured_explanation = details

    def _apply_workspace_maintenance_priority(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> MaintenanceAgenda:
        signal = self.agent.global_workspace.maintenance_signal(self.agent.last_workspace_state)
        priority_gain = float(signal.get("priority_gain", 0.0))
        active_tasks = list(agenda.active_tasks)
        for task in signal.get("active_tasks", []):
            if task not in active_tasks:
                active_tasks.insert(0, str(task))
        recommended_action = agenda.recommended_action
        if str(signal.get("recommended_action", "")):
            recommended_action = str(signal["recommended_action"])
        updated = replace(
            agenda,
            active_tasks=tuple(active_tasks),
            recommended_action=recommended_action,
            policy_shift_strength=min(1.0, agenda.policy_shift_strength + priority_gain),
        )
        details = dict(diagnostics.structured_explanation)
        details["workspace_maintenance_priority"] = {
            "priority_gain": round(priority_gain, 6),
            "active_tasks": list(active_tasks),
            "recommended_action": recommended_action,
        }
        diagnostics.structured_explanation = details
        return updated

    def _apply_prediction_ledger_maintenance_priority(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> MaintenanceAgenda:
        signal = self.agent.prediction_ledger.maintenance_signal()
        priority_gain = float(signal.get("priority_gain", 0.0))
        active_tasks = list(agenda.active_tasks)
        for task in signal.get("active_tasks", []):
            if task not in active_tasks:
                active_tasks.insert(0, str(task))
        recommended_action = agenda.recommended_action
        ledger_recommended = str(signal.get("recommended_action", ""))
        if ledger_recommended:
            recommended_action = ledger_recommended
        suppressed_actions = tuple(
            dict.fromkeys([*agenda.suppressed_actions, *[str(item) for item in signal.get("suppressed_actions", [])]])
        )
        updated = replace(
            agenda,
            active_tasks=tuple(active_tasks),
            recommended_action=recommended_action,
            suppressed_actions=suppressed_actions,
            policy_shift_strength=min(1.0, agenda.policy_shift_strength + priority_gain),
        )
        details = dict(diagnostics.structured_explanation)
        details["prediction_ledger_maintenance_priority"] = {
            "priority_gain": round(priority_gain, 6),
            "active_tasks": list(active_tasks),
            "recommended_action": recommended_action,
            "suppressed_actions": list(suppressed_actions),
        }
        diagnostics.structured_explanation = details
        return updated

    def _apply_verification_maintenance_priority(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> MaintenanceAgenda:
        signal = self.agent.verification_loop.maintenance_signal()
        priority_gain = float(signal.get("priority_gain", 0.0))
        active_tasks = list(agenda.active_tasks)
        for task in signal.get("active_tasks", []):
            if task not in active_tasks:
                active_tasks.insert(0, str(task))
        recommended_action = agenda.recommended_action
        verification_recommended = str(signal.get("recommended_action", ""))
        if verification_recommended:
            recommended_action = verification_recommended
        updated = replace(
            agenda,
            active_tasks=tuple(active_tasks),
            recommended_action=recommended_action,
            policy_shift_strength=min(1.0, agenda.policy_shift_strength + priority_gain),
        )
        details = dict(diagnostics.structured_explanation)
        details["verification_maintenance_priority"] = {
            "priority_gain": round(priority_gain, 6),
            "active_tasks": list(active_tasks),
            "recommended_action": recommended_action,
        }
        diagnostics.structured_explanation = details
        return updated

    def _apply_reconciliation_maintenance_priority(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> MaintenanceAgenda:
        signal = self.agent.reconciliation_engine.maintenance_signal()
        priority_gain = float(signal.get("priority_gain", 0.0))
        active_tasks = list(agenda.active_tasks)
        for task in signal.get("active_tasks", []):
            if task not in active_tasks:
                active_tasks.insert(0, str(task))
        recommended_action = agenda.recommended_action
        reconciliation_recommended = str(signal.get("recommended_action", ""))
        if reconciliation_recommended:
            recommended_action = reconciliation_recommended
        suppressed_actions = tuple(
            dict.fromkeys([*agenda.suppressed_actions, *[str(item) for item in signal.get("suppressed_actions", [])]])
        )
        updated = replace(
            agenda,
            active_tasks=tuple(active_tasks),
            recommended_action=recommended_action,
            suppressed_actions=suppressed_actions,
            policy_shift_strength=min(1.0, agenda.policy_shift_strength + priority_gain),
        )
        details = dict(diagnostics.structured_explanation)
        details["reconciliation_maintenance_priority"] = {
            "priority_gain": round(priority_gain, 6),
            "active_tasks": list(active_tasks),
            "recommended_action": recommended_action,
            "suppressed_actions": list(suppressed_actions),
        }
        diagnostics.structured_explanation = details
        return updated

    def _apply_subject_state_maintenance_priority(
        self,
        diagnostics: DecisionDiagnostics,
        agenda: MaintenanceAgenda,
    ) -> MaintenanceAgenda:
        subject_state = derive_subject_state(
            self.agent,
            diagnostics=diagnostics,
            maintenance_agenda=agenda,
            continuity_report=self.last_continuity_report,
            previous_state=self.subject_state,
            restart_anchors=self.restart_policy_anchors,
        )
        self.subject_state = subject_state
        self.agent.subject_state = subject_state
        if self.continuity_rebind_ticks_remaining > 0:
            details = dict(diagnostics.structured_explanation)
            details["subject_state"] = subject_state.explanation_payload()
            details["subject_state_maintenance_priority"] = {
                "priority_gain": 0.0,
                "recommended_action": agenda.recommended_action,
                "interrupt_action": agenda.interrupt_action,
                "active_tasks": list(agenda.active_tasks),
                "status_flags": dict(subject_state.status_flags),
                "rebind_window_active": True,
            }
            diagnostics.structured_explanation = details
            diagnostics.subject_state_summary = subject_state.summary_text()
            diagnostics.subject_status_flags = dict(subject_state.status_flags)
            diagnostics.subject_priority_stack = [
                item.to_dict() for item in subject_state.subject_priority_stack[:4]
            ]
            return agenda
        updated, maintenance_details = apply_subject_state_to_maintenance_agenda(
            subject_state,
            agenda,
        )
        details = dict(diagnostics.structured_explanation)
        details["subject_state"] = subject_state.explanation_payload()
        details["subject_state_maintenance_priority"] = maintenance_details
        diagnostics.structured_explanation = details
        diagnostics.subject_state_summary = subject_state.summary_text()
        diagnostics.subject_status_flags = dict(subject_state.status_flags)
        diagnostics.subject_priority_stack = [
            item.to_dict() for item in subject_state.subject_priority_stack[:4]
        ]
        return updated

    def _apply_continuity_rebind_prior(self, diagnostics: DecisionDiagnostics) -> None:
        if self.continuity_rebind_ticks_remaining <= 0 or not self.restart_policy_anchors:
            return
        if not bool(self.restart_policy_anchors.get("policy_rebind_enabled", False)):
            return
        decay = self.continuity_rebind_ticks_remaining / max(1, self.continuity_rebind_total_ticks)
        preferred_distribution = self.restart_policy_anchors.get("preferred_policy_distribution")
        if isinstance(preferred_distribution, dict):
            for option in diagnostics.ranked_options:
                option.policy_score += float(preferred_distribution.get(option.choice, 0.0)) * 1.35 * decay
        dominant_strategy = str(self.restart_policy_anchors.get("dominant_strategy", ""))
        if dominant_strategy and diagnostics.ranked_options:
            chosen_component = diagnostics.ranked_options[0].dominant_component
            if chosen_component != dominant_strategy:
                for option in diagnostics.ranked_options:
                    if option.dominant_component == dominant_strategy:
                        option.policy_score += 0.18 * decay
        learned_avoidances = {
            str(item) for item in self.restart_policy_anchors.get("learned_avoidances", [])
        }
        learned_preferences = {
            str(item) for item in self.restart_policy_anchors.get("learned_preferences", [])
        }
        commitment_priors = {
            str(item) for item in self.restart_policy_anchors.get("commitment_action_priors", [])
        }
        maintenance_agenda = self.restart_policy_anchors.get("maintenance_agenda")
        maintenance_recommended = ""
        maintenance_suppressed: set[str] = set()
        if isinstance(maintenance_agenda, dict):
            maintenance_recommended = str(
                maintenance_agenda.get("interrupt_action")
                or maintenance_agenda.get("recommended_action")
                or ""
            )
            maintenance_suppressed = {
                str(item) for item in maintenance_agenda.get("suppressed_actions", [])
            }
        recent_actions = [
            str(item) for item in self.restart_policy_anchors.get("recent_actions", [])
        ]
        recent_action_weights: dict[str, float] = {}
        for index, action in enumerate(reversed(recent_actions[-8:]), start=1):
            recent_action_weights[action] = max(recent_action_weights.get(action, 0.0), 1.0 / index)
        recent_priority_actions = {str(action) for action in recent_actions[-4:]}
        preferred_rebind_actions = set(recent_priority_actions) | set(commitment_priors)
        if maintenance_recommended:
            preferred_rebind_actions.add(maintenance_recommended)
        for option in diagnostics.ranked_options:
            if option.choice in learned_avoidances:
                option.policy_score -= 0.18 * decay
            if option.choice in learned_preferences:
                option.policy_score += 0.18 * decay
            if option.choice in commitment_priors:
                option.policy_score += 0.40 * decay
            option.policy_score += recent_action_weights.get(option.choice, 0.0) * 0.60 * decay
            if option.choice in recent_priority_actions:
                option.policy_score += 0.42 * decay
            elif recent_action_weights:
                option.policy_score -= 0.24 * decay
            if preferred_rebind_actions:
                if option.choice in preferred_rebind_actions:
                    option.policy_score += 0.62 * decay
                else:
                    option.policy_score -= 0.42 * decay
            if maintenance_recommended and option.choice == maintenance_recommended:
                option.policy_score += 0.34 * decay
            if option.choice in maintenance_suppressed:
                option.policy_score -= 0.26 * decay
        self._resort_diagnostics(diagnostics)
        diagnostics.explanation = (
            f"{diagnostics.explanation} Continuity rebind applied with decay={decay:.2f}."
        )
        details = dict(diagnostics.structured_explanation)
        details["continuity_rebind"] = {
            "ticks_remaining": self.continuity_rebind_ticks_remaining,
            "dominant_strategy": dominant_strategy,
            "learned_avoidances": sorted(learned_avoidances),
            "learned_preferences": sorted(learned_preferences),
            "commitment_action_priors": sorted(commitment_priors),
            "maintenance_recommended_action": maintenance_recommended,
            "maintenance_suppressed_actions": sorted(maintenance_suppressed),
        }
        diagnostics.structured_explanation = details

    def _apply_mature_continuity_anchor(self, diagnostics: DecisionDiagnostics) -> None:
        if self.agent.cycle < MATURE_CONTINUITY_MIN_CYCLE:
            return
        recent_actions = list(self.agent.action_history[-96:])
        if len(recent_actions) < 48:
            return
        counts: dict[str, int] = {}
        for action in recent_actions:
            counts[action] = counts.get(action, 0) + 1
        dominant_action, dominant_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        dominant_share = dominant_count / max(1, len(recent_actions))
        if dominant_share < 0.30:
            return
        stable_actions = {"rest", "hide", "exploit_shelter", "thermoregulate"}
        for option in diagnostics.ranked_options:
            if option.choice == dominant_action:
                option.policy_score += 0.42
            elif option.choice in stable_actions and dominant_action in stable_actions:
                option.policy_score += 0.14
            elif option.choice not in {dominant_action, "unstable_workspace_note"}:
                option.policy_score -= 0.22
        self._resort_diagnostics(diagnostics)
        details = dict(diagnostics.structured_explanation)
        details["mature_continuity_anchor"] = {
            "cycle": self.agent.cycle,
            "dominant_action": dominant_action,
            "dominant_share": round(dominant_share, 6),
        }
        diagnostics.structured_explanation = details


    @classmethod
    def load_world(cls, world_path: str | Path, *, seed: int | None = None) -> NarrativeWorld:
        path = Path(world_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = NarrativeWorldConfig.from_dict(payload)
        return NarrativeWorld(config, rng_seed=seed if seed is not None else config.seed)

    def run_world_episode(
        self,
        *,
        world: NarrativeWorld,
        cycles: int,
        ingest_events: bool = True,
    ) -> dict[str, object]:
        action_counts: dict[str, int] = {}
        selected_channel_counts: dict[str, int] = {}
        conditioned_prediction_errors: list[float] = []
        ingestion_records: list[dict[str, object]] = []
        event_count = 0

        for tick in range(cycles):
            self.agent.cycle += 1
            observation = world.observe(tick)
            decision = self.agent.decision_cycle(observation)
            diagnostics = cast(DecisionDiagnostics, decision["diagnostics"])
            action = diagnostics.chosen.choice
            action_counts[action] = action_counts.get(action, 0) + 1
            for channel in diagnostics.attention_selected_channels:
                selected_channel_counts[channel] = selected_channel_counts.get(channel, 0) + 1
            conditioned_prediction_errors.append(float(diagnostics.prediction_error))

            direct_feedback = world.apply_action(action, tick)
            self.agent.apply_action_feedback(direct_feedback)

            if ingest_events:
                episodes = world.narrative_episodes(tick)
                if episodes:
                    event_count += len(episodes)
                    ingestion_records.extend(
                        self.narrative_ingestion_service.ingest(
                            agent=self.agent,
                            episodes=episodes,
                        )
                    )

        total_actions = sum(action_counts.values()) or 1
        return {
            "world_id": world.config.world_id,
            "ticks": cycles,
            "event_count": event_count,
            "action_distribution": {
                action: count / total_actions
                for action, count in sorted(action_counts.items())
            },
            "mean_conditioned_prediction_error": (
                sum(conditioned_prediction_errors) / max(1, len(conditioned_prediction_errors))
            ),
            "selected_channel_statistics": dict(sorted(selected_channel_counts.items())),
            "narrative_ingestion_count": len(ingestion_records),
            "agent_state": {
                "energy": self.agent.energy,
                "stress": self.agent.stress,
                "fatigue": self.agent.fatigue,
                "temperature": self.agent.temperature,
                "narrative_priors": self.agent.self_model.narrative_priors.to_dict(),
                "social_memory": self.agent.social_memory.snapshot(),
            },
        }

    def _build_cycle_trace(
        self,
        observation_packet: PerceptionPacket,
        validation_packet: PerceptionPacket,
        observed: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        hierarchy,
        diagnostics: DecisionDiagnostics,
        choice: ActionSchema,
        action_dispatch: ActionDispatchRecord,
        expected_fe: float,
        choice_cost: float,
        free_energy_before: float,
        free_energy_after: float,
        memory_hits: int,
        memory_decision: MemoryDecision,
        sleep_summary: SleepSummary | None,
        maintenance_agenda: MaintenanceAgenda,
        maintenance_effects: dict[str, object],
        alive: bool,
        host_tick: dict[str, object] | None,
    ) -> dict[str, object]:
        trace_record: dict[str, object] = {
            "event": "cycle",
            "state_version": STATE_VERSION,
            "state_load_status": self.state_load_status,
            "cycle": self.agent.cycle,
            "alive": alive,
            "choice": choice.name,
            "action": choice.to_dict(),
            "expected_free_energy": expected_fe,
            "choice_cost": choice_cost,
            "free_energy_before": free_energy_before,
            "free_energy_after": free_energy_after,
            "memory_hits": memory_hits,
            "memory_hit": diagnostics.memory_hit,
            "retrieved_episode_ids": list(diagnostics.retrieved_episode_ids),
            "memory_context_summary": diagnostics.memory_context_summary,
            "prediction_before_memory": diagnostics.prediction_before_memory,
            "prediction_after_memory": diagnostics.prediction_after_memory,
            "prediction_delta": diagnostics.prediction_delta,
            "sleep_triggered": sleep_summary is not None,
            "body_state": {
                "energy": self.agent.energy,
                "stress": self.agent.stress,
                "fatigue": self.agent.fatigue,
                "temperature": self.agent.temperature,
                "dopamine": self.agent.dopamine,
            },
            "world_state": {
                "seed": self.world.seed,
                "tick": self.world.tick,
                "food_density": self.world.food_density,
                "threat_density": self.world.threat_density,
                "novelty_density": self.world.novelty_density,
                "shelter_density": self.world.shelter_density,
                "temperature": self.world.temperature,
                "social_density": self.world.social_density,
            },
            "observation": observed,
            "prediction": prediction,
            "errors": errors,
            "prediction_errors": self.agent.prediction_error_trace(observed, prediction),
            "hierarchy": asdict(hierarchy),
            "decision_ranking": [
                {
                    "choice": option.choice,
                    "action": dict(option.action_descriptor),
                    "policy_score": option.policy_score,
                    "expected_free_energy": option.expected_free_energy,
                    "predicted_error": option.predicted_error,
                    "ambiguity": option.action_ambiguity,
                    "action_ambiguity": option.action_ambiguity,
                    "risk": option.risk,
                    "preferred_probability": option.preferred_probability,
                    "memory_bias": option.memory_bias,
                    "pattern_bias": option.pattern_bias,
                    "policy_bias": option.policy_bias,
                    "epistemic_bonus": option.epistemic_bonus,
                    "workspace_bias": option.workspace_bias,
                    "social_bias": option.social_bias,
                    "commitment_bias": option.commitment_bias,
                    "identity_bias": option.identity_bias,
                    "ledger_bias": option.ledger_bias,
                    "subject_bias": option.subject_bias,
                    "reconciliation_bias": option.reconciliation_bias,
                    "verification_bias": option.verification_bias,
                    "goal_alignment": option.goal_alignment,
                    "value_score": option.value_score,
                    "dominant_component": option.dominant_component,
                    "predicted_outcome": option.predicted_outcome,
                    "predicted_effects": option.predicted_effects,
                    "cost": option.cost,
                }
                for option in diagnostics.ranked_options
            ],
            "decision_loop": {
                "prediction_error": diagnostics.chosen.predicted_error,
                "current_prediction_error": diagnostics.prediction_error,
                "predicted_outcome": diagnostics.chosen.predicted_outcome,
                "preferred_probability": diagnostics.chosen.preferred_probability,
                "risk": diagnostics.chosen.risk,
                "ambiguity": diagnostics.chosen.action_ambiguity,
                "expected_free_energy": diagnostics.chosen.expected_free_energy,
                "value_score": memory_decision.value_score,
                "memory_bias": diagnostics.chosen.memory_bias,
                "pattern_bias": diagnostics.chosen.pattern_bias,
                "policy_bias": diagnostics.chosen.policy_bias,
                "epistemic_bonus": diagnostics.chosen.epistemic_bonus,
                "workspace_bias": diagnostics.chosen.workspace_bias,
                "social_bias": diagnostics.chosen.social_bias,
                "commitment_bias": diagnostics.chosen.commitment_bias,
                "identity_bias": diagnostics.chosen.identity_bias,
                "ledger_bias": diagnostics.chosen.ledger_bias,
                "subject_bias": diagnostics.chosen.subject_bias,
                "reconciliation_bias": diagnostics.chosen.reconciliation_bias,
                "verification_bias": diagnostics.chosen.verification_bias,
                "goal_alignment": diagnostics.chosen.goal_alignment,
                "active_goal": diagnostics.active_goal,
                "goal_context": diagnostics.goal_context,
                "policy_score": diagnostics.chosen.policy_score,
                "policy_scores": diagnostics.policy_scores,
                "chosen_action": choice.to_dict(),
                "chosen_action_name": choice.name,
                "total_surprise": memory_decision.total_surprise,
                "explanation": diagnostics.explanation,
                "explanation_details": diagnostics.structured_explanation,
                "attention_selected_channels": list(diagnostics.attention_selected_channels),
                "attention_dropped_channels": list(diagnostics.attention_dropped_channels),
                "attention_salience_scores": dict(diagnostics.attention_salience_scores),
                "workspace_latent_channels": list(diagnostics.workspace_latent_channels),
                "workspace_attended_channels": list(diagnostics.workspace_attended_channels),
                "workspace_broadcast_channels": list(diagnostics.workspace_broadcast_channels),
                "workspace_suppressed_channels": list(diagnostics.workspace_suppressed_channels),
                "workspace_carry_over_channels": list(diagnostics.workspace_carry_over_channels),
                "workspace_broadcast_intensity": diagnostics.workspace_broadcast_intensity,
                "workspace_persistence_horizon": diagnostics.workspace_persistence_horizon,
                "conscious_report_channels": list(diagnostics.conscious_report_channels),
                "current_commitments": list(diagnostics.current_commitments),
                "relevant_commitments": list(diagnostics.relevant_commitments),
                "commitment_focus": list(diagnostics.commitment_focus),
                "violated_commitments": list(diagnostics.violated_commitments),
                "commitment_compatibility_score": diagnostics.commitment_compatibility_score,
                "self_inconsistency_error": diagnostics.self_inconsistency_error,
                "conflict_type": diagnostics.conflict_type,
                "severity_level": diagnostics.severity_level,
                "consistency_classification": diagnostics.consistency_classification,
                "behavioral_classification": diagnostics.behavioral_classification,
                "repair_triggered": diagnostics.repair_triggered,
                "repair_policy": diagnostics.repair_policy,
                "repair_result": dict(diagnostics.repair_result),
                "identity_tension": diagnostics.identity_tension,
                "identity_repair_policy": diagnostics.identity_repair_policy,
                "social_focus": list(diagnostics.social_focus),
                "social_alerts": list(diagnostics.social_alerts),
                "social_snapshot": dict(diagnostics.social_snapshot),
                "ledger_summary": diagnostics.ledger_summary,
                "ledger_payload": dict(diagnostics.ledger_payload),
                "verification_summary": diagnostics.verification_summary,
                "verification_payload": dict(diagnostics.verification_payload),
                "reconciliation_summary": diagnostics.reconciliation_summary,
                "reconciliation_payload": dict(diagnostics.reconciliation_payload),
                "subject_state_summary": diagnostics.subject_state_summary,
                "subject_status_flags": dict(diagnostics.subject_status_flags),
            },
            "retrieved_memories": diagnostics.retrieved_memories,
            "episodic_memory": memory_decision.to_dict(),
            "running_metrics": self.metrics.summary(),
            "recent_action_history": list(self.agent.action_history),
            "homeostasis": {
                "agenda": maintenance_agenda.to_dict(),
                "effects": dict(maintenance_effects),
            },
            "io": {
                "perception": {
                    "primary_observation": observation_packet.to_dict(),
                    "validation_observation": validation_packet.to_dict(),
                },
                "action": {
                    "dispatch": action_dispatch.to_dict(),
                    "acknowledgment": action_dispatch.acknowledgment.to_dict(),
                },
            },
        }
        if self.agent.last_attention_trace is not None:
            trace_record["attention"] = self.agent.last_attention_trace.to_dict()
            trace_record["attention"]["filtered_observation"] = dict(
                self.agent.last_attention_filtered_observation
            )
        if self.agent.last_workspace_state is not None:
            trace_record["workspace"] = self.agent.last_workspace_state.to_dict()
        if self.agent.self_model.identity_narrative is not None:
            narrative = self.agent.self_model.identity_narrative
            trace_record["identity"] = {
                "core_identity": narrative.core_identity,
                "core_summary": narrative.core_summary,
                "trait_self_model": dict(narrative.trait_self_model),
                "commitments": [commitment.to_dict() for commitment in narrative.commitments],
                "chapter_transition_evidence": list(narrative.chapter_transition_evidence),
                "identity_tension_history": list(self.agent.identity_tension_history[-8:]),
                "self_inconsistency_events": [
                    event.to_dict()
                    for event in self.agent.self_model.self_inconsistency_events[-8:]
                ],
                "repair_history": [
                    record.to_dict()
                    for record in self.agent.self_model.repair_history[-8:]
                ],
            }
        trace_record["social_memory"] = self.agent.social_memory.to_dict()
        trace_record["prediction_ledger"] = self.agent.prediction_ledger.to_dict()
        trace_record["reconciliation"] = self.agent.reconciliation_engine.to_dict()
        trace_record["verification_loop"] = self.agent.verification_loop.to_dict()
        trace_record["subject_state"] = self.subject_state.to_dict()
        trace_record["slow_learning"] = self.agent.slow_variable_learner.to_dict()
        trace_record["continuity"] = dict(self.last_continuity_report)
        if self.last_error_attribution is not None:
            trace_record["last_error_attribution"] = {
                "classification": self.last_error_attribution.classification,
                "attribution": self.last_error_attribution.attribution,
                "surprise_source": self.last_error_attribution.surprise_source,
                "detected_threats": list(self.last_error_attribution.detected_threats),
                "evidence": dict(self.last_error_attribution.evidence),
            }
        if sleep_summary:
            sleep_dict = asdict(sleep_summary)
            cm = sleep_summary.consolidation_metrics
            if cm is not None:
                sleep_dict["consolidation_metrics"] = cm.to_dict()
            trace_record["sleep_summary"] = sleep_dict
        if host_tick:
            policy = host_tick["policy"]
            input_packet = host_tick["input_packet"]
            tick_input = host_tick["tick_input"]
            state_after = host_tick["state_after"]
            assert isinstance(policy, PolicyTendency)
            assert isinstance(input_packet, PerceptionPacket)
            assert isinstance(tick_input, TickInput)
            assert isinstance(state_after, AgentState)
            trace_record["host_tick"] = {
                "strategy": policy.chosen_strategy.value,
                "input_packet": input_packet.to_dict(),
                "tick_input": asdict(tick_input),
                "state_after": asdict(state_after),
                "inner_speech": host_tick["inner_speech"],
            }
        return trace_record

    def _print_cycle(
        self,
        observed: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        hierarchy,
        diagnostics: DecisionDiagnostics,
        choice: ActionSchema,
        expected_fe: float,
        choice_cost: float,
        free_energy_before: float,
        free_energy_after: float,
        memory_hits: int,
        memory_decision: MemoryDecision,
        sleep_summary: SleepSummary | None,
        maintenance_agenda: MaintenanceAgenda,
        maintenance_effects: dict[str, object],
        host_tick: dict[str, object] | None,
    ) -> None:
        print(
            f"[cycle {self.agent.cycle:02d}] choice={choice.name:>15}  "
            f"score={diagnostics.chosen.policy_score:.3f}  "
            f"expected_fe={expected_fe:.3f}  cost={choice_cost:.2f}"
        )
        print(f"  sensed      {format_state(observed)}")
        print(f"  predicted   {format_state(prediction)}")
        print(f"  error       {format_state(errors)}")
        print(
            "  top-down    "
            f"prior->{format_state(hierarchy.strategic_prior)}"
        )
        print(
            "  top-down    "
            f"strategic->{format_state(hierarchy.strategic_prediction)}"
        )
        print(
            "  top-down    "
            f"sensorimotor->{format_state(hierarchy.sensorimotor_prediction)}"
        )
        print(
            "  top-down    "
            f"interoceptive->{format_state(hierarchy.interoceptive_prediction)}"
        )
        print(
            "  bottom-up   "
            f"intero obs={format_state(hierarchy.observation)}"
        )
        print(
            "  bottom-up   "
            f"{format_update_summary('intero', hierarchy.interoceptive_update)}"
        )
        print(
            "  bottom-up   "
            f"intero->sensor signal={format_state(hierarchy.sensorimotor_observation)}"
        )
        print(
            "  bottom-up   "
            f"{format_update_summary('sensor', hierarchy.sensorimotor_update)}"
        )
        print(
            "  bottom-up   "
            f"sensor->strategic signal={format_state(hierarchy.strategic_observation)}"
        )
        print(
            "  bottom-up   "
            f"{format_update_summary('strategic', hierarchy.strategic_update)}"
        )
        print(
            "  precision   "
            f"intero={format_state(hierarchy.interoceptive_update.error_precision)}"
        )
        print(
            "  precision   "
            f"sensor={format_state(hierarchy.sensorimotor_update.error_precision)}"
        )
        print(
            "  precision   "
            f"strategic={format_state(hierarchy.strategic_update.error_precision)}"
        )
        print(f"  scoring     {format_action_scores(diagnostics.ranked_options)}")

        drive_str = ", ".join(
            f"{drive.name}={drive.urgency:.2f}"
            for drive in self.agent.drive_system.drives
            if drive.urgency > 0.15
        )
        if drive_str:
            print(f"  drives      {drive_str}")

        print(
            "  body        "
            f"energy={self.agent.energy:.2f}, stress={self.agent.stress:.2f}, "
            f"fatigue={self.agent.fatigue:.2f}, temp={self.agent.temperature:.2f}, "
            f"dopamine={self.agent.dopamine:.2f}, "
            f"free_energy={free_energy_before:.3f}->{free_energy_after:.3f}"
        )

        if memory_hits:
            print(f"  memory      retrieved {memory_hits} similar episode(s)")
        print(
            "  episodic    "
            f"value={memory_decision.value_score:.2f}, "
            f"pe={memory_decision.prediction_error:.3f}, "
            f"total={memory_decision.total_surprise:.3f}, "
            f"created={memory_decision.episode_created}"
        )
        print(
            "  policy      "
            f"pe={diagnostics.prediction_error:.3f}, "
            f"memory={diagnostics.chosen.memory_bias:.3f}, "
            f"pattern={diagnostics.chosen.pattern_bias:.3f}, "
            f"policy_bias={diagnostics.chosen.policy_bias:.3f}, "
            f"epistemic={diagnostics.chosen.epistemic_bonus:.3f}, "
            f"workspace={diagnostics.chosen.workspace_bias:.3f}, "
            f"social={diagnostics.chosen.social_bias:.3f}, "
            f"commitment={diagnostics.chosen.commitment_bias:.3f}, "
            f"identity={diagnostics.chosen.identity_bias:.3f}, "
            f"ledger={diagnostics.chosen.ledger_bias:.3f}"
        )
        if diagnostics.workspace_broadcast_channels:
            print(
                "  workspace   "
                f"broadcast={', '.join(diagnostics.workspace_broadcast_channels)}, "
                f"suppressed={', '.join(diagnostics.workspace_suppressed_channels) or 'none'}, "
                f"intensity={diagnostics.workspace_broadcast_intensity:.3f}"
            )
        if diagnostics.current_commitments:
            print(
                "  identity    "
                f"focus={', '.join(diagnostics.commitment_focus) or 'none'}, "
                f"violations={', '.join(diagnostics.violated_commitments) or 'none'}, "
                f"tension={diagnostics.identity_tension:.3f}, "
                f"repair={diagnostics.identity_repair_policy or 'none'}"
            )
        if diagnostics.social_focus or diagnostics.social_alerts:
            print(
                "  social      "
                f"focus={', '.join(diagnostics.social_focus) or 'none'}, "
                f"alerts={', '.join(diagnostics.social_alerts) or 'none'}"
            )
        print(
            "  maintain    "
            f"tasks={', '.join(maintenance_agenda.active_tasks) or 'none'}, "
            f"recommended={maintenance_agenda.recommended_action}, "
            f"interrupt={maintenance_agenda.interrupt_action or 'none'}, "
            f"sleep={maintenance_agenda.sleep_recommended}, "
            f"effects={maintenance_effects}"
        )
        if diagnostics.ledger_summary:
            print(f"  ledger      {diagnostics.ledger_summary}")
        print(f"  explain     {diagnostics.explanation}")

        if sleep_summary:
            print(
                "  sleep       "
                f"avg_fe_drop={sleep_summary.average_free_energy_drop:.3f}, "
                f"preferred_action={sleep_summary.preferred_action}, "
                f"dreams={sleep_summary.dream_replay_count}, "
                f"consolidations={sleep_summary.memory_consolidations}, "
                f"sampled={sleep_summary.episodes_sampled}, "
                f"clusters={sleep_summary.clusters_created}, "
                f"patterns={sleep_summary.patterns_found}, "
                f"wm_updates={sleep_summary.world_model_updates}, "
                f"policy_updates={sleep_summary.policy_bias_updates}, "
                f"epistemic_updates={sleep_summary.epistemic_bonus_updates}, "
                f"archived={sleep_summary.episodes_archived}, "
                f"deleted={sleep_summary.episodes_deleted}, "
                f"pe={sleep_summary.prediction_error_before:.3f}->{sleep_summary.prediction_error_after:.3f}, "
                + (
                    f"cond_pe={sleep_summary.consolidation_metrics.conditioned_pe_before:.3f}"
                    f"->{sleep_summary.consolidation_metrics.conditioned_pe_after:.3f}, "
                    if sleep_summary.consolidation_metrics
                    else ""
                )
                + f"beliefs={format_state(sleep_summary.stable_beliefs)}"
            )

        if host_tick:
            policy = host_tick["policy"]
            tick_input = host_tick["tick_input"]
            host_state = host_tick["state_after"]
            assert isinstance(policy, PolicyTendency)
            assert isinstance(tick_input, TickInput)
            assert isinstance(host_state, AgentState)
            notes = ", ".join(tick_input.notes) if tick_input.notes else "stable"
            print(
                "  host        "
                f"strategy={policy.chosen_strategy.value}, "
                f"energy={host_state.internal_energy:.2f}, "
                f"error={host_state.prediction_error:.2f}, "
                f"surprise={host_state.surprise_load:.2f}, "
                f"input={notes}"
            )
            print(f"  speech      {host_tick['inner_speech']}")

    def export_snapshot(self) -> dict[str, object]:
        payload = self._snapshot_payload()
        payload["semantic_memory"] = [asdict(item) for item in self.agent.semantic_memory]
        payload["sleep_history"] = [asdict(item) for item in self.agent.sleep_history]
        return payload

