from __future__ import annotations

import asyncio
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path

from .agent import SegmentAgent
from .environment import SimulatedWorld
from .evaluation import RunMetrics
from .fep import advance_state, infer_policy
from .interoception import ProcessInteroceptor
from .llm import InnerSpeechEngine, build_inner_speech_engine
from .logging_utils import ConsciousnessLogger
from .persistence import SnapshotLoadError, atomic_write_json, load_snapshot, quarantine_snapshot
from .predictive_coding import PredictiveCodingHyperparameters
from .self_model import ClassificationResult, SelfModel, build_default_self_model
from .state import AgentState, PolicyTendency, TickInput
from .tracing import JsonlTraceWriter, derive_trace_path
from .types import InterventionScore, SleepSummary


STATE_VERSION = "0.3"
SUPPORTED_STATE_VERSIONS = {STATE_VERSION, "0.2", "0.1"}


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
        f"{option.choice} fe={option.expected_free_energy:.3f} cost={option.cost:.2f}"
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
        state_path: str | Path | None = None,
        trace_path: str | Path | None = None,
        state_load_status: str = "fresh",
    ) -> None:
        self.world = world or SimulatedWorld()
        self.agent = agent or SegmentAgent(rng=self.world.rng)
        self.agent.rng = self.world.rng
        self.metrics = metrics or RunMetrics()
        self.host_state = host_state or AgentState()
        self.interoceptor = interoceptor or ProcessInteroceptor()
        self.inner_speech_engine = inner_speech_engine or build_inner_speech_engine()
        self.consciousness_logger = consciousness_logger or ConsciousnessLogger()
        self.self_model = self_model or build_default_self_model()
        self.state_path = Path(state_path) if state_path else None
        resolved_trace_path = Path(trace_path) if trace_path else derive_trace_path(self.state_path)
        self.trace_writer = (
            JsonlTraceWriter(resolved_trace_path) if resolved_trace_path else None
        )
        self.state_load_status = state_load_status

    @classmethod
    def load_or_create(
        cls,
        state_path: str | Path | None = None,
        trace_path: str | Path | None = None,
        seed: int = 17,
        reset: bool = False,
        predictive_hyperparameters: PredictiveCodingHyperparameters | None = None,
        reset_predictive_precisions: bool = False,
    ) -> SegmentRuntime:
        path = Path(state_path) if state_path else None
        resolved_trace_path = (
            Path(trace_path) if trace_path else derive_trace_path(path)
        )
        if reset and resolved_trace_path:
            JsonlTraceWriter(resolved_trace_path).reset()
        if not path or reset or not path.exists():
            world = SimulatedWorld(seed=seed)
            return cls(
                agent=SegmentAgent(
                    rng=world.rng,
                    predictive_hyperparameters=predictive_hyperparameters,
                ),
                world=world,
                state_path=path,
                trace_path=resolved_trace_path,
                state_load_status="fresh" if not reset else "reset",
            )

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
            return cls(
                agent=SegmentAgent(rng=world.rng),
                world=world,
                state_path=path,
                trace_path=resolved_trace_path,
                state_load_status=f"recovered_from_{exc.reason}",
            )

        world = SimulatedWorld.from_dict(payload.get("world"))
        agent = SegmentAgent.from_dict(
            payload.get("agent"),
            rng=world.rng,
            predictive_hyperparameters=predictive_hyperparameters,
            reset_predictive_precisions=reset_predictive_precisions,
        )
        metrics = RunMetrics.from_dict(payload.get("metrics"))
        host_state = AgentState.from_dict(payload.get("host_state"))
        return cls(
            agent=agent,
            world=world,
            metrics=metrics,
            host_state=host_state,
            state_path=path,
            trace_path=resolved_trace_path,
            state_load_status="restored",
        )

    def save_snapshot(self) -> None:
        if not self.state_path:
            return

        atomic_write_json(self.state_path, self._snapshot_payload())

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
            print(f"Semantic summaries stored: {len(self.agent.semantic_memory)}")
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
        self.agent.cycle += 1
        observation = self.world.observe()
        observed, prediction, errors, free_energy_before, hierarchy = self.agent.perceive(
            observation
        )

        similar = self.agent.long_term_memory.retrieve_similar(
            observed,
            self._body_state(),
            k=1,
        )
        memory_hits = len(similar)

        diagnostics = self.agent.choose_intervention(prediction, errors)
        choice = diagnostics.chosen.choice
        expected_fe = diagnostics.chosen.expected_free_energy
        choice_cost = diagnostics.chosen.cost

        if choice == "internal_update":
            self.agent.apply_internal_update(errors)
        else:
            direct_feedback = self.world.apply_action(choice)
            self.agent.apply_action_feedback(direct_feedback)

        validation_observation = self.world.observe()
        _, _, _, free_energy_after, _ = self.agent.perceive(validation_observation)
        self.agent.integrate_outcome(
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

        alive = self.agent.energy > 0.01
        self.metrics.record_cycle(
            choice=choice,
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
                observed=observed,
                prediction=prediction,
                errors=errors,
                hierarchy=hierarchy,
                diagnostics=diagnostics.ranked_options,
                choice=choice,
                expected_fe=expected_fe,
                choice_cost=choice_cost,
                free_energy_before=free_energy_before,
                free_energy_after=free_energy_after,
                memory_hits=memory_hits,
                sleep_summary=sleep_summary,
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
                diagnostics=diagnostics.ranked_options,
                choice=choice,
                expected_fe=expected_fe,
                choice_cost=choice_cost,
                free_energy_before=free_energy_before,
                free_energy_after=free_energy_after,
                memory_hits=memory_hits,
                sleep_summary=sleep_summary,
                host_tick=host_tick,
            )

        return {
            "cycle": self.agent.cycle,
            "choice": choice,
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
        tick_input = self.interoceptor.sample().to_tick_input()
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
        result = self.self_model.inspect_event(exc)
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
                            "surprise_source": result.surprise_source,
                            "detected_threats": list(result.detected_threats),
                            "resource_state": dict(result.resource_state),
                        },
                    }
                )
            except Exception:
                pass
        return result

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
        }

    def _build_cycle_trace(
        self,
        observed: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        hierarchy,
        diagnostics: list[InterventionScore],
        choice: str,
        expected_fe: float,
        choice_cost: float,
        free_energy_before: float,
        free_energy_after: float,
        memory_hits: int,
        sleep_summary: SleepSummary | None,
        alive: bool,
        host_tick: dict[str, object] | None,
    ) -> dict[str, object]:
        trace_record: dict[str, object] = {
            "event": "cycle",
            "state_version": STATE_VERSION,
            "state_load_status": self.state_load_status,
            "cycle": self.agent.cycle,
            "alive": alive,
            "choice": choice,
            "expected_free_energy": expected_fe,
            "choice_cost": choice_cost,
            "free_energy_before": free_energy_before,
            "free_energy_after": free_energy_after,
            "memory_hits": memory_hits,
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
            "hierarchy": asdict(hierarchy),
            "decision_ranking": [
                {
                    "choice": option.choice,
                    "expected_free_energy": option.expected_free_energy,
                    "cost": option.cost,
                }
                for option in diagnostics
            ],
            "running_metrics": self.metrics.summary(),
            "recent_action_history": list(self.agent.action_history),
        }
        if sleep_summary:
            trace_record["sleep_summary"] = asdict(sleep_summary)
        if host_tick:
            policy = host_tick["policy"]
            tick_input = host_tick["tick_input"]
            state_after = host_tick["state_after"]
            assert isinstance(policy, PolicyTendency)
            assert isinstance(tick_input, TickInput)
            assert isinstance(state_after, AgentState)
            trace_record["host_tick"] = {
                "strategy": policy.chosen_strategy.value,
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
        diagnostics: list[InterventionScore],
        choice: str,
        expected_fe: float,
        choice_cost: float,
        free_energy_before: float,
        free_energy_after: float,
        memory_hits: int,
        sleep_summary: SleepSummary | None,
        host_tick: dict[str, object] | None,
    ) -> None:
        print(
            f"[cycle {self.agent.cycle:02d}] choice={choice:>15}  "
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
        print(f"  scoring     {format_action_scores(diagnostics)}")

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

        if sleep_summary:
            print(
                "  sleep       "
                f"avg_fe_drop={sleep_summary.average_free_energy_drop:.3f}, "
                f"preferred_action={sleep_summary.preferred_action}, "
                f"dreams={sleep_summary.dream_replay_count}, "
                f"consolidations={sleep_summary.memory_consolidations}, "
                f"beliefs={format_state(sleep_summary.stable_beliefs)}"
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
        return payload
