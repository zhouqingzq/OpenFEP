from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class HomeostasisState:
    acute_energy_debt: float = 0.0
    acute_stress_debt: float = 0.0
    acute_fatigue_debt: float = 0.0
    acute_thermal_debt: float = 0.0
    short_term_sleep_pressure: float = 0.0
    chronic_energy_debt: float = 0.0
    chronic_stress_debt: float = 0.0
    chronic_fatigue_debt: float = 0.0
    chronic_runtime_debt: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "acute_energy_debt": self.acute_energy_debt,
            "acute_stress_debt": self.acute_stress_debt,
            "acute_fatigue_debt": self.acute_fatigue_debt,
            "acute_thermal_debt": self.acute_thermal_debt,
            "short_term_sleep_pressure": self.short_term_sleep_pressure,
            "chronic_energy_debt": self.chronic_energy_debt,
            "chronic_stress_debt": self.chronic_stress_debt,
            "chronic_fatigue_debt": self.chronic_fatigue_debt,
            "chronic_runtime_debt": self.chronic_runtime_debt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "HomeostasisState":
        if not payload:
            return cls()
        return cls(
            acute_energy_debt=float(payload.get("acute_energy_debt", 0.0)),
            acute_stress_debt=float(payload.get("acute_stress_debt", 0.0)),
            acute_fatigue_debt=float(payload.get("acute_fatigue_debt", 0.0)),
            acute_thermal_debt=float(payload.get("acute_thermal_debt", 0.0)),
            short_term_sleep_pressure=float(payload.get("short_term_sleep_pressure", 0.0)),
            chronic_energy_debt=float(payload.get("chronic_energy_debt", 0.0)),
            chronic_stress_debt=float(payload.get("chronic_stress_debt", 0.0)),
            chronic_fatigue_debt=float(payload.get("chronic_fatigue_debt", 0.0)),
            chronic_runtime_debt=float(payload.get("chronic_runtime_debt", 0.0)),
        )


@dataclass(frozen=True)
class MaintenanceAgenda:
    cycle: int
    active_tasks: tuple[str, ...]
    recommended_action: str
    interrupt_action: str | None
    interrupt_reason: str = ""
    sleep_recommended: bool = False
    memory_compaction_recommended: bool = False
    telemetry_backoff_recommended: bool = False
    protected_mode: bool = False
    protected_mode_ticks_remaining: int = 0
    recovery_rebound_active: bool = False
    recovery_rebound_ticks_remaining: int = 0
    policy_shift_strength: float = 0.0
    suppressed_actions: tuple[str, ...] = ()
    recovery_focus: tuple[str, ...] = ()
    chronic_debt_pressure: float = 0.0
    state: HomeostasisState = field(default_factory=HomeostasisState)

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "active_tasks": list(self.active_tasks),
            "recommended_action": self.recommended_action,
            "interrupt_action": self.interrupt_action,
            "interrupt_reason": self.interrupt_reason,
            "sleep_recommended": self.sleep_recommended,
            "memory_compaction_recommended": self.memory_compaction_recommended,
            "telemetry_backoff_recommended": self.telemetry_backoff_recommended,
            "protected_mode": self.protected_mode,
            "protected_mode_ticks_remaining": self.protected_mode_ticks_remaining,
            "recovery_rebound_active": self.recovery_rebound_active,
            "recovery_rebound_ticks_remaining": self.recovery_rebound_ticks_remaining,
            "policy_shift_strength": self.policy_shift_strength,
            "suppressed_actions": list(self.suppressed_actions),
            "recovery_focus": list(self.recovery_focus),
            "chronic_debt_pressure": self.chronic_debt_pressure,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "MaintenanceAgenda | None":
        if not payload:
            return None
        return cls(
            cycle=int(payload.get("cycle", 0)),
            active_tasks=tuple(str(item) for item in payload.get("active_tasks", [])),
            recommended_action=str(payload.get("recommended_action", "rest")),
            interrupt_action=(
                str(payload.get("interrupt_action"))
                if payload.get("interrupt_action") is not None
                else None
            ),
            interrupt_reason=str(payload.get("interrupt_reason", "")),
            sleep_recommended=bool(payload.get("sleep_recommended", False)),
            memory_compaction_recommended=bool(
                payload.get("memory_compaction_recommended", False)
            ),
            telemetry_backoff_recommended=bool(
                payload.get("telemetry_backoff_recommended", False)
            ),
            protected_mode=bool(payload.get("protected_mode", False)),
            protected_mode_ticks_remaining=int(payload.get("protected_mode_ticks_remaining", 0)),
            recovery_rebound_active=bool(payload.get("recovery_rebound_active", False)),
            recovery_rebound_ticks_remaining=int(
                payload.get("recovery_rebound_ticks_remaining", 0)
            ),
            policy_shift_strength=float(payload.get("policy_shift_strength", 0.0)),
            suppressed_actions=tuple(
                str(item) for item in payload.get("suppressed_actions", [])
            ),
            recovery_focus=tuple(str(item) for item in payload.get("recovery_focus", [])),
            chronic_debt_pressure=float(payload.get("chronic_debt_pressure", 0.0)),
            state=HomeostasisState.from_dict(
                payload.get("state") if isinstance(payload.get("state"), Mapping) else None
            ),
        )


@dataclass
class HomeostasisScheduler:
    energy_floor: float = 0.30
    stress_ceiling: float = 0.72
    fatigue_ceiling: float = 0.70
    thermal_tolerance: float = 0.12
    chronic_gain: float = 0.08
    chronic_decay: float = 0.04
    maintenance_actions_triggered: int = 0
    interrupts_triggered: int = 0
    memory_compactions_triggered: int = 0
    telemetry_backoffs_triggered: int = 0
    protected_mode_activations: int = 0
    recovery_rebound_activations: int = 0
    protected_mode_ticks_remaining: int = 0
    recovery_rebound_ticks_remaining: int = 0
    last_guard_tick: int = 0
    last_rebound_tick: int = 0
    debt_relief_energy: float = 0.0
    debt_relief_stress: float = 0.0
    debt_relief_fatigue: float = 0.0
    debt_relief_runtime: float = 0.0
    last_agenda: MaintenanceAgenda | None = None
    history: list[dict[str, object]] = field(default_factory=list)

    def assess(
        self,
        *,
        cycle: int,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        telemetry_error_count: int,
        persistence_error_count: int,
        should_sleep: bool,
    ) -> MaintenanceAgenda:
        previous = self.last_agenda.state if self.last_agenda is not None else HomeostasisState()
        acute_energy_debt = max(0.0, self.energy_floor - energy)
        acute_stress_debt = max(0.0, stress - self.stress_ceiling)
        acute_fatigue_debt = max(0.0, fatigue - self.fatigue_ceiling)
        acute_thermal_debt = max(0.0, abs(temperature - 0.5) - self.thermal_tolerance)
        sleep_pressure = max(
            previous.short_term_sleep_pressure * 0.85,
            acute_fatigue_debt * 0.8 + acute_stress_debt * 0.4,
        )
        if should_sleep:
            sleep_pressure = max(sleep_pressure, 0.55)

        approach_energy = max(0.0, (self.energy_floor + 0.20) - energy)
        approach_stress = max(0.0, stress - (self.stress_ceiling - 0.24))
        approach_fatigue = max(0.0, fatigue - (self.fatigue_ceiling - 0.24))
        threat_pressure = max(
            acute_energy_debt * 1.35,
            acute_stress_debt * 1.20,
            acute_fatigue_debt * 1.10,
            acute_thermal_debt * 1.05,
            approach_energy * 0.95,
            approach_stress * 0.90,
            approach_fatigue * 0.90,
        )
        chronic_pressure = (
            previous.chronic_energy_debt
            + previous.chronic_stress_debt
            + previous.chronic_fatigue_debt
        )
        if threat_pressure > 0.08 or chronic_pressure > 0.18:
            self.protected_mode_ticks_remaining = max(
                self.protected_mode_ticks_remaining,
                5 + int(threat_pressure > 0.16) + int(chronic_pressure > 0.28),
            )
            self.last_guard_tick = cycle
            self.protected_mode_activations += 1
        elif self.protected_mode_ticks_remaining > 0:
            self.protected_mode_ticks_remaining -= 1

        if (
            self.protected_mode_ticks_remaining > 0
            and threat_pressure < 0.06
            and chronic_pressure > 0.10
        ):
            self.recovery_rebound_ticks_remaining = max(self.recovery_rebound_ticks_remaining, 5)
            self.last_rebound_tick = cycle
            self.recovery_rebound_activations += 1
        elif self.recovery_rebound_ticks_remaining > 0:
            self.recovery_rebound_ticks_remaining -= 1

        relief_multiplier = 1.0 + (0.85 if self.recovery_rebound_ticks_remaining > 0 else 0.0)
        guard_relief = 0.012 if self.protected_mode_ticks_remaining > 0 else 0.0

        chronic_energy = max(
            0.0,
            previous.chronic_energy_debt * (1.0 - self.chronic_decay)
            + acute_energy_debt * self.chronic_gain
            - min(previous.chronic_energy_debt, self.debt_relief_energy * relief_multiplier + guard_relief),
        )
        chronic_stress = max(
            0.0,
            previous.chronic_stress_debt * (1.0 - self.chronic_decay)
            + acute_stress_debt * self.chronic_gain
            - min(previous.chronic_stress_debt, self.debt_relief_stress * relief_multiplier + guard_relief),
        )
        chronic_fatigue = max(
            0.0,
            previous.chronic_fatigue_debt * (1.0 - self.chronic_decay)
            + acute_fatigue_debt * self.chronic_gain
            - min(previous.chronic_fatigue_debt, self.debt_relief_fatigue * relief_multiplier + guard_relief),
        )
        runtime_debt_signal = min(1.0, (telemetry_error_count + persistence_error_count) / 8.0)
        chronic_runtime = max(
            0.0,
            previous.chronic_runtime_debt * (1.0 - self.chronic_decay)
            + runtime_debt_signal * self.chronic_gain
            - min(previous.chronic_runtime_debt, self.debt_relief_runtime * relief_multiplier),
        )
        self.debt_relief_energy *= 0.55
        self.debt_relief_stress *= 0.55
        self.debt_relief_fatigue *= 0.55
        self.debt_relief_runtime *= 0.55

        state = HomeostasisState(
            acute_energy_debt=round(acute_energy_debt, 6),
            acute_stress_debt=round(acute_stress_debt, 6),
            acute_fatigue_debt=round(acute_fatigue_debt, 6),
            acute_thermal_debt=round(acute_thermal_debt, 6),
            short_term_sleep_pressure=round(sleep_pressure, 6),
            chronic_energy_debt=round(chronic_energy, 6),
            chronic_stress_debt=round(chronic_stress, 6),
            chronic_fatigue_debt=round(chronic_fatigue, 6),
            chronic_runtime_debt=round(chronic_runtime, 6),
        )

        tasks: list[str] = []
        recommended_action = "rest"
        interrupt_action: str | None = None
        interrupt_reason = ""
        protected_mode = self.protected_mode_ticks_remaining > 0
        recovery_rebound_active = self.recovery_rebound_ticks_remaining > 0
        suppressed_actions: list[str] = []
        recovery_focus: list[str] = []
        policy_shift_strength = min(
            1.0,
            threat_pressure * 2.8
            + chronic_pressure * 1.7
            + (0.20 if protected_mode else 0.0)
            + (0.16 if recovery_rebound_active else 0.0),
        )

        if state.acute_thermal_debt > 0.02:
            tasks.append("thermal_rebalance")
            recommended_action = "thermoregulate"
        if state.acute_energy_debt > 0.0:
            tasks.append("energy_recovery")
            recommended_action = "rest"
        if state.acute_stress_debt > 0.0:
            tasks.append("stress_relief")
            recommended_action = "hide"
        if state.acute_fatigue_debt > 0.0:
            tasks.append("fatigue_recovery")
            recommended_action = "rest"

        if state.chronic_runtime_debt > 0.08:
            tasks.append("runtime_stabilization")
        if state.short_term_sleep_pressure > 0.45:
            tasks.append("sleep_pressure")
        if state.chronic_energy_debt > 0.06 or state.chronic_fatigue_debt > 0.06:
            tasks.append("maintenance_budget_protection")
        if protected_mode:
            tasks.append("resource_guard")
            suppressed_actions.extend(["forage", "scan", "seek_contact"])
            if state.acute_stress_debt > state.acute_energy_debt:
                recommended_action = "hide"
            elif state.acute_thermal_debt > 0.04:
                recommended_action = "thermoregulate"
            elif state.acute_energy_debt > 0.02:
                recommended_action = "rest"
            else:
                recommended_action = "exploit_shelter"
        if recovery_rebound_active:
            tasks.append("recovery_rebound")
            recovery_focus.extend(["sleep_consolidation", "maintenance_agenda", "memory_compaction"])
            if recommended_action not in {"rest", "hide", "exploit_shelter", "thermoregulate"}:
                recommended_action = "rest"

        if state.acute_energy_debt > 0.10 or state.acute_fatigue_debt > 0.16 or approach_energy > 0.12:
            interrupt_action = "rest"
            interrupt_reason = "critical energy/fatigue debt"
        elif state.acute_thermal_debt > 0.08:
            interrupt_action = "thermoregulate"
            interrupt_reason = "critical thermal debt"
        elif state.acute_stress_debt > 0.14 or approach_stress > 0.10:
            interrupt_action = "hide"
            interrupt_reason = "critical stress debt"
        elif protected_mode and recommended_action in {"rest", "hide", "exploit_shelter", "thermoregulate"}:
            interrupt_action = recommended_action
            interrupt_reason = "protected mode resource guard"

        agenda = MaintenanceAgenda(
            cycle=cycle,
            active_tasks=tuple(tasks),
            recommended_action=recommended_action,
            interrupt_action=interrupt_action,
            interrupt_reason=interrupt_reason,
            sleep_recommended=state.short_term_sleep_pressure > 0.45,
            memory_compaction_recommended=state.chronic_runtime_debt > 0.10,
            telemetry_backoff_recommended=state.chronic_runtime_debt > 0.12,
            protected_mode=protected_mode,
            protected_mode_ticks_remaining=self.protected_mode_ticks_remaining,
            recovery_rebound_active=recovery_rebound_active,
            recovery_rebound_ticks_remaining=self.recovery_rebound_ticks_remaining,
            policy_shift_strength=round(policy_shift_strength, 6),
            suppressed_actions=tuple(dict.fromkeys(suppressed_actions)),
            recovery_focus=tuple(dict.fromkeys(recovery_focus)),
            chronic_debt_pressure=round(
                state.chronic_energy_debt
                + state.chronic_stress_debt
                + state.chronic_fatigue_debt
                + state.chronic_runtime_debt,
                6,
            ),
            state=state,
        )
        self.last_agenda = agenda
        self.history.append(agenda.to_dict())
        self.history = self.history[-64:]
        return agenda

    def note_interrupt(self, agenda: MaintenanceAgenda, previous_choice: str, final_choice: str) -> None:
        if agenda.interrupt_action and previous_choice != final_choice:
            self.interrupts_triggered += 1
            self.maintenance_actions_triggered += 1

    def apply_background_maintenance(self, agent) -> dict[str, object]:
        agenda = self.last_agenda
        if agenda is None:
            return {
                "memory_compaction_applied": False,
                "telemetry_backoff_applied": False,
                "stress_recovery_applied": False,
                "guard_activated_tick": None,
                "protected_mode_duration": 0,
                "suppressed_actions": [],
                "recovery_rebound_activated": False,
                "recovery_result": "none",
            }
        memory_compaction_applied = False
        telemetry_backoff_applied = False
        stress_recovery_applied = False
        energy_recovery_applied = False
        debt_reversal_applied = False

        if agenda.memory_compaction_recommended and len(agent.long_term_memory.episodes) > 4:
            removed = agent.long_term_memory.compress_episodes()
            memory_compaction_applied = removed > 0
            if memory_compaction_applied:
                self.memory_compactions_triggered += 1
                self.debt_relief_runtime += 0.035
        if agenda.telemetry_backoff_recommended:
            self.telemetry_backoffs_triggered += 1
            telemetry_backoff_applied = True
            self.debt_relief_runtime += 0.025
        if "stress_relief" in agenda.active_tasks or "fatigue_recovery" in agenda.active_tasks:
            stress_delta = 0.02 + (0.035 if agenda.protected_mode else 0.0) + (
                0.03 if agenda.recovery_rebound_active else 0.0
            )
            fatigue_delta = 0.015 + (0.025 if agenda.protected_mode else 0.0) + (
                0.035 if agenda.recovery_rebound_active else 0.0
            )
            agent.stress = max(0.0, min(1.0, agent.stress - stress_delta))
            agent.fatigue = max(0.0, min(1.0, agent.fatigue - fatigue_delta))
            stress_recovery_applied = True
            self.debt_relief_stress += stress_delta * 0.55
            self.debt_relief_fatigue += fatigue_delta * 0.55
            debt_reversal_applied = True
        if agenda.recommended_action in {"rest", "hide", "exploit_shelter", "thermoregulate"}:
            energy_delta = 0.0
            if agenda.recommended_action == "rest":
                energy_delta += 0.055
            elif agenda.recommended_action == "hide":
                energy_delta += 0.026
            elif agenda.recommended_action == "exploit_shelter":
                energy_delta += 0.022
            if agenda.protected_mode:
                energy_delta += 0.030
            if agenda.recovery_rebound_active:
                energy_delta += 0.040
            if energy_delta > 0.0:
                agent.energy = max(0.0, min(1.0, agent.energy + energy_delta))
                self.debt_relief_energy += energy_delta * 0.60
                energy_recovery_applied = True
                debt_reversal_applied = True
        if agenda.recovery_rebound_active:
            agent.temperature = max(0.0, min(1.0, agent.temperature + ((0.5 - agent.temperature) * 0.22)))
            self.debt_relief_stress += 0.018
            self.debt_relief_fatigue += 0.018
            self.debt_relief_energy += 0.016

        return {
            "memory_compaction_applied": memory_compaction_applied,
            "telemetry_backoff_applied": telemetry_backoff_applied,
            "stress_recovery_applied": stress_recovery_applied,
            "energy_recovery_applied": energy_recovery_applied,
            "debt_reversal_applied": debt_reversal_applied,
            "guard_activated_tick": self.last_guard_tick if agenda.protected_mode else None,
            "protected_mode_duration": agenda.protected_mode_ticks_remaining,
            "suppressed_actions": list(agenda.suppressed_actions),
            "recovery_rebound_activated": agenda.recovery_rebound_active,
            "recovery_result": (
                "success"
                if debt_reversal_applied or energy_recovery_applied
                else "partial"
                if stress_recovery_applied or memory_compaction_applied
                else "fail"
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "energy_floor": self.energy_floor,
            "stress_ceiling": self.stress_ceiling,
            "fatigue_ceiling": self.fatigue_ceiling,
            "thermal_tolerance": self.thermal_tolerance,
            "chronic_gain": self.chronic_gain,
            "chronic_decay": self.chronic_decay,
            "maintenance_actions_triggered": self.maintenance_actions_triggered,
            "interrupts_triggered": self.interrupts_triggered,
            "memory_compactions_triggered": self.memory_compactions_triggered,
            "telemetry_backoffs_triggered": self.telemetry_backoffs_triggered,
            "protected_mode_activations": self.protected_mode_activations,
            "recovery_rebound_activations": self.recovery_rebound_activations,
            "protected_mode_ticks_remaining": self.protected_mode_ticks_remaining,
            "recovery_rebound_ticks_remaining": self.recovery_rebound_ticks_remaining,
            "last_guard_tick": self.last_guard_tick,
            "last_rebound_tick": self.last_rebound_tick,
            "debt_relief_energy": self.debt_relief_energy,
            "debt_relief_stress": self.debt_relief_stress,
            "debt_relief_fatigue": self.debt_relief_fatigue,
            "debt_relief_runtime": self.debt_relief_runtime,
            "last_agenda": self.last_agenda.to_dict() if self.last_agenda is not None else None,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "HomeostasisScheduler":
        if not payload:
            return cls()
        scheduler = cls(
            energy_floor=float(payload.get("energy_floor", 0.30)),
            stress_ceiling=float(payload.get("stress_ceiling", 0.72)),
            fatigue_ceiling=float(payload.get("fatigue_ceiling", 0.70)),
            thermal_tolerance=float(payload.get("thermal_tolerance", 0.12)),
            chronic_gain=float(payload.get("chronic_gain", 0.08)),
            chronic_decay=float(payload.get("chronic_decay", 0.04)),
            maintenance_actions_triggered=int(payload.get("maintenance_actions_triggered", 0)),
            interrupts_triggered=int(payload.get("interrupts_triggered", 0)),
            memory_compactions_triggered=int(payload.get("memory_compactions_triggered", 0)),
            telemetry_backoffs_triggered=int(payload.get("telemetry_backoffs_triggered", 0)),
            protected_mode_activations=int(payload.get("protected_mode_activations", 0)),
            recovery_rebound_activations=int(payload.get("recovery_rebound_activations", 0)),
            protected_mode_ticks_remaining=int(payload.get("protected_mode_ticks_remaining", 0)),
            recovery_rebound_ticks_remaining=int(
                payload.get("recovery_rebound_ticks_remaining", 0)
            ),
            last_guard_tick=int(payload.get("last_guard_tick", 0)),
            last_rebound_tick=int(payload.get("last_rebound_tick", 0)),
            debt_relief_energy=float(payload.get("debt_relief_energy", 0.0)),
            debt_relief_stress=float(payload.get("debt_relief_stress", 0.0)),
            debt_relief_fatigue=float(payload.get("debt_relief_fatigue", 0.0)),
            debt_relief_runtime=float(payload.get("debt_relief_runtime", 0.0)),
        )
        scheduler.last_agenda = MaintenanceAgenda.from_dict(
            payload.get("last_agenda") if isinstance(payload.get("last_agenda"), Mapping) else None
        )
        history = payload.get("history", [])
        if isinstance(history, list):
            scheduler.history = [dict(item) for item in history if isinstance(item, Mapping)]
        return scheduler
