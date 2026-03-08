from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class RunMetrics:
    cycles_completed: int = 0
    survival_ticks: int = 0
    sleep_count: int = 0
    internal_update_count: int = 0
    memory_retrieval_count: int = 0
    telemetry_error_count: int = 0
    persistence_error_count: int = 0
    free_energy_before_total: float = 0.0
    free_energy_after_total: float = 0.0
    free_energy_drop_total: float = 0.0
    min_energy: float = 1.0
    max_stress: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    action_switch_count: int = 0
    current_action_streak: int = 0
    max_action_streak: int = 0
    last_choice: str = ""
    termination_reason: str = ""

    def record_cycle(
        self,
        choice: str,
        free_energy_before: float,
        free_energy_after: float,
        energy: float,
        stress: float,
        memory_hits: int,
        slept: bool,
        alive: bool,
    ) -> None:
        self.cycles_completed += 1
        if alive:
            self.survival_ticks += 1
        if slept:
            self.sleep_count += 1
        if choice == "internal_update":
            self.internal_update_count += 1
        if memory_hits > 0:
            self.memory_retrieval_count += 1

        self.free_energy_before_total += free_energy_before
        self.free_energy_after_total += free_energy_after
        self.free_energy_drop_total += max(0.0, free_energy_before - free_energy_after)
        self.min_energy = min(self.min_energy, energy)
        self.max_stress = max(self.max_stress, stress)
        self.action_counts[choice] = self.action_counts.get(choice, 0) + 1

        if self.last_choice == choice:
            self.current_action_streak += 1
        else:
            if self.last_choice:
                self.action_switch_count += 1
            self.current_action_streak = 1
        self.max_action_streak = max(self.max_action_streak, self.current_action_streak)
        self.last_choice = choice

    def summary(self) -> dict[str, object]:
        completed = max(1, self.cycles_completed)
        action_entropy = 0.0
        action_distribution: dict[str, float] = {}
        dominant_action = ""
        dominant_action_share = 0.0
        for count in self.action_counts.values():
            probability = count / completed
            if probability > 0.0:
                action_entropy -= probability * math.log2(probability)
        for action, count in sorted(self.action_counts.items()):
            probability = count / completed
            action_distribution[action] = probability
            if probability > dominant_action_share:
                dominant_action = action
                dominant_action_share = probability
        return {
            "cycles_completed": self.cycles_completed,
            "survival_ticks": self.survival_ticks,
            "sleep_count": self.sleep_count,
            "internal_update_count": self.internal_update_count,
            "memory_retrieval_count": self.memory_retrieval_count,
            "telemetry_error_count": self.telemetry_error_count,
            "persistence_error_count": self.persistence_error_count,
            "memory_hit_rate": self.memory_retrieval_count / completed,
            "avg_free_energy_before": self.free_energy_before_total / completed,
            "avg_free_energy_after": self.free_energy_after_total / completed,
            "avg_free_energy_drop": self.free_energy_drop_total / completed,
            "min_energy": self.min_energy,
            "max_stress": self.max_stress,
            "action_counts": dict(sorted(self.action_counts.items())),
            "action_distribution": action_distribution,
            "action_switch_count": self.action_switch_count,
            "unique_actions": len(self.action_counts),
            "action_entropy": action_entropy,
            "dominant_action": dominant_action,
            "dominant_action_share": dominant_action_share,
            "max_action_streak": self.max_action_streak,
            "last_choice": self.last_choice,
            "termination_reason": self.termination_reason,
        }

    def to_dict(self) -> dict:
        return {
            "cycles_completed": self.cycles_completed,
            "survival_ticks": self.survival_ticks,
            "sleep_count": self.sleep_count,
            "internal_update_count": self.internal_update_count,
            "memory_retrieval_count": self.memory_retrieval_count,
            "telemetry_error_count": self.telemetry_error_count,
            "persistence_error_count": self.persistence_error_count,
            "free_energy_before_total": self.free_energy_before_total,
            "free_energy_after_total": self.free_energy_after_total,
            "free_energy_drop_total": self.free_energy_drop_total,
            "min_energy": self.min_energy,
            "max_stress": self.max_stress,
            "action_counts": dict(self.action_counts),
            "action_switch_count": self.action_switch_count,
            "current_action_streak": self.current_action_streak,
            "max_action_streak": self.max_action_streak,
            "last_choice": self.last_choice,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> RunMetrics:
        if not payload:
            return cls()

        return cls(
            cycles_completed=int(payload.get("cycles_completed", 0)),
            survival_ticks=int(payload.get("survival_ticks", 0)),
            sleep_count=int(payload.get("sleep_count", 0)),
            internal_update_count=int(payload.get("internal_update_count", 0)),
            memory_retrieval_count=int(payload.get("memory_retrieval_count", 0)),
            telemetry_error_count=int(payload.get("telemetry_error_count", 0)),
            persistence_error_count=int(payload.get("persistence_error_count", 0)),
            free_energy_before_total=float(
                payload.get("free_energy_before_total", 0.0)
            ),
            free_energy_after_total=float(payload.get("free_energy_after_total", 0.0)),
            free_energy_drop_total=float(payload.get("free_energy_drop_total", 0.0)),
            min_energy=float(payload.get("min_energy", 1.0)),
            max_stress=float(payload.get("max_stress", 0.0)),
            action_counts=dict(payload.get("action_counts", {})),
            action_switch_count=int(payload.get("action_switch_count", 0)),
            current_action_streak=int(payload.get("current_action_streak", 0)),
            max_action_streak=int(payload.get("max_action_streak", 0)),
            last_choice=str(payload.get("last_choice", "")),
            termination_reason=str(payload.get("termination_reason", "")),
        )
