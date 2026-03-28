from __future__ import annotations

from dataclasses import dataclass, field

from .environment import clamp
from .predictive_coding import LayerBeliefUpdate, StrategicBeliefState
from .types import Drive


@dataclass
class ProcessValenceState:
    active_focus_id: str = ""
    recent_closed_focus_id: str = ""
    unresolved_tension: float = 0.0
    closure_satisfaction: float = 0.0
    post_closure_decay: float = 0.0
    boredom_pressure: float = 0.0
    process_reward: float = 0.0
    focus_persistence_ticks: int = 0
    closure_events: int = 0
    active_phase: str = "idle"

    def to_dict(self) -> dict[str, object]:
        return {
            "active_focus_id": self.active_focus_id,
            "recent_closed_focus_id": self.recent_closed_focus_id,
            "unresolved_tension": clamp(self.unresolved_tension),
            "closure_satisfaction": clamp(self.closure_satisfaction),
            "post_closure_decay": clamp(self.post_closure_decay),
            "boredom_pressure": clamp(self.boredom_pressure),
            "process_reward": clamp(self.process_reward),
            "focus_persistence_ticks": int(self.focus_persistence_ticks),
            "closure_events": int(self.closure_events),
            "active_phase": self.active_phase,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "ProcessValenceState":
        if not payload:
            return cls()
        return cls(
            active_focus_id=str(payload.get("active_focus_id", "")),
            recent_closed_focus_id=str(payload.get("recent_closed_focus_id", "")),
            unresolved_tension=float(payload.get("unresolved_tension", 0.0)),
            closure_satisfaction=float(payload.get("closure_satisfaction", 0.0)),
            post_closure_decay=float(payload.get("post_closure_decay", 0.0)),
            boredom_pressure=float(payload.get("boredom_pressure", 0.0)),
            process_reward=float(payload.get("process_reward", 0.0)),
            focus_persistence_ticks=int(payload.get("focus_persistence_ticks", 0)),
            closure_events=int(payload.get("closure_events", 0)),
            active_phase=str(payload.get("active_phase", "idle")),
        )


@dataclass
class DriveSystem:
    """Manages competing drives that create internal pressure."""

    drives: list[Drive] = field(
        default_factory=lambda: [
            Drive("hunger", 0.0, 1.3, "food"),
            Drive("safety", 0.0, 1.5, "danger"),
            Drive("exploration", 0.0, 0.8, "novelty"),
            Drive("comfort", 0.0, 1.0, "shelter"),
            Drive("thermal", 0.0, 0.9, "temperature"),
            Drive("social", 0.0, 0.7, "social"),
        ]
    )
    process_valence: ProcessValenceState = field(default_factory=ProcessValenceState)

    def update_urgencies(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        social_isolation: float,
        novelty_deficit: float,
        personality_modulation: dict[str, float] | None = None,
    ) -> None:
        """Update drive urgencies based on body state, deficits, and personality."""
        self.drives[0].urgency = clamp(1.0 - energy)
        self.drives[1].urgency = clamp(stress * 1.2)
        self.drives[2].urgency = clamp(novelty_deficit)
        self.drives[3].urgency = clamp(stress * 0.8 + (1.0 - energy) * 0.3)
        self.drives[4].urgency = clamp(abs(temperature - 0.5) * 2.0)
        self.drives[5].urgency = clamp(social_isolation)

        # M2.6: Apply personality-driven modulation to drive weights
        if personality_modulation:
            for drive in self.drives:
                delta = personality_modulation.get(drive.name, 0.0)
                if delta != 0.0:
                    drive.urgency = clamp(drive.urgency + delta)

    def compute_prior_modulation(self, base_prior: float, modality: str) -> float:
        """Modulate a prior based on competing drives."""
        total_pressure = 0.0
        for drive in self.drives:
            if drive.target_modality == modality:
                total_pressure += drive.urgency * drive.weight

        modulation = clamp(total_pressure * 0.15)
        if modality in ["danger", "temperature"]:
            return clamp(base_prior - modulation)
        return clamp(base_prior + modulation)

    def update_process_valence(
        self,
        *,
        current_focus_id: str = "",
        unresolved_targets: set[str] | None = None,
        focus_strength: float = 0.0,
        maintenance_pressure: float = 0.0,
        closure_signal: float = 0.0,
    ) -> ProcessValenceState:
        previous = self.process_valence
        unresolved_targets = {str(item) for item in (unresolved_targets or set()) if str(item)}
        current_focus_id = str(current_focus_id or "")
        focus_strength = clamp(focus_strength)
        maintenance_pressure = clamp(maintenance_pressure)
        closure_signal = clamp(closure_signal)

        if current_focus_id and current_focus_id in unresolved_targets:
            persistence = (
                previous.focus_persistence_ticks + 1
                if previous.active_focus_id == current_focus_id
                else 1
            )
            unresolved_tension = clamp(
                max(previous.unresolved_tension * 0.76, focus_strength * 0.92)
                + 0.04
                + min(0.18, max(0, len(unresolved_targets) - 1) * 0.06)
            )
            closure_satisfaction = clamp(previous.closure_satisfaction * 0.55)
            post_closure_decay = clamp(previous.post_closure_decay * 0.58)
            boredom_pressure = clamp(
                previous.boredom_pressure * 0.45
                - unresolved_tension * 0.18
                - focus_strength * 0.08
                - maintenance_pressure * 0.05
            )
            active_phase = (
                "reorientation"
                if previous.boredom_pressure >= 0.42 and previous.active_focus_id != current_focus_id
                else "wanting"
            )
            self.process_valence = ProcessValenceState(
                active_focus_id=current_focus_id,
                recent_closed_focus_id=(
                    ""
                    if previous.recent_closed_focus_id == current_focus_id
                    else previous.recent_closed_focus_id
                ),
                unresolved_tension=unresolved_tension,
                closure_satisfaction=closure_satisfaction,
                post_closure_decay=post_closure_decay,
                boredom_pressure=boredom_pressure,
                process_reward=clamp(unresolved_tension * 0.62 + focus_strength * 0.18),
                focus_persistence_ticks=persistence,
                closure_events=previous.closure_events,
                active_phase=active_phase,
            )
            return self.process_valence

        if previous.active_focus_id and previous.active_focus_id not in unresolved_targets:
            closure_satisfaction = clamp(
                0.42 + previous.unresolved_tension * 0.46 + closure_signal * 0.20
            )
            self.process_valence = ProcessValenceState(
                active_focus_id="",
                recent_closed_focus_id=previous.active_focus_id,
                unresolved_tension=0.0,
                closure_satisfaction=closure_satisfaction,
                post_closure_decay=1.0,
                boredom_pressure=clamp(previous.boredom_pressure * 0.30),
                process_reward=clamp(closure_satisfaction * 0.30),
                focus_persistence_ticks=0,
                closure_events=previous.closure_events + 1,
                active_phase="closure",
            )
            return self.process_valence

        closure_satisfaction = clamp(previous.closure_satisfaction * 0.65)
        post_closure_decay = clamp(max(0.0, previous.post_closure_decay * 0.78 - 0.10))
        boredom_pressure = clamp(
            previous.boredom_pressure * 0.82
            + 0.18
            - closure_satisfaction * 0.06
            - maintenance_pressure * 0.10
        )
        if closure_satisfaction > 0.18 or post_closure_decay > 0.18:
            phase = "satiation"
        elif boredom_pressure >= 0.40:
            phase = "boredom"
        else:
            phase = "idle"
        self.process_valence = ProcessValenceState(
            active_focus_id="",
            recent_closed_focus_id=previous.recent_closed_focus_id,
            unresolved_tension=0.0,
            closure_satisfaction=closure_satisfaction,
            post_closure_decay=post_closure_decay,
            boredom_pressure=boredom_pressure,
            process_reward=clamp(boredom_pressure * 0.24),
            focus_persistence_ticks=0,
            closure_events=previous.closure_events,
            active_phase=phase,
        )
        return self.process_valence

    def process_action_bias(self, action: str) -> float:
        state = self.process_valence
        bias = 0.0
        if state.active_phase in {"wanting", "reorientation"}:
            if action in {"scan", "seek_contact"}:
                bias += state.unresolved_tension * 0.24 + state.boredom_pressure * 0.10
            if action in {"rest", "hide"}:
                bias -= state.unresolved_tension * 0.08
        if state.active_phase in {"closure", "satiation"}:
            if action in {"scan", "seek_contact"}:
                bias -= state.closure_satisfaction * 0.18 + state.post_closure_decay * 0.08
            if action in {"rest", "exploit_shelter"}:
                bias += state.closure_satisfaction * 0.10
        if state.active_phase == "boredom":
            if action in {"scan", "seek_contact"}:
                bias += state.boredom_pressure * 0.22
            if action in {"rest", "hide"}:
                bias -= state.boredom_pressure * 0.06
        return max(-0.28, min(0.28, round(bias, 6)))

    def inquiry_focus_bonus(self, target_id: str, *, is_novel: bool = False) -> float:
        state = self.process_valence
        target_id = str(target_id or "")
        if not target_id:
            return round(max(0.0, min(0.12, state.boredom_pressure * 0.10)), 6)
        if target_id == state.active_focus_id:
            return round(
                min(
                    0.28,
                    0.08
                    + state.unresolved_tension * 0.20
                    + min(0.10, state.focus_persistence_ticks * 0.025),
                ),
                6,
            )
        if target_id == state.recent_closed_focus_id:
            return round(
                -min(
                    0.24,
                    state.closure_satisfaction * 0.18 + state.post_closure_decay * 0.16,
                ),
                6,
            )
        if is_novel and state.boredom_pressure >= 0.42:
            return round(min(0.16, state.boredom_pressure * 0.18), 6)
        return 0.0


@dataclass
class StrategicLayer:
    """Global priors that define what survival means right now."""

    energy_floor: float = 0.50
    danger_ceiling: float = 0.20
    novelty_floor: float = 0.30
    shelter_floor: float = 0.40
    temperature_ideal: float = 0.50
    social_floor: float = 0.25
    belief_state: StrategicBeliefState = field(default_factory=StrategicBeliefState)

    def priors(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        dopamine: float,
        drive_system: DriveSystem,
        personality_modulation: dict[str, float] | None = None,
    ) -> dict[str, float]:
        # M2.6: Apply personality modulation to strategic layer parameters
        energy_floor = self.energy_floor
        danger_ceiling = self.danger_ceiling
        novelty_floor = self.novelty_floor
        shelter_floor = self.shelter_floor
        temperature_ideal = self.temperature_ideal
        social_floor = self.social_floor
        if personality_modulation:
            energy_floor = clamp(energy_floor + personality_modulation.get("energy_floor", 0.0))
            danger_ceiling = clamp(danger_ceiling + personality_modulation.get("danger_ceiling", 0.0))
            novelty_floor = clamp(novelty_floor + personality_modulation.get("novelty_floor", 0.0))
            shelter_floor = clamp(shelter_floor + personality_modulation.get("shelter_floor", 0.0))
            temperature_ideal = clamp(temperature_ideal + personality_modulation.get("temperature_ideal", 0.0))
            social_floor = clamp(social_floor + personality_modulation.get("social_floor", 0.0))

        base = {
            "food": clamp(1.10 - energy),
            "danger": clamp(danger_ceiling + stress * 0.60),
            "novelty": clamp(novelty_floor + dopamine * 0.10),
            "shelter": clamp(shelter_floor + stress * 0.30),
            "temperature": clamp(
                temperature_ideal + abs(temperature - temperature_ideal) * 0.5
            ),
            "social": clamp(social_floor + (1.0 - dopamine) * 0.15),
        }

        return {
            key: drive_system.compute_prior_modulation(value, key)
            for key, value in base.items()
        }

    def dispatch_prediction(
        self,
        energy: float,
        stress: float,
        fatigue: float,
        temperature: float,
        dopamine: float,
        drive_system: DriveSystem,
        personality_modulation: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        base_priors = self.priors(
            energy,
            stress,
            fatigue,
            temperature,
            dopamine,
            drive_system,
            personality_modulation=personality_modulation,
        )
        prediction = self.belief_state.predict(base_priors)
        return base_priors, prediction

    def assimilate(
        self,
        lower_layer_signal: dict[str, float],
        base_priors: dict[str, float],
        predicted_state: dict[str, float] | None = None,
    ) -> LayerBeliefUpdate:
        return self.belief_state.posterior_update(
            lower_layer_signal,
            base_priors,
            predicted_state=predicted_state,
        )

    @property
    def beliefs(self) -> dict[str, float]:
        return self.belief_state.beliefs

    def absorb_error_signal(
        self,
        errors: dict[str, float],
        strength: float = 1.0,
    ) -> None:
        self.belief_state.absorb_error_signal(errors, strength=strength)

    def to_dict(self) -> dict[str, object]:
        return {
            "energy_floor": self.energy_floor,
            "danger_ceiling": self.danger_ceiling,
            "novelty_floor": self.novelty_floor,
            "shelter_floor": self.shelter_floor,
            "temperature_ideal": self.temperature_ideal,
            "social_floor": self.social_floor,
            "belief_state": self.belief_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> StrategicLayer:
        layer = cls()
        if not payload:
            return layer
        layer.energy_floor = float(payload.get("energy_floor", layer.energy_floor))
        layer.danger_ceiling = float(payload.get("danger_ceiling", layer.danger_ceiling))
        layer.novelty_floor = float(payload.get("novelty_floor", layer.novelty_floor))
        layer.shelter_floor = float(payload.get("shelter_floor", layer.shelter_floor))
        layer.temperature_ideal = float(
            payload.get("temperature_ideal", layer.temperature_ideal)
        )
        layer.social_floor = float(payload.get("social_floor", layer.social_floor))
        layer.belief_state = StrategicBeliefState.from_dict(payload.get("belief_state"))
        return layer
