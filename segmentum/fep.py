from __future__ import annotations

from dataclasses import replace

from .state import AgentState, PolicyTendency, Strategy, TickInput


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def infer_policy(state: AgentState, tick_input: TickInput) -> PolicyTendency:
    """
    Compute a minimal expected free energy vector.

    In predictive coding terms:
    - epistemic value rewards actions that resolve low-risk uncertainty
    - pragmatic value rewards actions that reduce survival-threatening load
    - the selected strategy is the one with the lowest expected free energy
    """

    combined_error = clamp01(
        (state.prediction_error * 0.55)
        + (tick_input.surprise_signal * 0.45)
        + (state.surprise_load * 0.20)
    )
    low_energy_pressure = clamp01(1.0 - state.internal_energy)

    # When the world is too predictable, curiosity should grow instead of letting
    # the agent collapse into a zero-error but information-poor attractor.
    epistemic_value = clamp01(
        (state.boredom * 0.55)
        + (tick_input.boredom_signal * 0.30)
        + (max(0.0, 0.35 - combined_error) * 0.45)
    )

    # Pragmatic value encodes the need to preserve the system when surprise or
    # metabolic strain becomes expensive to explain away.
    pragmatic_value = clamp01(
        (low_energy_pressure * 0.45)
        + (tick_input.resource_pressure * 0.35)
        + (combined_error * 0.35)
    )

    explore_efe = (
        (tick_input.resource_pressure * 0.40)
        + (low_energy_pressure * 0.35)
        + (combined_error * 0.20)
        - (epistemic_value * 0.85)
    )
    exploit_efe = (
        (tick_input.resource_pressure * 0.20)
        + (low_energy_pressure * 0.15)
        + (abs(combined_error - 0.28) * 0.50)
        - (state.internal_energy * 0.30)
    )
    escape_efe = (
        (0.20 - pragmatic_value * 1.05)
        - (combined_error * 0.45)
        + (epistemic_value * 0.10)
    )

    efe_by_strategy = {
        Strategy.EXPLORE: explore_efe,
        Strategy.EXPLOIT: exploit_efe,
        Strategy.ESCAPE: escape_efe,
    }
    chosen_strategy = min(efe_by_strategy, key=efe_by_strategy.get)

    return PolicyTendency(
        explore_efe=explore_efe,
        exploit_efe=exploit_efe,
        escape_efe=escape_efe,
        epistemic_value=epistemic_value,
        pragmatic_value=pragmatic_value,
        chosen_strategy=chosen_strategy,
    )


def advance_state(
    state: AgentState,
    tick_input: TickInput,
    policy: PolicyTendency,
) -> AgentState:
    """
    Integrate one heartbeat of active inference.

    This is intentionally simple: the chosen macro strategy perturbs the
    trade-off between conserving energy, reducing prediction error, and keeping
    enough uncertainty in play to avoid pathological boredom.
    """

    next_error = clamp01(
        (state.prediction_error * 0.45) + (tick_input.surprise_signal * 0.55)
    )
    next_surprise = clamp01(
        (state.surprise_load * 0.60)
        + (tick_input.resource_pressure * 0.25)
        + (tick_input.surprise_signal * 0.35)
    )
    next_boredom = clamp01(
        (state.boredom * 0.75)
        + (tick_input.boredom_signal * 0.35)
        + (max(0.0, 0.15 - next_error) * 0.50)
    )
    next_energy = clamp01(state.internal_energy - tick_input.energy_drain)

    if policy.chosen_strategy is Strategy.EXPLORE:
        next_energy = clamp01(next_energy - 0.08)
        next_error = clamp01(next_error + 0.06)
        next_surprise = clamp01(next_surprise + 0.04)
        next_boredom = clamp01(next_boredom - 0.28)
    elif policy.chosen_strategy is Strategy.EXPLOIT:
        next_energy = clamp01(next_energy - 0.03)
        next_error = clamp01(next_error - 0.10)
        next_surprise = clamp01(next_surprise - 0.06)
        next_boredom = clamp01(next_boredom - 0.05)
    else:
        # Escape is a defensive withdrawal: the agent stops expensive engagement,
        # lowers incoming perturbation, and regains a small amount of usable budget.
        next_energy = clamp01(next_energy + 0.05)
        next_error = clamp01(next_error - 0.18)
        next_surprise = clamp01(next_surprise - 0.14)
        next_boredom = clamp01(next_boredom + 0.08)

    return replace(
        state,
        internal_energy=next_energy,
        prediction_error=next_error,
        boredom=next_boredom,
        surprise_load=next_surprise,
        tick_count=state.tick_count + 1,
        last_strategy=policy.chosen_strategy,
    )
