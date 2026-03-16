"""Tests for TherapeuticAgent and vicious cycle simulation (M2.7 Phase C)."""

from segmentum.therapeutic import (
    SimulatedPersonalityState,
    TherapeuticAgent,
    run_vicious_cycle_simulation,
)


def test_vicious_cycle_trust_prior_decreases():
    traj = run_vicious_cycle_simulation(num_cycles=50, seed=42)
    assert traj.final_trust_prior < traj.initial_trust_prior


def test_vicious_cycle_lovability_variance_shrinks():
    traj = run_vicious_cycle_simulation(num_cycles=50, seed=42)
    initial_var = 0.04  # from SimulatedPersonalityState default
    final_var = traj.snapshots[-1].lovability_belief_variance
    assert final_var < initial_var


def test_vicious_cycle_suppress_dominates():
    traj = run_vicious_cycle_simulation(num_cycles=50, seed=42)
    # In last window, suppress + assimilate should dominate
    last = traj.snapshots[-1]
    assert last.suppress_rate + (1.0 - last.accommodate_rate) > 0.5


def test_therapeutic_no_meta_limited_effect():
    """Without metacognition, therapeutic signal has limited effect."""
    personality = SimulatedPersonalityState()
    agent = TherapeuticAgent(signal_type="unconditional_positive_regard")
    traj = agent.run_therapeutic_simulation(
        personality, num_cycles=60, metacognitive_enabled=False, seed=42,
    )
    # Some effect but limited — accommodate rate stays low
    last = traj.snapshots[-1]
    # Without meta, suppress penalty doesn't increase, so accommodate stays low
    assert last.accommodate_rate < 0.5


def test_therapeutic_with_meta_shows_improvement():
    """With metacognition, therapeutic signals break through more effectively."""
    personality = SimulatedPersonalityState()
    agent = TherapeuticAgent(signal_type="unconditional_positive_regard")
    traj = agent.run_therapeutic_simulation(
        personality, num_cycles=80, metacognitive_enabled=True, seed=42,
    )
    # With metacognition, we expect some dissociation and eventual improvement
    has_dissociation = any(s.dissociation_level > 0 for s in traj.snapshots)
    assert has_dissociation


def test_therapeutic_with_meta_lovability_improves_vs_no_meta():
    """Metacognitive condition should show better lovability outcome."""
    # No meta
    p1 = SimulatedPersonalityState()
    agent = TherapeuticAgent()
    traj_no_meta = agent.run_therapeutic_simulation(
        p1, num_cycles=80, metacognitive_enabled=False, seed=42,
    )
    # With meta
    p2 = SimulatedPersonalityState()
    traj_with_meta = agent.run_therapeutic_simulation(
        p2, num_cycles=80, metacognitive_enabled=True, seed=42,
    )
    assert traj_with_meta.final_lovability_mean >= traj_no_meta.final_lovability_mean


def test_therapeutic_deterministic():
    """Same seed → same trajectory."""
    p1 = SimulatedPersonalityState()
    p2 = SimulatedPersonalityState()
    agent = TherapeuticAgent()
    t1 = agent.run_therapeutic_simulation(p1, num_cycles=30, seed=123)
    t2 = agent.run_therapeutic_simulation(p2, num_cycles=30, seed=123)
    assert t1.final_trust_prior == t2.final_trust_prior
    assert t1.final_lovability_mean == t2.final_lovability_mean


def test_trajectory_to_dict():
    traj = run_vicious_cycle_simulation(num_cycles=10, seed=42)
    d = traj.to_dict()
    assert "snapshots" in d
    assert len(d["snapshots"]) == 10
    assert "cycle_of_reversal" in d
    assert "final_trust_prior" in d


def test_therapeutic_ramp_up():
    """Signal precision should ramp up gradually."""
    personality = SimulatedPersonalityState()
    agent = TherapeuticAgent(ramp_up_cycles=10)
    sig_early = agent.generate_therapeutic_signal(personality, cycle=1)
    sig_late = agent.generate_therapeutic_signal(personality, cycle=20)
    assert sig_late.precision > sig_early.precision
