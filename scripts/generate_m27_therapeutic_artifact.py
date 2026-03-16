"""Generate M2.7 acceptance artifacts.

Produces four JSON artifacts demonstrating:
1. Vicious cycle self-reinforcing dynamics
2. Therapeutic intervention without metacognition (control)
3. Therapeutic intervention with metacognition (experiment)
4. VIA 24-strength projection example

Usage:
    python scripts/generate_m27_therapeutic_artifact.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.therapeutic import (
    SimulatedPersonalityState,
    TherapeuticAgent,
    run_vicious_cycle_simulation,
)
from segmentum.via_projection import VIAProjection


def main(seed: int = 42) -> None:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Scenario 1: Vicious cycle (no intervention)
    # ------------------------------------------------------------------
    print("Scenario 1: Vicious cycle simulation...")
    vicious = run_vicious_cycle_simulation(num_cycles=50, seed=seed)
    vicious_path = os.path.join(artifacts_dir, "segment_m27_vicious_cycle.json")
    with open(vicious_path, "w", encoding="utf-8") as f:
        json.dump(vicious.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  Written: {vicious_path}")
    print(f"  trust_prior: {vicious.initial_trust_prior:.4f} → {vicious.final_trust_prior:.4f}")
    print(f"  lovability:  {vicious.initial_lovability_mean:.4f} → {vicious.final_lovability_mean:.4f}")

    # ------------------------------------------------------------------
    # Scenario 2: Therapeutic intervention WITHOUT metacognition
    # ------------------------------------------------------------------
    print("\nScenario 2: Therapeutic without metacognition...")
    personality_ctrl = SimulatedPersonalityState()
    agent = TherapeuticAgent(signal_type="unconditional_positive_regard")
    no_meta = agent.run_therapeutic_simulation(
        personality_ctrl, num_cycles=80, metacognitive_enabled=False, seed=seed,
    )
    no_meta_path = os.path.join(artifacts_dir, "segment_m27_therapeutic_no_meta.json")
    with open(no_meta_path, "w", encoding="utf-8") as f:
        json.dump(no_meta.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  Written: {no_meta_path}")
    print(f"  trust_prior: {no_meta.initial_trust_prior:.4f} → {no_meta.final_trust_prior:.4f}")
    print(f"  lovability:  {no_meta.initial_lovability_mean:.4f} → {no_meta.final_lovability_mean:.4f}")
    print(f"  reversal:    {no_meta.reversal_detected} (cycle={no_meta.cycle_of_reversal})")

    # ------------------------------------------------------------------
    # Scenario 3: Therapeutic intervention WITH metacognition
    # ------------------------------------------------------------------
    print("\nScenario 3: Therapeutic with metacognition...")
    personality_exp = SimulatedPersonalityState()
    with_meta = agent.run_therapeutic_simulation(
        personality_exp, num_cycles=80, metacognitive_enabled=True, seed=seed,
    )
    with_meta_path = os.path.join(artifacts_dir, "segment_m27_therapeutic_with_meta.json")
    with open(with_meta_path, "w", encoding="utf-8") as f:
        json.dump(with_meta.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  Written: {with_meta_path}")
    print(f"  trust_prior: {with_meta.initial_trust_prior:.4f} → {with_meta.final_trust_prior:.4f}")
    print(f"  lovability:  {with_meta.initial_lovability_mean:.4f} → {with_meta.final_lovability_mean:.4f}")
    print(f"  reversal:    {with_meta.reversal_detected} (cycle={with_meta.cycle_of_reversal})")

    # ------------------------------------------------------------------
    # Scenario 4: VIA Projection
    # ------------------------------------------------------------------
    print("\nScenario 4: VIA projection...")
    proj = VIAProjection()

    # Project "lovability deficit" personality
    deficit_via = proj.project(
        openness=0.35, conscientiousness=0.5, extraversion=0.4,
        agreeableness=0.55, neuroticism=0.8, trust_prior=-0.6,
        meaning_construction_tendency=0.4, emotional_regulation_style=0.3,
    )
    # Project neutral personality
    neutral_via = proj.project()
    # Project high-resilience personality
    resilient_via = proj.project(
        openness=0.75, conscientiousness=0.7, extraversion=0.6,
        agreeableness=0.65, neuroticism=0.2, trust_prior=0.6,
        meaning_construction_tendency=0.7, emotional_regulation_style=0.7,
    )

    via_data = {
        "lovability_deficit_personality": {
            "via_strengths": deficit_via.to_dict(),
            "top_5": deficit_via.top_strengths(5),
            "bottom_5": deficit_via.bottom_strengths(5),
        },
        "neutral_personality": {
            "via_strengths": neutral_via.to_dict(),
            "top_5": neutral_via.top_strengths(5),
            "bottom_5": neutral_via.bottom_strengths(5),
        },
        "resilient_personality": {
            "via_strengths": resilient_via.to_dict(),
            "top_5": resilient_via.top_strengths(5),
            "bottom_5": resilient_via.bottom_strengths(5),
        },
    }
    via_path = os.path.join(artifacts_dir, "segment_m27_via_projection.json")
    with open(via_path, "w", encoding="utf-8") as f:
        json.dump(via_data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {via_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== M2.7 Artifact Generation Complete ===")
    print(f"Seed: {seed}")
    print(f"\nVicious cycle: trust {vicious.initial_trust_prior:.3f}→{vicious.final_trust_prior:.3f}")
    print(f"No meta therapy: lovability {no_meta.initial_lovability_mean:.3f}→{no_meta.final_lovability_mean:.3f}")
    print(f"With meta therapy: lovability {with_meta.initial_lovability_mean:.3f}→{with_meta.final_lovability_mean:.3f}")
    improvement_diff = with_meta.final_lovability_mean - no_meta.final_lovability_mean
    print(f"Metacognitive advantage: {improvement_diff:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
