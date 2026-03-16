from __future__ import annotations

from segmentum.m28_benchmarks import run_transfer_benchmark


def test_transfer_benchmark_shows_benefit_on_required_directions() -> None:
    predator_to_valley = run_transfer_benchmark(
        seed=42,
        train_world="predator_river",
        eval_worlds=["foraging_valley"],
        train_cycles=60,
        eval_cycles=40,
    )
    valley_to_social = run_transfer_benchmark(
        seed=43,
        train_world="foraging_valley",
        eval_worlds=["social_shelter"],
        train_cycles=60,
        eval_cycles=40,
    )

    for result in (predator_to_valley, valley_to_social):
        improvements = result["comparisons"][0]["improvements"]
        assert (
            improvements["survival_score_lift"] >= 0.08
            or improvements["conditioned_prediction_error_reduction"] >= 0.10
            or improvements["first_50_cycle_regret_reduction"] >= 0.10
        )
