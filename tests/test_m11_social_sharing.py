from segmentum.user_model import (
    SocialSharingCandidate,
    boundary_strength_from_constraints,
    decide_social_sharing,
    detect_explicit_secrecy,
    update_regret_bias,
)


def test_default_social_memory_can_be_retold_to_reduce_reaction_prediction_free_energy() -> None:
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:reaction_expectation",
            source_user_id="user_a",
            audience_user_id="user_b",
            shareability="default_social",
            boundary_strength="none",
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
    )

    assert decision.action == "direct_share"
    assert decision.expected_free_energy_reduction > 0
    assert decision.net_free_energy_reduction > 0


def test_source_declared_secret_hard_blocks_retelling() -> None:
    explicit, marker = detect_explicit_secrecy("我告诉你一个秘密，你别告诉别人。")
    boundary = boundary_strength_from_constraints((), explicit_secrecy=explicit)
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:secret",
            source_user_id="user_a",
            audience_user_id="user_b",
            shareability="restricted_explicit",
            boundary_strength=boundary,
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
    )

    assert marker
    assert decision.action == "withhold"
    assert decision.allow_direct_disclosure is False


def test_soft_boundary_prefers_abstract_reference() -> None:
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:soft",
            source_user_id="user_a",
            audience_user_id="user_b",
            shareability="restricted_implicit",
            boundary_strength="soft",
            expected_audience_reaction="amused",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
    )

    assert decision.action == "abstract_reference"
    assert decision.abstract_only is True


def test_negative_feedback_raises_future_sharing_cost() -> None:
    regret = update_regret_bias(
        previous_regret_bias=0.0,
        negative_feedback=True,
        had_cross_user_share=True,
        same_audience_user=True,
    )
    before = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:reaction_expectation",
            source_user_id="user_a",
            audience_user_id="user_b",
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
        regret_bias=0.0,
    )
    after = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:reaction_expectation",
            source_user_id="user_a",
            audience_user_id="user_b",
            expected_audience_reaction="surprised",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
        regret_bias=regret,
    )

    assert regret > 0
    assert after.net_free_energy_reduction < before.net_free_energy_reduction


def test_verified_reaction_expectation_carries_less_free_energy() -> None:
    unverified = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:expectation",
            source_user_id="user_a",
            audience_user_id="user_b",
            expected_audience_reaction="approving",
            expectation_status="unverified",
        ),
        sharing_intent="social_share",
    )
    verified = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:expectation",
            source_user_id="user_a",
            audience_user_id="user_b",
            expected_audience_reaction="approving",
            expectation_status="verified",
        ),
        sharing_intent="social_share",
    )

    assert verified.current_free_energy < unverified.current_free_energy


def test_incomprehensible_reaction_requires_explanation_or_self_reframe() -> None:
    decision = decide_social_sharing(
        SocialSharingCandidate(
            memory_id="m:reaction_gap",
            source_user_id="user_a",
            audience_user_id="user_b",
            expected_audience_reaction="neutral",
            expectation_status="incomprehensible",
        ),
        sharing_intent="social_share",
    )

    assert decision.action == "explain_or_reframe"
    assert "rebuild_self_model" in decision.explanation_strategy
