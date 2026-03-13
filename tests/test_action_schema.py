from __future__ import annotations

from segmentum.action_schema import ActionSchema


def test_action_schema_round_trip_and_str_compatibility() -> None:
    action = ActionSchema(
        name="web_fetch",
        params={"timeout": 30, "url": "https://example.com"},
        cost_estimate=1.25,
        reversible=False,
    )
    restored = ActionSchema.from_dict(action.to_dict())

    assert restored == action
    assert restored != "hide"
    assert restored.name == ActionSchema.from_dict("web_fetch").name
    assert str(ActionSchema.from_dict("forage")) == "forage"


def test_action_schema_hash_is_deterministic_for_same_params() -> None:
    left = ActionSchema(name="web_fetch", params={"b": 2, "a": 1})
    right = ActionSchema(name="web_fetch", params={"a": 1, "b": 2})

    assert hash(left) == hash(right)
    assert left == right
