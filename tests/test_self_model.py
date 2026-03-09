from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from segmentum.self_model import (
    BodySchema,
    CapabilityModel,
    SelfModel,
    ResourceState,
    ThreatModel,
)


def build_self_model(
    *,
    tokens_remaining: int = 64,
    cpu_budget: float = 0.80,
    memory_free: float = 512.0,
    memory_usage: float = 128.0,
    compute_load: float = 0.30,
) -> SelfModel:
    return SelfModel(
        body_schema=BodySchema(
            energy=0.90,
            token_budget=256,
            memory_usage=memory_usage,
            compute_load=compute_load,
        ),
        capability_model=CapabilityModel(
            available_actions=("observe", "act", "reflect"),
            api_limits={"requests_per_minute": 60},
        ),
        resource_state=ResourceState(
            tokens_remaining=tokens_remaining,
            cpu_budget=cpu_budget,
            memory_free=memory_free,
        ),
        threat_model=ThreatModel(),
    )


class SelfModelTests(unittest.TestCase):
    def test_error_classification(self) -> None:
        model = build_self_model()

        self.assertEqual(model.classify_event("TokenLimitExceeded"), "self_error")
        self.assertEqual(model.classify_event("OutOfMemory"), "self_error")
        self.assertEqual(model.classify_event("HTTPTimeout"), "world_error")
        self.assertEqual(model.classify_event("NetworkFailure"), "world_error")
        self.assertEqual(model.classify_event("FatalException"), "existential_threat")

        result = model.inspect_event("TokenLimitExceeded")
        log_output = result.to_log_string()
        self.assertIn("[SelfModel]", log_output)
        self.assertIn("event=TokenLimitExceeded", log_output)
        self.assertIn("classification=self_error", log_output)
        self.assertIn("surprise_source=interoceptive", log_output)
        self.assertIn("tokens_remaining", log_output)

    def test_resource_prediction(self) -> None:
        model = build_self_model(
            tokens_remaining=0,
            cpu_budget=0.20,
            memory_free=64.0,
            memory_usage=128.0,
            compute_load=0.45,
        )

        prediction = model.predict_resource_state()

        self.assertEqual(
            prediction,
            {
                "token_exhaustion": True,
                "memory_overflow": True,
                "cpu_overload": True,
            },
        )

    def test_threat_detection(self) -> None:
        model = build_self_model(
            tokens_remaining=0,
            memory_free=64.0,
            memory_usage=128.0,
        )

        self.assertEqual(
            model.detect_threats("FatalException"),
            ("token_exhaustion", "memory_overflow", "fatal_exception"),
        )

        result = model.inspect_event("TokenLimitExceeded")
        self.assertEqual(result.classification, "self_error")
        self.assertIn("token_exhaustion", result.detected_threats)

    def test_core_beliefs_are_immutable(self) -> None:
        model = build_self_model()

        with self.assertRaises(FrozenInstanceError):
            model.body_schema.energy = 0.10  # type: ignore[misc]

        with self.assertRaises(FrozenInstanceError):
            model.threat_model.memory_overflow_threshold = 1.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
