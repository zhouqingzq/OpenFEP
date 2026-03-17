from __future__ import annotations

import unittest

from segmentum.action_schema import ActionSchema
from segmentum.environment import Observation, SimulatedWorld
from segmentum.interoception import InteroceptionReading
from segmentum.io_bus import ActionBus, ActionDispatchRecord, PerceptionBus, PerceptionPacket
from segmentum.narrative_types import NarrativeEpisode


class M212AdapterRoundtripTests(unittest.TestCase):
    def test_perception_packets_roundtrip_for_supported_sources(self) -> None:
        bus = PerceptionBus()

        world_packet = bus.capture_simulated_world(
            Observation(
                food=0.4,
                danger=0.7,
                novelty=0.2,
                shelter=0.8,
                temperature=0.5,
                social=0.3,
            ),
            cycle=3,
        )
        host_packet = bus.capture_interoception(
            InteroceptionReading(
                cpu_percent=12.0,
                memory_mb=144.0,
                cpu_prediction_error=0.1,
                memory_prediction_error=0.2,
                resource_pressure=0.16,
                energy_drain=0.04,
                boredom_signal=0.0,
                surprise_signal=0.12,
            ),
            cycle=3,
        )
        narrative_packet = bus.capture_narrative_episode(
            NarrativeEpisode(
                episode_id="ep-1",
                timestamp=9,
                source="story",
                raw_text="predator near the river",
                tags=["danger", "memory"],
                metadata={},
            ),
            cycle=3,
        )

        for packet in (world_packet, host_packet, narrative_packet):
            restored = PerceptionPacket.from_dict(packet.to_dict())
            self.assertEqual(restored.to_dict(), packet.to_dict())

    def test_action_dispatch_roundtrip(self) -> None:
        bus = ActionBus()
        world = SimulatedWorld(seed=7)
        dispatch = bus.dispatch_to_simulated_world(
            world,
            ActionSchema(name="hide"),
            cycle=2,
        )

        restored = ActionDispatchRecord.from_dict(dispatch.to_dict())
        self.assertEqual(restored.to_dict(), dispatch.to_dict())
        self.assertEqual(bus.dispatch_count, 1)
        self.assertGreaterEqual(bus.acknowledged_effects, 1)


if __name__ == "__main__":
    unittest.main()
