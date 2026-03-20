from __future__ import annotations

import unittest

from segmentum.memory import LongTermMemory


def _store_rest_episode(
    memory: LongTermMemory,
    *,
    cycle: int,
    danger: float = 0.20,
    temperature: float = 0.50,
) -> dict[str, object]:
    return memory.store_episode(
        cycle=cycle,
        observation={
            "food": 0.40,
            "danger": danger,
            "novelty": 0.20,
            "shelter": 0.60,
            "temperature": temperature,
            "social": 0.20,
        },
        prediction={
            "food": 0.42,
            "danger": 0.18,
            "novelty": 0.18,
            "shelter": 0.62,
            "temperature": 0.48,
            "social": 0.22,
        },
        errors={
            "food": -0.02,
            "danger": 0.02,
            "novelty": 0.02,
            "shelter": -0.02,
            "temperature": temperature - 0.48,
            "social": -0.02,
        },
        action="rest",
        outcome={
            "energy_delta": 0.01,
            "stress_delta": -0.01,
            "free_energy_drop": 0.02,
        },
        body_state={
            "energy": 0.55,
            "stress": 0.35,
            "fatigue": 0.25,
            "temperature": temperature,
        },
    )


class TestRestartMemoryProtection(unittest.TestCase):
    def test_memory_roundtrip_rehydrates_protected_metadata(self) -> None:
        memory = LongTermMemory()
        protected = _store_rest_episode(memory, cycle=10, temperature=0.66)
        protected["identity_critical"] = True
        protected["identity_commitment_ids"] = ["commitment-survival-priority"]
        protected["restart_protected"] = False
        protected["continuity_role"] = ""
        memory.archive_episode(protected, archive_cycle=11, reason="test")

        restored = LongTermMemory.from_dict(memory.to_dict())
        combined = [
            *restored.episodes,
            *restored.archived_episodes,
        ]
        archived = next(payload for payload in combined if payload.get("episode_id") == protected["episode_id"])

        self.assertTrue(bool(archived.get("identity_critical", False)))
        self.assertTrue(bool(archived.get("restart_protected", False)))
        self.assertEqual(str(archived.get("continuity_role", "")), "identity_critical_memory")

    def test_restart_window_protects_only_anchored_memories(self) -> None:
        memory = LongTermMemory()
        memory.max_active_age = 64
        protected = _store_rest_episode(memory, cycle=1, temperature=0.67)
        protected["identity_critical"] = True
        protected["identity_commitment_ids"] = ["commitment-survival-priority"]
        protected["restart_protected"] = False
        protected["continuity_role"] = ""

        first_routine = _store_rest_episode(memory, cycle=2)
        second_routine = _store_rest_episode(memory, cycle=3)
        first_routine["restart_protected"] = False
        first_routine["continuity_role"] = ""
        second_routine["restart_protected"] = False
        second_routine["continuity_role"] = ""

        memory.activate_restart_continuity_window(current_cycle=90, duration=24)

        removed = memory.compress_episodes(current_cycle=100)
        active_ids_after_compress = {
            str(payload.get("episode_id"))
            for payload in memory.episodes
            if payload.get("episode_id")
        }
        self.assertEqual(removed, 1)
        self.assertIn(str(protected["episode_id"]), active_ids_after_compress)

        retired = memory.retire_stale_episodes(current_cycle=100, retain_recent=1)
        active_ids_after_retire = {
            str(payload.get("episode_id"))
            for payload in memory.episodes
            if payload.get("episode_id")
        }
        archived_ids = {
            str(payload.get("episode_id"))
            for payload in memory.archived_episodes
            if payload.get("episode_id")
        }

        self.assertGreaterEqual(retired, 1)
        self.assertIn(str(protected["episode_id"]), active_ids_after_retire)
        self.assertGreaterEqual(len(memory.archived_episodes), 1)
        self.assertNotIn(str(protected["episode_id"]), archived_ids)


if __name__ == "__main__":
    unittest.main()
