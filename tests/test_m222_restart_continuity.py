from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from segmentum.m222_benchmarks import build_m222_protocols, run_m222_protocol
from segmentum.runtime import SegmentRuntime


class TestM222RestartContinuity(unittest.TestCase):
    def test_restart_preserves_required_continuity_fields(self) -> None:
        protocol = build_m222_protocols(long_run_cycles=48, restart_pre_cycles=24, restart_post_cycles=24)["restart_continuity"]
        result = run_m222_protocol(protocol, seed=222, system_variant="full_system")
        for field in (
            "restart_identity_continuity",
            "restart_policy_continuity",
            "restart_memory_integrity",
        ):
            self.assertIn(field, result.metrics)
            self.assertGreaterEqual(float(result.metrics[field]), 0.80)
        self.assertGreaterEqual(float(result.restart["identity_narrative_continuity"]["commitment_similarity"]), 0.55)
        preferred = result.restart["preferred_policy_continuity"]
        self.assertGreaterEqual(float(preferred["anchor_restore_score"]), 0.95)
        self.assertGreaterEqual(float(preferred["rebind_consistency"]), 0.70)
        memory = result.restart["long_term_memory_integrity"]
        self.assertGreaterEqual(float(memory["protected_memory_integrity"]), 0.95)
        self.assertGreaterEqual(float(memory["critical_memory_integrity"]), 0.95)

    def test_snapshot_restore_preserves_commitment_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=222,
                reset=True,
            )
            for _ in range(260):
                runtime.step(verbose=False)
                commitments = list(runtime.agent.self_model.continuity_audit.commitment_snapshot)
                if commitments:
                    break
            else:
                self.fail("expected a non-empty commitment snapshot before restart")

            runtime.save_snapshot()
            before_commitments = list(runtime.agent.self_model.continuity_audit.commitment_snapshot)
            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=222,
                reset=False,
            )
            after_commitments = list(restored.agent.self_model.continuity_audit.commitment_snapshot)
            restored_narrative = restored.agent.self_model.identity_narrative

            self.assertEqual(before_commitments, after_commitments)
            self.assertIsNotNone(restored_narrative)
            self.assertEqual(
                before_commitments,
                [commitment.commitment_id for commitment in restored_narrative.commitments],
            )

    def test_snapshot_restore_preserves_protected_and_critical_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "segment_state.json"
            trace_path = Path(tmp_dir) / "segment_trace.jsonl"
            runtime = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=True,
            )
            protected_before: set[str] = set()
            critical_before: set[str] = set()
            for _ in range(260):
                runtime.step(verbose=False)
                combined = [
                    *runtime.agent.long_term_memory.episodes,
                    *runtime.agent.long_term_memory.archived_episodes,
                ]
                protected_before = {
                    str(payload.get("episode_id"))
                    for payload in combined
                    if payload.get("episode_id") and bool(payload.get("restart_protected", False))
                }
                critical_before = {
                    str(payload.get("episode_id"))
                    for payload in combined
                    if payload.get("episode_id") and bool(payload.get("identity_critical", False))
                }
                if protected_before and critical_before:
                    break
            else:
                self.fail("expected protected and critical memories before restart")

            runtime.save_snapshot()
            restored = SegmentRuntime.load_or_create(
                state_path=state_path,
                trace_path=trace_path,
                seed=342,
                reset=False,
            )
            combined_after = [
                *restored.agent.long_term_memory.episodes,
                *restored.agent.long_term_memory.archived_episodes,
            ]
            protected_after = {
                str(payload.get("episode_id"))
                for payload in combined_after
                if payload.get("episode_id") and bool(payload.get("restart_protected", False))
            }
            critical_after = {
                str(payload.get("episode_id"))
                for payload in combined_after
                if payload.get("episode_id") and bool(payload.get("identity_critical", False))
            }

            self.assertSetEqual(protected_before, protected_after)
            self.assertSetEqual(critical_before, critical_after)


if __name__ == "__main__":
    unittest.main()
