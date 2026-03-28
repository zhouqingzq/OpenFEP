from __future__ import annotations

import unittest

from segmentum.drives import DriveSystem
from segmentum.inquiry_scheduler import InquiryCandidate, process_valence_priority_adjustment


class TestM34ProcessValenceMotivation(unittest.TestCase):
    def test_focus_persists_then_closes_then_boredom_reorients(self) -> None:
        drives = DriveSystem()
        for _ in range(3):
            wanting = drives.update_process_valence(
                current_focus_id="unknown:m34-focus",
                unresolved_targets={"unknown:m34-focus"},
                focus_strength=0.75,
                maintenance_pressure=0.12,
            )
        self.assertEqual(wanting.active_phase, "wanting")
        self.assertEqual(wanting.focus_persistence_ticks, 3)
        self.assertGreater(wanting.unresolved_tension, 0.7)

        closure = drives.update_process_valence(
            current_focus_id="",
            unresolved_targets=set(),
            focus_strength=0.0,
            maintenance_pressure=0.12,
            closure_signal=1.0,
        )
        self.assertEqual(closure.active_phase, "closure")
        self.assertEqual(closure.recent_closed_focus_id, "unknown:m34-focus")

        boredom = closure
        for _ in range(4):
            boredom = drives.update_process_valence(
                current_focus_id="",
                unresolved_targets=set(),
                focus_strength=0.0,
                maintenance_pressure=0.05,
            )
        self.assertEqual(boredom.active_phase, "boredom")
        self.assertGreater(boredom.boredom_pressure, 0.4)

    def test_closed_focus_is_penalized_and_novel_focus_is_boosted_under_boredom(self) -> None:
        drives = DriveSystem()
        drives.update_process_valence(
            current_focus_id="unknown:m34-focus",
            unresolved_targets={"unknown:m34-focus"},
            focus_strength=0.72,
            maintenance_pressure=0.12,
        )
        closure = drives.update_process_valence(
            current_focus_id="",
            unresolved_targets=set(),
            focus_strength=0.0,
            maintenance_pressure=0.12,
            closure_signal=1.0,
        )
        boredom = closure
        for _ in range(4):
            boredom = drives.update_process_valence(
                current_focus_id="",
                unresolved_targets=set(),
                focus_strength=0.0,
                maintenance_pressure=0.05,
            )
        focus_candidate = InquiryCandidate(
            candidate_id="focus",
            source_subsystem="test",
            linked_target_id="unknown:m34-focus",
            linked_unknown_id="unknown:m34-focus",
            expected_information_gain=0.65,
        )
        novel_candidate = InquiryCandidate(
            candidate_id="novel",
            source_subsystem="test",
            linked_target_id="unknown:m34-novel",
            linked_unknown_id="unknown:m34-novel",
            expected_information_gain=0.65,
        )
        closure_penalty = process_valence_priority_adjustment(
            candidate=focus_candidate,
            process_valence_state=closure,
        )
        boredom_bonus = process_valence_priority_adjustment(
            candidate=novel_candidate,
            process_valence_state=boredom,
        )
        self.assertGreater(closure_penalty["closure_penalty"], 0.0)
        self.assertGreater(boredom_bonus["process_bonus"], 0.0)


if __name__ == "__main__":
    unittest.main()
