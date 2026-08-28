import unittest
from types import SimpleNamespace

from material_usage_perception.domain import (
    BoundingBox,
    TrackObservation,
    VehicleState,
    VehicleStateConfig,
    VehicleWorkStateMachine,
    VisitOutcome,
)


class VehicleWorkStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.machine = VehicleWorkStateMachine(VehicleStateConfig(2.0, 5.0, 2.0, 1.0))
        self.key = ("cam", "session", "raw:7", "zone")

    def observation(self, frame, time, inside=True):
        bbox = BoundingBox(40, 30, 60, 50) if inside else BoundingBox(0, 0, 10, 10)
        tracking = TrackObservation(
            "cam", "session", frame, time, 7, "transporter", bbox, 0.9
        )
        return SimpleNamespace(
            tracking=tracking,
            area_id="zone",
            is_inside_area=inside,
        )

    def process(self, frame, time, inside=None):
        observations = () if inside is None else (self.observation(frame, time, inside),)
        return self.machine.process_frame(
            frame_index=frame, video_time_seconds=time, observations=observations
        )

    def test_completed_visit(self):
        self.process(0, 0.0, False)
        self.process(1, 1.0, True)
        self.process(2, 3.0, True)
        self.assertEqual(self.machine.state_for(self.key), VehicleState.INSIDE)
        self.process(3, 7.0, True)
        self.process(4, 8.0, False)
        events = self.process(5, 10.0, False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].outcome, VisitOutcome.COMPLETED)
        self.assertEqual(events[0].enter_time_seconds, 1.0)
        self.assertEqual(events[0].exit_time_seconds, 8.0)
        self.assertEqual(events[0].dwell_seconds, 7.0)

    def test_short_visit_is_pass_through(self):
        self.process(0, 0.0, True)
        self.process(1, 2.0, True)
        self.process(2, 3.0, False)
        events = self.process(3, 5.0, False)
        self.assertEqual(events[0].outcome, VisitOutcome.PASS_THROUGH)

    def test_boundary_reentry_cancels_leaving(self):
        self.process(0, 0.0, True)
        self.process(1, 2.0, True)
        self.process(2, 8.0, False)
        self.process(3, 9.0, True)
        self.assertEqual(self.machine.state_for(self.key), VehicleState.INSIDE)
        self.assertEqual(self.process(4, 10.0, True), ())

    def test_short_loss_is_tolerated_and_long_loss_completes(self):
        self.process(0, 0.0, True)
        self.process(1, 2.0, True)
        self.process(2, 7.0, True)
        self.assertEqual(self.process(3, 7.5, None), ())
        self.assertEqual(self.machine.state_for(self.key), VehicleState.INSIDE)
        self.assertEqual(self.process(4, 9.0, None), ())
        events = self.process(5, 10.1, None)
        self.assertEqual(events[0].outcome, VisitOutcome.COMPLETED)
        self.assertEqual(events[0].exit_time_seconds, 8.0)

    def test_finalize_marks_confirmed_open_visit_incomplete(self):
        self.process(0, 0.0, True)
        self.process(1, 2.0, True)
        self.process(2, 6.0, True)
        events = self.machine.finalize()
        self.assertEqual(events[0].outcome, VisitOutcome.INCOMPLETE)
        self.assertIsNone(events[0].exit_time_seconds)

    def test_unconfirmed_entry_is_not_an_event(self):
        self.process(0, 0.0, True)
        self.assertEqual(self.machine.finalize(), ())


if __name__ == "__main__":
    unittest.main()
