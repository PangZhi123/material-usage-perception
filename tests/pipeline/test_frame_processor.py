import unittest

import numpy as np

from material_usage_perception.domain import BoundingBox, TrackObservation, VehicleStateConfig
from material_usage_perception.estimation import MaterialEstimator, MaterialUsageAccumulator
from material_usage_perception.geometry import HomographyProjector, PolygonRegion, SpatialProjector
from material_usage_perception.identity import TrackStitchConfig
from material_usage_perception.pipeline import FrameProcessor


class FrameProcessorTest(unittest.TestCase):
    def setUp(self):
        self.projector = SpatialProjector(
            HomographyProjector(np.eye(3), (100, 100), (100, 100), (100, 100), (100, 100)),
            PolygonRegion("zone", ((10, 10), (90, 10), (90, 90), (10, 90))),
        )
        self.processor = FrameProcessor(
            TrackStitchConfig(2.0, 5.0, 30.0, 50.0, 1.0),
            VehicleStateConfig(1.0, 2.0, 1.0, 1.0),
        )

    def spatial(self, frame, time, raw_id, x=50):
        tracking = TrackObservation(
            "cam", "session", frame, time, raw_id, "transporter",
            BoundingBox(x - 5, 30, x + 5, 50), 0.9,
        )
        return self.projector.enrich(tracking)

    def test_id_switch_does_not_split_state_machine_visit(self):
        first = self.processor.process_frame(
            frame_index=0, video_time_seconds=0.0,
            observations=[self.spatial(0, 0.0, 5, 40)],
        )
        self.processor.process_frame(
            frame_index=1, video_time_seconds=1.0,
            observations=[self.spatial(1, 1.0, 5, 45)],
        )
        switched = self.processor.process_frame(
            frame_index=2, video_time_seconds=2.0,
            observations=[self.spatial(2, 2.0, 30, 50)],
        )
        self.assertEqual(
            first.observations[0].vehicle_visit_id,
            switched.observations[0].vehicle_visit_id,
        )
        events = self.processor.finalize()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_track_ids, (5, 30))

    def test_two_simultaneous_trucks_remain_separate_and_are_counted_twice(self):
        processor = FrameProcessor(
            TrackStitchConfig(2.0, 5.0, 30.0, 50.0, 1.0),
            VehicleStateConfig(0.0, 2.0, 1.0, 1.0),
        )
        for frame, time in ((0, 0.0), (1, 2.0)):
            processor.process_frame(
                frame_index=frame,
                video_time_seconds=time,
                observations=[
                    self.spatial(frame, time, 5, 30),
                    self.spatial(frame, time, 6, 70),
                ],
            )
        processor.process_frame(
            frame_index=2,
            video_time_seconds=3.0,
            observations=[
                self.spatial(2, 3.0, 5, 0),
                self.spatial(2, 3.0, 6, 100),
            ],
        )
        completed = processor.process_frame(
            frame_index=3,
            video_time_seconds=4.0,
            observations=[
                self.spatial(3, 4.0, 5, 0),
                self.spatial(3, 4.0, 6, 100),
            ],
        ).emitted_events
        self.assertEqual(len(completed), 2)
        self.assertEqual(len({event.vehicle_visit_id for event in completed}), 2)

        accumulator = MaterialUsageAccumulator()
        estimator = MaterialEstimator()
        for event in completed:
            accumulator.add(estimator.estimate(event))
        summary = accumulator.summaries()[0]
        self.assertEqual(summary.completed_vehicle_count, 2)
        self.assertAlmostEqual(summary.estimated_usage_m3, 22.8)


if __name__ == "__main__":
    unittest.main()
