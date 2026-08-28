import unittest

import numpy as np

from material_usage_perception.domain import BoundingBox, TrackObservation
from material_usage_perception.geometry import (
    HomographyProjector,
    PolygonRegion,
    SpatialProjector,
)
from material_usage_perception.identity import OnlineTrackStitcher, TrackStitchConfig


class OnlineTrackStitcherTest(unittest.TestCase):
    def setUp(self):
        self.projector = SpatialProjector(
            HomographyProjector(
                np.eye(3), (1000, 500), (1000, 500), (1000, 500), (1000, 500)
            ),
            PolygonRegion("zone", ((0, 0), (1000, 0), (1000, 500), (0, 500))),
        )
        self.stitcher = OnlineTrackStitcher(
            TrackStitchConfig(
                max_reconnect_gap_seconds=2.0,
                retention_seconds=5.0,
                max_scene_distance=80.0,
                max_image_distance=120.0,
                max_prediction_seconds=1.0,
            )
        )

    def observation(self, frame, time, track_id, x, y=200):
        tracking = TrackObservation(
            "cam", "session", frame, time, track_id, "transporter",
            BoundingBox(x - 20, y - 40, x + 20, y), 0.9,
        )
        return self.projector.enrich(tracking)

    def test_keeps_visit_id_for_same_raw_track(self):
        first = self.stitcher.process_frame([self.observation(0, 0.0, 5, 100)])[0]
        second = self.stitcher.process_frame([self.observation(1, 0.5, 5, 110)])[0]
        self.assertEqual(first.vehicle_visit_id, second.vehicle_visit_id)
        self.assertFalse(second.stitched)

    def test_stitches_nearby_new_raw_id_causally(self):
        first = self.stitcher.process_frame([self.observation(0, 0.0, 5, 100)])[0]
        self.stitcher.process_frame([self.observation(1, 0.5, 5, 110)])
        switched = self.stitcher.process_frame([self.observation(2, 1.0, 30, 121)])[0]
        self.assertEqual(first.vehicle_visit_id, switched.vehicle_visit_id)
        self.assertTrue(switched.stitched)
        self.assertEqual(
            self.stitcher.source_track_ids(switched.vehicle_visit_id), (5, 30)
        )

    def test_does_not_merge_far_new_track(self):
        first = self.stitcher.process_frame([self.observation(0, 0.0, 5, 100)])[0]
        far = self.stitcher.process_frame([self.observation(1, 0.5, 30, 500)])[0]
        self.assertNotEqual(first.vehicle_visit_id, far.vehicle_visit_id)

    def test_does_not_merge_simultaneous_tracks(self):
        resolved = self.stitcher.process_frame(
            [self.observation(0, 0.0, 5, 100), self.observation(0, 0.0, 30, 160)]
        )
        self.assertNotEqual(resolved[0].vehicle_visit_id, resolved[1].vehicle_visit_id)

    def test_suppresses_nested_duplicate_box_and_keeps_existing_track(self):
        first = self.stitcher.process_frame([self.observation(0, 0.0, 5, 100)])[0]
        main = self.observation(1, 0.5, 5, 110)
        duplicate_tracking = TrackObservation(
            "cam", "session", 1, 0.5, 30, "transporter",
            BoundingBox(100, 165, 130, 200), 0.9,
        )
        duplicate = self.projector.enrich(duplicate_tracking)
        resolved = self.stitcher.process_frame([main, duplicate])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].source_track_id, 5)
        self.assertEqual(resolved[0].vehicle_visit_id, first.vehicle_visit_id)

    def test_does_not_reconnect_after_gap(self):
        first = self.stitcher.process_frame([self.observation(0, 0.0, 5, 100)])[0]
        later = self.stitcher.process_frame([self.observation(100, 4.0, 30, 105)])[0]
        self.assertNotEqual(first.vehicle_visit_id, later.vehicle_visit_id)


if __name__ == "__main__":
    unittest.main()
