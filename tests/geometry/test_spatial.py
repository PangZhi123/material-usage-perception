import unittest

import numpy as np

from material_usage_perception.domain import BoundingBox, TrackObservation
from material_usage_perception.geometry import (
    HomographyProjector,
    PolygonRegion,
    SpatialProjector,
)


class SpatialGeometryTest(unittest.TestCase):
    def test_projects_between_explicit_sizes(self):
        projector = HomographyProjector(
            matrix=np.eye(3),
            source_frame_size=(200, 100),
            calibration_source_size=(100, 50),
            calibration_target_size=(100, 50),
            scene_size=(1000, 500),
        )
        self.assertEqual(projector.project((100, 50)), (500.0, 250.0))

    def test_scales_same_aspect_runtime_resolution(self):
        projector = HomographyProjector(
            matrix=np.eye(3),
            source_frame_size=(200, 100),
            calibration_source_size=(200, 100),
            calibration_target_size=(200, 100),
            scene_size=(200, 100),
        )
        self.assertEqual(
            projector.project((50, 25), runtime_frame_size=(100, 50)),
            (100.0, 50.0),
        )

    def test_rejects_runtime_aspect_ratio_mismatch(self):
        projector = HomographyProjector(
            matrix=np.eye(3),
            source_frame_size=(200, 100),
            calibration_source_size=(200, 100),
            calibration_target_size=(200, 100),
            scene_size=(200, 100),
        )
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            projector.project((50, 25), runtime_frame_size=(100, 100))

    def test_polygon_covers_inside_boundary_and_rejects_outside(self):
        region = PolygonRegion("zone", ((0, 0), (10, 0), (10, 10), (0, 10)))
        self.assertTrue(region.covers((5, 5)))
        self.assertTrue(region.covers((10, 5)))
        self.assertFalse(region.covers((11, 5)))

    def test_enriches_observation(self):
        spatial = SpatialProjector(
            HomographyProjector(
                np.eye(3), (100, 100), (100, 100), (100, 100), (100, 100)
            ),
            PolygonRegion("zone", ((0, 0), (20, 0), (20, 30), (0, 30))),
        )
        observation = TrackObservation(
            "cam", "session", 0, 0.0, 1, "transporter", BoundingBox(0, 0, 20, 20), 0.9
        )
        result = spatial.enrich(observation)
        self.assertEqual(result.image_ground_point, (10.0, 20))
        self.assertEqual(result.scene_ground_point, (10.0, 20.0))
        self.assertTrue(result.is_inside_area)

    def test_rejects_singular_matrix(self):
        with self.assertRaises(ValueError):
            HomographyProjector(
                np.zeros((3, 3)), (1, 1), (1, 1), (1, 1), (1, 1)
            )


if __name__ == "__main__":
    unittest.main()
