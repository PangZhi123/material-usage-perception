import unittest

import cv2
import numpy as np

from material_usage_perception.geometry import (
    CalibrationPointPair,
    compute_camera_to_scene_homography,
    validate_normalized_polygon,
)


class CalibrationTest(unittest.TestCase):
    def test_recovers_known_camera_to_scene_mapping(self):
        expected = np.array(
            [[1.2, 0.1, 30.0], [0.05, 0.9, 20.0], [0.0002, 0.0001, 1.0]]
        )
        camera_size = (1000.0, 500.0)
        scene_size = (1200.0, 800.0)
        camera_pixels = np.array(
            [[100, 100], [800, 80], [900, 400], [120, 420], [500, 250], [700, 300]],
            dtype=np.float64,
        )
        scene_pixels = cv2.perspectiveTransform(
            camera_pixels.reshape(-1, 1, 2), expected
        ).reshape(-1, 2)
        pairs = [
            CalibrationPointPair(
                (camera[0] / camera_size[0], camera[1] / camera_size[1]),
                (scene[0] / scene_size[0], scene[1] / scene_size[1]),
            )
            for camera, scene in zip(camera_pixels, scene_pixels)
        ]

        result = compute_camera_to_scene_homography(
            pairs, camera_image_size=camera_size, scene_image_size=scene_size
        )

        actual = result.matrix / result.matrix[2, 2]
        np.testing.assert_allclose(actual, expected, atol=2e-5)
        self.assertLess(result.quality.mean_reprojection_error_px, 1e-4)
        self.assertEqual(result.quality.inlier_count, 6)

    def test_rejects_too_few_and_collinear_pairs(self):
        pair = CalibrationPointPair((0.1, 0.1), (0.2, 0.2))
        with self.assertRaises(ValueError):
            compute_camera_to_scene_homography(
                [pair] * 3, camera_image_size=(100, 100), scene_image_size=(100, 100)
            )

        collinear = [
            CalibrationPointPair((v, v), (v, v)) for v in (0.1, 0.2, 0.3, 0.4)
        ]
        with self.assertRaises(ValueError):
            compute_camera_to_scene_homography(
                collinear, camera_image_size=(100, 100), scene_image_size=(100, 100)
            )

    def test_validates_polygon(self):
        validate_normalized_polygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        with self.assertRaises(ValueError):
            validate_normalized_polygon(((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)))
        with self.assertRaises(ValueError):
            validate_normalized_polygon(((0.1, 0.1), (1.1, 0.1), (0.5, 0.9)))


if __name__ == "__main__":
    unittest.main()
