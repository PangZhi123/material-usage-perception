import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from material_usage_perception.pipeline.legacy_tracking_reader import (
    LegacyTrackingReader,
)


class LegacyTrackingReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = LegacyTrackingReader(
            camera_id="cam_02",
            processing_session_id="session_001",
            fps=25.0,
        )

    def _read(self, document: object):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return self.reader.read(path)

    def test_reads_transporter_and_uses_video_time(self) -> None:
        result = self._read(
            [
                {
                    "frame": 100,
                    "timestamp": "2099-01-01T00:00:00",
                    "objects": [
                        {
                            "id": 7,
                            "class": 8,
                            "bbox": [10, 20, 30, 50],
                            "confidence": 0.91,
                        }
                    ],
                }
            ]
        )

        observation = result.observations[0]
        self.assertEqual(observation.track_id, 7)
        self.assertEqual(observation.vehicle_class, "transporter")
        self.assertEqual(observation.video_time_seconds, 4.0)
        self.assertEqual(observation.bbox.ground_point, (20.0, 50.0))

    def test_filters_non_transporter_and_accepts_empty_frames(self) -> None:
        result = self._read(
            [
                {"frame": 0, "objects": []},
                {
                    "frame": 1,
                    "objects": [
                        {
                            "id": 1,
                            "class": 2,
                            "bbox": [0, 0, 10, 10],
                            "confidence": 0.8,
                        }
                    ],
                },
            ]
        )

        self.assertEqual(result.observations, ())
        self.assertEqual(result.summary.frame_count, 2)
        self.assertEqual(result.summary.skipped_object_count, 0)

    def test_skips_malformed_object_in_non_strict_mode(self) -> None:
        result = self._read(
            [{"frame": 0, "objects": [{"class": 8, "bbox": [0, 0, 1, 1]}]}]
        )

        self.assertEqual(result.summary.observation_count, 0)
        self.assertEqual(result.summary.skipped_object_count, 1)

    def test_rejects_invalid_fps(self) -> None:
        with self.assertRaises(ValueError):
            LegacyTrackingReader(
                camera_id="cam_02",
                processing_session_id="session_001",
                fps=0,
            )


if __name__ == "__main__":
    unittest.main()
