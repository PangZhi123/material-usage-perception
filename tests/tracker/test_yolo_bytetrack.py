import unittest

from material_usage_perception.tracker import YoloByteTrackTracker


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[1, 2, 11, 22], [30, 40, 50, 60]])
        self.cls = FakeTensor([8, 8])
        self.id = FakeTensor([3, 9])
        self.conf = FakeTensor([0.61, 0.94])

    def __len__(self):
        return 2


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {8: "transporter"}

    def __init__(self):
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


class YoloByteTrackTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = YoloByteTrackTracker("unused.pt", model=FakeModel())

    def test_converts_each_box_with_its_own_confidence(self):
        observations = self.tracker._convert_result(
            FakeResult(),
            frame_index=50,
            video_time_seconds=2.0,
            camera_id="cam_02",
            processing_session_id="session_1",
            class_names={8: "transporter"},
        )

        self.assertEqual([item.track_id for item in observations], [3, 9])
        self.assertEqual([item.confidence for item in observations], [0.61, 0.94])
        self.assertEqual(observations[0].bbox.ground_point, (6.0, 22.0))
        self.assertEqual(observations[0].video_time_seconds, 2.0)

    def test_requires_target_class_in_model(self):
        with self.assertRaises(ValueError):
            self.tracker._target_class_id({0: "person"})

    def test_validates_confidence_threshold(self):
        with self.assertRaises(ValueError):
            YoloByteTrackTracker("unused.pt", confidence_threshold=1.1)

    def test_tracks_live_frame_with_persistent_bytetrack_state(self):
        observations = self.tracker.track_frame(
            object(), frame_index=4, video_time_seconds=1.5,
            camera_id="cam", processing_session_id="task",
        )
        self.assertEqual([item.track_id for item in observations], [3, 9])
        self.assertTrue(self.tracker.model.calls[0]["persist"])
        self.assertEqual(self.tracker.model.calls[0]["classes"], [8])


if __name__ == "__main__":
    unittest.main()
