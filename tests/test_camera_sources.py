import unittest

from material_usage_perception.sources import (
    CameraSource,
    MappingCameraSourceRegistry,
)


class CameraSourceRegistryTest(unittest.TestCase):
    def test_registers_and_resolves_camera_stream(self):
        registry = MappingCameraSourceRegistry()
        registry.register(CameraSource("cam_01", "rtsp://camera/live"))
        self.assertEqual(registry.resolve("cam_01").stream_url, "rtsp://camera/live")
        self.assertIsNone(registry.resolve("missing"))


if __name__ == "__main__":
    unittest.main()
