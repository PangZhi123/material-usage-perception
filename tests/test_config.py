import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from material_usage_perception.config import AlgorithmSettings, DetectionRuntimeConfig


class AlgorithmSettingsTest(unittest.TestCase):
    def test_operator_can_set_minimum_work_duration(self):
        settings = AlgorithmSettings.from_mapping(
            {"state": {"minimum_dwell_seconds": 23.0}}
        )
        self.assertEqual(settings.state.minimum_dwell_seconds, 23.0)
        self.assertEqual(settings.state.enter_confirm_seconds, 2.0)

    def test_rejects_negative_duration(self):
        with self.assertRaises(ValueError):
            AlgorithmSettings.from_mapping(
                {"state": {"minimum_dwell_seconds": -1.0}}
            )

    def test_rejects_unknown_setting_group(self):
        with self.assertRaises(ValueError):
            AlgorithmSettings.from_mapping({"unknown": {}})

    def test_detects_git_lfs_pointer_instead_of_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\nsize 136722099\n",
                encoding="ascii",
            )
            config = DetectionRuntimeConfig(model_path=path)
            self.assertIn("Git LFS pointer", config.model_file_error())


if __name__ == "__main__":
    unittest.main()
