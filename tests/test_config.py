import unittest

from material_usage_perception.config import AlgorithmSettings


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


if __name__ == "__main__":
    unittest.main()
