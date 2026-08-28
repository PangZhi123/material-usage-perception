import unittest

from material_usage_perception.domain import VehicleVisitEvent, VisitOutcome
from material_usage_perception.estimation import (
    MaterialEstimator,
    MaterialQuantityConfig,
    MaterialUsageAccumulator,
    QuantitySource,
)


def event(outcome=VisitOutcome.COMPLETED, visit_id="visit_1"):
    return VehicleVisitEvent(
        camera_id="cam",
        processing_session_id="session",
        vehicle_visit_id=visit_id,
        source_track_ids=(5, 30),
        area_id="zone",
        vehicle_class="transporter",
        enter_time_seconds=0.0,
        exit_time_seconds=60.0,
        dwell_seconds=60.0,
        enter_frame_index=0,
        exit_frame_index=1500,
        outcome=outcome,
    )


class MaterialEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.estimator = MaterialEstimator(MaterialQuantityConfig(12.0, 0.9, 0.95))

    def test_uses_actual_load_first(self):
        record = self.estimator.estimate(
            event(), actual_loaded_m3=11.0, vehicle_capacity_m3=14.0
        )
        self.assertEqual(record.quantity_source, QuantitySource.ACTUAL_LOADED)
        self.assertAlmostEqual(record.estimated_usage_m3, 10.45)

    def test_uses_vehicle_then_default_capacity(self):
        vehicle = self.estimator.estimate(event(), vehicle_capacity_m3=10.0)
        default = self.estimator.estimate(event(visit_id="visit_2"))
        self.assertEqual(vehicle.quantity_source, QuantitySource.VEHICLE_CAPACITY)
        self.assertAlmostEqual(vehicle.estimated_usage_m3, 8.55)
        self.assertEqual(default.quantity_source, QuantitySource.DEFAULT_CAPACITY)
        self.assertAlmostEqual(default.estimated_usage_m3, 10.26)

    def test_does_not_count_non_completed_events(self):
        for outcome in (
            VisitOutcome.ACTIVE, VisitOutcome.PASS_THROUGH,
            VisitOutcome.INCOMPLETE, VisitOutcome.REJECTED,
        ):
            record = self.estimator.estimate(event(outcome))
            self.assertFalse(record.counted)
            self.assertEqual(record.estimated_usage_m3, 0.0)
            self.assertEqual(record.quantity_source, QuantitySource.NOT_COUNTED)

    def test_accumulates_and_prevents_duplicate_counting(self):
        accumulator = MaterialUsageAccumulator()
        accumulator.add(self.estimator.estimate(event()))
        accumulator.add(
            self.estimator.estimate(event(VisitOutcome.PASS_THROUGH, "visit_2"))
        )
        summary = accumulator.summaries()[0]
        self.assertEqual(summary.completed_vehicle_count, 1)
        self.assertEqual(summary.ignored_event_count, 1)
        self.assertAlmostEqual(summary.estimated_usage_m3, 10.26)
        with self.assertRaises(ValueError):
            accumulator.add(self.estimator.estimate(event()))


if __name__ == "__main__":
    unittest.main()
