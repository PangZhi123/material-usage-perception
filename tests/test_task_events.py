import unittest
from datetime import datetime, timezone

from material_usage_perception.tasks import TaskEvent, TaskEventBus


class TaskEventBusTest(unittest.TestCase):
    def test_assigns_sequences_and_detects_truncated_frame_history(self):
        bus = TaskEventBus(history_size=2)
        now = datetime.now(timezone.utc)
        for seq in (1, 2, 3):
            event = bus.publish(TaskEvent(
                "task", "FRAME_RESULT", now,
                {"frame": {"seq": seq}, "objects": []},
            ))
            self.assertEqual(event.sequence, seq)
        self.assertFalse(bus.can_replay_frame("task", 1))
        self.assertTrue(bus.can_replay_frame("task", 2))
        self.assertEqual(bus.latest_sequence("task"), 3)

    def test_snapshot_tracks_latest_frame_event_and_status(self):
        bus = TaskEventBus()
        now = datetime.now(timezone.utc)
        bus.publish(TaskEvent(
            "task", "FRAME_RESULT", now,
            {"frame": {"seq": 7}, "objects": [{"vehicleVisitId": "visit"}]},
        ))
        bus.publish(TaskEvent(
            "task", "EVENT_UPDATE", now,
            {"event": {"vehicleVisitId": "visit", "status": "COMPLETED"}},
        ))
        bus.publish(TaskEvent(
            "task", "_TASK_STATUS", now, {"status": "COMPLETED"}
        ))
        snapshot = bus.snapshot_data("task")
        self.assertEqual(snapshot["taskStatus"], "COMPLETED")
        self.assertEqual(snapshot["frame"]["seq"], 7)
        self.assertEqual(snapshot["events"][0]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
