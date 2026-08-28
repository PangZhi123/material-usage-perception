import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from material_usage_perception.repositories import (
    EventQuery, SQLiteRepository, StatisticsQuery,
)


class SQLiteRepositoryTest(unittest.TestCase):
    def test_only_same_camera_active_task_is_rejected(self):
        with TemporaryDirectory() as directory:
            repository = SQLiteRepository(Path(directory) / "test.sqlite3")
            first = {"taskId": "task_1", "status": "RUNNING"}
            second = {"taskId": "task_2", "status": "RUNNING"}
            self.assertTrue(repository.claim_task("task_1", "camera:cam_1", first))
            repository.add_task("task_1", {**first, "startedAt": "now"})
            self.assertFalse(repository.claim_task("task_2", "camera:cam_1", second))
            self.assertTrue(repository.claim_task("task_3", "camera:cam_2", second))
            first["status"] = "COMPLETED"
            repository.add_task("task_1", first)
            self.assertTrue(repository.claim_task("task_2", "camera:cam_1", second))
            repository.close()

    def test_atomic_result_save_and_repository_queries(self):
        with TemporaryDirectory() as directory:
            repository = SQLiteRepository(Path(directory) / "test.sqlite3")
            now = datetime.now(timezone.utc)
            task = {"taskId": "task", "status": "COMPLETED"}
            events = [
                {
                    "vehicleVisitId": f"visit_{index}", "cameraId": "cam",
                    "areaId": "zone", "materialType": "CONCRETE",
                    "status": status,
                    "enteredAt": (now + timedelta(minutes=index)).isoformat(),
                    "estimatedVolumeM3": volume,
                }
                for index, status, volume in (
                    (1, "COMPLETED", 10.0),
                    (2, "PASS_THROUGH", 0.0),
                    (3, "COMPLETED", 12.0),
                )
            ]
            trajectories = {
                "visit_1": [{"seq": seq, "timestamp": now.isoformat()} for seq in (1, 2, 3)]
            }
            repository.add_task("task", {"taskId": "task", "status": "RUNNING"})
            repository.save_task_result(task, events, trajectories)
            repository.add_event("task", {
                **events[0], "status": "ACTIVE", "estimatedVolumeM3": 0.0,
            })
            self.assertEqual(
                repository.get_event("visit_1")["status"], "COMPLETED"
            )
            with self.assertRaises(ValueError):
                repository.add_event("task", {
                    **events[0], "status": "INCOMPLETE",
                    "estimatedVolumeM3": 0.0,
                })
            with self.assertRaises(ValueError):
                repository.save_task_result(
                    {**task, "status": "FAILED"},
                    [{**events[0], "status": "REJECTED"}],
                    {},
                )
            self.assertEqual(repository.get_task("task")["status"], "COMPLETED")

            page = repository.query_events(EventQuery(
                start_time=now - timedelta(minutes=1),
                end_time=now + timedelta(minutes=5),
                statuses=("COMPLETED",), page=1, page_size=1,
            ))
            self.assertEqual(page.total, 2)
            self.assertEqual(page.items[0]["vehicleVisitId"], "visit_3")
            statistics = repository.query_statistics(
                StatisticsQuery(
                    now - timedelta(minutes=1), now + timedelta(minutes=5),
                    material_type="CONCRETE",
                ),
                generated_at=now,
            )
            self.assertEqual(statistics["completedEventCount"], 2)
            self.assertEqual(statistics["totalEstimatedVolumeM3"], 22.0)
            trajectory = repository.query_trajectory("visit_1", limit=2)
            self.assertEqual([point["seq"] for point in trajectory.points], [1, 2])
            self.assertEqual(trajectory.next_seq, 3)
            self.assertEqual(repository.get_task("task")["status"], "COMPLETED")
            repository.close()


if __name__ == "__main__":
    unittest.main()
