import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from material_usage_perception.repositories import FilesystemRepository


class FilesystemRepositoryTest(unittest.TestCase):
    def test_task_events_and_trajectories_are_frontend_readable_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = FilesystemRepository(root)
            repository.add_configuration("cfg_1", {"cameraId": "cam_1"})
            task = {
                "taskId": "task_1", "configurationId": "cfg_1",
                "status": "RUNNING",
            }
            self.assertTrue(repository.claim_task("task_1", "camera:cam_1", task))
            event = {
                "taskId": "task_1", "vehicleVisitId": "visit_1",
                "status": "COMPLETED", "estimatedVolumeM3": 8.5,
            }
            repository.add_event("task_1", event)
            point = {"seq": 1, "scenePointNorm": {"x": 0.4, "y": 0.6}}
            repository.add_trajectory_points("task_1", "visit_1", [point])

            saved_task = json.loads(
                (root / "tasks" / "task_1" / "task.json").read_text(encoding="utf-8")
            )
            saved_events = json.loads(
                (root / "tasks" / "task_1" / "events.json").read_text(encoding="utf-8")
            )
            saved_points = json.loads(
                (root / "tasks" / "task_1" / "trajectories" / "visit_1.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(saved_task["configurationSnapshot"], {"cameraId": "cam_1"})
            self.assertEqual(saved_events, [event])
            self.assertEqual(saved_points, [point])


if __name__ == "__main__":
    unittest.main()
