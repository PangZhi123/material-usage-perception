from tempfile import TemporaryDirectory
from pathlib import Path
from fastapi.testclient import TestClient
from material_usage_perception.repositories import FilesystemRepository
from material_usage_perception.api.app import create_app


def test_samples_and_task_results_survive_restart():
    with TemporaryDirectory() as directory:
        repo = FilesystemRepository(directory)
        repo.add_configuration("cfg", {"configurationId": "cfg", "cameraId": "cam"})
        repo.add_task("task", {"taskId": "task", "status": "COMPLETED", "recordingStartedAt": "2020-01-01T00:00:00+08:00"})
        repo.add_event("task", {"taskId": "task", "vehicleVisitId": "visit", "status": "COMPLETED", "estimatedVolumeM3": 12})
        repo.add_trajectory_points("task", "visit", [{"seq": 1, "x": .5}])
        for seq in (1, 11, 21):
            repo.save_sample("task", {"frame": {"seq": seq}, "objects": [{"bboxPx": [1, 2, 3, 4]}]})
        repo.add_task("interrupted", {"taskId": "interrupted", "status": "RUNNING"})
        restored = FilesystemRepository(directory)
        assert restored.get_configuration("cfg")["cameraId"] == "cam"
        assert restored.get_trajectory("visit")[0]["x"] == .5
        assert restored.get_task("interrupted")["status"] == "FAILED"
        with TestClient(create_app(repository=restored, storage_root=directory)) as client:
            base = "/api/material-perception/v1/tasks/task"
            result = client.get(base + "/result").json()["data"]
            assert result["estimatedVolumeM3"] == 12
            page = client.get(base + "/frames?limit=2").json()["data"]
            assert page["hasMore"] and page["nextFromSeq"] == 12
            assert len(client.get(base + "/frames?fromSeq=12").json()["data"]["items"]) == 1
