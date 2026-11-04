from unittest.mock import Mock
import pytest
from material_usage_perception.tasks.manager import TaskManager, TaskConflictError
from material_usage_perception.repositories import MemoryRepository
from material_usage_perception.config import DetectionRuntimeConfig


def test_history_single_slot_is_independent_of_camera_and_snapshots_configuration():
    repository = MemoryRepository()
    manager = TaskManager(repository, DetectionRuntimeConfig())
    manager._executor.shutdown()
    manager._executor = Mock()
    configuration = {"cameraId": "cam", "settings": {"value": 1}}
    options = dict(configuration=configuration, persist_trajectory=True, push_interval_ms=200)
    manager.submit_camera(task={"taskId": "live", "status": "QUEUED"},
                          source=Mock(camera_id="cam"), **options)
    manager.submit_video(task={"taskId": "history", "status": "QUEUED"},
                         source=Mock(), **options)
    configuration["settings"]["value"] = 2
    assert repository.get_task("history")["configurationSnapshot"]["settings"]["value"] == 1
    with pytest.raises(TaskConflictError):
        manager.submit_video(task={"taskId": "other", "status": "QUEUED"}, source=Mock(), **options)
    repository.add_task("history", {"status": "STOPPING"})
    with pytest.raises(TaskConflictError):
        manager.submit_video(task={"taskId": "other", "status": "QUEUED"}, source=Mock(), **options)
    repository.add_task("history", {"status": "STOPPED"})
    manager.submit_video(task={"taskId": "other", "status": "QUEUED"}, source=Mock(), **options)
