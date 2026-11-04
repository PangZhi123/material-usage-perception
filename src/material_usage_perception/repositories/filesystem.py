"""JSON file repository for frontend-readable local persistence."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from .memory import MemoryRepository, SceneRecord
from .queries import EventQuery, StatisticsQuery
from material_usage_perception.sources import VideoSource


class FilesystemRepository(MemoryRepository):
    """Persist backend records as JSON files below the configured data folder."""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.tasks_root = self.root / "tasks"
        self.configurations_root = self.root / "configurations"
        self.scenes_root = self.root / "scenes"
        self.videos_root = self.root / "videos"
        for directory in (
            self.tasks_root, self.configurations_root, self.scenes_root,
            self.videos_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._file_lock = RLock()
        for path in self.configurations_root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.configurations[value.get("configurationId", path.stem)] = value
        for path in self.tasks_root.glob("*/task.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") in {"QUEUED", "RUNNING", "STOPPING"}:
                value.update(status="FAILED", error="service restarted before task completed")
                self._write(path, value)
            self.tasks[value["taskId"]] = value
            events_path = path.parent / "events.json"
            if events_path.exists():
                for event in json.loads(events_path.read_text(encoding="utf-8")):
                    MemoryRepository.add_event(self, value["taskId"], event)
            for trajectory in (path.parent / "trajectories").glob("*.json"):
                MemoryRepository.add_trajectory_points(self, value["taskId"], trajectory.stem,
                    json.loads(trajectory.read_text(encoding="utf-8")))
        for path in self.scenes_root.glob("*.json"):
            v = json.loads(path.read_text(encoding="utf-8"))
            self.scenes[v["sceneId"]] = SceneRecord(v["sceneId"], v["sceneName"], v["siteId"],
                v.get("description"), Path(v["imagePath"]), v["imageUrl"], v["width"],
                v["height"], v["sha256"], v.get("videoId"))
        for path in self.videos_root.glob("*.json"):
            v = json.loads(path.read_text(encoding="utf-8"))
            self.videos[v["videoId"]] = VideoSource(v["videoId"], Path(v["videoPath"]),
                camera_id=v.get("cameraId"), original_file_name=v.get("fileName"),
                content_type=v.get("contentType"), size_bytes=v.get("sizeBytes"),
                sha256=v.get("sha256"), width=v.get("width"), height=v.get("height"),
                fps=v.get("fps"), frame_count=v.get("frameCount"),
                duration_seconds=v.get("durationSeconds"), captured_at=v.get("capturedAt"),
                uploaded_at=v.get("uploadedAt"), status=v.get("status", "READY"))

    def _write(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def add_video(self, source: VideoSource) -> None:
        super().add_video(source)
        self._write(self.videos_root / f"{source.video_id}.json", {
            "videoId": source.video_id, "videoPath": str(source.video_path),
            "cameraId": source.camera_id, "fileName": source.original_file_name,
            "contentType": source.content_type, "sizeBytes": source.size_bytes,
            "sha256": source.sha256, "width": source.width, "height": source.height,
            "fps": source.fps, "frameCount": source.frame_count,
            "durationSeconds": source.duration_seconds,
            "capturedAt": source.captured_at, "uploadedAt": source.uploaded_at,
            "status": source.status,
        })

    def add_scene(self, record: SceneRecord) -> None:
        super().add_scene(record)
        self._write(self.scenes_root / f"{record.scene_id}.json", {
            "sceneId": record.scene_id, "sceneName": record.scene_name,
            "siteId": record.site_id, "description": record.description,
            "imagePath": str(record.image_path), "imageUrl": record.image_url,
            "width": record.width, "height": record.height, "sha256": record.sha256,
            "videoId": record.video_id,
        })

    def add_configuration(self, configuration_id: str, value: dict[str, Any]) -> None:
        with self._file_lock:
            self._write(self.configurations_root / f"{configuration_id}.json", value)
            super().add_configuration(configuration_id, value)

    def claim_task(self, task_id: str, resource_key: str, value: dict[str, Any]) -> bool:
        with self._file_lock:
            if not super().claim_task(task_id, resource_key, value):
                return False
            task = self.get_task(task_id) or value
            task.setdefault("configurationSnapshot", self.get_configuration(task.get("configurationId", "")))
            self._write(self.tasks_root / task_id / "task.json", task)
            self._write(self.tasks_root / "index.json", list(self.tasks.values()))
            return True

    def add_task(self, task_id: str, value: dict[str, Any]) -> None:
        super().add_task(task_id, value)
        task = self.get_task(task_id) or value
        self._write(self.tasks_root / task_id / "task.json", task)
        self._write(self.tasks_root / "index.json", list(self.tasks.values()))

    def add_event(self, task_id: str, event: dict[str, Any]) -> None:
        super().add_event(task_id, event)
        events = [item for item in self.list_events() if item.get("taskId") == task_id]
        self._write(self.tasks_root / task_id / "events.json", events)

    def add_trajectory_points(
        self, task_id: str, vehicle_visit_id: str, points: list[dict[str, Any]]
    ) -> None:
        super().add_trajectory_points(task_id, vehicle_visit_id, points)
        self._write(
            self.tasks_root / task_id / "trajectories" / f"{vehicle_visit_id}.json",
            self.get_trajectory(vehicle_visit_id, limit=1_000_000),
        )

    def save_task_result(
        self, task: dict[str, Any], events: list[dict[str, Any]],
        trajectories: dict[str, list[dict[str, Any]]],
    ) -> None:
        super().save_task_result(task, events, trajectories)
        for visit_id, points in trajectories.items():
            self.add_trajectory_points(task["taskId"], visit_id, points)

    def save_sample(self, task_id: str, frame: dict[str, Any]) -> None:
        self._write(self.tasks_root / task_id / "frames" / f"{frame['frame']['seq']:012d}.json", frame)

    def query_samples(self, task_id: str, from_seq: int, limit: int) -> list[dict[str, Any]]:
        paths = sorted((self.tasks_root / task_id / "frames").glob("*.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in
                [p for p in paths if int(p.stem) >= from_seq][:limit]]
