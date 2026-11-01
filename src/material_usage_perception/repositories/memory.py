"""Thread-safe development repository for V1 backend integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from material_usage_perception.sources import VideoSource

from .queries import (
    EventPage, EventQuery, StatisticsQuery, TrajectoryPage,
    event_page, statistics_data, trajectory_page,
    merge_event_state,
)


@dataclass(frozen=True, slots=True)
class SceneRecord:
    scene_id: str
    scene_name: str
    site_id: str
    description: str | None
    image_path: Path
    image_url: str
    width: int
    height: int
    sha256: str
    video_id: str | None = None


class MemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self.scenes: dict[str, SceneRecord] = {}
        self.configurations: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.events: dict[tuple[str, str], dict[str, Any]] = {}
        self.trajectories: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.videos: dict[str, VideoSource] = {}

    def add_video(self, source: VideoSource) -> None:
        with self._lock:
            self.videos[source.video_id] = source

    def get_video(self, video_id: str) -> VideoSource | None:
        with self._lock:
            return self.videos.get(video_id)

    def find_video_by_sha256(self, digest: str) -> VideoSource | None:
        with self._lock:
            return next((item for item in self.videos.values() if item.sha256 == digest), None)

    def list_videos(self) -> list[VideoSource]:
        with self._lock:
            return list(self.videos.values())

    def add_scene(self, record: SceneRecord) -> None:
        with self._lock:
            self.scenes[record.scene_id] = record

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        with self._lock:
            return self.scenes.get(scene_id)

    def add_configuration(self, configuration_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.configurations[configuration_id] = value

    def get_configuration(self, configuration_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.configurations.get(configuration_id)

    def claim_task(
        self, task_id: str, resource_key: str, value: dict[str, Any]
    ) -> bool:
        with self._lock:
            if any(
                item.get("status") in {"QUEUED", "RUNNING", "STOPPING"}
                and item.get("_resourceKey") == resource_key
                for item in self.tasks.values()
            ):
                return False
            self.tasks[task_id] = {**value, "_resourceKey": resource_key}
            return True

    def claim_single_task(self, task_id: str, value: dict[str, Any]) -> bool:
        return self.claim_task(task_id, "global", value)

    def add_task(self, task_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            existing = self.tasks.get(task_id, {})
            self.tasks[task_id] = {**existing, **value}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self.tasks.get(task_id)
            return None if value is None else dict(value)

    def add_event(self, task_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            key = (task_id, event["vehicleVisitId"])
            self.events[key] = merge_event_state(self.events.get(key), event)

    def list_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events.values())

    def query_events(self, query: EventQuery) -> EventPage:
        return event_page(self.list_events(), query)

    def query_statistics(
        self, query: StatisticsQuery, *, generated_at: Any
    ) -> dict[str, Any]:
        return statistics_data(self.list_events(), query, generated_at)

    def get_event(self, vehicle_visit_id: str) -> dict[str, Any] | None:
        with self._lock:
            matches = [
                event for (_, visit_id), event in self.events.items()
                if visit_id == vehicle_visit_id
            ]
        return matches[-1] if matches else None

    def add_trajectory_points(
        self, task_id: str, vehicle_visit_id: str, points: list[dict[str, Any]]
    ) -> None:
        with self._lock:
            for point in points:
                self.trajectories.setdefault(
                    (task_id, vehicle_visit_id, point["seq"]), point
                )

    def get_trajectory(
        self, vehicle_visit_id: str, *, from_seq: int = 0, limit: int = 2000
    ) -> list[dict[str, Any]]:
        with self._lock:
            values = [
                point for (_, visit_id, seq), point in self.trajectories.items()
                if visit_id == vehicle_visit_id and seq >= from_seq
            ]
        return sorted(values, key=lambda item: item["seq"])[:limit]

    def query_trajectory(
        self, vehicle_visit_id: str, *, from_seq: int = 0, limit: int = 2000
    ) -> TrajectoryPage:
        points = self.get_trajectory(
            vehicle_visit_id, from_seq=from_seq, limit=limit + 1
        )
        return trajectory_page(points, from_seq=from_seq, limit=limit)

    def save_task_result(
        self,
        task: dict[str, Any],
        events: list[dict[str, Any]],
        trajectories: dict[str, list[dict[str, Any]]],
    ) -> None:
        """One lock is the in-memory equivalent of the future MySQL transaction."""
        with self._lock:
            self.add_task(task["taskId"], task)
            for event in events:
                self.add_event(task["taskId"], event)
            for visit_id, points in trajectories.items():
                self.add_trajectory_points(task["taskId"], visit_id, points)
