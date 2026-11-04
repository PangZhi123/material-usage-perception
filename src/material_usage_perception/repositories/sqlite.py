"""Small SQLite repository for the single-node algorithm backend."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from material_usage_perception.sources import VideoSource
from .memory import SceneRecord
from .queries import (
    EventPage, EventQuery, StatisticsQuery, TrajectoryPage,
    event_page, statistics_data, trajectory_page,
    merge_event_state,
)


class SQLiteRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.database_path, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scene (
                    scene_id TEXT PRIMARY KEY, scene_name TEXT NOT NULL,
                    site_id TEXT NOT NULL, description TEXT, image_path TEXT NOT NULL,
                    image_url TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, video_id TEXT
                );
                CREATE TABLE IF NOT EXISTS configuration (
                    configuration_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS video_asset (
                    video_id TEXT PRIMARY KEY, camera_id TEXT,
                    original_file_name TEXT, storage_path TEXT NOT NULL,
                    tracking_json_path TEXT, content_type TEXT NOT NULL,
                    size_bytes INTEGER, sha256 TEXT, width INTEGER, height INTEGER,
                    fps REAL, frame_count INTEGER, duration_seconds REAL,
                    captured_at TEXT, uploaded_at TEXT, status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_video_asset_sha256
                    ON video_asset(sha256);
                CREATE TABLE IF NOT EXISTS analysis_task (
                    task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vehicle_event (
                    task_id TEXT NOT NULL, vehicle_visit_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(task_id, vehicle_visit_id)
                );
                CREATE TABLE IF NOT EXISTS trajectory_point (
                    task_id TEXT NOT NULL, vehicle_visit_id TEXT NOT NULL,
                    seq INTEGER NOT NULL, payload_json TEXT NOT NULL,
                    PRIMARY KEY(task_id, vehicle_visit_id, seq)
                );
                """
            )
            scene_columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(scene)")
            }
            if "video_id" not in scene_columns:
                self._connection.execute("ALTER TABLE scene ADD COLUMN video_id TEXT")

    def add_video(self, source: VideoSource) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR REPLACE INTO video_asset VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    source.video_id, source.camera_id, source.original_file_name,
                    str(source.video_path),
                    None if source.tracking_json_path is None else str(source.tracking_json_path),
                    source.content_type, source.size_bytes, source.sha256,
                    source.width, source.height, source.fps, source.frame_count,
                    source.duration_seconds, source.captured_at, source.uploaded_at,
                    source.status,
                ),
            )

    def get_video(self, video_id: str) -> VideoSource | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM video_asset WHERE video_id=?", (video_id,)
            ).fetchone()
        return None if row is None else self._video_from_row(row)

    def find_video_by_sha256(self, digest: str) -> VideoSource | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM video_asset WHERE sha256=? AND status='READY'
                   ORDER BY rowid LIMIT 1""", (digest,)
            ).fetchone()
        return None if row is None else self._video_from_row(row)

    def list_videos(self) -> list[VideoSource]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM video_asset WHERE status='READY' ORDER BY rowid"
            ).fetchall()
        return [self._video_from_row(row) for row in rows]

    @staticmethod
    def _video_from_row(row: sqlite3.Row) -> VideoSource:
        return VideoSource(
            video_id=row["video_id"], video_path=Path(row["storage_path"]),
            tracking_json_path=(
                None if row["tracking_json_path"] is None
                else Path(row["tracking_json_path"])
            ),
            camera_id=row["camera_id"], original_file_name=row["original_file_name"],
            content_type=row["content_type"], size_bytes=row["size_bytes"],
            sha256=row["sha256"], width=row["width"], height=row["height"],
            fps=row["fps"], frame_count=row["frame_count"],
            duration_seconds=row["duration_seconds"], captured_at=row["captured_at"],
            uploaded_at=row["uploaded_at"], status=row["status"],
        )

    def add_scene(self, record: SceneRecord) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR REPLACE INTO scene
                   (scene_id, scene_name, site_id, description, image_path,
                    image_url, width, height, sha256, video_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.scene_id, record.scene_name, record.site_id,
                    record.description, str(record.image_path), record.image_url,
                    record.width, record.height, record.sha256, record.video_id,
                ),
            )

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scene WHERE scene_id=?", (scene_id,)
            ).fetchone()
        if row is None:
            return None
        return SceneRecord(
            row["scene_id"], row["scene_name"], row["site_id"], row["description"],
            Path(row["image_path"]), row["image_url"], row["width"], row["height"],
            row["sha256"], row["video_id"],
        )

    def add_configuration(self, configuration_id: str, value: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO configuration (configuration_id, payload_json) VALUES (?,?)",
                (configuration_id, json.dumps(value, ensure_ascii=False)),
            )

    def get_configuration(self, configuration_id: str) -> dict[str, Any] | None:
        return self._get_json("configuration", "configuration_id", configuration_id)

    def list_configurations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(row[0]) for row in self._connection.execute(
                "SELECT payload_json FROM configuration"
            ).fetchall()]

    def add_task(self, task_id: str, value: dict[str, Any]) -> None:
        with self._lock, self._connection:
            existing = self.get_task(task_id) or {}
            stored = {**existing, **value}
            self._connection.execute(
                "INSERT OR REPLACE INTO analysis_task VALUES (?,?)",
                (task_id, json.dumps(stored, ensure_ascii=False)),
            )

    def claim_task(
        self, task_id: str, resource_key: str, value: dict[str, Any]
    ) -> bool:
        """Atomically enforce one active task for the same camera resource."""

        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT payload_json FROM analysis_task"
            ).fetchall()
            if any(
                json.loads(row[0]).get("status") in {"QUEUED", "RUNNING", "STOPPING"}
                and json.loads(row[0]).get("_resourceKey") == resource_key
                for row in rows
            ):
                return False
            stored = {**value, "_resourceKey": resource_key}
            self._connection.execute(
                "INSERT INTO analysis_task (task_id, payload_json) VALUES (?,?)",
                (task_id, json.dumps(stored, ensure_ascii=False)),
            )
            return True

    def claim_single_task(self, task_id: str, value: dict[str, Any]) -> bool:
        return self.claim_task(task_id, "global", value)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._get_json("analysis_task", "task_id", task_id)

    def add_event(self, task_id: str, event: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._upsert_event_locked(task_id, event)

    def save_task_result(
        self,
        task: dict[str, Any],
        events: list[dict[str, Any]],
        trajectories: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Atomically finish a task and persist its idempotent result set."""
        with self._lock, self._connection:
            existing = self.get_task(task["taskId"]) or {}
            stored_task = {**existing, **task}
            self._connection.execute(
                "INSERT OR REPLACE INTO analysis_task VALUES (?,?)",
                (task["taskId"], json.dumps(stored_task, ensure_ascii=False)),
            )
            for event in events:
                self._upsert_event_locked(task["taskId"], event)
            rows = [
                (task["taskId"], visit_id, point["seq"], json.dumps(point, ensure_ascii=False))
                for visit_id, points in trajectories.items()
                for point in points
            ]
            self._connection.executemany(
                "INSERT OR IGNORE INTO trajectory_point VALUES (?,?,?,?)", rows
            )

    def _upsert_event_locked(
        self, task_id: str, event: dict[str, Any]
    ) -> None:
        row = self._connection.execute(
            """SELECT payload_json FROM vehicle_event
               WHERE task_id=? AND vehicle_visit_id=?""",
            (task_id, event["vehicleVisitId"]),
        ).fetchone()
        existing = None if row is None else json.loads(row[0])
        stored = merge_event_state(existing, event)
        self._connection.execute(
            "INSERT OR REPLACE INTO vehicle_event VALUES (?,?,?)",
            (
                task_id, event["vehicleVisitId"],
                json.dumps(stored, ensure_ascii=False),
            ),
        )

    def list_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM vehicle_event ORDER BY rowid"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def query_events(self, query: EventQuery) -> EventPage:
        return event_page(self.list_events(), query)

    def query_statistics(
        self, query: StatisticsQuery, *, generated_at: Any
    ) -> dict[str, Any]:
        return statistics_data(self.list_events(), query, generated_at)

    def get_event(self, vehicle_visit_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT payload_json FROM vehicle_event
                   WHERE vehicle_visit_id=? ORDER BY rowid DESC LIMIT 1""",
                (vehicle_visit_id,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def add_trajectory_points(
        self, task_id: str, vehicle_visit_id: str, points: list[dict[str, Any]]
    ) -> None:
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO trajectory_point VALUES (?,?,?,?)",
                [
                    (task_id, vehicle_visit_id, point["seq"], json.dumps(point, ensure_ascii=False))
                    for point in points
                ],
            )

    def get_trajectory(
        self, vehicle_visit_id: str, *, from_seq: int = 0, limit: int = 2000
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT payload_json FROM trajectory_point
                   WHERE vehicle_visit_id=? AND seq>=? ORDER BY seq LIMIT ?""",
                (vehicle_visit_id, from_seq, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def query_trajectory(
        self, vehicle_visit_id: str, *, from_seq: int = 0, limit: int = 2000
    ) -> TrajectoryPage:
        points = self.get_trajectory(
            vehicle_visit_id, from_seq=from_seq, limit=limit + 1
        )
        return trajectory_page(points, from_seq=from_seq, limit=limit)

    def _put_json(self, table: str, key_name: str, key: str, value: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                f"INSERT OR REPLACE INTO {table} ({key_name}, payload_json) VALUES (?,?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def _get_json(self, table: str, key_name: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT payload_json FROM {table} WHERE {key_name}=?", (key,)
            ).fetchone()
        return None if row is None else json.loads(row[0])
