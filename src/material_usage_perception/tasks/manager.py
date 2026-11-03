"""Asynchronous, cancellable task orchestration independent of FastAPI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from material_usage_perception.config import DetectionRuntimeConfig
from material_usage_perception.pipeline.v1_task import execute_tracking_task
from material_usage_perception.repositories import BackendRepository
from material_usage_perception.sources import CameraSource, CameraStreamReader, VideoSource
from material_usage_perception.tracker import ModelLoadError

from .events import TaskEvent, TaskEventBus
from .models import CancellationToken, TaskStatus


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


class TaskConflictError(RuntimeError):
    pass


class _TaskPersistenceFailure(RuntimeError):
    pass


class TaskManager:
    def __init__(
        self,
        repository: BackendRepository,
        detection: DetectionRuntimeConfig,
        *,
        max_workers: int = 4,
        event_bus: TaskEventBus | None = None,
    ) -> None:
        self._repository = repository
        self._detection = detection
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="material-perception"
        )
        self._event_bus = event_bus or TaskEventBus()
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = RLock()

    @property
    def event_bus(self) -> TaskEventBus:
        return self._event_bus

    def submit_video(
        self,
        *,
        task: dict[str, Any],
        source: VideoSource,
        configuration: dict[str, Any],
        persist_trajectory: bool,
        push_interval_ms: int,
    ) -> None:
        configuration = deepcopy(configuration)
        task["configurationSnapshot"] = configuration
        resource_key = "history:single"
        if not self._repository.claim_task(task["taskId"], resource_key, task):
            raise TaskConflictError("another historical video task is running or stopping; wait for it to finish")
        token = CancellationToken()
        with self._lock:
            self._tokens[task["taskId"]] = token
        self._executor.submit(
            self._run_video, task, source, configuration, persist_trajectory,
            push_interval_ms, token, None,
        )

    def submit_camera(
        self,
        *,
        task: dict[str, Any],
        source: CameraSource,
        configuration: dict[str, Any],
        persist_trajectory: bool,
        push_interval_ms: int,
    ) -> None:
        configuration = deepcopy(configuration)
        task["configurationSnapshot"] = configuration
        resource_key = f"camera:{source.camera_id}"
        if not self._repository.claim_task(task["taskId"], resource_key, task):
            raise TaskConflictError("another analysis task is already running for this camera")
        token = CancellationToken()
        with self._lock:
            self._tokens[task["taskId"]] = token
        self._executor.submit(
            self._run_video, task, None, configuration, persist_trajectory,
            push_interval_ms, token, source,
        )

    def request_stop(self, task_id: str) -> dict[str, Any] | None:
        task = self._repository.get_task(task_id)
        if task is None:
            return None
        if task["status"] in {TaskStatus.QUEUED.value, TaskStatus.RUNNING.value}:
            task.update({"status": TaskStatus.STOPPING.value, "stopRequested": True})
            self._repository.add_task(task_id, task)
            with self._lock:
                token = self._tokens.get(task_id)
            if token is not None:
                token.request_stop()
        return task

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_video(
        self,
        task: dict[str, Any],
        source: VideoSource | None,
        configuration: dict[str, Any],
        persist_trajectory: bool,
        push_interval_ms: int,
        token: CancellationToken,
        camera_source: CameraSource | None,
    ) -> None:
        task_id = task["taskId"]
        started_at = _now()
        if token.is_stop_requested():
            task.update({
                "status": TaskStatus.STOPPED.value,
                "startedAt": None,
                "endedAt": started_at.isoformat(),
                "stopRequested": True,
            })
            try:
                self._repository.save_task_result(task, [], {})
                self._publish(
                    task_id, "_TASK_STATUS", {"status": TaskStatus.STOPPED.value}
                )
            except Exception as exc:
                self._publish(task_id, "ERROR", {
                    "code": "MP0401", "message": str(exc)
                })
            with self._lock:
                self._tokens.pop(task_id, None)
            return
        task.update({"status": TaskStatus.RUNNING.value, "startedAt": started_at.isoformat()})
        try:
            self._repository.add_task(task_id, task)
        except Exception as exc:
            self._publish(task_id, "ERROR", {
                "code": "MP0401", "message": str(exc)
            })
            with self._lock:
                self._tokens.pop(task_id, None)
            return
        self._publish(task_id, "_TASK_STATUS", {"status": TaskStatus.RUNNING.value})
        realtime = _RealtimePublisher(task_id, push_interval_ms, self._event_bus)

        def on_frame(frame: dict[str, Any]) -> None:
            interval = task.get("options", {}).get("sampleEveryFrames", 10)
            if (frame["frame"]["seq"] - 1) % interval == 0:
                saver = getattr(self._repository, "save_sample", None)
                if saver is not None:
                    try:
                        saver(task_id, frame)
                    except Exception as exc:
                        raise _TaskPersistenceFailure(str(exc)) from exc
            realtime.on_frame(frame)

        def persist_and_publish_event(event: dict[str, Any]) -> None:
            # The event already contains its material estimate. Persist it before
            # notifying clients so a pushed completion is immediately queryable.
            try:
                self._repository.add_event(task_id, event)
            except Exception as exc:
                raise _TaskPersistenceFailure(str(exc)) from exc
            self._publish(task_id, "EVENT_UPDATE", {"event": event})

        try:
            if camera_source is not None:
                with CameraStreamReader(camera_source.stream_url) as camera_stream:
                    result = execute_tracking_task(
                        task_id=task_id,
                        video_path=camera_source.stream_url,
                        tracking_json_path=None,
                        configuration=configuration,
                        started_at=(datetime.fromisoformat(task["recordingStartedAt"])
                                if task.get("recordingStartedAt") else started_at),
                        model_path=self._detection.model_path,
                        confidence_threshold=self._detection.confidence_threshold,
                        cancellation_token=token,
                        frame_callback=on_frame,
                        event_callback=persist_and_publish_event,
                        camera_stream=camera_stream,
                    )
            else:
                assert source is not None
                result = execute_tracking_task(
                    task_id=task_id,
                    video_path=source.video_path,
                    tracking_json_path=source.tracking_json_path,
                    configuration=configuration,
                    started_at=(datetime.fromisoformat(task["recordingStartedAt"])
                                if task.get("recordingStartedAt") else started_at),
                    model_path=self._detection.model_path,
                    confidence_threshold=self._detection.confidence_threshold,
                    cancellation_token=token,
                    frame_callback=on_frame,
                    event_callback=persist_and_publish_event,
                )
            realtime.flush()
            final_status = (
                TaskStatus.STOPPED.value if result.stopped else TaskStatus.COMPLETED.value
            )
            task.update({
                "status": final_status,
                "endedAt": _now().isoformat(),
                "stopRequested": result.stopped,
            })
            trajectories = result.trajectories if persist_trajectory else {}
            try:
                self._repository.save_task_result(task, result.events, trajectories)
            except Exception as exc:
                raise _TaskPersistenceFailure(str(exc)) from exc
            self._publish(task_id, "_TASK_STATUS", {"status": final_status})
        except Exception as exc:  # worker boundary: persist every unexpected failure
            realtime.flush()
            task.update({
                "status": TaskStatus.FAILED.value,
                "endedAt": _now().isoformat(),
                "error": str(exc),
            })
            try:
                self._repository.add_task(task_id, task)
            except Exception:
                pass
            code = (
                "MP0401" if isinstance(exc, _TaskPersistenceFailure)
                else "MP0301" if isinstance(exc, ModelLoadError)
                else "MP0303"
            )
            self._publish(task_id, "ERROR", {"code": code, "message": str(exc)})
            self._publish(task_id, "_TASK_STATUS", {"status": TaskStatus.FAILED.value})
        finally:
            with self._lock:
                self._tokens.pop(task_id, None)

    def _publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> None:
        self._event_bus.publish(TaskEvent(task_id, event_type, _now(), data))


class _RealtimePublisher:
    """Apply source-time throttling without dropping trajectory deltas."""

    def __init__(
        self, task_id: str, push_interval_ms: int, event_bus: TaskEventBus
    ) -> None:
        self._task_id = task_id
        self._interval_seconds = push_interval_ms / 1000.0
        self._event_bus = event_bus
        self._last_emitted_time: float | None = None
        self._pending: dict[str, Any] | None = None
        self._pending_objects: dict[str, dict[str, Any]] = {}

    def on_frame(self, frame_result: dict[str, Any]) -> None:
        video_time = float(frame_result.pop("_videoTimeSeconds"))
        self._pending = {"frame": frame_result["frame"], "objects": []}
        for incoming in frame_result["objects"]:
            visit_id = incoming["vehicleVisitId"]
            existing = self._pending_objects.get(visit_id)
            trajectory = list(existing.get("trajectoryDelta", ())) if existing else []
            trajectory.extend(incoming.get("trajectoryDelta", ()))
            self._pending_objects[visit_id] = {
                **incoming, "trajectoryDelta": trajectory,
            }
        due = (
            self._last_emitted_time is None
            or self._interval_seconds == 0
            or video_time - self._last_emitted_time >= self._interval_seconds
        )
        if due:
            self._emit(video_time)

    def flush(self) -> None:
        if self._pending is not None:
            self._emit(self._last_emitted_time or 0.0)

    def _emit(self, video_time: float) -> None:
        if self._pending is None:
            return
        payload = {
            "frame": self._pending["frame"],
            "objects": list(self._pending_objects.values()),
        }
        self._event_bus.publish(
            TaskEvent(self._task_id, "FRAME_RESULT", _now(), payload)
        )
        self._pending = None
        self._pending_objects = {}
        self._last_emitted_time = video_time
