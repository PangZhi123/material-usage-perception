"""Formal V1 backend endpoints aligned with the 2026-08-25 contract."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import cv2
import numpy as np
from fastapi import (
    APIRouter, File, Form, HTTPException, Query, Request, UploadFile,
    WebSocket, WebSocketDisconnect,
)

from material_usage_perception.api.v1_models import (
    ApiEnvelope, CameraRegistrationRequest, ConfigurationCreateRequest,
    LocalVideoRegistrationRequest, TaskActionRequest, TaskCreateRequest,
)
from material_usage_perception.config import AlgorithmSettings, DetectionRuntimeConfig
from material_usage_perception.domain import VehicleStateConfig
from material_usage_perception.estimation import MaterialQuantityConfig
from material_usage_perception.geometry import (
    CalibrationPointPair,
    compute_camera_to_scene_homography,
    validate_normalized_polygon,
)
from material_usage_perception.repositories import (
    BackendRepository, EventQuery, SceneRecord, StatisticsQuery,
)
from material_usage_perception.sources import (
    CameraSource, CameraSourceRegistry, VideoSource, VideoSourceRegistry,
)
from material_usage_perception.tasks import TaskConflictError, TaskManager, TaskStatus


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _request_id() -> str:
    return f"req_{uuid4().hex}"


def ok(data: Any, message: str = "success") -> ApiEnvelope:
    return ApiEnvelope(
        requestId=_request_id(), success=True, code="0", message=message,
        data=data, timestamp=_now(),
    )


def _fail(status: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status,
        detail={
            "requestId": _request_id(), "success": False, "code": code,
            "message": message, "data": None, "timestamp": _now().isoformat(),
        },
    )


def _optional_timestamp(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(422, "MP0101", "capturedAt must use ISO 8601")
    if parsed.utcoffset() is None:
        _fail(422, "MP0101", "capturedAt must include a timezone")
    return parsed.isoformat()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _video_metadata(path: Path) -> tuple[int, int, float, int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            _fail(422, "MP0201", "uploaded video cannot be decoded")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, _ = capture.read()
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0 or not success:
        _fail(422, "MP0201", "video metadata is invalid")
    return width, height, fps, frame_count, frame_count / fps


def _register_video_file(
    *,
    path: Path,
    camera_id: str,
    original_file_name: str,
    content_type: str,
    digest: str,
    size_bytes: int,
    captured_at: str | None,
    repository: BackendRepository,
    registry: VideoSourceRegistry,
    destination_root: Path,
    move_into_storage: bool,
) -> tuple[VideoSource, bool]:
    matching = [
        item for item in repository.list_videos()
        if item.sha256 == digest and item.video_path.is_file()
    ]
    same_camera = next(
        (item for item in matching if item.camera_id == camera_id), None
    )
    if same_camera is not None:
        if move_into_storage:
            path.unlink(missing_ok=True)
        registry.register(same_camera)
        return same_camera, True
    existing = matching[0] if matching else None
    if existing is not None:
        if move_into_storage:
            path.unlink(missing_ok=True)
        source = VideoSource(
            video_id=f"video_{uuid4().hex[:16]}", video_path=existing.video_path,
            camera_id=camera_id, original_file_name=original_file_name,
            content_type=content_type, size_bytes=existing.size_bytes,
            sha256=digest, width=existing.width, height=existing.height,
            fps=existing.fps, frame_count=existing.frame_count,
            duration_seconds=existing.duration_seconds, captured_at=captured_at,
            uploaded_at=_now().isoformat(), status="READY",
        )
        repository.add_video(source)
        registry.register(source)
        return source, True

    width, height, fps, frame_count, duration = _video_metadata(path)
    video_id = f"video_{uuid4().hex[:16]}"
    final_path = destination_root / f"{video_id}.mp4" if move_into_storage else path
    moved = False
    try:
        if move_into_storage:
            path.replace(final_path)
            moved = True
        source = VideoSource(
            video_id=video_id, video_path=final_path, camera_id=camera_id,
            original_file_name=original_file_name, content_type=content_type,
            size_bytes=size_bytes, sha256=digest, width=width, height=height,
            fps=fps, frame_count=frame_count, duration_seconds=duration,
            captured_at=captured_at, uploaded_at=_now().isoformat(), status="READY",
        )
        repository.add_video(source)
        registry.register(source)
        return source, False
    except Exception:
        if moved:
            final_path.unlink(missing_ok=True)
        raise


def _video_payload(source: VideoSource, *, deduplicated: bool) -> dict[str, Any]:
    return {
        "videoId": source.video_id, "cameraId": source.camera_id,
        "originalFileName": source.original_file_name,
        "contentType": source.content_type, "sizeBytes": source.size_bytes,
        "sha256": source.sha256, "width": source.width, "height": source.height,
        "fps": source.fps, "frameCount": source.frame_count,
        "durationSec": source.duration_seconds, "capturedAt": source.captured_at,
        "uploadedAt": source.uploaded_at, "status": source.status,
        "deduplicated": deduplicated,
    }


def _validate_time_range(start: datetime, end: datetime) -> None:
    if start.utcoffset() is None or end.utcoffset() is None:
        _fail(422, "MP0101", "startTime and endTime must include a timezone")
    if end <= start:
        _fail(422, "MP0101", "endTime must be greater than startTime")
    if end - start > timedelta(days=31):
        _fail(422, "MP0402", "query time range must not exceed 31 days")


def create_v1_router(
    repository: BackendRepository,
    storage_root: Path,
    detection: DetectionRuntimeConfig | None = None,
    task_manager: TaskManager | None = None,
    video_registry: VideoSourceRegistry | None = None,
    camera_registry: CameraSourceRegistry | None = None,
) -> APIRouter:
    router = APIRouter()
    api_prefix = "/api/material-perception/v1"
    scene_root = storage_root / "scenes"
    video_root = storage_root / "videos"
    detection = detection or DetectionRuntimeConfig.from_environment()
    if task_manager is None or video_registry is None or camera_registry is None:
        raise ValueError("task_manager, video_registry and camera_registry are required")

    @router.post(f"{api_prefix}/cameras", response_model=ApiEnvelope)
    def register_camera(payload: CameraRegistrationRequest) -> ApiEnvelope:
        camera_registry.register(CameraSource(payload.cameraId, payload.streamUrl))
        return ok({"cameraId": payload.cameraId, "status": "ACTIVE"})

    @router.get(f"{api_prefix}/videos", response_model=ApiEnvelope)
    def registered_videos() -> ApiEnvelope:
        return ok({"items": [_video_payload(source, deduplicated=False)
                             for source in repository.list_videos()]})

    @router.get(f"{api_prefix}/tasks/{{task_id}}", response_model=ApiEnvelope)
    def task_status(task_id: str) -> ApiEnvelope:
        task = repository.get_task(task_id)
        if task is None:
            _fail(404, "MP0204", "task does not exist")
        return ok(task)

    @router.get(f"{api_prefix}/configurations/{{configuration_id}}", response_model=ApiEnvelope)
    def configuration_detail(configuration_id: str) -> ApiEnvelope:
        configuration = repository.get_configuration(configuration_id)
        if configuration is None:
            _fail(404, "MP0102", "configuration does not exist")
        return ok(configuration)

    @router.get(f"{api_prefix}/scenes/{{scene_id}}", response_model=ApiEnvelope)
    def scene_detail(scene_id: str) -> ApiEnvelope:
        scene = repository.get_scene(scene_id)
        if scene is None:
            _fail(404, "MP0102", "scene does not exist")
        return ok({"sceneId": scene.scene_id, "sceneName": scene.scene_name,
                   "imageUrl": scene.image_url, "width": scene.width,
                   "height": scene.height, "videoId": scene.video_id})

    @router.post(f"{api_prefix}/videos", response_model=ApiEnvelope, status_code=201)
    async def upload_video(
        file: UploadFile = File(...),
        cameraId: str = Form(...),
        capturedAt: str | None = Form(default=None),
    ) -> ApiEnvelope:
        if not cameraId.strip() or len(cameraId) > 64:
            _fail(422, "MP0101", "cameraId must contain between 1 and 64 characters")
        original_name = Path(file.filename or "upload.mp4").name
        content_type = (file.content_type or "application/octet-stream").lower()
        if Path(original_name).suffix.lower() != ".mp4" or content_type not in {
            "video/mp4", "application/mp4", "application/octet-stream",
        }:
            _fail(415, "MP0101", "only MP4 video is supported")
        captured_at = _optional_timestamp(capturedAt)
        maximum_bytes = int(os.getenv(
            "MATERIAL_PERCEPTION_MAX_VIDEO_BYTES", str(5 * 1024 * 1024 * 1024)
        ))
        video_root.mkdir(parents=True, exist_ok=True)
        temporary = video_root / f".upload_{uuid4().hex}.mp4"
        digest = sha256()
        size_bytes = 0
        try:
            with temporary.open("xb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size_bytes += len(chunk)
                    if size_bytes > maximum_bytes:
                        _fail(413, "MP0101", "video exceeds maximum upload size")
                    destination.write(chunk)
                    digest.update(chunk)
            if size_bytes == 0:
                _fail(422, "MP0101", "video file must not be empty")
            source, deduplicated = _register_video_file(
                path=temporary, camera_id=cameraId.strip(),
                original_file_name=original_name, content_type="video/mp4",
                digest=digest.hexdigest(), size_bytes=size_bytes,
                captured_at=captured_at, repository=repository,
                registry=video_registry, destination_root=video_root,
                move_into_storage=True,
            )
        except HTTPException:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            _fail(500, "MP0401", f"video registration failed: {exc}")
        return ok(_video_payload(source, deduplicated=deduplicated))

    @router.post(
        f"{api_prefix}/video-registrations",
        response_model=ApiEnvelope,
        status_code=201,
    )
    def register_local_video(payload: LocalVideoRegistrationRequest) -> ApiEnvelope:
        local_root = Path(os.getenv(
            "MATERIAL_PERCEPTION_LOCAL_VIDEO_ROOT", "data/videos"
        )).resolve()
        relative = Path(payload.relativePath)
        if relative.is_absolute():
            _fail(422, "MP0101", "relativePath must be relative to the configured video root")
        candidate = (local_root / relative).resolve()
        try:
            candidate.relative_to(local_root)
        except ValueError:
            _fail(422, "MP0101", "relativePath escapes the configured video root")
        if candidate.suffix.lower() != ".mp4" or not candidate.is_file():
            _fail(404, "MP0201", "local MP4 video is not readable")
        try:
            digest = _file_sha256(candidate)
            source, deduplicated = _register_video_file(
                path=candidate, camera_id=payload.cameraId,
                original_file_name=candidate.name, content_type="video/mp4",
                digest=digest, size_bytes=candidate.stat().st_size,
                captured_at=(
                    None if payload.capturedAt is None else payload.capturedAt.isoformat()
                ),
                repository=repository, registry=video_registry,
                destination_root=video_root, move_into_storage=False,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _fail(500, "MP0401", f"video registration failed: {exc}")
        return ok(_video_payload(source, deduplicated=deduplicated))

    @router.post(f"{api_prefix}/scenes", response_model=ApiEnvelope)
    async def create_scene(
        file: UploadFile = File(...),
        sceneName: str = Form(...),
        siteId: str = Form(...),
        description: str | None = Form(default=None),
        videoId: str | None = Form(default=None),
    ) -> ApiEnvelope:
        if videoId is not None and video_registry.resolve(videoId) is None:
            _fail(404, "MP0201", f"videoId is not registered: {videoId}")
        content = await file.read()
        if not content or len(content) > 20 * 1024 * 1024:
            _fail(422, "MP0101", "scene image must be between 1 byte and 20 MB")
        content_type = (file.content_type or "").lower()
        if content_type not in {"image/png", "image/jpeg"}:
            _fail(422, "MP0101", "scene image content type must be PNG or JPEG")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            _fail(422, "MP0101", "scene image must be a valid PNG or JPEG")
        scene_id = f"scene_{uuid4().hex[:16]}"
        suffix = ".png" if content_type == "image/png" else ".jpg"
        destination = scene_root / f"{scene_id}{suffix}"
        record = SceneRecord(
            scene_id, sceneName, siteId, description, destination,
            f"/files/scenes/{destination.name}", image.shape[1], image.shape[0],
            sha256(content).hexdigest(), videoId,
        )
        try:
            scene_root.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            repository.add_scene(record)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            _fail(500, "MP0401", f"scene save failed: {exc}")
        return ok({
            "sceneId": record.scene_id, "sceneName": record.scene_name,
            "imageUrl": record.image_url, "width": record.width,
            "height": record.height, "sha256": record.sha256,
            "videoId": record.video_id,
        })

    @router.post(f"{api_prefix}/configurations", response_model=ApiEnvelope)
    def create_configuration(payload: ConfigurationCreateRequest, request: Request) -> ApiEnvelope:
        try:
            scene = repository.get_scene(payload.sceneId)
        except Exception as exc:
            _fail(500, "MP0401", f"scene query failed: {exc}")
        if scene is None:
            _fail(404, "MP0102", f"scene does not exist: {payload.sceneId}")
        if payload.videoId is not None:
            video_source = video_registry.resolve(payload.videoId)
            if video_source is None:
                _fail(404, "MP0201", f"videoId is not registered: {payload.videoId}")
            if scene.video_id != payload.videoId:
                _fail(409, "MP0202", "sceneId is not bound to this videoId")
            if (
                video_source.camera_id is not None
                and video_source.camera_id != payload.cameraId
            ):
                _fail(422, "MP0201", "video cameraId does not match configuration")
        elif scene.video_id is not None:
            _fail(409, "MP0202", "video-bound scene requires videoId")
        if len(payload.workAreas) != 1:
            _fail(
                422, "MP0104",
                "current backend supports exactly one work area per configuration",
            )
        try:
            pairs = [
                CalibrationPointPair(
                    (item.cameraPoint.x / payload.cameraFrameWidth,
                     item.cameraPoint.y / payload.cameraFrameHeight),
                    (item.scenePoint.x / scene.width, item.scenePoint.y / scene.height),
                )
                for item in payload.featurePairs
            ]
            calibration = compute_camera_to_scene_homography(
                pairs,
                camera_image_size=(payload.cameraFrameWidth, payload.cameraFrameHeight),
                scene_image_size=(scene.width, scene.height),
            )
        except ValueError as exc:
            _fail(422, "MP0103", str(exc))
        if calibration.quality.max_reprojection_error_px > payload.maximumCalibrationErrorPx:
            _fail(422, "MP0105", "maximum calibration reprojection error exceeded")

        normalized_areas = []
        try:
            for area in payload.workAreas:
                vertices = tuple(
                    (point.x, point.y) if area.coordinateType == "NORMALIZED" else
                    (point.x / scene.width, point.y / scene.height)
                    for point in area.polygon
                )
                validate_normalized_polygon(vertices)
                normalized_areas.append({"areaId": area.areaId, "polygon": vertices})
        except ValueError as exc:
            _fail(422, "MP0104", str(exc))

        configuration_id = f"cfg_{uuid4().hex[:16]}"
        calibration_version = f"cal_{uuid4().hex[:16]}"
        area_version = f"area_{uuid4().hex[:16]}"
        material_version = f"mat_{uuid4().hex[:16]}"
        capacity_source = (
            "USER"
            if "defaultCapacityM3" in payload.material.model_fields_set
            else "SYSTEM_DEFAULT"
        )
        state_fields = payload.stateMachine.model_fields_set
        settings = AlgorithmSettings(
            state=VehicleStateConfig(
                enter_confirm_seconds=payload.stateMachine.entryDetectionDurationSec,
                minimum_dwell_seconds=payload.stateMachine.minOperationDurationSec,
                exit_confirm_seconds=payload.stateMachine.exitDetectionDurationSec,
                lost_track_tolerance_seconds=payload.stateMachine.lostTrackToleranceSec,
            ),
            material=MaterialQuantityConfig(
                default_capacity_m3=payload.material.defaultCapacityM3,
            ),
        )
        data = {
            "configurationId": configuration_id,
            "cameraId": payload.cameraId,
            "sceneId": payload.sceneId,
            "videoId": payload.videoId,
            "calibrationVersion": calibration_version,
            "areaVersion": area_version,
            "materialConfigVersion": material_version,
            "homography": calibration.matrix.reshape(-1).tolist(),
            "quality": {
                "inlierCount": calibration.quality.inlier_count,
                "meanErrorPx": calibration.quality.mean_reprojection_error_px,
                "maxErrorPx": calibration.quality.max_reprojection_error_px,
            },
            "status": "ACTIVE",
            "effectiveMaterial": {
                "materialType": payload.material.materialType,
                "defaultCapacityM3": payload.material.defaultCapacityM3,
                "capacitySource": capacity_source,
            },
            "effectiveStateMachine": {
                "minOperationDurationSec": payload.stateMachine.minOperationDurationSec,
                "minOperationDurationSource": (
                    "USER" if "minOperationDurationSec" in state_fields
                    else "SYSTEM_DEFAULT"
                ),
                "entryDetectionDurationSec": payload.stateMachine.entryDetectionDurationSec,
                "entryDetectionDurationSource": (
                    "USER" if "entryDetectionDurationSec" in state_fields
                    else "SYSTEM_DEFAULT"
                ),
                "exitDetectionDurationSec": payload.stateMachine.exitDetectionDurationSec,
                "exitDetectionDurationSource": (
                    "USER" if "exitDetectionDurationSec" in state_fields
                    else "SYSTEM_DEFAULT"
                ),
            },
        }
        try:
            repository.add_configuration(configuration_id, {
                **data, "areas": normalized_areas, "settings": settings.to_dict(),
                "materialType": payload.material.materialType,
                "vehicleOverrides": {
                    item.vehicleKey.strip(): item.capacityM3
                    for item in payload.material.vehicleOverrides
                },
                "cameraFrameSize": (
                    payload.cameraFrameWidth, payload.cameraFrameHeight
                ),
                "sceneFrameSize": (scene.width, scene.height),
                "modelVersion": detection.model_version,
            })
        except Exception as exc:
            _fail(500, "MP0401", f"configuration save failed: {exc}")
        return ok(data)

    @router.post(f"{api_prefix}/tasks", response_model=ApiEnvelope)
    def create_task(payload: TaskCreateRequest) -> ApiEnvelope:
        try:
            configuration = repository.get_configuration(payload.configurationId)
        except Exception as exc:
            _fail(500, "MP0401", f"configuration query failed: {exc}")
        if configuration is None or configuration.get("status") != "ACTIVE":
            _fail(409, "MP0202", "configurationId is not ACTIVE")
        video_source = None
        camera_source = None
        if payload.inputType == "CAMERA_STREAM":
            assert payload.cameraId is not None
            camera_source = camera_registry.resolve(payload.cameraId)
            if camera_source is None:
                _fail(404, "MP0201", f"cameraId is not registered: {payload.cameraId}")
            if configuration["cameraId"] != payload.cameraId:
                _fail(422, "MP0201", "cameraId does not match configuration")
            if configuration.get("videoId") is not None:
                _fail(409, "MP0202", "video configuration cannot be used for camera stream")
            model_error = detection.model_file_error()
            if model_error:
                _fail(500, "MP0301", model_error)
        else:
            assert payload.videoId is not None
            video_source = video_registry.resolve(payload.videoId)
            if video_source is None:
                _fail(404, "MP0201", f"videoId is not registered: {payload.videoId}")
            if not video_source.video_path.is_file() or (
                video_source.tracking_json_path is not None
                and not video_source.tracking_json_path.is_file()
            ):
                _fail(404, "MP0201", "registered video source is not readable")
            if video_source.tracking_json_path is None:
                model_error = detection.model_file_error()
                if model_error:
                    _fail(500, "MP0301", model_error)

        recording_started_at = payload.recordingStartedAt
        if video_source is not None and recording_started_at is None:
            raw_time = video_source.captured_at
            if raw_time:
                recording_started_at = datetime.fromisoformat(str(raw_time))
            if recording_started_at is None or recording_started_at.utcoffset() is None:
                _fail(422, "MP0101", "historical video requires recordingStartedAt with timezone")
        task_id = f"task_{uuid4().hex[:16]}"
        task = {
            "taskId": task_id, "inputType": payload.inputType,
            "cameraId": configuration["cameraId"], "videoId": payload.videoId,
            "configurationId": payload.configurationId,
            "modelVersion": configuration.get("modelVersion", detection.model_version),
            "recordingStartedAt": recording_started_at.isoformat() if recording_started_at else None,
            "options": payload.options.model_dump(),
            "status": TaskStatus.QUEUED.value,
            "createdAt": _now().isoformat(), "startedAt": None,
            "endedAt": None, "error": None, "stopRequested": False,
        }
        try:
            if camera_source is not None:
                task_manager.submit_camera(
                    task=task, source=camera_source, configuration=configuration,
                    persist_trajectory=payload.options.persistTrajectory,
                    push_interval_ms=payload.options.pushIntervalMs,
                )
            else:
                assert video_source is not None
                task_manager.submit_video(
                    task=task, source=video_source, configuration=configuration,
                    persist_trajectory=payload.options.persistTrajectory,
                    push_interval_ms=payload.options.pushIntervalMs,
                )
        except TaskConflictError as exc:
            _fail(409, "MP0203", str(exc))
        except Exception as exc:
            _fail(500, "MP0401", f"task save failed: {exc}")
        return ok({
            "taskId": task_id, "inputType": payload.inputType,
            "cameraId": configuration["cameraId"], "videoId": payload.videoId,
            "configurationId": payload.configurationId,
        })

    @router.get(f"{api_prefix}/tasks/{{task_id}}/result", response_model=ApiEnvelope)
    def task_result(task_id: str) -> ApiEnvelope:
        task = repository.get_task(task_id)
        if task is None:
            _fail(404, "MP0204", "task does not exist")
        events = [e for e in repository.list_events() if e.get("taskId") == task_id]
        return ok({"task": task, "events": events,
                   "estimatedVolumeM3": sum(e.get("estimatedVolumeM3") or 0
                       for e in events if e.get("status") == "COMPLETED")})

    @router.get(f"{api_prefix}/tasks/{{task_id}}/frames", response_model=ApiEnvelope)
    def sampled_frames(task_id: str, fromSeq: int = Query(default=0, ge=0),
                       limit: int = Query(default=100, ge=1, le=1000)) -> ApiEnvelope:
        if repository.get_task(task_id) is None:
            _fail(404, "MP0204", "task does not exist")
        reader = getattr(repository, "query_samples", None)
        items = reader(task_id, fromSeq, limit + 1) if reader else []
        return ok({"items": items[:limit], "hasMore": len(items) > limit,
                   "nextFromSeq": items[limit - 1]["frame"]["seq"] + 1 if len(items) > limit else None})

    @router.patch(f"{api_prefix}/tasks/{{task_id}}", response_model=ApiEnvelope)
    def stop_task(task_id: str, payload: TaskActionRequest) -> ApiEnvelope:
        try:
            task = task_manager.request_stop(task_id)
        except Exception as exc:
            _fail(500, "MP0401", f"task update failed: {exc}")
        if task is None:
            _fail(404, "MP0204", "task does not exist")
        return ok({"taskId": task_id, "action": payload.action})

    @router.websocket("/ws/material-perception/v1/tasks/{task_id}")
    async def task_results(websocket: WebSocket, task_id: str) -> None:
        await websocket.accept()
        task = repository.get_task(task_id)
        if task is None:
            await websocket.send_json({
                "type": "ERROR", "taskId": task_id, "code": "MP0204",
                "message": "task does not exist", "fatal": True,
            })
            await websocket.close(code=4404)
            return
        try:
            last_frame_seq = _non_negative_ws_int(
                websocket.query_params.get("lastFrameSeq"), "lastFrameSeq"
            )
            last_trajectory_seq = _non_negative_ws_int(
                websocket.query_params.get("lastTrajectorySeq"), "lastTrajectorySeq"
            )
        except ValueError as exc:
            await websocket.send_json({
                "type": "ERROR", "taskId": task_id, "code": "MP0101",
                "message": str(exc), "fatal": True,
            })
            await websocket.close(code=4400)
            return

        bus = task_manager.event_bus
        cursor = 0
        if not bus.can_replay_frame(task_id, last_frame_seq):
            await websocket.send_json({
                "type": "SNAPSHOT", "taskId": task_id,
                **bus.snapshot_data(task_id),
            })
            cursor = bus.latest_sequence(task_id)
        else:
            initial = bus.snapshot(task_id)
            for event in initial:
                await _send_realtime_event(
                    websocket, event, last_frame_seq, last_trajectory_seq
                )
                cursor = event.sequence

        terminal = {"STOPPED", "COMPLETED", "FAILED"}
        try:
            while True:
                current = repository.get_task(task_id)
                if (
                    current is not None
                    and current.get("status") in terminal
                    and cursor >= bus.latest_sequence(task_id)
                ):
                    await websocket.close(code=1000)
                    return
                events = await asyncio.to_thread(bus.wait_after, task_id, cursor, 15.0)
                if not events:
                    await websocket.send_json({
                        "type": "HEARTBEAT", "taskId": task_id,
                        "timestamp": _now().isoformat(),
                    })
                    continue
                for event in events:
                    await _send_realtime_event(
                        websocket, event, last_frame_seq, last_trajectory_seq
                    )
                    cursor = event.sequence
        except WebSocketDisconnect:
            return

    @router.get(f"{api_prefix}/events", response_model=ApiEnvelope)
    def list_events(
        startTime: datetime, endTime: datetime, cameraId: str | None = None,
        areaId: str | None = None,
        status: list[Literal[
            "ACTIVE", "COMPLETED", "PASS_THROUGH", "INCOMPLETE", "REJECTED"
        ]] | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        pageSize: int = Query(default=20, ge=1, le=100),
    ) -> ApiEnvelope:
        _validate_time_range(startTime, endTime)
        try:
            result = repository.query_events(EventQuery(
                start_time=startTime, end_time=endTime,
                camera_id=cameraId, area_id=areaId,
                statuses=tuple(status or ()), page=page, page_size=pageSize,
            ))
        except Exception as exc:
            _fail(500, "MP0401", f"event query failed: {exc}")
        return ok({
            "items": list(result.items), "page": result.page,
            "pageSize": result.page_size, "total": result.total,
        })

    @router.get(f"{api_prefix}/statistics", response_model=ApiEnvelope)
    def statistics(
        startTime: datetime, endTime: datetime, cameraId: str | None = None,
        areaId: str | None = None, materialType: str | None = None,
        groupBy: Literal["NONE", "HOUR", "DAY"] = "NONE",
    ) -> ApiEnvelope:
        _validate_time_range(startTime, endTime)
        try:
            data = repository.query_statistics(
                StatisticsQuery(
                    start_time=startTime, end_time=endTime,
                    camera_id=cameraId, area_id=areaId,
                    material_type=materialType, group_by=groupBy,
                ),
                generated_at=_now(),
            )
        except Exception as exc:
            _fail(500, "MP0401", f"statistics query failed: {exc}")
        return ok(data)

    @router.get(
        f"{api_prefix}/trajectories/{{vehicle_visit_id}}",
        response_model=ApiEnvelope,
    )
    def trajectory(
        vehicle_visit_id: str,
        coordinateType: Literal["PIXEL", "NORMALIZED", "BOTH"] = "BOTH",
        samplingMs: int = Query(default=500, ge=0, le=86_400_000),
        fromSeq: int = Query(default=0, ge=0),
        limit: int = Query(default=2000, ge=1, le=5000),
    ) -> ApiEnvelope:
        try:
            trajectory_result = repository.query_trajectory(
                vehicle_visit_id, from_seq=fromSeq, limit=limit
            )
            event = repository.get_event(vehicle_visit_id)
        except Exception as exc:
            _fail(500, "MP0401", f"trajectory query failed: {exc}")
        points = list(trajectory_result.points)
        if not points:
            _fail(404, "MP0204", "vehicleVisitId trajectory does not exist")
        if samplingMs > 0:
            sampled, last = [], None
            for point in points:
                current = datetime.fromisoformat(point["timestamp"])
                if last is None or (current - last).total_seconds() * 1000 >= samplingMs:
                    sampled.append(point)
                    last = current
            points = sampled
        points = [
            _trajectory_coordinates(point, coordinateType) for point in points
        ]
        return ok({
            "vehicleVisitId": vehicle_visit_id,
            "cameraId": event["cameraId"] if event else None,
            "sceneId": event["sceneId"] if event else None,
            "areaId": event["areaId"] if event else None,
            "versions": event["versions"] if event else {},
            "points": points,
            "nextSeq": trajectory_result.next_seq,
        })

    return router


def _non_negative_ws_int(value: str | None, name: str) -> int:
    if value is None:
        return 0
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


async def _send_realtime_event(
    websocket: WebSocket,
    event: Any,
    last_frame_seq: int,
    last_trajectory_seq: int,
) -> None:
    if event.type == "_TASK_STATUS":
        return
    if event.type == "FRAME_RESULT":
        frame_seq = int(event.data["frame"]["seq"])
        if frame_seq <= last_frame_seq:
            return
        objects = []
        for item in event.data["objects"]:
            filtered = {
                **item,
                "trajectoryDelta": [
                    point for point in item.get("trajectoryDelta", ())
                    if int(point["seq"]) > last_trajectory_seq
                ],
            }
            objects.append(filtered)
        await websocket.send_json({
            "type": "FRAME_RESULT", "taskId": event.task_id,
            "frame": event.data["frame"], "objects": objects,
        })
        return
    if event.type == "EVENT_UPDATE":
        await websocket.send_json({
            "type": "EVENT_UPDATE", "taskId": event.task_id,
            "event": event.data["event"],
        })
        return
    if event.type == "ERROR":
        await websocket.send_json({
            "type": "ERROR", "taskId": event.task_id,
            "code": event.data.get("code", "MP0303"),
            "message": event.data.get("message", "task processing failed"),
            "fatal": True,
        })


def _trajectory_coordinates(
    point: dict[str, Any], coordinate_type: str
) -> dict[str, Any]:
    common = {
        key: value for key, value in point.items()
        if key not in {"cameraAnchorPx", "scenePointPx", "scenePointNorm"}
    }
    if coordinate_type == "PIXEL":
        return {
            **common,
            "cameraAnchorPx": point.get("cameraAnchorPx"),
            "scenePointPx": point.get("scenePointPx"),
        }
    if coordinate_type == "NORMALIZED":
        return {**common, "scenePointNorm": point.get("scenePointNorm")}
    return {
        **common,
        "cameraAnchorPx": point.get("cameraAnchorPx"),
        "scenePointPx": point.get("scenePointPx"),
        "scenePointNorm": point.get("scenePointNorm"),
    }
