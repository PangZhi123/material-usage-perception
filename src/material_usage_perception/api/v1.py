"""Formal V1 backend endpoints aligned with the 2026-08-25 contract."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
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
    ApiEnvelope, ConfigurationCreateRequest, TaskActionRequest, TaskCreateRequest,
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
from material_usage_perception.sources import VideoSourceRegistry
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
) -> APIRouter:
    router = APIRouter()
    api_prefix = "/api/material-perception/v1"
    scene_root = storage_root / "scenes"
    detection = detection or DetectionRuntimeConfig.from_environment()
    if task_manager is None or video_registry is None:
        raise ValueError("task_manager and video_registry are required")

    @router.post(f"{api_prefix}/scenes", response_model=ApiEnvelope)
    async def create_scene(
        file: UploadFile = File(...),
        sceneName: str = Form(...),
        siteId: str = Form(...),
        description: str | None = Form(default=None),
    ) -> ApiEnvelope:
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
            sha256(content).hexdigest(),
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
        })

    @router.post(f"{api_prefix}/configurations", response_model=ApiEnvelope)
    def create_configuration(payload: ConfigurationCreateRequest, request: Request) -> ApiEnvelope:
        try:
            scene = repository.get_scene(payload.sceneId)
        except Exception as exc:
            _fail(500, "MP0401", f"scene query failed: {exc}")
        if scene is None:
            _fail(404, "MP0102", f"scene does not exist: {payload.sceneId}")
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
        if payload.inputType == "CAMERA_STREAM":
            _fail(503, "MP0201", "camera source adapter is not configured")
        assert payload.videoId is not None
        source = video_registry.resolve(payload.videoId)
        if source is None:
            _fail(404, "MP0201", f"videoId is not registered: {payload.videoId}")
        if not source.video_path.is_file() or (
            source.tracking_json_path is not None
            and not source.tracking_json_path.is_file()
        ):
            _fail(404, "MP0201", "registered video source is not readable")
        if source.tracking_json_path is None and (
            detection.model_path is None or not detection.model_path.is_file()
        ):
            _fail(500, "MP0301", "configured detection model is not readable")

        task_id = f"task_{uuid4().hex[:16]}"
        task = {
            "taskId": task_id, "inputType": payload.inputType,
            "cameraId": None, "videoId": payload.videoId,
            "configurationId": payload.configurationId,
            "modelVersion": configuration.get("modelVersion", detection.model_version),
            "options": payload.options.model_dump(),
            "status": TaskStatus.QUEUED.value,
            "createdAt": _now().isoformat(), "startedAt": None,
            "endedAt": None, "error": None, "stopRequested": False,
        }
        try:
            task_manager.submit_video(
                task=task, source=source, configuration=configuration,
                persist_trajectory=payload.options.persistTrajectory,
                push_interval_ms=payload.options.pushIntervalMs,
            )
        except TaskConflictError:
            _fail(409, "MP0203", "another analysis task is already running")
        except Exception as exc:
            _fail(500, "MP0401", f"task save failed: {exc}")
        return ok({
            "taskId": task_id, "inputType": payload.inputType,
            "cameraId": None, "videoId": payload.videoId,
            "configurationId": payload.configurationId,
        })

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
