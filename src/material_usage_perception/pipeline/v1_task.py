"""Execute a persisted V1 historical-video task from existing tracking output."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from material_usage_perception.config import AlgorithmSettings
from material_usage_perception.domain import VehicleStateConfig
from material_usage_perception.estimation import MaterialEstimator, MaterialQuantityConfig
from material_usage_perception.geometry import HomographyProjector, PolygonRegion, SpatialProjector
from material_usage_perception.identity import TrackStitchConfig
from material_usage_perception.pipeline.frame_processor import FrameProcessor
from material_usage_perception.pipeline.legacy_tracking_reader import LegacyTrackingReader
from material_usage_perception.tracker import YoloByteTrackTracker


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    events: list[dict[str, Any]]
    trajectories: dict[str, list[dict[str, Any]]]
    stopped: bool = False


def execute_tracking_task(
    *,
    task_id: str,
    video_path: str | Path,
    tracking_json_path: str | Path | None,
    configuration: dict[str, Any],
    started_at: datetime,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.5,
    cancellation_token: Any | None = None,
    frame_callback: Callable[[dict[str, Any]], None] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TaskExecutionResult:
    """Return contract-shaped events and trajectories for one configured area."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("source video reports invalid metadata")

    camera_size = tuple(configuration["cameraFrameSize"])
    scene_size = tuple(configuration["sceneFrameSize"])
    area = configuration["areas"][0]
    scene_polygon = tuple(
        (point[0] * scene_size[0], point[1] * scene_size[1])
        for point in area["polygon"]
    )
    projector = SpatialProjector(
        HomographyProjector(
            np.asarray(configuration["homography"], dtype=np.float64).reshape(3, 3),
            camera_size, camera_size, scene_size, scene_size,
        ),
        PolygonRegion(area["areaId"], scene_polygon),
    )
    settings = _settings(configuration["settings"])
    processor = FrameProcessor(settings.stitch, settings.state)
    if tracking_json_path is not None:
        observations = LegacyTrackingReader(
            camera_id=configuration["cameraId"], processing_session_id=task_id, fps=fps
        ).read(tracking_json_path).observations
        by_frame: dict[int, list] = defaultdict(list)
        for item in observations:
            by_frame[item.frame_index].append(item)
        frame_stream = (
            (index, index / fps, by_frame.get(index, ()))
            for index in range(frame_count)
        )
    else:
        if model_path is None:
            raise ValueError("model path is required when trackingJsonPath is omitted")
        tracker = YoloByteTrackTracker(
            model_path, confidence_threshold=confidence_threshold
        )
        frame_stream = (
            (frame.frame_index, frame.video_time_seconds, frame.observations)
            for frame in tracker.iter_video(
                video_path, camera_id=configuration["cameraId"],
                processing_session_id=task_id, fps=fps,
            )
        )

    estimator = MaterialEstimator(settings.material)
    contract_events: list[dict[str, Any]] = []
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sequence: dict[str, int] = defaultdict(int)
    published_frame_seq = 0
    announced_active_visits: set[str] = set()
    stopped = False
    for frame_index, video_time_seconds, tracked_items in frame_stream:
        if cancellation_token is not None and cancellation_token.is_stop_requested():
            stopped = True
            break
        spatial = [
            projector.enrich(item, runtime_frame_size=(width, height))
            for item in tracked_items
        ]
        processed = processor.process_frame(
            frame_index=frame_index,
            video_time_seconds=video_time_seconds,
            observations=spatial,
        )
        emitted_by_visit = {
            event.vehicle_visit_id: event for event in processed.emitted_events
        }
        frame_objects = []
        for item in processed.observations:
            sequence[item.vehicle_visit_id] += 1
            state_key = (
                item.tracking.camera_id, item.tracking.processing_session_id,
                item.vehicle_visit_id, item.area_id,
            )
            scene_x, scene_y = item.scene_ground_point
            point = {
                "seq": sequence[item.vehicle_visit_id],
                "timestamp": (started_at + timedelta(seconds=item.tracking.video_time_seconds)).isoformat(),
                "cameraAnchorPx": {"x": item.image_ground_point[0], "y": item.image_ground_point[1]},
                "scenePointPx": {"x": scene_x, "y": scene_y},
                "scenePointNorm": {"x": scene_x / scene_size[0], "y": scene_y / scene_size[1]},
                "zoneState": processor._state_machine.state_for(state_key).value.upper(),
            }
            trajectories[item.vehicle_visit_id].append(point)
            bbox = item.tracking.bbox
            frame_objects.append({
                "trackId": str(item.source_track_id),
                "vehicleVisitId": item.vehicle_visit_id,
                "className": item.tracking.vehicle_class,
                "bboxPx": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "bboxNorm": [
                    bbox.x1 / width, bbox.y1 / height,
                    bbox.x2 / width, bbox.y2 / height,
                ],
                "anchorPx": {
                    "x": item.image_ground_point[0], "y": item.image_ground_point[1]
                },
                "scenePointPx": {"x": scene_x, "y": scene_y},
                "scenePointNorm": {
                    "x": scene_x / scene_size[0], "y": scene_y / scene_size[1]
                },
                "trajectoryDelta": [{
                    "seq": point["seq"], "t": point["timestamp"],
                    "x": point["scenePointNorm"]["x"],
                    "y": point["scenePointNorm"]["y"],
                }],
                "areaId": item.area_id,
                "zoneState": point["zoneState"],
                "eventStatus": (
                    emitted_by_visit[item.vehicle_visit_id].outcome.value.upper()
                    if item.vehicle_visit_id in emitted_by_visit else "ACTIVE"
                ),
                "confidence": item.tracking.confidence,
                "estimatedVolumeM3": None,
            })
        published_frame_seq += 1
        if frame_callback is not None:
            frame_callback({
                "_videoTimeSeconds": video_time_seconds,
                "frame": {
                    "seq": published_frame_seq,
                    "capturedAt": (
                        started_at + timedelta(seconds=video_time_seconds)
                    ).isoformat(),
                    "width": width,
                    "height": height,
                },
                "objects": frame_objects,
            })
        if event_callback is not None:
            for item in processed.observations:
                if item.vehicle_visit_id in announced_active_visits:
                    continue
                key = (
                    item.tracking.camera_id, item.tracking.processing_session_id,
                    item.vehicle_visit_id, item.area_id,
                )
                active_event = processor._state_machine.active_event_for(key)
                if active_event is None:
                    continue
                announced_active_visits.add(item.vehicle_visit_id)
                event_callback(_contract_event(
                    task_id, active_event, estimator, configuration, started_at
                ))
        for event in processed.emitted_events:
            contract_event = _contract_event(
                task_id, event, estimator, configuration, started_at
            )
            contract_events.append(contract_event)
            if event_callback is not None:
                event_callback(contract_event)

    for event in processor.finalize():
        contract_event = _contract_event(
            task_id, event, estimator, configuration, started_at
        )
        contract_events.append(contract_event)
        if event_callback is not None:
            event_callback(contract_event)
    return TaskExecutionResult(contract_events, dict(trajectories), stopped)


def _contract_event(
    task_id: str,
    event: Any,
    estimator: MaterialEstimator,
    configuration: dict[str, Any],
    started_at: datetime,
) -> dict[str, Any]:
    vehicle_key, vehicle_capacity = _vehicle_capacity_override(event, configuration)
    material = estimator.estimate(event, vehicle_capacity_m3=vehicle_capacity)
    return {
        "taskId": task_id,
        "vehicleVisitId": event.vehicle_visit_id,
        "trackIds": [str(value) for value in event.source_track_ids],
        "vehicleKey": vehicle_key,
        "status": event.outcome.value.upper(),
        "cameraId": event.camera_id,
        "sceneId": configuration["sceneId"],
        "areaId": event.area_id,
        "enteredAt": (
            started_at + timedelta(seconds=event.enter_time_seconds)
        ).isoformat(),
        "leftAt": None if event.exit_time_seconds is None else (
            started_at + timedelta(seconds=event.exit_time_seconds)
        ).isoformat(),
        "insideDurationSec": event.dwell_seconds,
        "estimatedVolumeM3": material.estimated_usage_m3,
        "volumeSource": _volume_source(
            material.quantity_source.value,
            has_vehicle_override=vehicle_capacity is not None,
        ),
        "materialType": configuration.get("materialType", "CONCRETE"),
        "versions": {
            "calibrationVersion": configuration["calibrationVersion"],
            "areaVersion": configuration["areaVersion"],
            "materialConfigVersion": configuration["materialConfigVersion"],
            "modelVersion": configuration.get("modelVersion", "unassigned"),
        },
        "evidence": {
            "enterFrame": event.enter_frame_index,
            "exitFrame": event.exit_frame_index,
            "sourceTrackCount": len(event.source_track_ids),
            "minConfidence": event.min_confidence,
            "boundaryCrossing": event.boundary_crossing,
            "stateTransitions": event.state_transition_count,
        },
    }


def _vehicle_capacity_override(
    event: Any, configuration: dict[str, Any]
) -> tuple[str | None, float | None]:
    overrides = configuration.get("vehicleOverrides", {})
    candidates = [
        event.vehicle_key,
        event.vehicle_visit_id,
        *(str(value) for value in event.source_track_ids),
        *(f"track:{value}" for value in event.source_track_ids),
    ]
    for candidate in candidates:
        if candidate is not None and candidate in overrides:
            return candidate, float(overrides[candidate])
    return event.vehicle_key, None


def _settings(value: dict[str, Any]) -> AlgorithmSettings:
    return AlgorithmSettings(
        state=VehicleStateConfig(**value["state"]),
        stitch=TrackStitchConfig(**value["stitch"]),
        material=MaterialQuantityConfig(**value["material"]),
    )


def _volume_source(value: str, *, has_vehicle_override: bool = False) -> str:
    if value == "not_counted":
        return "VEHICLE_CAPACITY" if has_vehicle_override else "DEFAULT_CAPACITY"
    return {
        "actual_loaded": "ACTUAL_LOAD",
        "vehicle_capacity": "VEHICLE_CAPACITY",
        "default_capacity": "DEFAULT_CAPACITY",
    }[value]
