"""Offline video entry point for the production detection/tracking path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from material_usage_perception.config import AlgorithmSettings
from material_usage_perception.estimation import MaterialEstimator, MaterialUsageAccumulator
from material_usage_perception.geometry import load_spatial_projector
from material_usage_perception.pipeline.frame_processor import FrameProcessor
from material_usage_perception.tracker import YoloByteTrackTracker


@dataclass(frozen=True, slots=True)
class OfflineTrackingSummary:
    camera_id: str
    processing_session_id: str
    fps: float
    frame_width: int
    frame_height: int
    processed_frame_count: int
    observation_count: int
    distinct_track_ids: tuple[int, ...]
    inside_area_observation_count: int
    output_path: str
    distinct_vehicle_visit_count: int = 0
    event_count: int = 0
    completed_vehicle_count: int = 0
    estimated_usage_m3: float = 0.0
    output_events_path: str | None = None
    output_material_path: str | None = None


def read_video_metadata(video_path: str | Path) -> tuple[float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"video reports invalid FPS: {fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"video reports invalid frame size: {width}x{height}")
    return fps, (width, height)


def read_video_fps(video_path: str | Path) -> float:
    """Backward-compatible helper for callers that only need FPS."""

    return read_video_metadata(video_path)[0]


def run_offline_tracking(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    camera_id: str,
    processing_session_id: str | None = None,
    confidence_threshold: float = 0.5,
    device: str | int | None = None,
    spatial_config_path: str | Path | None = None,
    output_events_path: str | Path | None = None,
    output_material_path: str | Path | None = None,
    settings: AlgorithmSettings | None = None,
) -> OfflineTrackingSummary:
    """Run the production offline chain and optionally write events and usage."""

    video_path = Path(video_path)
    output_path = Path(output_path)
    session_id = processing_session_id or f"offline_{uuid4().hex}"
    fps, runtime_frame_size = read_video_metadata(video_path)
    tracker = YoloByteTrackTracker(
        model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    spatial_projector = (
        load_spatial_projector(spatial_config_path)
        if spatial_config_path is not None
        else None
    )
    settings = settings or AlgorithmSettings()
    frame_processor = FrameProcessor(settings.stitch, settings.state)
    visit_events = []
    vehicle_visit_ids: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    observation_count = 0
    inside_area_observation_count = 0
    track_ids: set[int] = set()

    with output_path.open("w", encoding="utf-8") as output:
        for frame in tracker.iter_video(
            video_path,
            camera_id=camera_id,
            processing_session_id=session_id,
            fps=fps,
        ):
            frame_count += 1
            observation_count += len(frame.observations)
            track_ids.update(item.track_id for item in frame.observations)
            spatial_observations = (
                [
                    spatial_projector.enrich(
                        item, runtime_frame_size=runtime_frame_size
                    )
                    for item in frame.observations
                ]
                if spatial_projector is not None
                else []
            )
            inside_area_observation_count += sum(
                item.is_inside_area for item in spatial_observations
            )
            record = {
                "frame_index": frame.frame_index,
                "video_time_seconds": frame.video_time_seconds,
                "observations": [asdict(item) for item in frame.observations],
            }
            if spatial_projector is not None:
                processed = frame_processor.process_frame(
                    frame_index=frame.frame_index,
                    video_time_seconds=frame.video_time_seconds,
                    observations=spatial_observations,
                )
                visit_events.extend(processed.emitted_events)
                vehicle_visit_ids.update(
                    item.vehicle_visit_id for item in processed.observations
                )
                record["spatial_observations"] = [
                    asdict(item) for item in spatial_observations
                ]
                record["resolved_observations"] = [
                    _enum_safe(asdict(item)) for item in processed.observations
                ]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    if spatial_projector is not None:
        visit_events.extend(frame_processor.finalize())
    if (
        output_events_path is not None or output_material_path is not None
    ) and spatial_projector is None:
        raise ValueError("event/material outputs require spatial_config_path")

    event_destination = _write_json(output_events_path, [
        _enum_safe(asdict(event)) for event in visit_events
    ])
    estimator = MaterialEstimator(settings.material)
    accumulator = MaterialUsageAccumulator()
    material_records = []
    for event in visit_events:
        material_record = estimator.estimate(event)
        accumulator.add(material_record)
        material_records.append(material_record)
    summaries = accumulator.summaries()
    material_destination = _write_json(
        output_material_path,
        {
            "records": [_enum_safe(asdict(item)) for item in material_records],
            "summaries": [_enum_safe(asdict(item)) for item in summaries],
        },
    )

    return OfflineTrackingSummary(
        camera_id=camera_id,
        processing_session_id=session_id,
        fps=fps,
        frame_width=runtime_frame_size[0],
        frame_height=runtime_frame_size[1],
        processed_frame_count=frame_count,
        observation_count=observation_count,
        distinct_track_ids=tuple(sorted(track_ids)),
        inside_area_observation_count=inside_area_observation_count,
        output_path=str(output_path.resolve()),
        distinct_vehicle_visit_count=len(vehicle_visit_ids),
        event_count=len(visit_events),
        completed_vehicle_count=sum(item.completed_vehicle_count for item in summaries),
        estimated_usage_m3=sum(item.estimated_usage_m3 for item in summaries),
        output_events_path=event_destination,
        output_material_path=material_destination,
    )


def _enum_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _enum_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(item) for item in value]
    return getattr(value, "value", value)


def _write_json(path: str | Path | None, value: Any) -> str | None:
    if path is None:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(destination.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mixer-truck detection and ByteTrack on a video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device")
    parser.add_argument("--output-events")
    parser.add_argument("--output-material")
    parser.add_argument("--minimum-dwell-seconds", type=float, default=60.0)
    parser.add_argument("--enter-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--exit-confirm-seconds", type=float, default=5.0)
    parser.add_argument("--lost-track-tolerance-seconds", type=float, default=3.0)
    parser.add_argument(
        "--spatial-config",
        help="YAML containing camera-to-scene homography dimensions and area polygon",
    )
    args = parser.parse_args()

    summary = run_offline_tracking(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        camera_id=args.camera_id,
        processing_session_id=args.session_id,
        confidence_threshold=args.confidence,
        device=args.device,
        spatial_config_path=args.spatial_config,
        output_events_path=args.output_events,
        output_material_path=args.output_material,
        settings=AlgorithmSettings.from_mapping(
            {
                "state": {
                    "minimum_dwell_seconds": args.minimum_dwell_seconds,
                    "enter_confirm_seconds": args.enter_confirm_seconds,
                    "exit_confirm_seconds": args.exit_confirm_seconds,
                    "lost_track_tolerance_seconds": args.lost_track_tolerance_seconds,
                }
            }
        ),
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
