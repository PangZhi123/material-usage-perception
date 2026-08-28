"""Headless offline analysis using existing tracking results.

This is the lightweight acceptance path: it avoids rerunning YOLO and converts
one tracking file into stable vehicle visits and material totals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from material_usage_perception.config import AlgorithmSettings
from material_usage_perception.domain import VehicleStateConfig
from material_usage_perception.estimation import (
    MaterialEstimator,
    MaterialQuantityConfig,
    MaterialUsageAccumulator,
)
from material_usage_perception.geometry import load_spatial_projector
from material_usage_perception.pipeline.frame_processor import FrameProcessor
from material_usage_perception.pipeline.legacy_tracking_reader import LegacyTrackingReader


@dataclass(frozen=True, slots=True)
class OfflineAnalysisResult:
    camera_id: str
    processing_session_id: str
    processed_frame_count: int
    input_observation_count: int
    resolved_observation_count: int
    distinct_vehicle_count: int
    events: tuple[dict[str, Any], ...]
    material_records: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]
    settings: dict[str, Any]


def run_offline_analysis(
    *,
    video_path: str | Path,
    tracking_json_path: str | Path,
    spatial_config_path: str | Path,
    camera_id: str,
    processing_session_id: str = "offline_acceptance",
    settings: AlgorithmSettings | None = None,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
) -> OfflineAnalysisResult:
    """Analyze all frames, including empty frames needed for exit confirmation."""

    settings = settings or AlgorithmSettings()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("source video reports invalid metadata")

    read_result = LegacyTrackingReader(
        camera_id=camera_id,
        processing_session_id=processing_session_id,
        fps=fps,
    ).read(tracking_json_path)
    by_frame: dict[int, list] = defaultdict(list)
    for item in read_result.observations:
        by_frame[item.frame_index].append(item)

    last_observation_frame = max(by_frame, default=-1)
    frame_count = max(video_frame_count, last_observation_frame + 1)
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        frame_count = min(frame_count, max_frames)

    projector = load_spatial_projector(spatial_config_path)
    processor = FrameProcessor(settings.stitch, settings.state)
    events = []
    resolved_count = 0
    visit_ids: set[str] = set()
    for frame_index in range(frame_count):
        spatial = [
            projector.enrich(item, runtime_frame_size=(width, height))
            for item in by_frame.get(frame_index, ())
        ]
        processed = processor.process_frame(
            frame_index=frame_index,
            video_time_seconds=frame_index / fps,
            observations=spatial,
        )
        events.extend(processed.emitted_events)
        resolved_count += len(processed.observations)
        visit_ids.update(item.vehicle_visit_id for item in processed.observations)
    events.extend(processor.finalize())

    estimator = MaterialEstimator(settings.material)
    accumulator = MaterialUsageAccumulator()
    records = []
    for event in events:
        record = estimator.estimate(event)
        accumulator.add(record)
        records.append(record)

    result = OfflineAnalysisResult(
        camera_id=camera_id,
        processing_session_id=processing_session_id,
        processed_frame_count=frame_count,
        input_observation_count=sum(
            len(items) for index, items in by_frame.items() if index < frame_count
        ),
        resolved_observation_count=resolved_count,
        distinct_vehicle_count=len(visit_ids),
        events=tuple(_enum_safe(asdict(item)) for item in events),
        material_records=tuple(_enum_safe(asdict(item)) for item in records),
        summaries=tuple(_enum_safe(asdict(item)) for item in accumulator.summaries()),
        settings=settings.to_dict(),
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def _enum_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _enum_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(item) for item in value]
    return getattr(value, "value", value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an existing tracking JSON into vehicle events and usage"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--spatial-config", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--session-id", default="offline_acceptance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=60.0)
    parser.add_argument("--enter-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--exit-confirm-seconds", type=float, default=5.0)
    parser.add_argument("--lost-track-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--default-capacity-m3", type=float, default=12.0)
    parser.add_argument("--default-load-factor", type=float, default=1.0)
    parser.add_argument("--default-unload-factor", type=float, default=0.95)
    args = parser.parse_args()
    settings = AlgorithmSettings(
        state=VehicleStateConfig(
            enter_confirm_seconds=args.enter_confirm_seconds,
            minimum_dwell_seconds=args.minimum_dwell_seconds,
            exit_confirm_seconds=args.exit_confirm_seconds,
            lost_track_tolerance_seconds=args.lost_track_tolerance_seconds,
        ),
        material=MaterialQuantityConfig(
            default_capacity_m3=args.default_capacity_m3,
            default_load_factor=args.default_load_factor,
            default_unload_factor=args.default_unload_factor,
        ),
    )
    result = run_offline_analysis(
        video_path=args.video,
        tracking_json_path=args.tracking_json,
        spatial_config_path=args.spatial_config,
        camera_id=args.camera_id,
        processing_session_id=args.session_id,
        settings=settings,
        output_path=args.output,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
