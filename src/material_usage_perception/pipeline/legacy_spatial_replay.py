"""CPU-only spatial replay from legacy ByteTrack JSON and source video."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from material_usage_perception.geometry import SpatialObservation, load_spatial_projector
from material_usage_perception.domain import VehicleStateConfig
from material_usage_perception.identity import IdentityResolvedObservation, TrackStitchConfig
from material_usage_perception.estimation import (
    MaterialEstimator,
    MaterialQuantityConfig,
    MaterialUsageAccumulator,
)
from material_usage_perception.pipeline.frame_processor import FrameProcessor
from material_usage_perception.pipeline.legacy_tracking_reader import LegacyTrackingReader


@dataclass(frozen=True, slots=True)
class LegacySpatialReplaySummary:
    processed_frame_count: int
    observation_count: int
    inside_area_observation_count: int
    distinct_track_ids: tuple[int, ...]
    distinct_vehicle_visit_ids: tuple[str, ...]
    stitched_observation_count: int
    suppressed_duplicate_observation_count: int
    output_video_path: str
    output_jsonl_path: str | None
    output_events_path: str | None
    output_material_path: str | None
    completed_visit_count: int
    pass_through_count: int
    incomplete_visit_count: int
    counted_vehicle_count: int
    estimated_usage_m3: float
    source_notice: str = (
        "Legacy bbox and track_id are reused; legacy confidence values may be inaccurate."
    )


def run_legacy_spatial_replay(
    *,
    video_path: str | Path,
    tracking_json_path: str | Path,
    scene_image_path: str | Path,
    spatial_config_path: str | Path,
    output_video_path: str | Path,
    camera_id: str,
    processing_session_id: str = "legacy_replay",
    output_jsonl_path: str | Path | None = None,
    output_events_path: str | Path | None = None,
    output_material_path: str | Path | None = None,
    max_frames: int | None = None,
    trail_length: int = 100,
    state_config: VehicleStateConfig | None = None,
    stitch_config: TrackStitchConfig | None = None,
    material_config: MaterialQuantityConfig | None = None,
) -> LegacySpatialReplaySummary:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("source video reports invalid metadata")

    scene_original = cv2.imread(str(scene_image_path))
    if scene_original is None:
        capture.release()
        raise ValueError(f"cannot load scene image: {scene_image_path}")
    projector = load_spatial_projector(spatial_config_path)
    read_result = LegacyTrackingReader(
        camera_id=camera_id,
        processing_session_id=processing_session_id,
        fps=fps,
    ).read(tracking_json_path)
    by_frame: dict[int, list] = defaultdict(list)
    for observation in read_result.observations:
        by_frame[observation.frame_index].append(observation)

    display_height = min(frame_height, scene_original.shape[0], 720)
    camera_width = max(1, round(frame_width * display_height / frame_height))
    scene_width = max(1, round(scene_original.shape[1] * display_height / scene_original.shape[0]))
    output_size = (camera_width + scene_width, display_height)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create output video: {output_video_path}")

    jsonl_file = None
    if output_jsonl_path is not None:
        output_jsonl_path = Path(output_jsonl_path)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = output_jsonl_path.open("w", encoding="utf-8")

    trails: dict[str, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )
    frame_processor = FrameProcessor(
        stitch_config or TrackStitchConfig(), state_config or VehicleStateConfig()
    )
    visit_events = []
    processed = observations_count = inside_count = stitched_count = 0
    resolved_observation_count = 0
    track_ids: set[int] = set()
    vehicle_visit_ids: set[str] = set()
    polygon = np.asarray(projector.region.vertices, dtype=np.float32)

    try:
        frame_index = 0
        while max_frames is None or frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            spatial_items: list[SpatialObservation] = []
            for observation in by_frame.get(frame_index, []):
                spatial = projector.enrich(
                    observation, runtime_frame_size=(frame_width, frame_height)
                )
                spatial_items.append(spatial)
                track_ids.add(observation.track_id)
                observations_count += 1
                inside_count += int(spatial.is_inside_area)

            processed_frame = frame_processor.process_frame(
                frame_index=frame_index,
                video_time_seconds=frame_index / fps,
                observations=spatial_items,
            )
            resolved_items = processed_frame.observations
            resolved_observation_count += len(resolved_items)
            visit_events.extend(processed_frame.emitted_events)
            for item in resolved_items:
                vehicle_visit_ids.add(item.vehicle_visit_id)
                trails[item.vehicle_visit_id].append(item.scene_ground_point)
                stitched_count += int(item.stitched)

            rendered = _render_frame(
                frame,
                scene_original,
                resolved_items,
                trails,
                polygon,
                camera_width=camera_width,
                scene_width=scene_width,
                display_height=display_height,
                frame_index=frame_index,
                video_time_seconds=frame_index / fps,
            )
            writer.write(rendered)
            if jsonl_file is not None:
                jsonl_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "video_time_seconds": frame_index / fps,
                            "resolved_observations": [asdict(item) for item in resolved_items],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if jsonl_file is not None:
            jsonl_file.close()

    visit_events.extend(frame_processor.finalize())
    if output_events_path is not None:
        output_events_path = Path(output_events_path)
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text(
            json.dumps([asdict(event) for event in visit_events], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    outcome_counts = defaultdict(int)
    for event in visit_events:
        outcome_counts[event.outcome.value] += 1

    estimator = MaterialEstimator(material_config or MaterialQuantityConfig())
    accumulator = MaterialUsageAccumulator()
    material_records = []
    for event in visit_events:
        record = estimator.estimate(event)
        accumulator.add(record)
        material_records.append(record)
    material_summaries = accumulator.summaries()
    if output_material_path is not None:
        output_material_path = Path(output_material_path)
        output_material_path.parent.mkdir(parents=True, exist_ok=True)
        output_material_path.write_text(
            json.dumps(
                {
                    "records": [asdict(record) for record in material_records],
                    "summaries": [asdict(summary) for summary in material_summaries],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    counted_vehicle_count = sum(
        summary.completed_vehicle_count for summary in material_summaries
    )
    estimated_usage_m3 = sum(
        summary.estimated_usage_m3 for summary in material_summaries
    )

    return LegacySpatialReplaySummary(
        processed_frame_count=processed,
        observation_count=observations_count,
        inside_area_observation_count=inside_count,
        distinct_track_ids=tuple(sorted(track_ids)),
        distinct_vehicle_visit_ids=tuple(sorted(vehicle_visit_ids)),
        stitched_observation_count=stitched_count,
        suppressed_duplicate_observation_count=(
            observations_count - resolved_observation_count
        ),
        output_video_path=str(output_video_path.resolve()),
        output_jsonl_path=(
            str(Path(output_jsonl_path).resolve()) if output_jsonl_path is not None else None
        ),
        output_events_path=(
            str(Path(output_events_path).resolve()) if output_events_path is not None else None
        ),
        output_material_path=(
            str(Path(output_material_path).resolve())
            if output_material_path is not None
            else None
        ),
        completed_visit_count=outcome_counts["completed"],
        pass_through_count=outcome_counts["pass_through"],
        incomplete_visit_count=outcome_counts["incomplete"],
        counted_vehicle_count=counted_vehicle_count,
        estimated_usage_m3=estimated_usage_m3,
    )


def _render_frame(
    frame: np.ndarray,
    scene_image: np.ndarray,
    observations: tuple[IdentityResolvedObservation, ...],
    trails: dict[str, deque[tuple[float, float]]],
    polygon: np.ndarray,
    *,
    camera_width: int,
    scene_width: int,
    display_height: int,
    frame_index: int,
    video_time_seconds: float,
) -> np.ndarray:
    camera = frame.copy()
    scene = scene_image.copy()
    cv2.polylines(scene, [polygon.astype(np.int32)], True, (255, 0, 255), 3)
    cv2.putText(scene, "measurement area", tuple(polygon[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    for item in observations:
        tracking = item.tracking
        bbox = tracking.bbox
        color = (0, 220, 0) if item.is_inside_area else (0, 165, 255)
        cv2.rectangle(camera, (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2)), color, 3)
        ground = tuple(round(value) for value in item.image_ground_point)
        cv2.circle(camera, ground, 7, color, -1)
        visit_label = item.vehicle_visit_id.rsplit(":", 1)[-1]
        cv2.putText(camera, f"raw={tracking.track_id} {visit_label} {'IN' if item.is_inside_area else 'OUT'}", (round(bbox.x1), max(25, round(bbox.y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

        scene_point = tuple(round(value) for value in item.scene_ground_point)
        cv2.circle(scene, scene_point, 8, color, -1)
        cv2.putText(scene, f"raw={tracking.track_id} {visit_label}", (scene_point[0] + 8, scene_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        trail = np.asarray(trails[item.vehicle_visit_id], dtype=np.int32)
        if len(trail) >= 2:
            cv2.polylines(scene, [trail], False, color, 2)

    camera = cv2.resize(camera, (camera_width, display_height))
    scene = cv2.resize(scene, (scene_width, display_height))
    combined = np.hstack((camera, scene))
    cv2.rectangle(combined, (0, 0), (combined.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(combined, f"frame={frame_index} time={video_time_seconds:.2f}s | legacy tracking replay (CPU)", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay legacy tracking in camera and scene views without YOLO/GPU.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--scene-image", required=True)
    parser.add_argument("--spatial-config", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-events")
    parser.add_argument("--output-material")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--session-id", default="legacy_replay")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trail-length", type=int, default=100)
    parser.add_argument("--enter-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=60.0)
    parser.add_argument("--exit-confirm-seconds", type=float, default=5.0)
    parser.add_argument("--lost-track-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--max-reconnect-gap-seconds", type=float, default=3.0)
    parser.add_argument("--stitch-retention-seconds", type=float, default=8.0)
    parser.add_argument("--max-scene-distance", type=float, default=120.0)
    parser.add_argument("--max-image-distance", type=float, default=500.0)
    parser.add_argument("--default-capacity-m3", type=float, default=12.0)
    parser.add_argument("--default-load-factor", type=float, default=1.0)
    parser.add_argument("--default-unload-factor", type=float, default=0.95)
    args = parser.parse_args()
    summary = run_legacy_spatial_replay(
        video_path=args.video,
        tracking_json_path=args.tracking_json,
        scene_image_path=args.scene_image,
        spatial_config_path=args.spatial_config,
        output_video_path=args.output_video,
        output_jsonl_path=args.output_jsonl,
        output_events_path=args.output_events,
        output_material_path=args.output_material,
        camera_id=args.camera_id,
        processing_session_id=args.session_id,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        state_config=VehicleStateConfig(
            enter_confirm_seconds=args.enter_confirm_seconds,
            minimum_dwell_seconds=args.minimum_dwell_seconds,
            exit_confirm_seconds=args.exit_confirm_seconds,
            lost_track_tolerance_seconds=args.lost_track_tolerance_seconds,
        ),
        stitch_config=TrackStitchConfig(
            max_reconnect_gap_seconds=args.max_reconnect_gap_seconds,
            retention_seconds=args.stitch_retention_seconds,
            max_scene_distance=args.max_scene_distance,
            max_image_distance=args.max_image_distance,
        ),
        material_config=MaterialQuantityConfig(
            default_capacity_m3=args.default_capacity_m3,
            default_load_factor=args.default_load_factor,
            default_unload_factor=args.default_unload_factor,
        ),
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
