"""Adapter for tracking JSON files produced by the former ML prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from material_usage_perception.domain import BoundingBox, TrackObservation


LEGACY_MODEL_CLASS_NAMES: dict[int, str] = {
    0: "heavy_equipment-4",
    1: "construction_worker",
    2: "heavy_equipment-1",
    3: "heavy_equipment-3",
    4: "casting_pipe",
    5: "heavy_equipment-2",
    6: "object-1",
    7: "forklift",
    8: "transporter",
    9: "object-2",
}


@dataclass(frozen=True, slots=True)
class LegacyTrackingSummary:
    frame_count: int
    observation_count: int
    distinct_track_ids: tuple[int, ...]
    first_video_time_seconds: float | None
    last_video_time_seconds: float | None
    skipped_object_count: int


@dataclass(frozen=True, slots=True)
class LegacyTrackingReadResult:
    observations: tuple[TrackObservation, ...]
    summary: LegacyTrackingSummary


class LegacyTrackingReader:
    """Convert a legacy frame/object JSON document into stable observations.

    Legacy wall-clock ``timestamp`` values are deliberately ignored because
    they describe inference time, not the position in the source video.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        processing_session_id: str,
        fps: float,
        class_names: Mapping[int, str] = LEGACY_MODEL_CLASS_NAMES,
        target_classes: frozenset[str] = frozenset({"transporter"}),
        strict: bool = False,
    ) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not processing_session_id.strip():
            raise ValueError("processing_session_id must not be empty")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if not target_classes:
            raise ValueError("target_classes must not be empty")

        self.camera_id = camera_id
        self.processing_session_id = processing_session_id
        self.fps = float(fps)
        self.class_names = dict(class_names)
        self.target_classes = target_classes
        self.strict = strict

    def read(self, path: str | Path) -> LegacyTrackingReadResult:
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, list):
            raise ValueError("legacy tracking JSON root must be a list")

        observations: list[TrackObservation] = []
        skipped = 0
        frame_indexes: list[int] = []

        for frame_record in document:
            try:
                frame_index, objects = self._parse_frame(frame_record)
            except (TypeError, ValueError, KeyError):
                if self.strict:
                    raise
                skipped += 1
                continue

            frame_indexes.append(frame_index)
            for raw_object in objects:
                try:
                    observation = self._parse_object(frame_index, raw_object)
                except (TypeError, ValueError, KeyError):
                    if self.strict:
                        raise
                    skipped += 1
                    continue
                if observation is not None:
                    observations.append(observation)

        track_ids = tuple(sorted({item.track_id for item in observations}))
        times = [item.video_time_seconds for item in observations]
        summary = LegacyTrackingSummary(
            frame_count=len(set(frame_indexes)),
            observation_count=len(observations),
            distinct_track_ids=track_ids,
            first_video_time_seconds=min(times) if times else None,
            last_video_time_seconds=max(times) if times else None,
            skipped_object_count=skipped,
        )
        return LegacyTrackingReadResult(tuple(observations), summary)

    @staticmethod
    def _parse_frame(frame_record: object) -> tuple[int, list[object]]:
        if not isinstance(frame_record, dict):
            raise TypeError("frame record must be an object")
        frame_index = int(frame_record["frame"])
        if frame_index < 0:
            raise ValueError("frame index must be non-negative")
        objects = frame_record.get("objects", [])
        if not isinstance(objects, list):
            raise TypeError("frame objects must be a list")
        return frame_index, objects

    def _parse_object(
        self, frame_index: int, raw_object: object
    ) -> TrackObservation | None:
        if not isinstance(raw_object, dict):
            raise TypeError("tracked object must be an object")

        raw_class = raw_object["class"]
        if isinstance(raw_class, str) and not raw_class.isdigit():
            class_name = raw_class
        else:
            class_id = int(raw_class)
            if class_id not in self.class_names:
                raise ValueError(f"unknown legacy class id: {class_id}")
            class_name = self.class_names[class_id]
        if class_name not in self.target_classes:
            return None

        confidence_value = raw_object.get("confidence")
        confidence = (
            None if confidence_value is None else float(confidence_value)
        )
        return TrackObservation(
            camera_id=self.camera_id,
            processing_session_id=self.processing_session_id,
            frame_index=frame_index,
            video_time_seconds=frame_index / self.fps,
            track_id=int(raw_object["id"]),
            vehicle_class=class_name,
            bbox=BoundingBox.from_sequence(raw_object["bbox"]),
            confidence=confidence,
            vehicle_key=(
                str(raw_object.get("vehicleKey") or raw_object.get("plateNumber")).strip()
                if raw_object.get("vehicleKey") or raw_object.get("plateNumber")
                else None
            ),
        )
