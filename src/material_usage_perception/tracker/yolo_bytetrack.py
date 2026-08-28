"""Ultralytics YOLO + ByteTrack adapter for mixer-truck observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from material_usage_perception.domain import BoundingBox, TrackObservation


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    """Tracking output for one source-video frame, including empty frames."""

    frame_index: int
    video_time_seconds: float
    observations: tuple[TrackObservation, ...]


class YoloByteTrackTracker:
    """Run a YOLO detector and ByteTrack while exposing domain observations."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        target_class_name: str = "transporter",
        confidence_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: str | int | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not target_class_name.strip():
            raise ValueError("target_class_name must not be empty")

        self.model_path = Path(model_path)
        self.target_class_name = target_class_name
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def iter_video(
        self,
        video_path: str | Path,
        *,
        camera_id: str,
        processing_session_id: str,
        fps: float,
    ) -> Iterator[TrackingFrame]:
        """Yield standard observations in source-video order.

        Time is derived from source frame index and FPS. Ultralytics inference
        wall-clock timestamps are never used as video timestamps.
        """

        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        source_path = Path(video_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"video not found: {source_path}")

        class_names = self._class_names()
        target_class_id = self._target_class_id(class_names)
        track_kwargs: dict[str, Any] = {
            "source": str(source_path),
            "stream": True,
            "show": False,
            "tracker": self.tracker_config,
            "classes": [target_class_id],
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            track_kwargs["device"] = self.device

        results = self.model.track(**track_kwargs)
        for frame_index, result in enumerate(results):
            video_time = frame_index / fps
            observations = self._convert_result(
                result,
                frame_index=frame_index,
                video_time_seconds=video_time,
                camera_id=camera_id,
                processing_session_id=processing_session_id,
                class_names=class_names,
            )
            yield TrackingFrame(frame_index, video_time, observations)

    def _class_names(self) -> dict[int, str]:
        raw_names: Mapping[int, str] | list[str] = self.model.names
        if isinstance(raw_names, Mapping):
            return {int(key): str(value) for key, value in raw_names.items()}
        return {index: str(value) for index, value in enumerate(raw_names)}

    def _target_class_id(self, class_names: Mapping[int, str]) -> int:
        matches = [
            class_id
            for class_id, name in class_names.items()
            if name == self.target_class_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"model must contain exactly one class named "
                f"{self.target_class_name!r}; found {len(matches)}"
            )
        return matches[0]

    def _convert_result(
        self,
        result: Any,
        *,
        frame_index: int,
        video_time_seconds: float,
        camera_id: str,
        processing_session_id: str,
        class_names: Mapping[int, str],
    ) -> tuple[TrackObservation, ...]:
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return ()

        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        track_ids = boxes.id.cpu().tolist()
        confidences = (
            boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(boxes)
        )

        observations: list[TrackObservation] = []
        for bbox, class_id, track_id, confidence in zip(
            coordinates, class_ids, track_ids, confidences
        ):
            class_name = class_names[int(class_id)]
            if class_name != self.target_class_name:
                continue
            observations.append(
                TrackObservation(
                    camera_id=camera_id,
                    processing_session_id=processing_session_id,
                    frame_index=frame_index,
                    video_time_seconds=video_time_seconds,
                    track_id=int(track_id),
                    vehicle_class=class_name,
                    bbox=BoundingBox.from_sequence(bbox),
                    confidence=None if confidence is None else float(confidence),
                )
            )
        return tuple(observations)
