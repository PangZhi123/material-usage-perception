"""Stable domain inputs shared by offline and real-time pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned detection box in source-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box must satisfy x1 <= x2 and y1 <= y2")

    @classmethod
    def from_sequence(cls, values: object) -> BoundingBox:
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        try:
            return cls(*(float(value) for value in values))
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox coordinates must be numeric") from exc

    @property
    def ground_point(self) -> tuple[float, float]:
        """Return the bottom-center point used as the vehicle ground position."""

        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One tracked vehicle observation at a specific video time."""

    camera_id: str
    processing_session_id: str
    frame_index: int
    video_time_seconds: float
    track_id: int
    vehicle_class: str
    bbox: BoundingBox
    confidence: float | None = None
    vehicle_key: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not self.processing_session_id.strip():
            raise ValueError("processing_session_id must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not isfinite(self.video_time_seconds) or self.video_time_seconds < 0:
            raise ValueError("video_time_seconds must be finite and non-negative")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if not self.vehicle_class.strip():
            raise ValueError("vehicle_class must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.vehicle_key is not None and not self.vehicle_key.strip():
            raise ValueError("vehicle_key must not be blank")

    @property
    def observation_key(self) -> tuple[str, str, int, int]:
        return (
            self.camera_id,
            self.processing_session_id,
            self.frame_index,
            self.track_id,
        )
