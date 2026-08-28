"""Map tracked vehicle ground points into a scene and test an area polygon."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from material_usage_perception.domain import TrackObservation

Point = tuple[float, float]
Size = tuple[float, float]


def _size(values: Sequence[object], name: str) -> Size:
    if len(values) != 2:
        raise ValueError(f"{name} must contain width and height")
    width, height = float(values[0]), float(values[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be greater than zero")
    return width, height


@dataclass(frozen=True, slots=True)
class HomographyProjector:
    """Camera-to-scene projection with explicit calibration dimensions.

    A legacy matrix may have been calibrated on resized images. The source and
    target calibration sizes make those resize operations explicit.
    """

    matrix: np.ndarray
    source_frame_size: Size
    calibration_source_size: Size
    calibration_target_size: Size
    scene_size: Size

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("homography matrix must have shape (3, 3)")
        if not np.isfinite(matrix).all():
            raise ValueError("homography matrix must contain finite values")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography matrix must be invertible")
        for size in (
            self.source_frame_size,
            self.calibration_source_size,
            self.calibration_target_size,
            self.scene_size,
        ):
            _size(size, "projection size")
        object.__setattr__(self, "matrix", matrix)

    def project(
        self,
        point: Point,
        *,
        runtime_frame_size: Size | None = None,
        aspect_ratio_tolerance: float = 1e-3,
    ) -> Point:
        x, y = point
        if not isfinite(x) or not isfinite(y):
            raise ValueError("source point must be finite")

        source_w, source_h = self.source_frame_size
        runtime_w, runtime_h = (
            self.source_frame_size
            if runtime_frame_size is None
            else _size(runtime_frame_size, "runtime_frame_size")
        )
        source_ratio = source_w / source_h
        runtime_ratio = runtime_w / runtime_h
        if abs(source_ratio - runtime_ratio) > aspect_ratio_tolerance:
            raise ValueError(
                "runtime frame aspect ratio differs from calibration image; "
                "cropped, padded, or rotated video requires a new calibration"
            )
        calibration_w, calibration_h = self.calibration_source_size
        source_vector = np.array(
            [x * calibration_w / runtime_w, y * calibration_h / runtime_h, 1.0],
            dtype=np.float64,
        )
        mapped = self.matrix @ source_vector
        if abs(float(mapped[2])) < 1e-12:
            raise ValueError("point projects to infinity")
        mapped_x, mapped_y = mapped[0] / mapped[2], mapped[1] / mapped[2]

        target_w, target_h = self.calibration_target_size
        scene_w, scene_h = self.scene_size
        result = (
            float(mapped_x * scene_w / target_w),
            float(mapped_y * scene_h / target_h),
        )
        if not all(isfinite(value) for value in result):
            raise ValueError("projected point is not finite")
        return result


@dataclass(frozen=True, slots=True)
class PolygonRegion:
    area_id: str
    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.area_id.strip():
            raise ValueError("area_id must not be empty")
        if len(self.vertices) < 3:
            raise ValueError("polygon must contain at least three vertices")
        if not all(isfinite(value) for point in self.vertices for value in point):
            raise ValueError("polygon vertices must be finite")
        area = 0.0
        for index, (x1, y1) in enumerate(self.vertices):
            x2, y2 = self.vertices[(index + 1) % len(self.vertices)]
            area += x1 * y2 - x2 * y1
        if abs(area) < 1e-9:
            raise ValueError("polygon area must be greater than zero")

    def covers(self, point: Point, *, tolerance: float = 1e-9) -> bool:
        """Return True inside the polygon or on its boundary."""

        x, y = point
        inside = False
        for index, (x1, y1) in enumerate(self.vertices):
            x2, y2 = self.vertices[(index + 1) % len(self.vertices)]
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if (
                abs(cross) <= tolerance
                and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
                and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
            ):
                return True
            if (y1 > y) != (y2 > y):
                intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < intersection_x:
                    inside = not inside
        return inside


@dataclass(frozen=True, slots=True)
class SpatialObservation:
    tracking: TrackObservation
    image_ground_point: Point
    scene_ground_point: Point
    area_id: str
    is_inside_area: bool


@dataclass(frozen=True, slots=True)
class SpatialProjector:
    homography: HomographyProjector
    region: PolygonRegion

    def enrich(
        self,
        observation: TrackObservation,
        *,
        runtime_frame_size: Size | None = None,
    ) -> SpatialObservation:
        image_point = observation.bbox.ground_point
        scene_point = self.homography.project(
            image_point, runtime_frame_size=runtime_frame_size
        )
        return SpatialObservation(
            tracking=observation,
            image_ground_point=image_point,
            scene_ground_point=scene_point,
            area_id=self.region.area_id,
            is_inside_area=self.region.covers(scene_point),
        )


def load_spatial_projector(config_path: str | Path) -> SpatialProjector:
    """Load an explicit camera-to-scene projection and polygon YAML config."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    projection = config["projection"]
    direction = projection.get("direction")
    if direction != "camera_to_scene":
        raise ValueError("projection.direction must be 'camera_to_scene'")
    matrix_path = Path(projection["matrix_path"])
    if not matrix_path.is_absolute():
        matrix_path = config_path.parent / matrix_path
    matrix = np.load(matrix_path)

    homography = HomographyProjector(
        matrix=matrix,
        source_frame_size=_size(projection["source_frame_size"], "source_frame_size"),
        calibration_source_size=_size(
            projection["calibration_source_size"], "calibration_source_size"
        ),
        calibration_target_size=_size(
            projection["calibration_target_size"], "calibration_target_size"
        ),
        scene_size=_size(projection["scene_size"], "scene_size"),
    )
    area = config["area"]
    if area.get("coordinate_system") != "scene":
        raise ValueError("area.coordinate_system must be 'scene'")
    vertices = tuple((float(x), float(y)) for x, y in area["polygon"])
    return SpatialProjector(homography, PolygonRegion(area["area_id"], vertices))
