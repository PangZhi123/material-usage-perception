"""Camera-to-scene homography calibration and polygon validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

import cv2
import numpy as np

from .spatial import Point, Size


@dataclass(frozen=True, slots=True)
class CalibrationPointPair:
    camera_normalized: Point
    scene_normalized: Point

    def __post_init__(self) -> None:
        for name, point in (
            ("camera_normalized", self.camera_normalized),
            ("scene_normalized", self.scene_normalized),
        ):
            if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in point):
                raise ValueError(f"{name} coordinates must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    point_count: int
    inlier_count: int
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    reprojection_errors_px: tuple[float, ...]
    inlier_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    matrix: np.ndarray
    quality: CalibrationQuality
    camera_points_px: tuple[Point, ...]
    scene_points_px: tuple[Point, ...]


def normalized_to_pixel(point: Point, image_size: Size) -> Point:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be greater than zero")
    return point[0] * width, point[1] * height


def compute_camera_to_scene_homography(
    point_pairs: Sequence[CalibrationPointPair],
    *,
    camera_image_size: Size,
    scene_image_size: Size,
    ransac_reprojection_threshold_px: float = 5.0,
) -> CalibrationResult:
    """Compute a camera-pixel to scene-pixel homography and quality report."""

    if len(point_pairs) < 4:
        raise ValueError("at least four point pairs are required")
    if ransac_reprojection_threshold_px <= 0:
        raise ValueError("RANSAC threshold must be greater than zero")

    camera_points = tuple(
        normalized_to_pixel(pair.camera_normalized, camera_image_size)
        for pair in point_pairs
    )
    scene_points = tuple(
        normalized_to_pixel(pair.scene_normalized, scene_image_size)
        for pair in point_pairs
    )
    camera_array = np.asarray(camera_points, dtype=np.float64)
    scene_array = np.asarray(scene_points, dtype=np.float64)
    _validate_point_distribution(camera_array, "camera")
    _validate_point_distribution(scene_array, "scene")

    matrix, mask = cv2.findHomography(
        camera_array,
        scene_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold_px,
    )
    if matrix is None or mask is None:
        raise ValueError("homography calculation failed")
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("calculated homography is invalid or singular")

    projected = cv2.perspectiveTransform(
        camera_array.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = tuple(
        float(hypot(*(actual - expected)))
        for actual, expected in zip(projected, scene_array)
    )
    inlier_mask = tuple(bool(value) for value in mask.reshape(-1))
    inlier_errors = [
        error for error, is_inlier in zip(errors, inlier_mask) if is_inlier
    ]
    if len(inlier_errors) < 4:
        raise ValueError("fewer than four calibration points are RANSAC inliers")

    quality = CalibrationQuality(
        point_count=len(point_pairs),
        inlier_count=sum(inlier_mask),
        mean_reprojection_error_px=float(np.mean(inlier_errors)),
        max_reprojection_error_px=max(inlier_errors),
        reprojection_errors_px=errors,
        inlier_mask=inlier_mask,
    )
    return CalibrationResult(matrix, quality, camera_points, scene_points)


def validate_normalized_polygon(vertices: Sequence[Point]) -> None:
    """Reject invalid, degenerate, or self-intersecting normalized polygons."""

    if len(vertices) < 3:
        raise ValueError("polygon must contain at least three vertices")
    for point in vertices:
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in point):
            raise ValueError("polygon coordinates must be within [0, 1]")
    if len(set(vertices)) != len(vertices):
        raise ValueError("polygon must not contain duplicate vertices")

    signed_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(vertices, (*vertices[1:], vertices[0]))
    ) / 2.0
    if abs(signed_area) < 1e-8:
        raise ValueError("polygon area is too small")

    edge_count = len(vertices)
    for first in range(edge_count):
        a1, a2 = vertices[first], vertices[(first + 1) % edge_count]
        for second in range(first + 1, edge_count):
            if second in (first, (first + 1) % edge_count):
                continue
            if first == 0 and second == edge_count - 1:
                continue
            b1, b2 = vertices[second], vertices[(second + 1) % edge_count]
            if _segments_intersect(a1, a2, b1, b2):
                raise ValueError("polygon edges must not self-intersect")


def _validate_point_distribution(points: np.ndarray, label: str) -> None:
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{label} calibration points must be unique")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError(f"{label} calibration points must not be collinear")


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
