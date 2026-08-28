"""Coordinate projection and construction-area geometry."""

from .spatial import (
    Point,
    Size,
    HomographyProjector,
    PolygonRegion,
    SpatialObservation,
    SpatialProjector,
    load_spatial_projector,
)
from .calibration import (
    CalibrationPointPair,
    CalibrationQuality,
    CalibrationResult,
    compute_camera_to_scene_homography,
    normalized_to_pixel,
    validate_normalized_polygon,
)

__all__ = [
    "Point",
    "Size",
    "HomographyProjector",
    "PolygonRegion",
    "SpatialObservation",
    "SpatialProjector",
    "load_spatial_projector",
    "CalibrationPointPair",
    "CalibrationQuality",
    "CalibrationResult",
    "compute_camera_to_scene_homography",
    "normalized_to_pixel",
    "validate_normalized_polygon",
]
