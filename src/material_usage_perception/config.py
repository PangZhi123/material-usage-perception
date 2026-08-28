"""Validated runtime settings shared by the API and offline jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

from material_usage_perception.domain import VehicleStateConfig
from material_usage_perception.estimation import MaterialQuantityConfig
from material_usage_perception.identity import TrackStitchConfig


@dataclass(frozen=True, slots=True)
class AlgorithmSettings:
    """Operator-editable algorithm thresholds for one processing job."""

    state: VehicleStateConfig = field(default_factory=VehicleStateConfig)
    stitch: TrackStitchConfig = field(default_factory=TrackStitchConfig)
    material: MaterialQuantityConfig = field(default_factory=MaterialQuantityConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AlgorithmSettings":
        value = value or {}
        unknown = set(value) - {"state", "stitch", "material"}
        if unknown:
            raise ValueError(f"unknown algorithm setting groups: {sorted(unknown)}")
        return cls(
            state=VehicleStateConfig(**dict(value.get("state", {}))),
            stitch=TrackStitchConfig(**dict(value.get("stitch", {}))),
            material=MaterialQuantityConfig(**dict(value.get("material", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionRuntimeConfig:
    """Deployment-owned model location; the API never returns model_path."""

    model_path: Path | None = None
    model_version: str = "unassigned"
    confidence_threshold: float = 0.5

    @classmethod
    def from_environment(cls) -> "DetectionRuntimeConfig":
        raw_path = os.getenv("MATERIAL_PERCEPTION_MODEL_PATH")
        confidence = float(os.getenv("MATERIAL_PERCEPTION_CONFIDENCE", "0.5"))
        if not 0 <= confidence <= 1:
            raise ValueError("MATERIAL_PERCEPTION_CONFIDENCE must be within [0, 1]")
        return cls(
            model_path=Path(raw_path) if raw_path else None,
            model_version=os.getenv("MATERIAL_PERCEPTION_MODEL_VERSION", "unassigned"),
            confidence_threshold=confidence,
        )
