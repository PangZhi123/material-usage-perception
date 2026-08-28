"""Estimate concrete usage from completed vehicle work events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from material_usage_perception.domain import VehicleVisitEvent, VisitOutcome


class QuantitySource(str, Enum):
    ACTUAL_LOADED = "actual_loaded"
    VEHICLE_CAPACITY = "vehicle_capacity"
    DEFAULT_CAPACITY = "default_capacity"
    NOT_COUNTED = "not_counted"


@dataclass(frozen=True, slots=True)
class MaterialQuantityConfig:
    default_capacity_m3: float = 12.0
    default_load_factor: float = 1.0
    default_unload_factor: float = 0.95

    def __post_init__(self) -> None:
        if not isfinite(self.default_capacity_m3) or self.default_capacity_m3 <= 0:
            raise ValueError("default_capacity_m3 must be finite and greater than zero")
        for name, value in (
            ("default_load_factor", self.default_load_factor),
            ("default_unload_factor", self.default_unload_factor),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class MaterialUsageRecord:
    camera_id: str
    area_id: str
    vehicle_visit_id: str
    event_outcome: VisitOutcome
    counted: bool
    estimated_usage_m3: float
    quantity_source: QuantitySource
    base_quantity_m3: float | None
    load_factor: float
    unload_factor: float
    reason: str


@dataclass(frozen=True, slots=True)
class MaterialUsageSummary:
    camera_id: str
    area_id: str
    completed_vehicle_count: int
    ignored_event_count: int
    estimated_usage_m3: float


@dataclass(frozen=True, slots=True)
class MaterialEstimator:
    config: MaterialQuantityConfig = field(default_factory=MaterialQuantityConfig)

    def estimate(
        self,
        event: VehicleVisitEvent,
        *,
        actual_loaded_m3: float | None = None,
        vehicle_capacity_m3: float | None = None,
    ) -> MaterialUsageRecord:
        if event.outcome != VisitOutcome.COMPLETED:
            return MaterialUsageRecord(
                camera_id=event.camera_id,
                area_id=event.area_id,
                vehicle_visit_id=event.vehicle_visit_id,
                event_outcome=event.outcome,
                counted=False,
                estimated_usage_m3=0.0,
                quantity_source=QuantitySource.NOT_COUNTED,
                base_quantity_m3=None,
                load_factor=0.0,
                unload_factor=0.0,
                reason=f"event outcome {event.outcome.value!r} is not countable",
            )

        actual_loaded_m3 = _positive_optional(actual_loaded_m3, "actual_loaded_m3")
        vehicle_capacity_m3 = _positive_optional(
            vehicle_capacity_m3, "vehicle_capacity_m3"
        )
        if actual_loaded_m3 is not None:
            base = actual_loaded_m3
            load_factor = 1.0
            source = QuantitySource.ACTUAL_LOADED
        elif vehicle_capacity_m3 is not None:
            base = vehicle_capacity_m3
            load_factor = self.config.default_load_factor
            source = QuantitySource.VEHICLE_CAPACITY
        else:
            base = self.config.default_capacity_m3
            load_factor = self.config.default_load_factor
            source = QuantitySource.DEFAULT_CAPACITY
        usage = base * load_factor * self.config.default_unload_factor
        return MaterialUsageRecord(
            camera_id=event.camera_id,
            area_id=event.area_id,
            vehicle_visit_id=event.vehicle_visit_id,
            event_outcome=event.outcome,
            counted=True,
            estimated_usage_m3=usage,
            quantity_source=source,
            base_quantity_m3=base,
            load_factor=load_factor,
            unload_factor=self.config.default_unload_factor,
            reason="completed vehicle work event",
        )


@dataclass(slots=True)
class MaterialUsageAccumulator:
    _records: list[MaterialUsageRecord] = field(default_factory=list, init=False)
    _counted_visit_ids: set[str] = field(default_factory=set, init=False)

    def add(self, record: MaterialUsageRecord) -> None:
        if record.counted:
            if record.vehicle_visit_id in self._counted_visit_ids:
                raise ValueError(
                    f"vehicle visit already counted: {record.vehicle_visit_id}"
                )
            self._counted_visit_ids.add(record.vehicle_visit_id)
        self._records.append(record)

    def summaries(self) -> tuple[MaterialUsageSummary, ...]:
        groups: dict[tuple[str, str], list[MaterialUsageRecord]] = {}
        for record in self._records:
            groups.setdefault((record.camera_id, record.area_id), []).append(record)
        return tuple(
            MaterialUsageSummary(
                camera_id=camera_id,
                area_id=area_id,
                completed_vehicle_count=sum(item.counted for item in records),
                ignored_event_count=sum(not item.counted for item in records),
                estimated_usage_m3=sum(
                    item.estimated_usage_m3 for item in records if item.counted
                ),
            )
            for (camera_id, area_id), records in sorted(groups.items())
        )


def _positive_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return converted
