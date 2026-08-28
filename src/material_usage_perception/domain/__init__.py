"""Domain models used by the perception and usage pipelines."""

from .models import BoundingBox, TrackObservation
from .vehicle_state import (
    VehicleState,
    VehicleStateConfig,
    VehicleVisitEvent,
    VehicleWorkStateMachine,
    VisitOutcome,
)

__all__ = [
    "BoundingBox",
    "TrackObservation",
    "VehicleState",
    "VehicleStateConfig",
    "VehicleVisitEvent",
    "VehicleWorkStateMachine",
    "VisitOutcome",
]
