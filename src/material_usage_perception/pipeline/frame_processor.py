"""Shared causal frame processor for files, legacy results, and RTSP cameras."""

from __future__ import annotations

from dataclasses import dataclass, field

from material_usage_perception.domain import (
    VehicleStateConfig,
    VehicleVisitEvent,
    VehicleWorkStateMachine,
)
from material_usage_perception.geometry import SpatialObservation
from material_usage_perception.identity import (
    IdentityResolvedObservation,
    OnlineTrackStitcher,
    TrackStitchConfig,
)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    frame_index: int
    video_time_seconds: float
    observations: tuple[IdentityResolvedObservation, ...]
    emitted_events: tuple[VehicleVisitEvent, ...]


@dataclass(slots=True)
class FrameProcessor:
    """Resolve IDs and update work state using only present and past frames."""

    stitch_config: TrackStitchConfig = field(default_factory=TrackStitchConfig)
    state_config: VehicleStateConfig = field(default_factory=VehicleStateConfig)
    _stitcher: OnlineTrackStitcher = field(init=False)
    _state_machine: VehicleWorkStateMachine = field(init=False)

    def __post_init__(self) -> None:
        self._stitcher = OnlineTrackStitcher(self.stitch_config)
        self._state_machine = VehicleWorkStateMachine(self.state_config)

    def process_frame(
        self,
        *,
        frame_index: int,
        video_time_seconds: float,
        observations: tuple[SpatialObservation, ...] | list[SpatialObservation],
    ) -> ProcessedFrame:
        resolved = self._stitcher.process_frame(observations)
        events = self._state_machine.process_frame(
            frame_index=frame_index,
            video_time_seconds=video_time_seconds,
            observations=resolved,
        )
        return ProcessedFrame(frame_index, video_time_seconds, resolved, events)

    def finalize(self) -> tuple[VehicleVisitEvent, ...]:
        return self._state_machine.finalize()
