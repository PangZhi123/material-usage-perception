"""Time-based vehicle work-state machine for one measurement area."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from material_usage_perception.geometry import SpatialObservation
    from material_usage_perception.identity import IdentityResolvedObservation


class VehicleState(str, Enum):
    OUTSIDE = "outside"
    ENTERING = "entering"
    INSIDE = "inside"
    LEAVING = "leaving"


class VisitOutcome(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PASS_THROUGH = "pass_through"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class VehicleStateConfig:
    enter_confirm_seconds: float = 2.0
    minimum_dwell_seconds: float = 60.0
    exit_confirm_seconds: float = 5.0
    lost_track_tolerance_seconds: float = 3.0

    def __post_init__(self) -> None:
        values = (
            self.enter_confirm_seconds,
            self.minimum_dwell_seconds,
            self.exit_confirm_seconds,
            self.lost_track_tolerance_seconds,
        )
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("state-machine thresholds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VehicleVisitEvent:
    camera_id: str
    processing_session_id: str
    vehicle_visit_id: str
    source_track_ids: tuple[int, ...]
    area_id: str
    vehicle_class: str
    enter_time_seconds: float
    exit_time_seconds: float | None
    dwell_seconds: float
    enter_frame_index: int
    exit_frame_index: int | None
    outcome: VisitOutcome
    vehicle_key: str | None = None
    min_confidence: float | None = None
    boundary_crossing: bool = False
    state_transition_count: int = 0


TrackAreaKey = tuple[str, str, str, str]


@dataclass(slots=True)
class _VisitContext:
    key: TrackAreaKey
    vehicle_class: str
    source_track_ids: set[int] = field(default_factory=set)
    state: VehicleState = VehicleState.OUTSIDE
    candidate_enter_time: float | None = None
    candidate_enter_frame: int | None = None
    confirmed_enter_time: float | None = None
    confirmed_enter_frame: int | None = None
    candidate_exit_time: float | None = None
    candidate_exit_frame: int | None = None
    last_seen_time: float | None = None
    last_seen_frame: int | None = None
    last_inside_time: float | None = None
    last_inside_frame: int | None = None
    vehicle_key: str | None = None
    min_confidence: float | None = None
    state_transition_count: int = 0


@dataclass(slots=True)
class VehicleWorkStateMachine:
    """Consume spatial observations in video-time order and emit visit events."""

    config: VehicleStateConfig = field(default_factory=VehicleStateConfig)
    _contexts: dict[TrackAreaKey, _VisitContext] = field(default_factory=dict, init=False)
    _last_time_seconds: float | None = field(default=None, init=False)
    _last_frame_index: int | None = field(default=None, init=False)

    def process_frame(
        self,
        *,
        frame_index: int,
        video_time_seconds: float,
        observations: tuple[SpatialObservation | IdentityResolvedObservation, ...]
        | list[SpatialObservation | IdentityResolvedObservation],
    ) -> tuple[VehicleVisitEvent, ...]:
        if frame_index < 0 or not isfinite(video_time_seconds) or video_time_seconds < 0:
            raise ValueError("frame index and video time must be non-negative")
        if self._last_time_seconds is not None and video_time_seconds < self._last_time_seconds:
            raise ValueError("frames must be processed in non-decreasing video time")

        events: list[VehicleVisitEvent] = []
        observed_keys: set[TrackAreaKey] = set()
        for observation in observations:
            tracking = observation.tracking
            if tracking.frame_index != frame_index:
                raise ValueError("observation frame does not match process_frame frame")
            vehicle_visit_id = getattr(
                observation, "vehicle_visit_id", f"raw:{tracking.track_id}"
            )
            key = (
                tracking.camera_id,
                tracking.processing_session_id,
                vehicle_visit_id,
                observation.area_id,
            )
            if key in observed_keys:
                raise ValueError("duplicate track/area observation in one frame")
            observed_keys.add(key)
            context = self._contexts.setdefault(
                key, _VisitContext(key=key, vehicle_class=tracking.vehicle_class)
            )
            event = self._update_seen(context, observation)
            if event is not None:
                events.append(event)

        for key, context in tuple(self._contexts.items()):
            if key in observed_keys or context.last_seen_time is None:
                continue
            lost_duration = video_time_seconds - context.last_seen_time
            if lost_duration <= self.config.lost_track_tolerance_seconds:
                continue
            effective_exit_time = (
                context.last_seen_time + self.config.lost_track_tolerance_seconds
            )
            event = self._update_outside(
                context,
                video_time_seconds=video_time_seconds,
                frame_index=frame_index,
                exit_started_at=effective_exit_time,
                exit_started_frame=context.last_seen_frame,
            )
            if event is not None:
                events.append(event)

        self._last_time_seconds = video_time_seconds
        self._last_frame_index = frame_index
        return tuple(events)

    def finalize(self) -> tuple[VehicleVisitEvent, ...]:
        """Close active visits at end-of-video without claiming a confirmed exit."""

        events: list[VehicleVisitEvent] = []
        for context in self._contexts.values():
            if context.state in (VehicleState.INSIDE, VehicleState.LEAVING):
                events.append(self._build_event(context, VisitOutcome.INCOMPLETE, None, None))
            self._reset(context)
        return tuple(events)

    def state_for(self, key: TrackAreaKey) -> VehicleState:
        context = self._contexts.get(key)
        return VehicleState.OUTSIDE if context is None else context.state

    def active_event_for(self, key: TrackAreaKey) -> VehicleVisitEvent | None:
        context = self._contexts.get(key)
        if context is None or context.state not in {
            VehicleState.INSIDE, VehicleState.LEAVING
        }:
            return None
        return self._build_event(context, VisitOutcome.ACTIVE, None, None)

    def _update_seen(
        self,
        context: _VisitContext,
        observation: SpatialObservation | IdentityResolvedObservation,
    ) -> VehicleVisitEvent | None:
        tracking = observation.tracking
        context.source_track_ids.add(
            int(getattr(observation, "source_track_id", tracking.track_id))
        )
        if context.vehicle_key is None:
            context.vehicle_key = tracking.vehicle_key
        if tracking.confidence is not None:
            context.min_confidence = (
                tracking.confidence
                if context.min_confidence is None
                else min(context.min_confidence, tracking.confidence)
            )
        context.last_seen_time = tracking.video_time_seconds
        context.last_seen_frame = tracking.frame_index
        if observation.is_inside_area:
            context.last_inside_time = tracking.video_time_seconds
            context.last_inside_frame = tracking.frame_index
            return self._update_inside(context, tracking.video_time_seconds, tracking.frame_index)
        return self._update_outside(
            context,
            video_time_seconds=tracking.video_time_seconds,
            frame_index=tracking.frame_index,
            exit_started_at=tracking.video_time_seconds,
            exit_started_frame=tracking.frame_index,
        )

    def _update_inside(
        self, context: _VisitContext, video_time_seconds: float, frame_index: int
    ) -> VehicleVisitEvent | None:
        if context.state == VehicleState.OUTSIDE:
            self._transition(context, VehicleState.ENTERING)
            context.candidate_enter_time = video_time_seconds
            context.candidate_enter_frame = frame_index
        elif context.state == VehicleState.LEAVING:
            self._transition(context, VehicleState.INSIDE)
            context.candidate_exit_time = None
            context.candidate_exit_frame = None

        if context.state == VehicleState.ENTERING:
            assert context.candidate_enter_time is not None
            if video_time_seconds - context.candidate_enter_time >= self.config.enter_confirm_seconds:
                self._transition(context, VehicleState.INSIDE)
                context.confirmed_enter_time = context.candidate_enter_time
                context.confirmed_enter_frame = context.candidate_enter_frame
        return None

    def _update_outside(
        self,
        context: _VisitContext,
        *,
        video_time_seconds: float,
        frame_index: int,
        exit_started_at: float,
        exit_started_frame: int | None,
    ) -> VehicleVisitEvent | None:
        if context.state == VehicleState.ENTERING:
            self._reset(context)
            return None
        if context.state == VehicleState.INSIDE:
            self._transition(context, VehicleState.LEAVING)
            context.candidate_exit_time = exit_started_at
            context.candidate_exit_frame = exit_started_frame
        if context.state != VehicleState.LEAVING:
            return None

        assert context.candidate_exit_time is not None
        if video_time_seconds - context.candidate_exit_time < self.config.exit_confirm_seconds:
            return None
        assert context.confirmed_enter_time is not None
        dwell = max(0.0, context.candidate_exit_time - context.confirmed_enter_time)
        outcome = (
            VisitOutcome.COMPLETED
            if dwell >= self.config.minimum_dwell_seconds
            else VisitOutcome.PASS_THROUGH
        )
        event = self._build_event(
            context, outcome, context.candidate_exit_time, context.candidate_exit_frame
        )
        self._reset(context)
        return event

    @staticmethod
    def _build_event(
        context: _VisitContext,
        outcome: VisitOutcome,
        exit_time: float | None,
        exit_frame: int | None,
    ) -> VehicleVisitEvent:
        assert context.confirmed_enter_time is not None
        assert context.confirmed_enter_frame is not None
        end_time = exit_time if exit_time is not None else context.last_inside_time
        dwell = max(0.0, (end_time or context.confirmed_enter_time) - context.confirmed_enter_time)
        camera_id, session_id, vehicle_visit_id, area_id = context.key
        return VehicleVisitEvent(
            camera_id=camera_id,
            processing_session_id=session_id,
            vehicle_visit_id=vehicle_visit_id,
            source_track_ids=tuple(sorted(context.source_track_ids)),
            area_id=area_id,
            vehicle_class=context.vehicle_class,
            enter_time_seconds=context.confirmed_enter_time,
            exit_time_seconds=exit_time,
            dwell_seconds=dwell,
            enter_frame_index=context.confirmed_enter_frame,
            exit_frame_index=exit_frame,
            outcome=outcome,
            vehicle_key=context.vehicle_key,
            min_confidence=context.min_confidence,
            boundary_crossing=context.confirmed_enter_time is not None,
            state_transition_count=(
                context.state_transition_count + (1 if exit_time is not None else 0)
            ),
        )

    @staticmethod
    def _transition(context: _VisitContext, state: VehicleState) -> None:
        if context.state != state:
            context.state = state
            context.state_transition_count += 1

    @staticmethod
    def _reset(context: _VisitContext) -> None:
        context.state = VehicleState.OUTSIDE
        context.candidate_enter_time = None
        context.candidate_enter_frame = None
        context.confirmed_enter_time = None
        context.confirmed_enter_frame = None
        context.candidate_exit_time = None
        context.candidate_exit_frame = None
        context.last_inside_time = None
        context.last_inside_frame = None
        context.source_track_ids.clear()
        context.vehicle_key = None
        context.min_confidence = None
        context.state_transition_count = 0
