"""Causal, bounded-memory repair of short tracker-ID switches."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

from material_usage_perception.geometry import Point, SpatialObservation


@dataclass(frozen=True, slots=True)
class TrackStitchConfig:
    max_reconnect_gap_seconds: float = 3.0
    retention_seconds: float = 8.0
    max_scene_distance: float = 120.0
    max_image_distance: float = 500.0
    max_prediction_seconds: float = 1.5
    duplicate_iou_threshold: float = 0.5
    duplicate_containment_threshold: float = 0.8

    def __post_init__(self) -> None:
        values = (
            self.max_reconnect_gap_seconds,
            self.retention_seconds,
            self.max_scene_distance,
            self.max_image_distance,
            self.max_prediction_seconds,
            self.duplicate_iou_threshold,
            self.duplicate_containment_threshold,
        )
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("track-stitch thresholds must be finite and non-negative")
        if self.retention_seconds < self.max_reconnect_gap_seconds:
            raise ValueError("retention_seconds must cover max_reconnect_gap_seconds")
        if not 0 <= self.duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be within [0, 1]")
        if not 0 <= self.duplicate_containment_threshold <= 1:
            raise ValueError("duplicate_containment_threshold must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class IdentityResolvedObservation:
    spatial: SpatialObservation
    vehicle_visit_id: str
    source_track_id: int
    stitched: bool
    stitch_score: float | None = None

    @property
    def tracking(self):
        return self.spatial.tracking

    @property
    def area_id(self) -> str:
        return self.spatial.area_id

    @property
    def is_inside_area(self) -> bool:
        return self.spatial.is_inside_area

    @property
    def scene_ground_point(self) -> Point:
        return self.spatial.scene_ground_point

    @property
    def image_ground_point(self) -> Point:
        return self.spatial.image_ground_point


IdentityScope = tuple[str, str, str]
RawTrackKey = tuple[IdentityScope, int]


@dataclass(slots=True)
class _VisitMemory:
    vehicle_visit_id: str
    scope: IdentityScope
    vehicle_class: str
    source_track_ids: set[int]
    current_raw_track_id: int
    last_time_seconds: float
    last_frame_index: int
    last_scene_point: Point
    last_image_point: Point
    previous_scene_point: Point | None = None
    previous_time_seconds: float | None = None

    def predicted_scene_point(self, at_time: float, max_seconds: float) -> Point:
        if self.previous_scene_point is None or self.previous_time_seconds is None:
            return self.last_scene_point
        elapsed = self.last_time_seconds - self.previous_time_seconds
        if elapsed <= 1e-9:
            return self.last_scene_point
        prediction_time = min(max(0.0, at_time - self.last_time_seconds), max_seconds)
        velocity_x = (self.last_scene_point[0] - self.previous_scene_point[0]) / elapsed
        velocity_y = (self.last_scene_point[1] - self.previous_scene_point[1]) / elapsed
        return (
            self.last_scene_point[0] + velocity_x * prediction_time,
            self.last_scene_point[1] + velocity_y * prediction_time,
        )

    def update(self, observation: SpatialObservation, raw_track_id: int) -> None:
        tracking = observation.tracking
        if tracking.video_time_seconds > self.last_time_seconds:
            self.previous_scene_point = self.last_scene_point
            self.previous_time_seconds = self.last_time_seconds
        self.last_scene_point = observation.scene_ground_point
        self.last_image_point = observation.image_ground_point
        self.last_time_seconds = tracking.video_time_seconds
        self.last_frame_index = tracking.frame_index
        self.current_raw_track_id = raw_track_id
        self.source_track_ids.add(raw_track_id)


@dataclass(slots=True)
class OnlineTrackStitcher:
    """Resolve raw tracker IDs online without future frames or global clustering."""

    config: TrackStitchConfig = field(default_factory=TrackStitchConfig)
    _visits: dict[str, _VisitMemory] = field(default_factory=dict, init=False)
    _raw_to_visit: dict[RawTrackKey, str] = field(default_factory=dict, init=False)
    _sequence_by_scope: dict[IdentityScope, int] = field(default_factory=dict, init=False)
    _last_time_seconds: float | None = field(default=None, init=False)

    def process_frame(
        self, observations: tuple[SpatialObservation, ...] | list[SpatialObservation]
    ) -> tuple[IdentityResolvedObservation, ...]:
        if not observations:
            return ()
        times = {item.tracking.video_time_seconds for item in observations}
        frames = {item.tracking.frame_index for item in observations}
        if len(times) != 1 or len(frames) != 1:
            raise ValueError("all observations must belong to one frame and video time")
        current_time = next(iter(times))
        if self._last_time_seconds is not None and current_time < self._last_time_seconds:
            raise ValueError("frames must be processed in non-decreasing video time")
        self._last_time_seconds = current_time
        self._expire(current_time)
        observations = self._suppress_duplicates(observations)

        raw_ids_by_scope: dict[IdentityScope, set[int]] = {}
        for item in observations:
            raw_ids_by_scope.setdefault(self._scope(item), set()).add(item.tracking.track_id)

        resolved: list[IdentityResolvedObservation] = []
        claimed_visits: set[str] = set()
        for item in observations:
            tracking = item.tracking
            scope = self._scope(item)
            raw_key = (scope, tracking.track_id)
            visit_id = self._raw_to_visit.get(raw_key)
            memory = self._visits.get(visit_id) if visit_id is not None else None
            stitched = False
            score = None

            if memory is not None:
                gap = tracking.video_time_seconds - memory.last_time_seconds
                if (
                    gap > self.config.max_reconnect_gap_seconds
                    or memory.vehicle_visit_id in claimed_visits
                ):
                    memory = None
                    visit_id = None
            if memory is None:
                match = self._best_recent_match(
                    item,
                    current_raw_ids=raw_ids_by_scope[scope],
                    claimed_visits=claimed_visits,
                )
                if match is None:
                    memory = self._create_visit(item)
                    visit_id = memory.vehicle_visit_id
                else:
                    memory, score = match
                    visit_id = memory.vehicle_visit_id
                    stitched = tracking.track_id not in memory.source_track_ids
                self._raw_to_visit[raw_key] = visit_id

            claimed_visits.add(visit_id)
            memory.update(item, tracking.track_id)
            resolved.append(
                IdentityResolvedObservation(
                    item, visit_id, tracking.track_id, stitched, score
                )
            )
        return tuple(resolved)

    def _suppress_duplicates(
        self, observations: tuple[SpatialObservation, ...] | list[SpatialObservation]
    ) -> list[SpatialObservation]:
        """Keep one box when legacy/tracker output duplicates the same vehicle."""

        def priority(item: SpatialObservation) -> tuple[int, float, float]:
            tracking = item.tracking
            raw_key = (self._scope(item), tracking.track_id)
            known = int(raw_key in self._raw_to_visit)
            confidence = -1.0 if tracking.confidence is None else tracking.confidence
            bbox = tracking.bbox
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            return known, confidence, area

        kept: list[SpatialObservation] = []
        for candidate in sorted(observations, key=priority, reverse=True):
            duplicate = False
            for selected in kept:
                if self._scope(candidate) != self._scope(selected):
                    continue
                if candidate.tracking.vehicle_class != selected.tracking.vehicle_class:
                    continue
                iou, containment = _box_overlap(
                    candidate.tracking.bbox, selected.tracking.bbox
                )
                if (
                    iou >= self.config.duplicate_iou_threshold
                    or containment >= self.config.duplicate_containment_threshold
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return sorted(kept, key=lambda item: item.tracking.track_id)

    def source_track_ids(self, vehicle_visit_id: str) -> tuple[int, ...]:
        memory = self._visits.get(vehicle_visit_id)
        return () if memory is None else tuple(sorted(memory.source_track_ids))

    @staticmethod
    def _scope(observation: SpatialObservation) -> IdentityScope:
        tracking = observation.tracking
        return tracking.camera_id, tracking.processing_session_id, observation.area_id

    def _create_visit(self, observation: SpatialObservation) -> _VisitMemory:
        tracking = observation.tracking
        scope = self._scope(observation)
        sequence = self._sequence_by_scope.get(scope, 0) + 1
        self._sequence_by_scope[scope] = sequence
        visit_id = f"{scope[0]}:{scope[1]}:{scope[2]}:visit_{sequence:06d}"
        memory = _VisitMemory(
            visit_id,
            scope,
            tracking.vehicle_class,
            set(),
            tracking.track_id,
            tracking.video_time_seconds,
            tracking.frame_index,
            observation.scene_ground_point,
            observation.image_ground_point,
        )
        self._visits[visit_id] = memory
        return memory

    def _best_recent_match(
        self,
        observation: SpatialObservation,
        *,
        current_raw_ids: set[int],
        claimed_visits: set[str],
    ) -> tuple[_VisitMemory, float] | None:
        tracking = observation.tracking
        scope = self._scope(observation)
        candidates: list[tuple[float, _VisitMemory]] = []
        for memory in self._visits.values():
            if memory.scope != scope or memory.vehicle_class != tracking.vehicle_class:
                continue
            if memory.vehicle_visit_id in claimed_visits:
                continue
            if memory.current_raw_track_id in current_raw_ids:
                continue
            gap = tracking.video_time_seconds - memory.last_time_seconds
            if gap < 0 or gap > self.config.max_reconnect_gap_seconds:
                continue
            predicted = memory.predicted_scene_point(
                tracking.video_time_seconds, self.config.max_prediction_seconds
            )
            scene_distance = _distance(predicted, observation.scene_ground_point)
            image_distance = _distance(memory.last_image_point, observation.image_ground_point)
            if scene_distance > self.config.max_scene_distance:
                continue
            if image_distance > self.config.max_image_distance:
                continue
            score = (
                0.45 * gap / max(self.config.max_reconnect_gap_seconds, 1e-9)
                + 0.40 * scene_distance / max(self.config.max_scene_distance, 1e-9)
                + 0.15 * image_distance / max(self.config.max_image_distance, 1e-9)
            )
            candidates.append((score, memory))
        if not candidates:
            return None
        score, memory = min(candidates, key=lambda item: item[0])
        return memory, max(0.0, 1.0 - score)

    def _expire(self, current_time: float) -> None:
        expired = [
            visit_id
            for visit_id, memory in self._visits.items()
            if current_time - memory.last_time_seconds > self.config.retention_seconds
        ]
        for visit_id in expired:
            del self._visits[visit_id]
        if expired:
            expired_set = set(expired)
            self._raw_to_visit = {
                key: visit_id
                for key, visit_id in self._raw_to_visit.items()
                if visit_id not in expired_set
            }


def _distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _box_overlap(first, second) -> tuple[float, float]:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, (first.x2 - first.x1) * (first.y2 - first.y1))
    second_area = max(0.0, (second.x2 - second.x1) * (second.y2 - second.y1))
    union = first_area + second_area - intersection
    iou = 0.0 if union <= 0 else intersection / union
    smaller = min(first_area, second_area)
    containment = 0.0 if smaller <= 0 else intersection / smaller
    return iou, containment
