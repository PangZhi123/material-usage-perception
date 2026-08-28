"""Database-neutral query objects and deterministic result calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


FINAL_EVENT_STATUSES = {"COMPLETED", "PASS_THROUGH", "INCOMPLETE", "REJECTED"}


class EventStateConflictError(ValueError):
    pass


def merge_event_state(
    existing: dict[str, Any] | None, incoming: dict[str, Any]
) -> dict[str, Any]:
    """Protect a finalized event from replay-driven downgrade or contradiction."""
    if existing is None:
        return incoming
    old_status = existing.get("status")
    new_status = incoming.get("status")
    if old_status in FINAL_EVENT_STATUSES:
        if new_status == "ACTIVE":
            return existing
        if new_status != old_status:
            raise EventStateConflictError(
                f"final event status conflict: {old_status} -> {new_status}"
            )
    return incoming


@dataclass(frozen=True, slots=True)
class EventQuery:
    start_time: datetime
    end_time: datetime
    camera_id: str | None = None
    area_id: str | None = None
    statuses: tuple[str, ...] = ()
    material_type: str | None = None
    page: int = 1
    page_size: int = 20

    def __post_init__(self) -> None:
        _validate_time_range(self.start_time, self.end_time)
        if self.page < 1 or not 1 <= self.page_size <= 100:
            raise ValueError("event query page/page_size is out of range")


@dataclass(frozen=True, slots=True)
class EventPage:
    items: tuple[dict[str, Any], ...]
    page: int
    page_size: int
    total: int


@dataclass(frozen=True, slots=True)
class StatisticsQuery:
    start_time: datetime
    end_time: datetime
    camera_id: str | None = None
    area_id: str | None = None
    material_type: str | None = None
    group_by: str = "NONE"

    def __post_init__(self) -> None:
        _validate_time_range(self.start_time, self.end_time)
        if self.group_by not in {"NONE", "HOUR", "DAY"}:
            raise ValueError("statistics group_by is invalid")


@dataclass(frozen=True, slots=True)
class TrajectoryPage:
    points: tuple[dict[str, Any], ...]
    next_seq: int | None


def filter_events(
    events: list[dict[str, Any]], query: EventQuery | StatisticsQuery
) -> list[dict[str, Any]]:
    result = []
    statuses = query.statuses if isinstance(query, EventQuery) else ()
    for event in events:
        entered = datetime.fromisoformat(event["enteredAt"])
        if not query.start_time <= entered < query.end_time:
            continue
        if query.camera_id and event.get("cameraId") != query.camera_id:
            continue
        if query.area_id and event.get("areaId") != query.area_id:
            continue
        if statuses and event.get("status") not in statuses:
            continue
        if query.material_type and event.get("materialType") != query.material_type:
            continue
        result.append(event)
    return sorted(
        result,
        key=lambda item: (item["enteredAt"], item["vehicleVisitId"]),
        reverse=True,
    )


def _validate_time_range(start: datetime, end: datetime) -> None:
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("query timestamps must include a timezone")
    if end <= start:
        raise ValueError("query end must be after start")
    if end - start > timedelta(days=31):
        raise ValueError("query time range must not exceed 31 days")


def event_page(events: list[dict[str, Any]], query: EventQuery) -> EventPage:
    filtered = filter_events(events, query)
    offset = (query.page - 1) * query.page_size
    return EventPage(
        tuple(filtered[offset:offset + query.page_size]),
        query.page,
        query.page_size,
        len(filtered),
    )


def statistics_data(
    events: list[dict[str, Any]], query: StatisticsQuery, generated_at: datetime
) -> dict[str, Any]:
    filtered = filter_events(events, query)
    completed = [item for item in filtered if item["status"] == "COMPLETED"]
    return {
        "totalEstimatedVolumeM3": round(
            sum(item["estimatedVolumeM3"] for item in completed), 6
        ),
        "completedEventCount": len(completed),
        "passThroughCount": sum(
            item["status"] == "PASS_THROUGH" for item in filtered
        ),
        "incompleteCount": sum(
            item["status"] == "INCOMPLETE" for item in filtered
        ),
        "activeCount": sum(item["status"] == "ACTIVE" for item in filtered),
        "buckets": _buckets(completed, query.group_by),
        "generatedAt": generated_at.isoformat(),
    }


def trajectory_page(
    points: list[dict[str, Any]], *, from_seq: int, limit: int
) -> TrajectoryPage:
    selected = sorted(
        (point for point in points if int(point["seq"]) >= from_seq),
        key=lambda point: int(point["seq"]),
    )[:limit + 1]
    has_more = len(selected) > limit
    selected = selected[:limit]
    next_seq = selected[-1]["seq"] + 1 if has_more and selected else None
    return TrajectoryPage(tuple(selected), next_seq)


def _buckets(events: list[dict[str, Any]], group_by: str) -> list[dict[str, Any]]:
    if group_by == "NONE":
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        time = datetime.fromisoformat(event["enteredAt"])
        start = time.replace(
            minute=0, second=0, microsecond=0
        ) if group_by == "HOUR" else time.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        grouped.setdefault(start.isoformat(), []).append(event)
    return [
        {
            "start": start,
            "estimatedVolumeM3": round(
                sum(item["estimatedVolumeM3"] for item in items), 6
            ),
            "completedEventCount": len(items),
        }
        for start, items in sorted(grouped.items())
    ]
