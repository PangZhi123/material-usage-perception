"""Small in-process event bus prepared for the WebSocket delivery layer."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Condition, RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskEvent:
    task_id: str
    type: str
    timestamp: datetime
    data: dict[str, Any]
    sequence: int = 0


class TaskEventBus:
    def __init__(self, history_size: int = 1000) -> None:
        self._history_size = history_size
        self._events: dict[str, deque[TaskEvent]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._sequences: dict[str, int] = defaultdict(int)
        self._latest_frame: dict[str, dict[str, Any]] = {}
        self._latest_task_status: dict[str, str] = {}
        self._latest_events: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._history_dropped: set[str] = set()

    def publish(self, event: TaskEvent) -> TaskEvent:
        with self._condition:
            self._sequences[event.task_id] += 1
            stored = replace(event, sequence=self._sequences[event.task_id])
            if len(self._events[event.task_id]) == self._history_size:
                self._history_dropped.add(event.task_id)
            self._events[event.task_id].append(stored)
            if stored.type == "FRAME_RESULT":
                self._latest_frame[event.task_id] = stored.data
            elif stored.type == "EVENT_UPDATE":
                visit_id = str(stored.data.get("event", {}).get("vehicleVisitId", ""))
                if visit_id:
                    self._latest_events[event.task_id][visit_id] = stored.data["event"]
            elif stored.type == "_TASK_STATUS":
                self._latest_task_status[event.task_id] = str(stored.data["status"])
            self._condition.notify_all()
            return stored

    def snapshot(self, task_id: str) -> tuple[TaskEvent, ...]:
        with self._lock:
            return tuple(self._events.get(task_id, ()))

    def after(self, task_id: str, sequence: int) -> tuple[TaskEvent, ...]:
        with self._lock:
            return tuple(
                event for event in self._events.get(task_id, ())
                if event.sequence > sequence
            )

    def wait_after(
        self, task_id: str, sequence: int, timeout_seconds: float
    ) -> tuple[TaskEvent, ...]:
        with self._condition:
            events = self.after(task_id, sequence)
            if events:
                return events
            self._condition.wait(timeout_seconds)
            return self.after(task_id, sequence)

    def latest_sequence(self, task_id: str) -> int:
        with self._lock:
            return self._sequences.get(task_id, 0)

    def can_replay_frame(self, task_id: str, last_frame_seq: int) -> bool:
        if last_frame_seq <= 0:
            return True
        with self._lock:
            frame_sequences = [
                int(event.data["frame"]["seq"])
                for event in self._events.get(task_id, ())
                if event.type == "FRAME_RESULT"
            ]
            dropped = task_id in self._history_dropped
        return not dropped or not frame_sequences or last_frame_seq >= min(frame_sequences)

    def snapshot_data(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            frame = self._latest_frame.get(task_id)
            return {
                "taskStatus": self._latest_task_status.get(task_id),
                "frame": frame.get("frame") if frame else None,
                "objects": frame.get("objects", []) if frame else [],
                "events": list(self._latest_events.get(task_id, {}).values()),
            }
