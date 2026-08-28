"""Task lifecycle primitives shared by HTTP, workers and future WebSockets."""

from __future__ import annotations

from enum import Enum
from threading import Event


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


ACTIVE_TASK_STATUSES = {
    TaskStatus.QUEUED.value, TaskStatus.RUNNING.value, TaskStatus.STOPPING.value,
}


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def request_stop(self) -> None:
        self._event.set()

    def is_stop_requested(self) -> bool:
        return self._event.is_set()
