"""Resolve public camera/video identifiers without exposing server file paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Condition, Event, RLock, Thread, current_thread
from time import monotonic, time
from typing import Any, Iterator, Mapping, Protocol

import cv2
import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class VideoSource:
    video_id: str
    video_path: Path
    tracking_json_path: Path | None = None
    camera_id: str | None = None
    original_file_name: str | None = None
    content_type: str = "video/mp4"
    size_bytes: int | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    captured_at: str | None = None
    uploaded_at: str | None = None
    status: str = "READY"


class VideoSourceRegistry(Protocol):
    def resolve(self, video_id: str) -> VideoSource | None: ...
    def register(self, source: VideoSource) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraSource:
    camera_id: str
    stream_url: str


class CameraSourceRegistry(Protocol):
    def resolve(self, camera_id: str) -> CameraSource | None: ...
    def register(self, source: CameraSource) -> None: ...


class MappingCameraSourceRegistry:
    """Thread-safe camera registry populated through YAML or the REST API."""

    def __init__(self, sources: Mapping[str, CameraSource] | None = None) -> None:
        self._sources = dict(sources or {})
        self._lock = RLock()

    def resolve(self, camera_id: str) -> CameraSource | None:
        with self._lock:
            return self._sources.get(camera_id)

    def register(self, source: CameraSource) -> None:
        with self._lock:
            self._sources[source.camera_id] = source

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MappingCameraSourceRegistry":
        registry_path = Path(path)
        if not registry_path.is_file():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        values = raw.get("cameras", {})
        if not isinstance(values, dict):
            raise ValueError("camera registry 'cameras' must be an object")
        sources: dict[str, CameraSource] = {}
        for camera_id, item in values.items():
            if not isinstance(item, dict) or not item.get("stream_url"):
                raise ValueError(f"camera registry entry is invalid: {camera_id}")
            sources[str(camera_id)] = CameraSource(str(camera_id), str(item["stream_url"]))
        return cls(sources)


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class StreamFrame:
    sequence: int
    timestamp_seconds: float
    captured_at_seconds: float
    image: np.ndarray


class CameraStreamReader:
    """OpenCV/FFmpeg RTSP reader with stale-frame dropping and reconnects."""

    def __init__(
        self,
        stream_url: str,
        *,
        reconnect_delay_seconds: float = 1.0,
        open_timeout_seconds: float = 15.0,
        capture_factory: Any | None = None,
    ) -> None:
        if not stream_url:
            raise ValueError("stream_url must not be blank")
        self.stream_url = stream_url
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.open_timeout_seconds = open_timeout_seconds
        self._capture_factory = capture_factory or _open_capture
        self._stop = Event()
        self._condition = Condition(RLock())
        self._metadata: StreamMetadata | None = None
        self._latest: StreamFrame | None = None
        self._version = 0
        self._capture: Any | None = None
        self._thread: Thread | None = None
        self._started_monotonic: float | None = None

    @property
    def metadata(self) -> StreamMetadata:
        if self._metadata is None:
            raise RuntimeError("camera stream is not connected")
        return self._metadata

    def __enter__(self) -> "CameraStreamReader":
        self._stop.clear()
        self._started_monotonic = monotonic()
        self._thread = Thread(target=self._capture_loop, name="material-camera", daemon=True)
        self._thread.start()
        deadline = monotonic() + self.open_timeout_seconds
        with self._condition:
            while self._metadata is None and not self._stop.is_set():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self.close()
                    raise ConnectionError(f"camera stream cannot be opened: {self.stream_url}")
                self._condition.wait(timeout=min(remaining, 0.1))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def iter_frames(self, stop_event: Any | None = None) -> Iterator[StreamFrame]:
        version = 0
        while not self._stop.is_set() and not _stop_requested(stop_event):
            with self._condition:
                self._condition.wait_for(
                    lambda: self._version > version or self._stop.is_set() or _stop_requested(stop_event),
                    timeout=0.5,
                )
                if self._stop.is_set() or _stop_requested(stop_event):
                    break
                frame = self._latest
                version = self._version
            if frame is not None:
                yield frame

    def close(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2)
        self._thread = None
        self._capture = None

    def _capture_loop(self) -> None:
        sequence = 0
        while not self._stop.is_set():
            capture = self._capture_factory(self.stream_url)
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._stop.wait(self.reconnect_delay_seconds)
                continue
            metadata = StreamMetadata(
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                max(float(capture.get(cv2.CAP_PROP_FPS)), 0.0),
            )
            if metadata.width <= 0 or metadata.height <= 0:
                capture.release()
                self._stop.wait(self.reconnect_delay_seconds)
                continue
            with self._condition:
                self._metadata = metadata
                self._condition.notify_all()
            while not self._stop.is_set():
                success, image = capture.read()
                if not success:
                    break
                current_monotonic = monotonic()
                started = self._started_monotonic or current_monotonic
                frame = StreamFrame(sequence, current_monotonic - started, time(), image)
                sequence += 1
                with self._condition:
                    self._latest = frame
                    self._version += 1
                    self._condition.notify_all()
            capture.release()
            self._capture = None
            self._stop.wait(self.reconnect_delay_seconds)


def _open_capture(stream_url: str) -> Any:
    if stream_url.lower().startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000])
    try:
        return cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, parameters)
    except TypeError:  # Compatibility with older OpenCV Python bindings.
        return cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)


def _stop_requested(token: Any | None) -> bool:
    return bool(token is not None and token.is_stop_requested())


class MappingVideoSourceRegistry:
    """Thread-safe immutable registry used by local deployments and tests."""

    def __init__(self, sources: Mapping[str, VideoSource] | None = None) -> None:
        self._sources = dict(sources or {})
        self._lock = RLock()

    def resolve(self, video_id: str) -> VideoSource | None:
        with self._lock:
            return self._sources.get(video_id)

    def register(self, source: VideoSource) -> None:
        with self._lock:
            self._sources[source.video_id] = source

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MappingVideoSourceRegistry":
        registry_path = Path(path)
        if not registry_path.is_file():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        values = raw.get("videos", {})
        if not isinstance(values, dict):
            raise ValueError("video registry 'videos' must be an object")
        sources: dict[str, VideoSource] = {}
        for video_id, item in values.items():
            if not isinstance(item, dict) or not item.get("video_path"):
                raise ValueError(f"video registry entry is invalid: {video_id}")
            video_path = _resolve_path(registry_path, item["video_path"])
            tracking = item.get("tracking_json_path")
            sources[str(video_id)] = VideoSource(
                video_id=str(video_id),
                video_path=video_path,
                tracking_json_path=(
                    _resolve_path(registry_path, tracking) if tracking else None
                ),
            )
        return cls(sources)


def video_registry_from_environment() -> MappingVideoSourceRegistry:
    return MappingVideoSourceRegistry.from_yaml(
        os.getenv("MATERIAL_PERCEPTION_VIDEO_REGISTRY", "config/video_registry.yaml")
    )


def camera_registry_from_environment() -> MappingCameraSourceRegistry:
    return MappingCameraSourceRegistry.from_yaml(
        os.getenv("MATERIAL_PERCEPTION_CAMERA_REGISTRY", "config/camera_registry.yaml")
    )


def _resolve_path(registry_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else registry_path.parent / path
