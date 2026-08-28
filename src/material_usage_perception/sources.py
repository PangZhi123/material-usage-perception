"""Resolve public camera/video identifiers without exposing server file paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml


@dataclass(frozen=True, slots=True)
class VideoSource:
    video_id: str
    video_path: Path
    tracking_json_path: Path | None = None


class VideoSourceRegistry(Protocol):
    def resolve(self, video_id: str) -> VideoSource | None: ...


class MappingVideoSourceRegistry:
    """Thread-safe immutable registry used by local deployments and tests."""

    def __init__(self, sources: Mapping[str, VideoSource] | None = None) -> None:
        self._sources = dict(sources or {})

    def resolve(self, video_id: str) -> VideoSource | None:
        return self._sources.get(video_id)

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


def _resolve_path(registry_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else registry_path.parent / path
