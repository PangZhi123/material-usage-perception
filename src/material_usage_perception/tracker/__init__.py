"""Vehicle tracking adapters."""

from .yolo_bytetrack import ModelLoadError, TrackingFrame, YoloByteTrackTracker

__all__ = ["ModelLoadError", "TrackingFrame", "YoloByteTrackTracker"]
