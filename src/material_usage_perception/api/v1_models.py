"""Public V1 request/response models from the 2026-08-25 interface contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PointPayload(BaseModel):
    x: float
    y: float


class FeaturePairPayload(BaseModel):
    cameraPoint: PointPayload
    scenePoint: PointPayload


class WorkAreaPayload(BaseModel):
    areaId: str = Field(min_length=1, max_length=64)
    polygon: list[PointPayload] = Field(min_length=3)
    coordinateType: Literal["NORMALIZED", "PIXEL"]


class VehicleCapacityOverridePayload(BaseModel):
    vehicleKey: str = Field(min_length=1, max_length=64)
    capacityM3: float = Field(gt=0)


class MaterialPayload(BaseModel):
    materialType: Literal["CONCRETE"] = "CONCRETE"
    defaultCapacityM3: float = Field(default=12.0, gt=0)
    vehicleOverrides: list[VehicleCapacityOverridePayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_vehicle_overrides(self) -> "MaterialPayload":
        keys = [item.vehicleKey.strip() for item in self.vehicleOverrides]
        if any(not key for key in keys):
            raise ValueError("vehicle override keys must not be blank")
        if len(set(keys)) != len(keys):
            raise ValueError("vehicle override keys must be unique")
        return self


class StateMachinePayload(BaseModel):
    minOperationDurationSec: float = Field(default=60.0, ge=0)
    entryDetectionDurationSec: float = Field(default=3.0, ge=0)
    exitDetectionDurationSec: float = Field(default=5.0, ge=0)
    lostTrackToleranceSec: float = Field(default=3.0, ge=0)


class ConfigurationCreateRequest(BaseModel):
    cameraId: str = Field(min_length=1, max_length=64)
    sceneId: str = Field(min_length=1, max_length=64)
    featurePairs: list[FeaturePairPayload] = Field(min_length=4)
    workAreas: list[WorkAreaPayload] = Field(min_length=1)
    material: MaterialPayload = Field(default_factory=MaterialPayload)
    stateMachine: StateMachinePayload = Field(default_factory=StateMachinePayload)
    # Camera dimensions normally come from the upper-level camera registry.
    # Defaults match the V1 example until that registry is connected.
    cameraFrameWidth: int = Field(default=1920, gt=0)
    cameraFrameHeight: int = Field(default=1080, gt=0)
    maximumCalibrationErrorPx: float = Field(default=10.0, gt=0)


class TaskOptionsPayload(BaseModel):
    persistTrajectory: bool = True
    pushIntervalMs: int = Field(default=200, ge=0, le=60_000)


class CameraRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cameraId: str = Field(min_length=1, max_length=64)
    streamUrl: str = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_stream_url(self) -> "CameraRegistrationRequest":
        if not self.streamUrl.lower().startswith(("rtsp://", "rtmp://")):
            raise ValueError("streamUrl must use RTSP or RTMP")
        return self


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputType: Literal["CAMERA_STREAM", "VIDEO_FILE"]
    cameraId: str | None = None
    videoId: str | None = None
    configurationId: str
    options: TaskOptionsPayload = Field(default_factory=TaskOptionsPayload)

    @model_validator(mode="after")
    def validate_source_reference(self) -> "TaskCreateRequest":
        if self.inputType == "CAMERA_STREAM":
            if not self.cameraId or self.videoId is not None:
                raise ValueError(
                    "CAMERA_STREAM requires cameraId and forbids videoId"
                )
        elif not self.videoId or self.cameraId is not None:
            raise ValueError("VIDEO_FILE requires videoId and forbids cameraId")
        return self


class TaskActionRequest(BaseModel):
    action: Literal["STOP"]


class ApiEnvelope(BaseModel):
    requestId: str
    success: bool
    code: str
    message: str
    data: Any = None
    timestamp: datetime
