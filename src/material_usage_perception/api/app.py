"""FastAPI application for manual configuration and offline acceptance runs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from material_usage_perception.config import AlgorithmSettings, DetectionRuntimeConfig
from material_usage_perception.api.v1 import create_v1_router
from material_usage_perception.pipeline.offline_analysis import run_offline_analysis
from material_usage_perception.repositories import FilesystemRepository
from material_usage_perception.sources import (
    CameraSourceRegistry, VideoSourceRegistry,
    camera_registry_from_environment, video_registry_from_environment,
)
from material_usage_perception.tasks import TaskManager


class SettingsPayload(BaseModel):
    state: dict[str, Any] = Field(default_factory=dict)
    stitch: dict[str, Any] = Field(default_factory=dict)
    material: dict[str, Any] = Field(default_factory=dict)


class OfflineAnalysisRequest(BaseModel):
    video_path: str
    tracking_json_path: str
    spatial_config_path: str
    camera_id: str = "cam_02"
    processing_session_id: str | None = None
    output_path: str | None = None
    max_frames: int | None = Field(default=None, ge=0)
    settings: SettingsPayload | None = None


class SettingsStore:
    def __init__(self) -> None:
        self._value = AlgorithmSettings()
        self._lock = RLock()

    def get(self) -> AlgorithmSettings:
        with self._lock:
            return self._value

    def replace(self, value: AlgorithmSettings) -> AlgorithmSettings:
        with self._lock:
            self._value = value
            return self._value


def _payload_dict(payload: BaseModel) -> dict[str, Any]:
    dump = getattr(payload, "model_dump", None)
    return dump(exclude_none=True) if dump else payload.dict(exclude_none=True)


def create_app(
    *,
    repository: Any | None = None,
    video_registry: VideoSourceRegistry | None = None,
    camera_registry: CameraSourceRegistry | None = None,
    storage_root: str | Path = Path("outputs") / "backend_data",
) -> FastAPI:
    store = SettingsStore()
    storage_root = Path(storage_root)
    resolved_repository = repository or FilesystemRepository(storage_root)
    detection = DetectionRuntimeConfig.from_environment()
    resolved_video_registry = video_registry or video_registry_from_environment()
    for registered_video in resolved_repository.list_videos():
        resolved_video_registry.register(registered_video)
    resolved_camera_registry = camera_registry or camera_registry_from_environment()
    task_manager = TaskManager(resolved_repository, detection)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            task_manager.shutdown(wait=True)

    application = FastAPI(
        title="Material Usage Perception Backend",
        version="0.1.0",
        description="Manual settings and single-video offline algorithm acceptance API",
        lifespan=lifespan,
    )
    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "MATERIAL_PERCEPTION_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.settings_store = store
    application.state.repository = resolved_repository
    application.state.detection = detection
    application.state.video_registry = resolved_video_registry
    application.state.camera_registry = resolved_camera_registry
    application.state.task_manager = task_manager
    application.include_router(
        create_v1_router(
            resolved_repository, storage_root, detection,
            task_manager=task_manager,
            video_registry=resolved_video_registry,
            camera_registry=resolved_camera_registry,
        )
    )
    application.mount(
        "/files/scenes",
        StaticFiles(directory=storage_root / "scenes", check_dir=False),
        name="scene-files",
    )
    application.mount(
        "/test-console",
        StaticFiles(directory=Path(__file__).parent / "static", html=True),
        name="test-console",
    )
    application.mount(
        "/files/data",
        StaticFiles(directory=storage_root, check_dir=False),
        name="local-data-files",
    )

    def error_content(code: str, message: str, data: Any = None) -> dict[str, Any]:
        return {
            "requestId": f"req_{uuid4().hex}", "success": False,
            "code": code, "message": message, "data": data,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        }

    @application.exception_handler(HTTPException)
    async def contract_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        content = (
            exc.detail
            if isinstance(exc.detail, dict) and "code" in exc.detail
            else error_content("MP0101", str(exc.detail))
        )
        return JSONResponse(status_code=exc.status_code, content=content)

    @application.exception_handler(RequestValidationError)
    async def contract_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = []
        for raw_error in exc.errors():
            error = dict(raw_error)
            if "ctx" in error:
                error["ctx"] = {
                    key: str(value) for key, value in error["ctx"].items()
                }
            errors.append(error)
        query_limit_fields = {"page", "pageSize", "limit", "fromSeq", "samplingMs"}
        code = (
            "MP0402"
            if any(
                len(error.get("loc", ())) >= 2
                and error["loc"][0] == "query"
                and error["loc"][1] in query_limit_fields
                for error in errors
            )
            else "MP0101"
        )
        return JSONResponse(
            status_code=422,
            content=error_content(
                code, "invalid request parameters", {"errors": errors}
            ),
        )

    @application.get("/health")
    def health() -> dict[str, str]:
        model_error = detection.model_file_error()
        return {
            "status": "ok",
            "model_status": "ready" if model_error is None else "invalid",
            "model_version": detection.model_version,
            "model_error": model_error or "",
        }

    @application.get("/settings")
    def get_settings() -> dict[str, Any]:
        return store.get().to_dict()

    @application.put("/settings")
    def put_settings(payload: SettingsPayload) -> dict[str, Any]:
        try:
            settings = AlgorithmSettings.from_mapping(_payload_dict(payload))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return store.replace(settings).to_dict()

    @application.post("/analysis/offline")
    def analyze(request: OfflineAnalysisRequest) -> dict[str, Any]:
        for name in ("video_path", "tracking_json_path", "spatial_config_path"):
            path = Path(getattr(request, name))
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"{name} does not exist: {path}")
        try:
            settings = (
                AlgorithmSettings.from_mapping(_payload_dict(request.settings))
                if request.settings is not None
                else store.get()
            )
            result = run_offline_analysis(
                video_path=request.video_path,
                tracking_json_path=request.tracking_json_path,
                spatial_config_path=request.spatial_config_path,
                camera_id=request.camera_id,
                processing_session_id=(
                    request.processing_session_id or f"manual_{uuid4().hex[:12]}"
                ),
                output_path=request.output_path,
                max_frames=request.max_frames,
                settings=settings,
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    return application


app = create_app()
