"""Smoke tests for the local console and its persisted-reference read APIs."""
import unittest
from tempfile import TemporaryDirectory

import cv2
import httpx
import numpy as np

from material_usage_perception.api.app import create_app
from material_usage_perception.repositories import MemoryRepository
from material_usage_perception.sources import MappingVideoSourceRegistry


class ConsoleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory()
        self.repository = MemoryRepository()
        self.app = create_app(repository=self.repository,
                              video_registry=MappingVideoSourceRegistry(),
                              storage_root=self.directory.name)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.app.state.task_manager.shutdown()
        self.directory.cleanup()

    async def test_console_assets_and_missing_references(self):
        for name in ("", "app.js", "style.css"):
            response = await self.client.get("/test-console/" + name)
            self.assertEqual(response.status_code, 200)
        for path in ("tasks/missing", "scenes/missing", "configurations/missing"):
            response = await self.client.get("/api/material-perception/v1/" + path)
            self.assertEqual(response.status_code, 404)
            self.assertFalse(response.json()["success"])
        response = await self.client.get("/api/material-perception/v1/videos")
        self.assertEqual(response.json()["data"]["items"], [])

    async def test_scene_configuration_and_task_references_can_be_reopened(self):
        _, image = cv2.imencode(".png", np.zeros((100, 200, 3), np.uint8))
        response = await self.client.post(
            "/api/material-perception/v1/scenes",
            files={"file": ("scene.png", image.tobytes(), "image/png")},
            data={"sceneName": "console", "siteId": "test"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        scene = response.json()["data"]
        detail = await self.client.get("/api/material-perception/v1/scenes/" + scene["sceneId"])
        self.assertEqual(detail.json()["data"]["imageUrl"], scene["imageUrl"])
        response = await self.client.post(
            "/api/material-perception/v1/configurations",
            json={"cameraId": "test", "sceneId": scene["sceneId"],
                  "cameraFrameWidth": 200, "cameraFrameHeight": 100,
                  "featurePairs": [{"cameraPoint": {"x": x, "y": y},
                                    "scenePoint": {"x": x, "y": y}}
                                   for x, y in ((0, 0), (200, 0), (200, 100), (0, 100))],
                  "workAreas": [{"areaId": "zone", "coordinateType": "NORMALIZED",
                                 "polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0},
                                             {"x": 1, "y": 1}, {"x": 0, "y": 1}]}]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        config_id = response.json()["data"]["configurationId"]
        detail = await self.client.get("/api/material-perception/v1/configurations/" + config_id)
        self.assertEqual(detail.json()["data"]["cameraFrameSize"], [200, 100])
        self.repository.add_task("test-task", {"taskId": "test-task", "status": "COMPLETED"})
        detail = await self.client.get("/api/material-perception/v1/tasks/test-task")
        self.assertEqual(detail.json()["data"]["status"], "COMPLETED")
