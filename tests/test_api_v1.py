import unittest
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from material_usage_perception.api.app import create_app
from material_usage_perception.repositories import MemoryRepository
from material_usage_perception.sources import MappingVideoSourceRegistry, VideoSource


class ApiV1Test(unittest.TestCase):
    def setUp(self):
        self.repository = MemoryRepository()
        self.client = TestClient(
            create_app(
                repository=self.repository,
                video_registry=MappingVideoSourceRegistry(),
            )
        )

    def create_scene(self):
        ok, encoded = cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertTrue(ok)
        response = self.client.post(
            "/api/material-perception/v1/scenes",
            files={"file": ("scene.png", encoded.tobytes(), "image/png")},
            data={"sceneName": "test scene", "siteId": "site_001"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        document = response.json()
        self.assertEqual(document["code"], "0")
        self.assertEqual(document["data"]["width"], 200)
        self.assertEqual(document["data"]["height"], 100)
        self.assertEqual(len(document["data"]["sha256"]), 64)
        return document["data"]["sceneId"]

    def test_scene_and_configuration_contract(self):
        scene_id = self.create_scene()
        response = self.client.post(
            "/api/material-perception/v1/configurations",
            json={
                "cameraId": "cam_02",
                "sceneId": scene_id,
                "cameraFrameWidth": 200,
                "cameraFrameHeight": 100,
                "featurePairs": [
                    {"cameraPoint": {"x": 0, "y": 0}, "scenePoint": {"x": 0, "y": 0}},
                    {"cameraPoint": {"x": 200, "y": 0}, "scenePoint": {"x": 200, "y": 0}},
                    {"cameraPoint": {"x": 200, "y": 100}, "scenePoint": {"x": 200, "y": 100}},
                    {"cameraPoint": {"x": 0, "y": 100}, "scenePoint": {"x": 0, "y": 100}},
                ],
                "workAreas": [{
                    "areaId": "pouring_zone_01",
                    "coordinateType": "NORMALIZED",
                    "polygon": [
                        {"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
                        {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9},
                    ],
                }],
                "material": {"defaultCapacityM3": 12.0},
                "stateMachine": {"minOperationDurationSec": 23.0},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["status"], "ACTIVE")
        self.assertEqual(data["quality"]["inlierCount"], 4)
        self.assertEqual(data["effectiveStateMachine"]["minOperationDurationSec"], 23.0)
        self.assertEqual(
            data["effectiveStateMachine"]["entryDetectionDurationSource"],
            "SYSTEM_DEFAULT",
        )
        self.assertIn("configurationId", data)
        self.assertIn("calibrationVersion", data)
        self.assertIn("areaVersion", data)
        self.assertIn("materialConfigVersion", data)

    def test_missing_scene_uses_contract_error_code(self):
        response = self.client.post(
            "/api/material-perception/v1/configurations",
            json={
                "cameraId": "cam",
                "sceneId": "missing",
                "featurePairs": [
                    {"cameraPoint": {"x": x, "y": y}, "scenePoint": {"x": x, "y": y}}
                    for x, y in ((0, 0), (100, 0), (100, 100), (0, 100))
                ],
                "workAreas": [{
                    "areaId": "zone", "coordinateType": "NORMALIZED",
                    "polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                }],
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "MP0102")

    def test_rejects_more_than_one_work_area(self):
        scene_id = self.create_scene()
        area = {
            "areaId": "zone_a", "coordinateType": "NORMALIZED",
            "polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
        }
        second = {**area, "areaId": "zone_b"}
        response = self.client.post(
            "/api/material-perception/v1/configurations",
            json={
                "cameraId": "cam", "sceneId": scene_id,
                "cameraFrameWidth": 200, "cameraFrameHeight": 100,
                "featurePairs": [
                    {"cameraPoint": {"x": x, "y": y}, "scenePoint": {"x": x, "y": y}}
                    for x, y in ((0, 0), (200, 0), (200, 100), (0, 100))
                ],
                "workAreas": [area, second],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "MP0104")
        self.assertIn("exactly one work area", response.json()["message"])

    def test_video_task_persists_event_statistics_and_trajectory(self):
        with TemporaryDirectory() as directory:
            video_path = Path(directory) / "test.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100)
            )
            for _ in range(8):
                writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
            writer.release()
            tracking_path = Path(directory) / "tracking.json"
            tracking_path.write_text(json.dumps([
                {"frame": frame, "objects": [{
                    "id": 7, "class": 8,
                    "bbox": [40, 20, 60, 50] if frame < 4 else [0, 0, 10, 5],
                    "confidence": .9,
                    "vehicleKey": "truck_7",
                }]}
                for frame in range(7)
            ]), encoding="utf-8")
            self.repository = MemoryRepository()
            self.client = TestClient(create_app(
                repository=self.repository,
                video_registry=MappingVideoSourceRegistry({
                    "video_test": VideoSource(
                        "video_test", video_path, tracking_path
                    )
                }),
            ))
            scene_id = self.create_scene()
            configuration = self.client.post(
            "/api/material-perception/v1/configurations",
            json={
                "cameraId": "cam_task", "sceneId": scene_id,
                "cameraFrameWidth": 100, "cameraFrameHeight": 100,
                "featurePairs": [
                    {"cameraPoint": {"x": x, "y": y}, "scenePoint": {"x": x * 2, "y": y}}
                    for x, y in ((0, 0), (100, 0), (100, 100), (0, 100))
                ],
                "workAreas": [{
                    "areaId": "zone", "coordinateType": "NORMALIZED",
                    "polygon": [
                        {"x": .1, "y": .1}, {"x": .9, "y": .1},
                        {"x": .9, "y": .9}, {"x": .1, "y": .9},
                    ],
                }],
                "stateMachine": {
                    "minOperationDurationSec": 0,
                    "entryDetectionDurationSec": 0,
                    "exitDetectionDurationSec": 0,
                },
                "material": {
                    "materialType": "CONCRETE",
                    "defaultCapacityM3": 12,
                    "vehicleOverrides": [
                        {"vehicleKey": "truck_7", "capacityM3": 10}
                    ],
                },
            },
            ).json()["data"]
            response = self.client.post(
                "/api/material-perception/v1/tasks",
                json={
                    "inputType": "VIDEO_FILE", "videoId": "video_test",
                    "configurationId": configuration["configurationId"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            task_id = response.json()["data"]["taskId"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                task = self.repository.get_task(task_id)
                if task and task["status"] in {"COMPLETED", "FAILED"}:
                    break
                time.sleep(0.01)
            self.assertEqual(self.repository.get_task(task_id)["status"], "COMPLETED")

        now = datetime.now(timezone.utc)
        params = {
            "startTime": (now - timedelta(minutes=5)).isoformat(),
            "endTime": (now + timedelta(minutes=5)).isoformat(),
            "cameraId": "cam_task",
        }
        event_response = self.client.get(
            "/api/material-perception/v1/events", params=params
        )
        self.assertEqual(event_response.status_code, 200, event_response.text)
        event = event_response.json()["data"]["items"][-1]
        self.assertEqual(event["status"], "COMPLETED")
        self.assertAlmostEqual(event["estimatedVolumeM3"], 9.5)
        self.assertEqual(event["volumeSource"], "VEHICLE_CAPACITY")
        self.assertEqual(event["vehicleKey"], "truck_7")
        self.assertEqual(event["materialType"], "CONCRETE")
        self.assertAlmostEqual(event["evidence"]["minConfidence"], 0.9)
        self.assertTrue(event["evidence"]["boundaryCrossing"])
        self.assertGreaterEqual(event["evidence"]["stateTransitions"], 3)

        statistics = self.client.get(
            "/api/material-perception/v1/statistics", params=params
        ).json()["data"]
        self.assertGreaterEqual(statistics["completedEventCount"], 1)
        self.assertGreaterEqual(statistics["totalEstimatedVolumeM3"], 9.5)
        mismatched_statistics = self.client.get(
            "/api/material-perception/v1/statistics",
            params={**params, "materialType": "OTHER"},
        ).json()["data"]
        self.assertEqual(mismatched_statistics["completedEventCount"], 0)
        self.assertEqual(mismatched_statistics["totalEstimatedVolumeM3"], 0)

        trajectory = self.client.get(
            f"/api/material-perception/v1/trajectories/{event['vehicleVisitId']}",
            params={"samplingMs": 0},
        )
        self.assertEqual(trajectory.status_code, 200, trajectory.text)
        self.assertGreater(len(trajectory.json()["data"]["points"]), 0)
        normalized = self.client.get(
            f"/api/material-perception/v1/trajectories/{event['vehicleVisitId']}",
            params={"samplingMs": 0, "coordinateType": "NORMALIZED"},
        ).json()["data"]["points"][0]
        self.assertIn("scenePointNorm", normalized)
        self.assertNotIn("scenePointPx", normalized)
        self.assertNotIn("cameraAnchorPx", normalized)
        pixel = self.client.get(
            f"/api/material-perception/v1/trajectories/{event['vehicleVisitId']}",
            params={"samplingMs": 0, "coordinateType": "PIXEL"},
        ).json()["data"]["points"][0]
        self.assertIn("scenePointPx", pixel)
        self.assertIn("cameraAnchorPx", pixel)
        self.assertNotIn("scenePointNorm", pixel)

        messages = []
        with self.client.websocket_connect(
            f"/ws/material-perception/v1/tasks/{task_id}"
        ) as websocket:
            while True:
                try:
                    messages.append(websocket.receive_json())
                except WebSocketDisconnect:
                    break
        message_types = {message["type"] for message in messages}
        self.assertIn("FRAME_RESULT", message_types)
        self.assertIn("EVENT_UPDATE", message_types)
        event_statuses = {
            message["event"]["status"]
            for message in messages if message["type"] == "EVENT_UPDATE"
        }
        self.assertIn("ACTIVE", event_statuses)
        self.assertIn("COMPLETED", event_statuses)
        frame_message = next(
            message for message in messages if message["type"] == "FRAME_RESULT"
        )
        self.assertIn("frame", frame_message)
        self.assertIn("objects", frame_message)
        object_message = next(
            item
            for message in messages if message["type"] == "FRAME_RESULT"
            for item in message["objects"]
        )
        for field in (
            "trackId", "vehicleVisitId", "bboxPx", "bboxNorm", "anchorPx",
            "scenePointPx", "scenePointNorm", "trajectoryDelta", "zoneState",
        ):
            self.assertIn(field, object_message)
        frame_sequences = [
            message["frame"]["seq"]
            for message in messages if message["type"] == "FRAME_RESULT"
        ]
        self.assertEqual(frame_sequences, sorted(set(frame_sequences)))
        self.assertTrue(any(
            len(item["trajectoryDelta"]) >= 2
            for message in messages if message["type"] == "FRAME_RESULT"
            for item in message["objects"]
        ))

    def test_websocket_missing_task_uses_mp0204(self):
        with self.client.websocket_connect(
            "/ws/material-perception/v1/tasks/task_missing"
        ) as websocket:
            message = websocket.receive_json()
            self.assertEqual(message["type"], "ERROR")
            self.assertEqual(message["code"], "MP0204")

    def test_task_request_rejects_development_file_paths(self):
        response = self.client.post(
            "/api/material-perception/v1/tasks",
            json={
                "inputType": "VIDEO_FILE", "videoId": "missing",
                "configurationId": "cfg_missing", "videoPath": "C:/private/test.mp4",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "MP0101")

    def test_task_source_fields_are_conditionally_required(self):
        for payload in (
            {"inputType": "VIDEO_FILE", "configurationId": "cfg"},
            {
                "inputType": "CAMERA_STREAM", "cameraId": "cam",
                "videoId": "video", "configurationId": "cfg",
            },
        ):
            response = self.client.post(
                "/api/material-perception/v1/tasks", json=payload
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["code"], "MP0101")

    def test_duplicate_vehicle_override_keys_are_rejected(self):
        scene_id = self.create_scene()
        response = self.client.post(
            "/api/material-perception/v1/configurations",
            json={
                "cameraId": "cam", "sceneId": scene_id,
                "featurePairs": [
                    {"cameraPoint": {"x": x, "y": y},
                     "scenePoint": {"x": x, "y": y}}
                    for x, y in ((0, 0), (100, 0), (100, 100), (0, 100))
                ],
                "workAreas": [{
                    "areaId": "zone", "coordinateType": "NORMALIZED",
                    "polygon": [
                        {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}
                    ],
                }],
                "material": {"vehicleOverrides": [
                    {"vehicleKey": "truck", "capacityM3": 10},
                    {"vehicleKey": "truck", "capacityM3": 12},
                ]},
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "MP0101")

    def test_query_limits_use_mp0402(self):
        now = datetime.now(timezone.utc)
        response = self.client.get(
            "/api/material-perception/v1/events",
            params={
                "startTime": (now - timedelta(days=32)).isoformat(),
                "endTime": now.isoformat(),
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "MP0402")

    def test_repository_query_failure_uses_mp0401(self):
        class FailingQueryRepository(MemoryRepository):
            def query_events(self, query):
                raise RuntimeError("database unavailable")

        client = TestClient(create_app(
            repository=FailingQueryRepository(),
            video_registry=MappingVideoSourceRegistry(),
        ))
        now = datetime.now(timezone.utc)
        response = client.get(
            "/api/material-perception/v1/events",
            params={
                "startTime": (now - timedelta(hours=1)).isoformat(),
                "endTime": now.isoformat(),
            },
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "MP0401")
        response = self.client.get(
            "/api/material-perception/v1/events",
            params={
                "startTime": (now - timedelta(days=1)).isoformat(),
                "endTime": now.isoformat(), "pageSize": 101,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "MP0402")


if __name__ == "__main__":
    unittest.main()
