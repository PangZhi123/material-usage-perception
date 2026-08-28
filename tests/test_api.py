import unittest

from fastapi.testclient import TestClient

from material_usage_perception.api.app import create_app


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_and_manual_minimum_duration_setting(self):
        health = self.client.get("/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertIn(health["model_status"], ("ready", "not_configured"))
        response = self.client.put(
            "/settings",
            json={
                "state": {"minimum_dwell_seconds": 23.0},
                "stitch": {},
                "material": {},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"]["minimum_dwell_seconds"], 23.0)
        self.assertEqual(
            self.client.get("/settings").json()["state"]["minimum_dwell_seconds"],
            23.0,
        )

    def test_local_frontend_cors_preflight(self):
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:5173",
        )

    def test_invalid_negative_duration_is_rejected(self):
        response = self.client.put(
            "/settings",
            json={
                "state": {"minimum_dwell_seconds": -1.0},
                "stitch": {},
                "material": {},
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
