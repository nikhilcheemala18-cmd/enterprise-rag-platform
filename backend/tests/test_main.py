import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app


class TestFastAPIApplicationCreation(unittest.TestCase):
    def test_app_is_a_fastapi_instance(self):
        self.assertIsInstance(app, FastAPI)

    def test_app_metadata(self):
        self.assertEqual(app.title, "Enterprise RAG Platform")
        self.assertEqual(app.version, "0.1.0")
        self.assertTrue(app.description)

    def test_openapi_schema_is_generated(self):
        schema = app.openapi()
        self.assertEqual(schema["info"]["title"], "Enterprise RAG Platform")
        self.assertEqual(schema["info"]["version"], "0.1.0")
        self.assertIn("/health", schema["paths"])


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_returns_expected_body(self):
        response = self.client.get("/health")
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_health_response_is_json(self):
        response = self.client.get("/health")
        self.assertEqual(response.headers["content-type"], "application/json")


class TestSwaggerDocsAvailable(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_docs_endpoint_available(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)

    def test_openapi_json_endpoint_available(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
