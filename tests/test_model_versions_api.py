import asyncio

import httpx

from ai_platform.api.app import create_app


class FakeVersionComparisonService:
    def __init__(self, result=None):
        self.result = result
        self.requested_name = None

    def compare_versions(self, name):
        self.requested_name = name

        if not name.strip():
            raise ValueError("Model name must be a non-empty string.")

        if self.result is None:
            return {
                "model_name": name,
                "version_count": 0,
                "versions": [],
            }

        return {
            "model_name": name,
            "version_count": len(self.result),
            "versions": self.result,
        }


def make_request(app, method, path):
    async def send():
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path)

    return asyncio.run(send())


def make_version(model_id, version, accuracy, status="ready"):
    return {
        "model_id": model_id,
        "version": version,
        "status": status,
        "model_type": "classifier",
        "provider": "local",
        "task_type": "classification",
        "framework": "scikit-learn",
        "metrics": {"accuracy": accuracy},
        "parameters": {"depth": model_id + 2},
        "created_at": "2026-10-10 10:00:00",
    }


def test_compare_model_versions_returns_metadata_and_metrics():
    versions = [
        make_version(7, "1.0.0", 0.85),
        make_version(8, "2.0.0", 0.93, status="disabled"),
    ]
    service = FakeVersionComparisonService(versions)
    app = create_app(version_comparison_service=service)

    response = make_request(
        app,
        "GET",
        "/v1/models/by-name/platform-model/versions",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "platform-model"
    assert data["version_count"] == 2
    assert [item["version"] for item in data["versions"]] == [
        "1.0.0",
        "2.0.0",
    ]
    assert data["versions"][0]["metrics"] == {"accuracy": 0.85}
    assert data["versions"][1]["metrics"] == {"accuracy": 0.93}
    assert data["versions"][1]["status"] == "disabled"
    assert service.requested_name == "platform-model"


def test_compare_model_versions_returns_empty_list():
    service = FakeVersionComparisonService()
    app = create_app(version_comparison_service=service)

    response = make_request(
        app,
        "GET",
        "/v1/models/by-name/missing-model/versions",
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "missing-model",
        "version_count": 0,
        "versions": [],
    }


def test_compare_model_versions_rejects_blank_name():
    service = FakeVersionComparisonService()
    app = create_app(version_comparison_service=service)

    response = make_request(
        app,
        "GET",
        "/v1/models/by-name/%20/versions",
    )

    assert response.status_code == 422


def test_compare_model_versions_rejects_name_over_limit():
    service = FakeVersionComparisonService()
    app = create_app(version_comparison_service=service)
    long_name = "a" * 201

    response = make_request(
        app,
        "GET",
        f"/v1/models/by-name/{long_name}/versions",
    )

    assert response.status_code == 422
