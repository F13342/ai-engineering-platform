import asyncio

import httpx

from ai_platform.api.app import create_app


class FakeRegistry:
    def __init__(self, models=None):
        self.models = [] if models is None else models

    def get_models(self):
        return self.models


class FakeRuntime:
    def __init__(self, models=None):
        self.registry = FakeRegistry(models)


def make_request(app, method, path):
    async def send():
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path)

    return asyncio.run(send())


def make_model(model_id, name, status="ready"):
    return {
        "model_id": model_id,
        "name": name,
        "model_type": "classifier",
        "version": "1.0.0",
        "provider": "local",
        "task_type": "classification",
        "framework": "scikit-learn",
        "artifact_uri": f"{model_id}/1.0.0.joblib",
        "status": status,
        "metrics": {"accuracy": 0.95},
        "parameters": {"purpose": "platform"},
        "created_at": "2026-10-10 10:00:00",
    }


def test_list_models_returns_registered_models():
    models = [
        make_model(7, "classification-model"),
        make_model(8, "ranking-model", status="registered"),
    ]
    app = create_app(runtime=FakeRuntime(models))

    response = make_request(app, "GET", "/v1/models")

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert [item["model_id"] for item in data] == [7, 8]
    assert data[0]["name"] == "classification-model"
    assert data[0]["metrics"] == {"accuracy": 0.95}
    assert data[0]["parameters"] == {"purpose": "platform"}
    assert data[1]["status"] == "registered"


def test_list_models_returns_empty_list_when_catalog_is_empty():
    app = create_app(runtime=FakeRuntime())

    response = make_request(app, "GET", "/v1/models")

    assert response.status_code == 200
    assert response.json() == []
