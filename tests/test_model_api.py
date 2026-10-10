import asyncio

import httpx

from ai_platform.api.app import create_app


class FakeRegistry:
    def __init__(self, status: str = "ready") -> None:
        self.record = {
            "model_id": 1,
            "name": "Internal Test Model",
            "model_type": "general",
            "version": "1.0.0",
            "provider": "local",
            "task_type": "inference",
            "framework": "test",
            "artifact_uri": "1/1.0.0.joblib",
            "status": status,
            "metrics": {},
            "parameters": {},
            "created_at": "2026-01-01 00:00:00",
        }

    def get_model(self, model_id: int):
        if model_id != 1:
            return None
        return self.record


class FakeRuntime:
    def __init__(self, status: str = "ready") -> None:
        self.registry = FakeRegistry(status)
        self.status = status
        self.calls = 0

    def predict(self, model_id: int, inputs: object):
        if model_id != 1:
            raise LookupError(f"Model not found: {model_id}")

        if self.status != "ready":
            raise ValueError(f"Model is not ready: {model_id}")

        self.calls += 1
        return {
            "model_id": model_id,
            "model_name": "Internal Test Model",
            "model_version": "1.0.0",
            "task_type": "inference",
            "predictions": [4, 8],
            "inference_latency_ms": 0.5,
        }


def make_request(app, method: str, path: str, **kwargs):
    """Send an HTTP request to the ASGI application without TestClient."""
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_endpoint() -> None:
    client_app = create_app(runtime=FakeRuntime())

    response = make_request(client_app, "GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_registered_model_metadata() -> None:
    client_app = create_app(runtime=FakeRuntime())

    response = make_request(client_app, "GET", "/v1/models/1")

    assert response.status_code == 200
    assert response.json()["name"] == "Internal Test Model"
    assert response.json()["status"] == "ready"


def test_unknown_model_returns_404() -> None:
    client_app = create_app(runtime=FakeRuntime())

    response = make_request(client_app, "GET", "/v1/models/999")

    assert response.status_code == 404


def test_predict_endpoint_returns_structured_result() -> None:
    runtime = FakeRuntime()
    client_app = create_app(runtime=runtime)

    response = make_request(
        client_app,
        "POST",
        "/v1/models/1/predict",
        json={"inputs": [2, 4]},
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == [4, 8]
    assert response.json()["model_version"] == "1.0.0"
    assert runtime.calls == 1


def test_predict_rejects_model_that_is_not_ready() -> None:
    client_app = create_app(runtime=FakeRuntime(status="registered"))

    response = make_request(
        client_app,
        "POST",
        "/v1/models/1/predict",
        json={"inputs": [2]},
    )

    assert response.status_code == 409


def test_predict_requires_inputs() -> None:
    client_app = create_app(runtime=FakeRuntime())

    response = make_request(
        client_app,
        "POST",
        "/v1/models/1/predict",
        json={},
    )

    assert response.status_code == 422
