import asyncio

import httpx
import pytest

from ai_platform.api.app import create_app


def active_record(model_id=17, name="text-model"):
    return {
        "model_name": name,
        "model_id": model_id,
        "version": "1.2.0",
        "status": "ready",
        "activated_at": "2026-10-10 20:00:00",
        "metrics": {"accuracy": 0.94},
        "parameters": {"max_depth": 4},
        "artifact_uri": "models/text-model/1.2.0/model.joblib",
    }


class FakeActiveService:
    def __init__(self, record=None, activate_error=None, get_error=None):
        self.record = dict(record) if record is not None else None
        self.activate_error = activate_error
        self.get_error = get_error
        self.calls = []

    def get_active_version(self, name):
        if self.get_error:
            raise self.get_error
        if self.record is None or self.record["model_name"] != name:
            return None
        return dict(self.record)

    def activate_version(self, name, model_id):
        self.calls.append((name, model_id))
        if self.activate_error:
            raise self.activate_error
        self.record = active_record(model_id, name)
        return dict(self.record)


class FakeRuntime:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def predict_active(self, name, inputs):
        self.calls.append((name, inputs))
        if self.error:
            raise self.error
        return {
            "model_id": 17,
            "model_name": "text-model",
            "model_version": "1.2.0",
            "task_type": "classification",
            "predictions": ["positive"],
            "inference_latency_ms": 4.2,
        }


def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def make_app(service=None, runtime=None):
    return create_app(
        runtime=runtime or FakeRuntime(),
        active_version_service=service or FakeActiveService(
            record=active_record()
        ),
    )


def test_get_active_version_returns_metadata():
    response = request(
        make_app(),
        "GET",
        "/v1/models/by-name/text-model/active",
    )
    assert response.status_code == 200
    assert response.json()["model_id"] == 17
    assert response.json()["parameters"] == {"max_depth": 4}


def test_get_active_version_returns_404_when_missing():
    response = request(
        make_app(service=FakeActiveService()),
        "GET",
        "/v1/models/by-name/text-model/active",
    )
    assert response.status_code == 404


def test_get_active_version_maps_value_error_to_422():
    service = FakeActiveService(get_error=ValueError("Invalid name"))
    response = request(
        make_app(service=service),
        "GET",
        "/v1/models/by-name/text-model/active",
    )
    assert response.status_code == 422


def test_activate_version_succeeds_with_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    service = FakeActiveService()
    response = request(
        make_app(service=service),
        "PUT",
        "/v1/models/by-name/text-model/active",
        headers={"X-API-Key": "test-secret"},
        json={"model_id": 17},
    )
    assert response.status_code == 200
    assert response.json()["model_id"] == 17
    assert service.calls == [("text-model", 17)]


@pytest.mark.parametrize(
    ("key", "configured_key", "expected"),
    [
        (None, "test-secret", 401),
        ("wrong", "test-secret", 401),
        ("any-key", None, 503),
    ],
)
def test_activate_version_enforces_api_key(
    monkeypatch, key, configured_key, expected
):
    if configured_key is None:
        monkeypatch.delenv("AI_PLATFORM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AI_PLATFORM_API_KEY", configured_key)

    headers = {} if key is None else {"X-API-Key": key}
    response = request(
        make_app(service=FakeActiveService()),
        "PUT",
        "/v1/models/by-name/text-model/active",
        headers=headers,
        json={"model_id": 17},
    )
    assert response.status_code == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"model_id": 0},
        {"model_id": -1},
        {"model_id": "17"},
        {"model_id": 17, "extra": True},
    ],
)
def test_activate_version_validates_payload(monkeypatch, payload):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    response = request(
        make_app(service=FakeActiveService()),
        "PUT",
        "/v1/models/by-name/text-model/active",
        headers={"X-API-Key": "test-secret"},
        json=payload,
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LookupError("Not found"), 404),
        (ValueError("Name mismatch"), 409),
        (RuntimeError("Storage failure"), 500),
    ],
)
def test_activate_version_maps_errors(monkeypatch, error, expected):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    service = FakeActiveService(activate_error=error)
    response = request(
        make_app(service=service),
        "PUT",
        "/v1/models/by-name/text-model/active",
        headers={"X-API-Key": "test-secret"},
        json={"model_id": 17},
    )
    assert response.status_code == expected


def test_predict_uses_active_version():
    runtime = FakeRuntime()
    response = request(
        make_app(runtime=runtime),
        "POST",
        "/v1/models/by-name/text-model/predict",
        json={"inputs": ["hello"]},
    )
    assert response.status_code == 200
    assert response.json()["predictions"] == ["positive"]
    assert runtime.calls == [("text-model", ["hello"])]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LookupError("No active version configured"), 404),
        (ValueError("Active model is not ready"), 409),
        (ValueError("Invalid input"), 422),
        (RuntimeError("Inference failed"), 500),
    ],
)
def test_predict_active_maps_errors(error, expected):
    response = request(
        make_app(runtime=FakeRuntime(error=error)),
        "POST",
        "/v1/models/by-name/text-model/predict",
        json={"inputs": ["hello"]},
    )
    assert response.status_code == expected


def test_predict_active_requires_inputs():
    response = request(
        make_app(),
        "POST",
        "/v1/models/by-name/text-model/predict",
        json={},
    )
    assert response.status_code == 422
