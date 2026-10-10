import asyncio

import httpx
import pytest

from ai_platform.api.app import create_app


class FakeRegistry:
    def __init__(self):
        self.record = {
            "model_id": 1,
            "name": "platform-model",
            "model_type": "classifier",
            "version": "1.0.0",
            "provider": "local",
            "task_type": "classification",
            "framework": "scikit-learn",
            "artifact_uri": "1/1.0.0.joblib",
            "status": "ready",
            "metrics": {"accuracy": 0.95},
            "parameters": {"purpose": "platform"},
            "created_at": "2026-10-10 10:00:00",
        }

    def set_model_status(self, model_id, status):
        if model_id != 1:
            raise LookupError(f"Model not found: {model_id}")

        if status not in {"ready", "disabled"}:
            raise ValueError("Status must be either 'ready' or 'disabled'.")

        self.record["status"] = status
        return dict(self.record)


class FakeRuntime:
    def __init__(self):
        self.registry = FakeRegistry()


def make_request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def make_app():
    runtime = FakeRuntime()
    return create_app(runtime=runtime), runtime


def test_status_update_succeeds_with_valid_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": "disabled"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert runtime.registry.record["status"] == "disabled"


def test_status_update_can_reenable_model(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()
    runtime.registry.record["status"] = "disabled"

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": "ready"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_status_update_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "wrong-secret"},
        json={"status": "disabled"},
    )

    assert response.status_code == 401
    assert runtime.registry.record["status"] == "ready"


def test_status_update_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        json={"status": "disabled"},
    )

    assert response.status_code == 401
    assert runtime.registry.record["status"] == "ready"


def test_status_update_fails_closed_when_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("AI_PLATFORM_API_KEY", raising=False)
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "any-key"},
        json={"status": "disabled"},
    )

    assert response.status_code == 503
    assert runtime.registry.record["status"] == "ready"


def test_status_update_returns_404_for_unknown_model(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, _ = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/99/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": "disabled"},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("status", ["registered", "failed", "unknown", ""])
def test_status_update_rejects_invalid_status(monkeypatch, status):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": status},
    )

    assert response.status_code == 422
    assert runtime.registry.record["status"] == "ready"


def test_status_update_rejects_extra_request_fields(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/1/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": "disabled", "admin": True},
    )

    assert response.status_code == 422
    assert runtime.registry.record["status"] == "ready"


def test_status_update_rejects_non_positive_model_id(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    app, runtime = make_app()

    response = make_request(
        app,
        "PUT",
        "/v1/models/0/status",
        headers={"X-API-Key": "test-secret"},
        json={"status": "disabled"},
    )

    assert response.status_code == 422
    assert runtime.registry.record["status"] == "ready"
