import asyncio

import httpx
import pytest
from fastapi import Depends, FastAPI

from ai_platform.api.security import require_api_key


def create_test_app():
    app = FastAPI()

    @app.get("/admin", dependencies=[Depends(require_api_key)])
    def protected_endpoint():
        return {"status": "authorized"}

    return app


def make_request(app, headers=None):
    async def send():
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/admin", headers=headers)

    return asyncio.run(send())


def test_accepts_correct_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret-key")
    response = make_request(
        create_test_app(),
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "authorized"}


def test_rejects_incorrect_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "expected-secret")
    response = make_request(
        create_test_app(),
        headers={"X-API-Key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "expected-secret")
    response = make_request(create_test_app())

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_rejects_requests_when_api_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("AI_PLATFORM_API_KEY", raising=False)
    response = make_request(create_test_app())

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Administrative API authentication is not configured."
    )
