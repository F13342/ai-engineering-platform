import asyncio

import httpx
import pytest

from ai_platform.api.app import create_app


@pytest.fixture(autouse=True)
def configure_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")


class FakeGenerationService:
    def __init__(self, result: str = "Hello from the test model.") -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        })
        if self.error is not None:
            raise self.error
        return self.result


def make_request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def generate_request(app, payload, api_key: str | None = "test-secret"):
    headers = {} if api_key is None else {"X-API-Key": api_key}
    return make_request(
        app,
        "POST",
        "/v1/language-model/generate",
        json=payload,
        headers=headers,
    )


def test_language_model_generate_returns_text_and_metrics() -> None:
    service = FakeGenerationService()
    app = create_app(language_model_generation_service=service)

    response = generate_request(app, {
        "prompt": "Hello",
        "max_new_tokens": 24,
        "temperature": 0.8,
        "top_k": 10,
    })

    assert response.status_code == 200
    assert response.json()["text"] == "Hello from the test model."
    assert response.json()["generation_latency_ms"] >= 0
    assert service.calls == [{
        "prompt": "Hello",
        "max_new_tokens": 24,
        "temperature": 0.8,
        "top_k": 10,
    }]


def test_language_model_generate_returns_503_when_unconfigured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AI_PLATFORM_LANGUAGE_MODEL_CHECKPOINT", raising=False)
    app = create_app(language_model_generation_service=None)

    response = generate_request(app, {"prompt": "Hello"})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_language_model_generate_rejects_missing_api_key() -> None:
    app = create_app(language_model_generation_service=FakeGenerationService())

    response = generate_request(app, {"prompt": "Hello"}, api_key=None)

    assert response.status_code == 401


def test_language_model_generate_rejects_invalid_api_key() -> None:
    app = create_app(language_model_generation_service=FakeGenerationService())

    response = generate_request(app, {"prompt": "Hello"}, api_key="wrong-key")

    assert response.status_code == 401


def test_language_model_generate_returns_503_if_api_key_not_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AI_PLATFORM_API_KEY", raising=False)
    app = create_app(language_model_generation_service=FakeGenerationService())

    response = generate_request(app, {"prompt": "Hello"})

    assert response.status_code == 503


def test_language_model_generate_rejects_blank_prompt() -> None:
    service = FakeGenerationService()
    app = create_app(language_model_generation_service=service)

    response = generate_request(app, {"prompt": "   "})

    assert response.status_code == 422
    assert service.calls == []


def test_language_model_generate_rejects_excessive_token_count() -> None:
    service = FakeGenerationService()
    app = create_app(language_model_generation_service=service)

    response = generate_request(app, {
        "prompt": "Hello",
        "max_new_tokens": 513,
    })

    assert response.status_code == 422
    assert service.calls == []


def test_language_model_generate_hides_internal_runtime_errors() -> None:
    service = FakeGenerationService()
    service.error = RuntimeError("internal checkpoint detail")
    app = create_app(language_model_generation_service=service)

    response = generate_request(app, {"prompt": "Hello"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Language model generation failed."
    assert "internal checkpoint detail" not in response.text


def test_language_model_generate_returns_503_for_missing_checkpoint() -> None:
    service = FakeGenerationService()
    service.error = FileNotFoundError("private server path")
    app = create_app(language_model_generation_service=service)

    response = generate_request(app, {"prompt": "Hello"})

    assert response.status_code == 503
    assert "private server path" not in response.text
