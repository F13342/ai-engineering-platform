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
        "history": [
            {"role": "user", "content": "My project is Atlas"},
            {"role": "assistant", "content": "I will help with Atlas"},
        ],
        "max_new_tokens": 24,
        "temperature": 0.8,
        "top_k": 10,
    })

    assert response.status_code == 200
    assert response.json()["text"] == "Hello from the test model."
    assert response.json()["generation_latency_ms"] >= 0
    assert len(service.calls) == 1

    call = service.calls[0]
    assert "AI Engineering Platform" in call["prompt"]
    assert "My project is Atlas" in call["prompt"]
    assert "I will help with Atlas" in call["prompt"]
    assert "Hello" in call["prompt"]

    assert call["max_new_tokens"] == 24
    assert call["temperature"] == 0.8
    assert call["top_k"] == 10


def test_language_model_generate_returns_503_when_unconfigured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AI_PLATFORM_LLM_PROVIDER", raising=False)
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

def test_language_model_generate_uses_local_ollama(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_LLM_PROVIDER", "ollama")
    monkeypatch.delenv(
        "AI_PLATFORM_LANGUAGE_MODEL_CHECKPOINT",
        raising=False,
    )

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "مرحبًا من Gemma"}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    app = create_app()
    response = generate_request(app, {
        "prompt": "اشرح الذكاء الاصطناعي",
        "max_new_tokens": 40,
    })

    assert response.status_code == 200
    assert response.json()["text"] == "مرحبًا من Gemma"
    assert captured["url"] == (
        "http://127.0.0.1:11434/api/generate"
    )
    assert captured["payload"]["model"] == "gemma3:4b"


def test_language_model_generate_returns_503_when_ollama_is_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("AI_PLATFORM_LLM_PROVIDER", "ollama")

    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    app = create_app()
    response = generate_request(app, {"prompt": "Hello"})

    assert response.status_code == 503
    assert "Cannot reach" in response.json()["detail"]

def test_prompt_builder_preserves_arabic_identity_instructions():
    from ai_platform.api.app import _build_language_model_prompt
    from ai_platform.api.schemas import LanguageModelGenerationRequest

    request = LanguageModelGenerationRequest(
        prompt="What is my name?",
    )
    prompt = _build_language_model_prompt(request)

    arabic_count = sum(
        chr(0x0600) <= char <= chr(0x06FF)
        for char in prompt
    )

    assert arabic_count > 100
    assert "????" not in prompt
    assert "AI Engineering Platform" in prompt
