import httpx
import pytest

from ai_platform.machine_learning.language_model.ollama_provider import (
    OllamaGenerationService,
    OllamaServiceUnavailableError,
)


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self.data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/generate")
            response = httpx.Response(
                self.status_code,
                request=request,
            )
            raise httpx.HTTPStatusError(
                "HTTP error",
                request=request,
                response=response,
            )

    def json(self):
        return self.data


def test_generate_sends_request_to_local_ollama(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"response": "مرحبًا من Gemma"})

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    service = OllamaGenerationService()
    result = service.generate(
        prompt="اشرح الذكاء الاصطناعي",
        max_new_tokens=120,
        temperature=0.7,
        top_k=40,
    )

    assert result == "مرحبًا من Gemma"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["json"]["model"] == "gemma3:4b"
    assert captured["json"]["prompt"] == "اشرح الذكاء الاصطناعي"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"] == {
        "num_predict": 120,
        "temperature": 0.7,
        "top_k": 40,
    }


def test_generate_omits_top_k_when_not_configured(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse({"response": "Hello"})

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    service = OllamaGenerationService()
    assert service.generate("Hello") == "Hello"
    assert "top_k" not in captured["options"]


def test_provider_rejects_non_local_host():
    with pytest.raises(ValueError, match="local loopback"):
        OllamaGenerationService(base_url="https://example.com")


def test_provider_reports_unavailable_service(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    service = OllamaGenerationService()

    with pytest.raises(OllamaServiceUnavailableError, match="Cannot reach"):
        service.generate("Hello")


def test_provider_reports_missing_local_model(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse({"error": "model not found"}, status_code=404)

    monkeypatch.setattr(
        "ai_platform.machine_learning.language_model.ollama_provider.httpx.post",
        fake_post,
    )

    service = OllamaGenerationService()

    with pytest.raises(OllamaServiceUnavailableError):
        service.generate("Hello")


def test_provider_rejects_invalid_timeout():
    with pytest.raises(ValueError, match="Timeout"):
        OllamaGenerationService(timeout_seconds=0)
