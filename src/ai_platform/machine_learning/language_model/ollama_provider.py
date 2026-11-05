"""Local Ollama provider for language model generation."""

import os
from urllib.parse import urlparse

import httpx


class OllamaServiceUnavailableError(RuntimeError):
    """Raised when the local Ollama service or model is unavailable."""


class OllamaGenerationService:
    """Generate text using a locally running Ollama model."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv(
                "AI_PLATFORM_OLLAMA_URL",
                "http://127.0.0.1:11434",
            )
        ).rstrip("/")

        self.model = model or os.getenv(
            "AI_PLATFORM_OLLAMA_MODEL",
            "gemma3:4b",
        )

        parsed_url = urlparse(self.base_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("Ollama must use a local loopback address.")

        if not self.model.strip():
            raise ValueError("Ollama model name cannot be empty.")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate a response through the local Ollama HTTP interface."""

        options: dict[str, int | float] = {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        }

        if top_k is not None:
            options["top_k"] = top_k

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaServiceUnavailableError(
                    "The configured local Ollama model is unavailable."
                ) from exc
            raise RuntimeError(
                f"Local Ollama returned HTTP {exc.response.status_code}."
            ) from exc

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
            raise OllamaServiceUnavailableError(
                "Cannot reach the local Ollama service. "
                "Make sure Ollama is running."
            ) from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned an invalid response.") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an unexpected response format.")

        if result.get("error"):
            raise RuntimeError("Local Ollama generation failed.")

        generated_text = result.get("response")

        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response does not contain generated text.")

        return generated_text
