"""Cached runtime for local language-model text generation."""

import math
from pathlib import Path
from threading import Lock
from typing import Any


class LanguageModelGenerationService:
    """Load a trusted local checkpoint once and reuse it for generation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = Lock()

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from ai_platform.machine_learning.language_model.training import (
                load_language_model,
            )

            model, tokenizer, _ = load_language_model(
                self._checkpoint_path,
                device=self._device,
            )
            self._model = model
            self._tokenizer = tokenizer

        return self._model, self._tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate a continuation without reloading the checkpoint."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must not be empty or whitespace only.")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 512
        ):
            raise ValueError("max_new_tokens must be between 1 and 512.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 < temperature <= 2.0
        ):
            raise ValueError("temperature must be greater than 0 and at most 2.")
        if top_k is not None and (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer when provided.")

        import torch

        with self._lock:
            model, tokenizer = self._ensure_loaded()
            token_ids = tokenizer.encode(prompt)
            if not token_ids:
                raise ValueError("Prompt produced no tokens.")

            device = next(model.parameters()).device
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature),
                    top_k=top_k,
                )
            return tokenizer.decode(generated[0].tolist())
