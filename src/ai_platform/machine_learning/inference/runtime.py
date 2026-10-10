"""Model inference runtime for the AI platform."""

import math
import time
from typing import Protocol

from ai_platform.machine_learning.registry import ModelRegistryService
from ai_platform.machine_learning.versions import ActiveModelVersionService


class PredictionAdapter(Protocol):
    """Interface for model-specific inference adapters."""

    def predict(self, model: object, inputs: object) -> object:
        """Run inference using a loaded model."""


class PredictMethodAdapter:
    """Adapter for models exposing a predict(inputs) method."""

    def predict(self, model: object, inputs: object) -> object:
        method = getattr(model, "predict", None)

        if not callable(method):
            raise TypeError(
                "Model does not expose a callable predict(inputs) method."
            )

        return method(inputs)


def _to_json_compatible(value: object) -> object:
    """Convert common model outputs into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Model output contains a non-finite number.")
        return value

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _to_json_compatible(tolist())

    raise TypeError(
        f"Unsupported model output type: {type(value).__name__}"
    )


class ModelInferenceRuntime:
    """Load registered models and execute inference requests."""

    def __init__(
        self,
        registry: ModelRegistryService | None = None,
        adapter: PredictionAdapter | None = None,
        active_version_service: ActiveModelVersionService | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ModelRegistryService()
        self.adapter = adapter if adapter is not None else PredictMethodAdapter()
        self.active_versions = (
            active_version_service
            if active_version_service is not None
            else ActiveModelVersionService()
        )
        self._loaded_models: dict[tuple[int, str, str], object] = {}

    def predict(
        self,
        model_id: int,
        inputs: object,
    ) -> dict[str, object]:
        """Execute inference and return a structured result."""
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise ValueError("model_id must be an integer.")

        if model_id <= 0:
            raise ValueError("model_id must be positive.")

        record = self.registry.get_model(model_id)

        if record is None:
            raise LookupError(f"Model not found: {model_id}")

        if record["status"] != "ready":
            raise ValueError(f"Model is not ready: {model_id}")

        artifact_uri = record["artifact_uri"]
        version = record["version"]

        if not isinstance(artifact_uri, str) or not artifact_uri:
            raise ValueError(f"Model has no artifact URI: {model_id}")

        if not isinstance(version, str):
            raise ValueError(f"Model has an invalid version: {model_id}")

        cache_key = (model_id, version, artifact_uri)

        if cache_key not in self._loaded_models:
            self._loaded_models[cache_key] = (
                self.registry.load_model(model_id)
            )

        model = self._loaded_models[cache_key]

        started_at = time.perf_counter()

        try:
            raw_predictions = self.adapter.predict(model, inputs)
            predictions = _to_json_compatible(raw_predictions)
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed for model {model_id}."
            ) from exc

        latency_ms = (time.perf_counter() - started_at) * 1000

        return {
            "model_id": model_id,
            "model_name": record["name"],
            "model_version": version,
            "task_type": record["task_type"],
            "predictions": predictions,
            "inference_latency_ms": round(latency_ms, 3),
        }

    def predict_active(
        self,
        model_name: str,
        inputs: object,
    ) -> dict[str, object]:
        """Run inference using the active ready version of a named model."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

        active = self.active_versions.get_active_version(model_name)

        if active is None:
            raise LookupError(
                f"No active version configured for model: {model_name}"
            )

        model_id = active["model_id"]
        if (
            isinstance(model_id, bool)
            or not isinstance(model_id, int)
            or model_id <= 0
        ):
            raise ValueError("Active model selection has an invalid model ID.")

        if active["status"] != "ready":
            raise ValueError(
                f"Active model is not ready: {model_id}"
            )

        return self.predict(model_id, inputs)

    def unload_model(self, model_id: int) -> bool:
        """Remove a model's cached instances from this runtime."""
        keys = [
            key for key in self._loaded_models
            if key[0] == model_id
        ]

        for key in keys:
            del self._loaded_models[key]

        return bool(keys)
