import pytest

from ai_platform.machine_learning.inference.runtime import (
    ModelInferenceRuntime,
)


class FakeModel:
    def predict(self, inputs):
        return [value * 2 for value in inputs]


class FakeRegistry:
    def __init__(self, status="ready", model=None):
        self.record = {
            "model_id": 1,
            "name": "Internal Test Model",
            "version": "1.0.0",
            "task_type": "inference",
            "status": status,
            "artifact_uri": "1/1.0.0.joblib",
        }
        self.model = model or FakeModel()
        self.load_calls = 0

    def get_model(self, model_id):
        if model_id != self.record["model_id"]:
            return None
        return self.record

    def load_model(self, model_id):
        self.load_calls += 1
        return self.model


def test_runtime_predict_returns_structured_result():
    registry = FakeRegistry()
    runtime = ModelInferenceRuntime(registry=registry)

    result = runtime.predict(1, [2, 4])

    assert result["model_id"] == 1
    assert result["model_name"] == "Internal Test Model"
    assert result["model_version"] == "1.0.0"
    assert result["task_type"] == "inference"
    assert result["predictions"] == [4, 8]
    assert result["inference_latency_ms"] >= 0


def test_runtime_caches_model_between_predictions():
    registry = FakeRegistry()
    runtime = ModelInferenceRuntime(registry=registry)

    runtime.predict(1, [1])
    runtime.predict(1, [2])

    assert registry.load_calls == 1


def test_runtime_rejects_model_that_is_not_ready():
    registry = FakeRegistry(status="registered")
    runtime = ModelInferenceRuntime(registry=registry)

    with pytest.raises(ValueError, match="not ready"):
        runtime.predict(1, [1])

    assert registry.load_calls == 0


def test_runtime_reports_unknown_model():
    registry = FakeRegistry()
    runtime = ModelInferenceRuntime(registry=registry)

    with pytest.raises(LookupError, match="Model not found"):
        runtime.predict(999, [1])


def test_runtime_can_unload_cached_model():
    registry = FakeRegistry()
    runtime = ModelInferenceRuntime(registry=registry)

    runtime.predict(1, [1])
    assert runtime.unload_model(1) is True
    assert runtime.unload_model(1) is False

    runtime.predict(1, [2])

    assert registry.load_calls == 2
