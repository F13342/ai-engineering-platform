import pytest

from ai_platform.machine_learning.inference.runtime import ModelInferenceRuntime


class FakePredictor:
    def predict(self, inputs):
        return [42]


class FakeRegistry:
    def __init__(self):
        self.load_calls = []

    def get_model(self, model_id):
        if model_id != 7:
            return None

        return {
            "model_id": 7,
            "name": "platform-model",
            "version": "2.0.0",
            "artifact_uri": "7/2.0.0.joblib",
            "status": "ready",
            "task_type": "classification",
        }

    def load_model(self, model_id):
        self.load_calls.append(model_id)
        return FakePredictor()


class FakeActiveVersionService:
    def __init__(self, active=None):
        self.active = active
        self.requested_name = None

    def get_active_version(self, model_name):
        self.requested_name = model_name
        return self.active


def make_runtime(active=None):
    registry = FakeRegistry()
    active_service = FakeActiveVersionService(active)
    runtime = ModelInferenceRuntime(
        registry=registry,
        active_version_service=active_service,
    )
    return runtime, registry, active_service


def test_predict_active_uses_selected_model_version():
    runtime, registry, active_service = make_runtime(
        {
            "model_name": "platform-model",
            "model_id": 7,
            "version": "2.0.0",
            "status": "ready",
        }
    )

    result = runtime.predict_active("platform-model", [[1, 2]])

    assert active_service.requested_name == "platform-model"
    assert registry.load_calls == [7]
    assert result["model_id"] == 7
    assert result["model_name"] == "platform-model"
    assert result["model_version"] == "2.0.0"
    assert result["task_type"] == "classification"
    assert result["predictions"] == [42]


def test_predict_active_rejects_missing_active_version():
    runtime, registry, _ = make_runtime(active=None)

    with pytest.raises(LookupError, match="No active version configured"):
        runtime.predict_active("platform-model", [[1, 2]])

    assert registry.load_calls == []


def test_predict_active_rejects_disabled_active_version():
    runtime, registry, _ = make_runtime(
        {
            "model_name": "platform-model",
            "model_id": 7,
            "version": "2.0.0",
            "status": "disabled",
        }
    )

    with pytest.raises(ValueError, match="Active model is not ready"):
        runtime.predict_active("platform-model", [[1, 2]])

    assert registry.load_calls == []


@pytest.mark.parametrize("model_id", [0, -1, True, "7"])
def test_predict_active_rejects_invalid_active_model_id(model_id):
    runtime, registry, _ = make_runtime(
        {
            "model_name": "platform-model",
            "model_id": model_id,
            "version": "2.0.0",
            "status": "ready",
        }
    )

    with pytest.raises(ValueError, match="invalid model ID"):
        runtime.predict_active("platform-model", [[1, 2]])

    assert registry.load_calls == []


@pytest.mark.parametrize("model_name", ["", " ", None, 123])
def test_predict_active_rejects_invalid_model_name(model_name):
    runtime, registry, _ = make_runtime()

    with pytest.raises(ValueError, match="non-empty string"):
        runtime.predict_active(model_name, [[1, 2]])

    assert registry.load_calls == []
