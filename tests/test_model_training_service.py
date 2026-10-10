import pytest

from ai_platform.machine_learning.training.service import (
    EstimatorTrainingAdapter,
    ModelTrainingService,
)


class FakeEstimator:
    def __init__(self, predictions=None):
        self.predictions = [1, 1] if predictions is None else predictions
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, features, targets):
        self.fit_calls.append((features, targets))
        return self

    def predict(self, features):
        self.predict_calls.append(features)
        return self.predictions


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def register_model(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return {
            "model_id": 42,
            "name": kwargs["name"],
            "status": "ready",
            "metrics": kwargs["metrics"],
        }


def make_service(model=None):
    registry = FakeRegistry()
    estimator = model if model is not None else FakeEstimator()
    service = ModelTrainingService(registry=registry)
    return service, registry, estimator


def accuracy(y_true, y_pred):
    return sum(
        expected == actual
        for expected, actual in zip(y_true, y_pred, strict=True)
    ) / len(y_true)


def test_trains_evaluates_and_registers_model():
    service, registry, model = make_service()

    result = service.train_and_register(
        model,
        [[1], [2]],
        [1, 0],
        name="platform-model",
        model_type="classifier",
        version="1.0.0",
        task_type="classification",
        validation_features=[[3], [4]],
        validation_targets=[1, 0],
        metrics={"accuracy": accuracy},
        parameters={"purpose": "platform"},
    )

    assert model.fit_calls == [([[1], [2]], [1, 0])]
    assert model.predict_calls == [[[3], [4]]]
    assert len(registry.calls) == 1
    assert registry.calls[0]["model"] is model
    assert registry.calls[0]["metrics"] == {"accuracy": 0.5}
    assert registry.calls[0]["parameters"] == {"purpose": "platform"}
    assert result["model_id"] == 42
    assert result["status"] == "ready"


def test_registers_trained_model_without_optional_evaluation():
    service, registry, model = make_service()

    service.train_and_register(
        model,
        [[1], [2]],
        [1, 0],
        name="platform-model",
        model_type="classifier",
        version="1.0.0",
    )

    assert len(registry.calls) == 1
    assert registry.calls[0]["metrics"] == {}
    assert model.predict_calls == []


@pytest.mark.parametrize(
    ("validation_features", "validation_targets"),
    [
        ([[3]], None),
        (None, [1]),
    ],
)
def test_requires_validation_inputs_as_a_pair(
    validation_features,
    validation_targets,
):
    service, registry, model = make_service()

    with pytest.raises(ValueError, match="provided together"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=validation_features,
            validation_targets=validation_targets,
        )

    assert registry.calls == []
    assert model.fit_calls == []


def test_metrics_require_validation_data():
    service, registry, model = make_service()

    with pytest.raises(ValueError, match="required to calculate metrics"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            metrics={"accuracy": accuracy},
        )

    assert registry.calls == []
    assert model.fit_calls == []


def test_validation_data_requires_metrics():
    service, registry, model = make_service()

    with pytest.raises(ValueError, match="at least one metric"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[2]],
            validation_targets=[1],
        )

    assert registry.calls == []
    assert model.fit_calls == []


def test_rejects_empty_metric_name():
    service, registry, model = make_service()

    with pytest.raises(ValueError, match="non-empty strings"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[2]],
            validation_targets=[1],
            metrics={" ": accuracy},
        )

    assert registry.calls == []


def test_rejects_non_callable_metric():
    service, registry, model = make_service()

    with pytest.raises(TypeError, match="must be callable"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[2]],
            validation_targets=[1],
            metrics={"accuracy": 123},
        )

    assert registry.calls == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_metrics(score):
    service, registry, model = make_service()

    with pytest.raises(ValueError, match="finite"):
        service.train_and_register(
            model,
            [[1], [2]],
            [1, 0],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[3], [4]],
            validation_targets=[1, 0],
            metrics={"invalid_metric": lambda _y, _p: score},
        )

    assert registry.calls == []


def test_does_not_register_when_training_fails():
    class BrokenEstimator(FakeEstimator):
        def fit(self, features, targets):
            raise RuntimeError("training failed")

    service, registry, model = make_service(BrokenEstimator())

    with pytest.raises(RuntimeError, match="training failed"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
        )

    assert registry.calls == []


def test_does_not_register_when_prediction_fails():
    class BrokenEstimator(FakeEstimator):
        def predict(self, features):
            raise RuntimeError("prediction failed")

    service, registry, model = make_service(BrokenEstimator())

    with pytest.raises(RuntimeError, match="prediction failed"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[2]],
            validation_targets=[1],
            metrics={"accuracy": accuracy},
        )

    assert registry.calls == []


def test_does_not_register_when_metric_fails():
    service, registry, model = make_service()

    def broken_metric(_targets, _predictions):
        raise RuntimeError("metric failed")

    with pytest.raises(RuntimeError, match="metric failed"):
        service.train_and_register(
            model,
            [[1]],
            [1],
            name="platform-model",
            model_type="classifier",
            version="1.0.0",
            validation_features=[[2], [3]],
            validation_targets=[1, 0],
            metrics={"broken": broken_metric},
        )

    assert registry.calls == []


def test_adapter_preserves_model_when_fit_returns_none():
    class NoneReturningEstimator:
        def fit(self, features, targets):
            return None

    model = NoneReturningEstimator()
    adapter = EstimatorTrainingAdapter()

    result = adapter.fit(model, [[1]], [1])

    assert result is model
