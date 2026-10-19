import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ai_platform.machine_learning.pipelines import build_tabular_pipeline
from ai_platform.machine_learning.training.tuned_service import (
    TunedModelTrainingService,
)


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def register_model(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return {
            "model_id": 71,
            "name": kwargs["name"],
            "status": "ready",
            "metrics": kwargs["metrics"],
            "parameters": kwargs["parameters"],
        }


class FakeRunRepository:
    def __init__(self):
        self.runs = {}
        self.next_id = 1

    def create_training_run(
        self,
        *,
        model_name,
        model_type,
        model_version,
        parameters=None,
    ):
        run_id = self.next_id
        self.next_id += 1
        self.runs[run_id] = {
            "run_id": run_id,
            "model_id": None,
            "model_name": model_name,
            "model_type": model_type,
            "model_version": model_version,
            "status": "running",
            "metrics": {},
            "parameters": parameters or {},
            "error_message": None,
            "finished_at": None,
        }
        return run_id

    def complete_training_run(self, run_id, *, model_id, metrics=None):
        run = self.runs[run_id]
        run.update(
            status="completed",
            model_id=model_id,
            metrics=metrics or {},
            finished_at="finished",
        )
        return run

    def fail_training_run(self, run_id, *, error_message):
        run = self.runs[run_id]
        run.update(
            status="failed",
            error_message=error_message,
            finished_at="finished",
        )
        return run


def data():
    features = pd.DataFrame({"value": np.arange(18, dtype=float)})
    targets = 2.5 * features["value"] + 3
    validation_features = pd.DataFrame({"value": [20.0, 21.0, 22.0]})
    validation_targets = 2.5 * validation_features["value"] + 3
    return features, targets, validation_features, validation_targets


def make_service():
    registry = FakeRegistry()
    runs = FakeRunRepository()
    service = TunedModelTrainingService(
        registry=registry,
        run_repository=runs,
    )
    model = build_tabular_pipeline(
        LinearRegression(),
        numeric_features=["value"],
    )
    return service, registry, runs, model


def mae(targets, predictions):
    return float(np.mean(np.abs(np.asarray(targets) - np.asarray(predictions))))


def test_tunes_evaluates_registers_and_tracks_model():
    service, registry, runs, model = make_service()
    x, y, validation_x, validation_y = data()

    result = service.tune_train_and_register(
        model,
        x,
        y,
        name="optimized-regressor",
        model_type="regressor",
        version="1.0.0",
        param_distributions={"model__fit_intercept": [True, False]},
        cv=3,
        n_iter=2,
        n_jobs=1,
        validation_features=validation_x,
        validation_targets=validation_y,
        metrics={"mae": mae},
        parameters={"purpose": "integration-test"},
    )

    assert result["model_id"] == 71
    assert result["training_run_id"] == 1
    assert result["optimization"]["candidates_evaluated"] == 2
    assert "best_params" in registry.calls[0]["parameters"]["optimization"]
    assert registry.calls[0]["metrics"]["mae"] < 1e-6
    assert runs.runs[1]["status"] == "completed"
    assert runs.runs[1]["model_id"] == 71
    assert runs.runs[1]["metrics"]["mae"] < 1e-6


def test_failed_search_marks_training_run_failed():
    service, registry, runs, model = make_service()
    x, y, _, _ = data()

    with pytest.raises(ValueError):
        service.tune_train_and_register(
            model,
            x,
            y,
            name="broken-search",
            model_type="regressor",
            version="1.0.0",
            param_distributions={"model__parameter_that_does_not_exist": [1, 2]},
            cv=3,
            n_iter=1,
            n_jobs=1,
        )

    assert registry.calls == []
    assert runs.runs[1]["status"] == "failed"
    assert runs.runs[1]["error_message"]


def test_validation_features_and_targets_must_be_a_pair():
    service, registry, runs, model = make_service()
    x, y, validation_x, _ = data()

    with pytest.raises(ValueError, match="provided together"):
        service.tune_train_and_register(
            model,
            x,
            y,
            name="invalid-validation",
            model_type="regressor",
            version="1.0.0",
            param_distributions={"model__fit_intercept": [True, False]},
            cv=3,
            n_iter=2,
            n_jobs=1,
            validation_features=validation_x,
        )

    assert registry.calls == []
    assert runs.runs[1]["status"] == "failed"
