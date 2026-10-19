import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from ai_platform.database import schema as schema_module
from ai_platform.database.repositories import (
    model_repository as model_repository_module,
)
from ai_platform.database.repositories import (
    training_run_repository as training_run_repository_module,
)
from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.machine_learning.artifacts.store import ModelArtifactStore
from ai_platform.machine_learning.inference.runtime import ModelInferenceRuntime
from ai_platform.machine_learning.pipelines import build_tabular_pipeline
from ai_platform.machine_learning.registry import ModelRegistryService
from ai_platform.machine_learning.training.tuned_service import (
    TunedModelTrainingService,
)


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route schema and repository operations to a temporary SQLite database."""
    database_path = tmp_path / "integration.sqlite3"

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    monkeypatch.setattr(schema_module, "get_connection", connect)
    monkeypatch.setattr(
        model_repository_module, "get_connection", connect
    )
    monkeypatch.setattr(
        training_run_repository_module, "get_connection", connect
    )

    schema_module.initialize_schema()
    return database_path


def test_tabular_tuning_registration_reload_and_inference(
    isolated_database: Path,
    tmp_path: Path,
) -> None:
    # Arrange: construct a deterministic mixed-type regression dataset.
    features = pd.DataFrame(
        {
            "size": np.arange(1, 31, dtype=float),
            "age": (np.arange(30) % 9 + 1).astype(float),
            "category": ["A", "B", "C"] * 10,
        }
    )
    category_effect = features["category"].map(
        {"A": 0.0, "B": 4.0, "C": -3.0}
    )
    targets = (
        2.5 * features["size"]
        - 0.7 * features["age"]
        + category_effect
    )

    train_features = features.iloc[:24].copy()
    validation_features = features.iloc[24:].copy()
    train_targets = targets.iloc[:24].copy()
    validation_targets = targets.iloc[24:].copy()

    pipeline = build_tabular_pipeline(
        Ridge(),
        numeric_features=["size", "age"],
        categorical_features=["category"],
    )

    repository = ModelRepository()
    run_repository = TrainingRunRepository()
    artifact_store = ModelArtifactStore(tmp_path / "model-artifacts")
    registry = ModelRegistryService(
        repository=repository,
        artifact_store=artifact_store,
    )
    training_service = TunedModelTrainingService(
        registry=registry,
        run_repository=run_repository,
    )

    # Act: tune, evaluate on held-out validation data, and register the model.
    result = training_service.tune_train_and_register(
        pipeline,
        train_features,
        train_targets,
        name="tabular-integration-test",
        model_type="sklearn_pipeline",
        version="1.0.0",
        param_distributions={
            "model__alpha": [0.01, 0.1, 1.0],
            "model__fit_intercept": [True, False],
        },
        scoring="neg_mean_absolute_error",
        cv=3,
        n_iter=4,
        random_state=42,
        n_jobs=1,
        task_type="regression",
        validation_features=validation_features,
        validation_targets=validation_targets,
        metrics={"mae": mean_absolute_error},
        parameters={"purpose": "integration-test"},
    )

    # Assert: tuning and validation produced usable results.
    model_id = result["model_id"]
    run_id = result["training_run_id"]

    assert isinstance(model_id, int) and model_id > 0
    assert isinstance(run_id, int) and run_id > 0
    assert result["status"] == "ready"
    assert result["artifact_uri"]
    assert result["optimization"]["candidates_evaluated"] == 4
    assert result["optimization"]["best_params"]
    assert np.isfinite(float(result["metrics"]["mae"]))

    # Assert: the training run was persisted as completed and linked to the model.
    with sqlite3.connect(isolated_database) as connection:
        run_row = connection.execute(
            """
            SELECT status, model_id, metrics_json, parameters_json
            FROM training_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    assert run_row is not None
    assert run_row[0] == "completed"
    assert run_row[1] == model_id
    saved_parameters = json.loads(run_row[3])
    assert saved_parameters["purpose"] == "integration-test"

    saved_optimization = saved_parameters["optimization"]
    assert (
        saved_optimization["best_params"]
        == result["optimization"]["best_params"]
    )
    assert np.isclose(
        saved_optimization["best_cv_score"],
        result["optimization"]["best_cv_score"],
    )
    assert (
        saved_optimization["candidates_evaluated"]
        == result["optimization"]["candidates_evaluated"]
    )

    saved_leaderboard = saved_optimization["leaderboard"]
    assert saved_leaderboard == result["optimization"]["leaderboard"]
    assert len(saved_leaderboard) == saved_optimization["candidates_evaluated"]

    ranks = [entry["rank"] for entry in saved_leaderboard]
    assert ranks[0] == 1
    assert ranks == sorted(ranks)
    assert all(
        "params" in entry and "mean_test_score" in entry
        for entry in saved_leaderboard
    )

    # Assert: model metadata and serialized artifact survive a reload.
    registered_record = registry.get_model(model_id)
    assert registered_record is not None
    assert registered_record["status"] == "ready"

    loaded_model = registry.load_model(model_id)
    reloaded_predictions = loaded_model.predict(validation_features)

    # Assert: the real inference runtime can load and use the registered model.
    runtime = ModelInferenceRuntime(registry=registry)
    inference_result = runtime.predict(model_id, validation_features)

    runtime_predictions = np.asarray(
        inference_result["predictions"],
        dtype=float,
    )

    np.testing.assert_allclose(
        runtime_predictions,
        np.asarray(reloaded_predictions, dtype=float),
    )
    assert len(runtime_predictions) == len(validation_features)
    assert np.isfinite(runtime_predictions).all()
    assert inference_result["model_id"] == model_id
