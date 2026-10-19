import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ai_platform.machine_learning.pipelines import build_tabular_pipeline
from ai_platform.machine_learning.tuning import HyperparameterTuningService


def sample_data():
    features = pd.DataFrame({
        "value": np.arange(18, dtype=float),
        "group": ["a", "b", "c"] * 6,
    })
    targets = 2.5 * features["value"] + 3
    return features, targets


def pipeline():
    return build_tabular_pipeline(
        LinearRegression(),
        numeric_features=["value"],
        categorical_features=["group"],
    )


def test_tuning_returns_fitted_pipeline_and_ranked_candidates():
    features, targets = sample_data()

    result = HyperparameterTuningService().tune(
        pipeline(),
        features,
        targets,
        param_distributions={"model__fit_intercept": [True, False]},
        scoring="neg_mean_absolute_error",
        cv=3,
        n_iter=2,
        random_state=42,
        n_jobs=1,
    )

    assert result.candidates_evaluated == 2
    assert result.leaderboard[0]["rank"] == 1
    assert result.leaderboard[0]["params"] == result.best_params
    assert np.isfinite(result.best_score)
    assert np.isfinite(result.best_estimator.predict(features)).all()


def test_tuning_is_reproducible():
    features, targets = sample_data()
    service = HyperparameterTuningService()
    params = {
        "model__fit_intercept": [True, False],
        "model__positive": [True, False],
    }

    first = service.tune(
        pipeline(), features, targets,
        param_distributions=params, cv=3, n_iter=3,
        random_state=7, n_jobs=1,
    )
    second = service.tune(
        pipeline(), features, targets,
        param_distributions=params, cv=3, n_iter=3,
        random_state=7, n_jobs=1,
    )

    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)


@pytest.mark.parametrize(
    ("params", "cv", "n_iter", "message"),
    [
        ({}, 3, 1, "non-empty mapping"),
        ({"model__fit_intercept": []}, 3, 1, "cannot be empty"),
        ({"model__fit_intercept": [True]}, 1, 1, "cv must be at least 2"),
        ({"model__fit_intercept": [True]}, 3, 0, "n_iter must be at least 1"),
    ],
)
def test_tuning_rejects_invalid_configuration(params, cv, n_iter, message):
    features, targets = sample_data()

    with pytest.raises(ValueError, match=message):
        HyperparameterTuningService().tune(
            pipeline(),
            features,
            targets,
            param_distributions=params,
            cv=cv,
            n_iter=n_iter,
            n_jobs=1,
        )


def test_tuning_rejects_estimator_without_fit():
    features, targets = sample_data()

    with pytest.raises(TypeError, match="fit method"):
        HyperparameterTuningService().tune(
            object(),
            features,
            targets,
            param_distributions={"some_parameter": [1, 2]},
            cv=3,
            n_iter=1,
        )
