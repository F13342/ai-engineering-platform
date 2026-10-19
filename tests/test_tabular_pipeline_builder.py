import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ai_platform.machine_learning.pipelines import build_tabular_pipeline


def test_pipeline_imputes_missing_values_and_handles_unseen_categories():
    training_features = pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan, 40.0],
            "income": [10.0, 20.0, 15.0, 30.0],
            "region": ["north", "south", "north", None],
            "unused_column": [100, 200, 300, 400],
        }
    )
    targets = [1.0, 2.0, 1.5, 3.0]

    pipeline = build_tabular_pipeline(
        LinearRegression(),
        numeric_features=["age", "income"],
        categorical_features=["region"],
    )
    pipeline.fit(training_features, targets)

    new_features = pd.DataFrame(
        {
            "age": [np.nan],
            "income": [22.0],
            "region": ["unseen-region"],
            "unused_column": [999999],
        }
    )
    predictions = pipeline.predict(new_features)

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_numeric_imputer_is_fitted_from_training_data_only():
    training_features = pd.DataFrame({"value": [1.0, 3.0, np.nan]})
    pipeline = build_tabular_pipeline(
        LinearRegression(),
        numeric_features=["value"],
    )

    pipeline.fit(training_features, [1.0, 3.0, 2.0])

    numeric_pipeline = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
    )
    statistics = numeric_pipeline.named_steps["imputer"].statistics_

    assert statistics[0] == pytest.approx(2.0)

    held_out = pd.DataFrame({"value": [10000.0]})
    pipeline.predict(held_out)

    assert statistics[0] == pytest.approx(2.0)


def test_numeric_only_pipeline_can_skip_scaling():
    pipeline = build_tabular_pipeline(
        LinearRegression(),
        numeric_features=["value"],
        scale_numeric=False,
    )

    pipeline.fit(
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}),
        [2.0, 4.0, 6.0],
    )
    predictions = pipeline.predict(pd.DataFrame({"value": [4.0]}))

    numeric_pipeline = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
    )
    assert "scaler" not in numeric_pipeline.named_steps
    assert predictions[0] == pytest.approx(8.0)


def test_categorical_only_pipeline():
    pipeline = build_tabular_pipeline(
        LinearRegression(),
        categorical_features=["category"],
    )
    training = pd.DataFrame({"category": ["a", "b", "a", "b"]})

    pipeline.fit(training, [1.0, 3.0, 1.0, 3.0])
    predictions = pipeline.predict(
        pd.DataFrame({"category": ["a", "new-category"]})
    )

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    ("numeric", "categorical", "message"),
    [
        ([], [], "At least one"),
        (["feature"], ["feature"], "must not overlap"),
        (["feature", "feature"], [], "duplicate"),
        (["  "], [], "empty column names"),
    ],
)
def test_pipeline_rejects_invalid_feature_configuration(
    numeric,
    categorical,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_tabular_pipeline(
            LinearRegression(),
            numeric_features=numeric,
            categorical_features=categorical,
        )


def test_pipeline_rejects_non_string_column_names():
    with pytest.raises(TypeError, match="must be a string"):
        build_tabular_pipeline(
            LinearRegression(),
            numeric_features=[123],
        )


@pytest.mark.parametrize(
    ("estimator", "message"),
    [
        (object(), "fit method"),
        (type("NoPredict", (), {"fit": lambda self, x, y: self})(), "predict method"),
    ],
)
def test_pipeline_rejects_invalid_estimators(estimator, message):
    with pytest.raises(TypeError, match=message):
        build_tabular_pipeline(
            estimator,
            numeric_features=["value"],
        )


def test_scale_numeric_must_be_boolean():
    with pytest.raises(TypeError, match="boolean"):
        build_tabular_pipeline(
            LinearRegression(),
            numeric_features=["value"],
            scale_numeric=1,
        )
