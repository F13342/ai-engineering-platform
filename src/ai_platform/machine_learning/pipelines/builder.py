"""Build reusable preprocessing and estimator pipelines for tabular ML."""

from collections.abc import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _validate_columns(label: str, columns: Iterable[str]) -> list[str]:
    """Validate and normalize a collection of DataFrame column names."""
    if isinstance(columns, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of column names, not a string.")

    try:
        result = list(columns)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of column names.") from exc

    if any(not isinstance(column, str) for column in result):
        raise TypeError(f"Every column in {label} must be a string.")

    if any(not column.strip() for column in result):
        raise ValueError(f"{label} cannot contain empty column names.")

    if len(result) != len(set(result)):
        raise ValueError(f"{label} cannot contain duplicate column names.")

    return result


def build_tabular_pipeline(
    estimator: object,
    *,
    numeric_features: Iterable[str] = (),
    categorical_features: Iterable[str] = (),
    scale_numeric: bool = True,
) -> Pipeline:
    """Build one fitted-at-training-time preprocessing and estimator workflow.

    Input data should be a pandas DataFrame whose columns match the supplied
    feature names. Unlisted columns are excluded from the pipeline.
    """
    if not callable(getattr(estimator, "fit", None)):
        raise TypeError("Estimator must expose a callable fit method.")

    if not callable(getattr(estimator, "predict", None)):
        raise TypeError("Estimator must expose a callable predict method.")

    if not isinstance(scale_numeric, bool):
        raise TypeError("scale_numeric must be a boolean.")

    numeric = _validate_columns("numeric_features", numeric_features)
    categorical = _validate_columns(
        "categorical_features",
        categorical_features,
    )

    if not numeric and not categorical:
        raise ValueError("At least one numeric or categorical feature is required.")

    overlap = set(numeric).intersection(categorical)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(
            f"Numeric and categorical features must not overlap: {names}."
        )

    transformers = []

    if numeric:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        transformers.append(
            ("numeric", Pipeline(steps=numeric_steps), numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
