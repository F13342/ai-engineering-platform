"""Orchestrate hyperparameter tuning, evaluation, registration, and tracking."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.machine_learning.evaluation import (
    MetricFunction,
    ModelEvaluationService,
)
from ai_platform.machine_learning.registry import ModelRegistryService
from ai_platform.machine_learning.tuning import HyperparameterTuningService


def _json_safe(value: Any) -> Any:
    """Convert common parameter values into metadata-safe JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass

    return repr(value)


class TunedModelTrainingService:
    """Tune a model, evaluate it, register it, and track the whole attempt."""

    def __init__(
        self,
        registry: ModelRegistryService | None = None,
        tuner: HyperparameterTuningService | None = None,
        evaluator: ModelEvaluationService | None = None,
        run_repository: TrainingRunRepository | None = None,
    ) -> None:
        self.registry = (
            registry if registry is not None else ModelRegistryService()
        )
        self.tuner = (
            tuner if tuner is not None else HyperparameterTuningService()
        )
        self.evaluator = (
            evaluator if evaluator is not None else ModelEvaluationService()
        )
        self.run_repository = (
            run_repository
            if run_repository is not None
            else TrainingRunRepository()
        )

    def tune_train_and_register(
        self,
        model: object,
        features: object,
        targets: object,
        *,
        name: str,
        model_type: str,
        version: str,
        param_distributions: Mapping[str, object],
        scoring: str | MetricFunction = "neg_mean_absolute_error",
        cv: int = 5,
        n_iter: int = 20,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        provider: str = "local",
        task_type: str = "regression",
        framework: str = "scikit-learn",
        validation_features: object | None = None,
        validation_targets: object | None = None,
        metrics: Mapping[str, MetricFunction] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Tune on training data, evaluate on separate validation data, and register.

        The tuning process uses cross-validation only within the supplied
        training data. Validation data is kept separate for final evaluation.
        The test set should remain untouched until final model assessment.
        """
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary or None.")

        scoring_name = (
            scoring
            if isinstance(scoring, str)
            else getattr(scoring, "__name__", type(scoring).__name__)
        )

        tuning_config = {
            "scoring": scoring_name,
            "cv": cv,
            "n_iter": n_iter,
            "random_state": random_state,
            "search_space": _json_safe(param_distributions),
        }

        run_parameters = dict(parameters or {})
        run_parameters["optimization"] = tuning_config

        run_id = self.run_repository.create_training_run(
            model_name=name,
            model_type=model_type,
            model_version=version,
            parameters=_json_safe(run_parameters),
        )

        try:
            metric_functions = self.evaluator.validate_metrics(metrics)

            has_validation_features = validation_features is not None
            has_validation_targets = validation_targets is not None

            if has_validation_features != has_validation_targets:
                raise ValueError(
                    "Validation features and targets must be provided together."
                )

            has_validation_data = (
                has_validation_features and has_validation_targets
            )

            if metric_functions and not has_validation_data:
                raise ValueError(
                    "Validation features and targets are required "
                    "to calculate metrics."
                )

            if has_validation_data and not metric_functions:
                raise ValueError(
                    "Provide at least one metric when validation data is supplied."
                )

            tuning_result = self.tuner.tune(
                model,
                features,
                targets,
                param_distributions=param_distributions,
                scoring=scoring,
                cv=cv,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            trained_model = tuning_result.best_estimator

            if not callable(getattr(trained_model, "predict", None)):
                raise TypeError(
                    "The selected estimator must expose a callable predict method."
                )

            calculated_metrics: dict[str, float] = {}
            if metric_functions:
                predictions = trained_model.predict(validation_features)
                calculated_metrics = self.evaluator.evaluate(
                    targets=validation_targets,
                    predictions=predictions,
                    metrics=metric_functions,
                )

            optimization_metadata = {
                **tuning_config,
                "best_params": _json_safe(tuning_result.best_params),
                "best_cv_score": tuning_result.best_score,
                "candidates_evaluated": tuning_result.candidates_evaluated,
            }

            registered_parameters = dict(parameters or {})
            registered_parameters["optimization"] = optimization_metadata

            registered_model = self.registry.register_model(
                trained_model,
                name=name,
                model_type=model_type,
                version=version,
                provider=provider,
                task_type=task_type,
                framework=framework,
                metrics=calculated_metrics,
                parameters=_json_safe(registered_parameters),
            )

            if not isinstance(registered_model, dict):
                raise TypeError(
                    "Model registry must return a model metadata dictionary."
                )

            model_id = registered_model.get("model_id")
            if (
                isinstance(model_id, bool)
                or not isinstance(model_id, int)
                or model_id <= 0
            ):
                raise ValueError("Model registry returned an invalid model ID.")

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}".strip()
            try:
                self.run_repository.fail_training_run(
                    run_id,
                    error_message=error_message,
                )
            except Exception as tracking_error:
                raise RuntimeError(
                    f"Optimization failed ({error_message}), and training run "
                    f"{run_id} could not be marked failed: {tracking_error}"
                ) from exc
            raise

        self.run_repository.complete_training_run(
            run_id,
            model_id=model_id,
            metrics=calculated_metrics,
            parameters=_json_safe(registered_parameters),
        )

        return {
            **registered_model,
            "training_run_id": run_id,
            "optimization": optimization_metadata,
        }
