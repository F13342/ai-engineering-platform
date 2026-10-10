"""Reusable model training, evaluation, and run tracking service."""

from collections.abc import Mapping
from typing import Protocol

from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.machine_learning.evaluation import (
    MetricFunction,
    ModelEvaluationService,
)
from ai_platform.machine_learning.registry import ModelRegistryService


class TrainingAdapter(Protocol):
    """Interface for pluggable model training and evaluation."""

    def fit(self, model: object, features: object, targets: object) -> object:
        """Train a model and return the trained model."""

    def predict(self, model: object, features: object) -> object:
        """Generate predictions for evaluation."""


class EstimatorTrainingAdapter:
    """Adapter for estimators exposing fit(features, targets) and predict(features)."""

    def fit(self, model: object, features: object, targets: object) -> object:
        fit_method = getattr(model, "fit", None)

        if not callable(fit_method):
            raise TypeError(
                "Model must expose a callable fit(features, targets) method."
            )

        fitted_model = fit_method(features, targets)
        return model if fitted_model is None else fitted_model

    def predict(self, model: object, features: object) -> object:
        predict_method = getattr(model, "predict", None)

        if not callable(predict_method):
            raise TypeError(
                "Model must expose a callable predict(features) method."
            )

        return predict_method(features)


class ModelTrainingService:
    """Train, evaluate, register, and track model training runs."""

    def __init__(
        self,
        registry: ModelRegistryService | None = None,
        adapter: TrainingAdapter | None = None,
        evaluator: ModelEvaluationService | None = None,
        run_repository: TrainingRunRepository | None = None,
    ) -> None:
        self.registry = (
            registry if registry is not None else ModelRegistryService()
        )
        self.adapter = (
            adapter if adapter is not None else EstimatorTrainingAdapter()
        )
        self.evaluator = (
            evaluator if evaluator is not None else ModelEvaluationService()
        )
        self.run_repository = (
            run_repository
            if run_repository is not None
            else TrainingRunRepository()
        )

    def train_and_register(
        self,
        model: object,
        features: object,
        targets: object,
        *,
        name: str,
        model_type: str,
        version: str,
        provider: str = "local",
        task_type: str = "unknown",
        framework: str = "unknown",
        validation_features: object | None = None,
        validation_targets: object | None = None,
        metrics: Mapping[str, MetricFunction] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Track a training attempt from creation through its final state."""
        run_id = self.run_repository.create_training_run(
            model_name=name,
            model_type=model_type,
            model_version=version,
            parameters=parameters,
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

            trained_model = self.adapter.fit(model, features, targets)
            calculated_metrics: dict[str, float] = {}

            if metric_functions:
                predictions = self.adapter.predict(
                    trained_model,
                    validation_features,
                )
                calculated_metrics = self.evaluator.evaluate(
                    targets=validation_targets,
                    predictions=predictions,
                    metrics=metric_functions,
                )

            registered_model = self.registry.register_model(
                trained_model,
                name=name,
                model_type=model_type,
                version=version,
                provider=provider,
                task_type=task_type,
                framework=framework,
                metrics=calculated_metrics,
                parameters=parameters,
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
                raise ValueError(
                    "Model registry returned an invalid model ID."
                )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}".strip()

            try:
                self.run_repository.fail_training_run(
                    run_id,
                    error_message=error_message,
                )
            except Exception as tracking_error:
                raise RuntimeError(
                    f"Training failed ({error_message}), and training run "
                    f"{run_id} could not be marked failed: {tracking_error}"
                ) from exc

            raise

        self.run_repository.complete_training_run(
            run_id,
            model_id=model_id,
            metrics=calculated_metrics,
        )

        return {**registered_model, "training_run_id": run_id}
