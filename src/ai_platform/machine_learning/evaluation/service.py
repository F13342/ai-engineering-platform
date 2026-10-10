"""Reusable model evaluation service."""

import math
from collections.abc import Callable, Mapping
from typing import TypeAlias


MetricFunction: TypeAlias = Callable[[object, object], object]


class ModelEvaluationService:
    """Validate metrics and evaluate predictions using supplied functions."""

    @staticmethod
    def validate_metrics(
        metrics: Mapping[str, MetricFunction] | None,
    ) -> dict[str, MetricFunction]:
        """Validate metric definitions before training or evaluation."""
        if metrics is None:
            return {}

        if not isinstance(metrics, Mapping):
            raise TypeError("Metrics must be provided as a mapping.")

        validated: dict[str, MetricFunction] = {}

        for name, function in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Metric names must be non-empty strings.")

            if not callable(function):
                raise TypeError(f"Metric '{name}' must be callable.")

            validated[name] = function

        return validated

    @staticmethod
    def _validate_score(name: str, score: object) -> float:
        """Require each metric to return a finite numeric scalar."""
        if isinstance(score, (str, bytes, bool)):
            raise ValueError(
                f"Metric '{name}' must return a finite numeric scalar."
            )

        try:
            numeric_score = float(score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Metric '{name}' must return a finite numeric scalar."
            ) from exc

        if not math.isfinite(numeric_score):
            raise ValueError(f"Metric '{name}' returned a non-finite value.")

        return numeric_score

    def evaluate(
        self,
        targets: object,
        predictions: object,
        metrics: Mapping[str, MetricFunction],
    ) -> dict[str, float]:
        """Calculate and validate metrics for a set of predictions."""
        validated_metrics = self.validate_metrics(metrics)

        if not validated_metrics:
            raise ValueError("At least one metric is required for evaluation.")

        results: dict[str, float] = {}

        for name, function in validated_metrics.items():
            score = function(targets, predictions)
            results[name] = self._validate_score(name, score)

        return results
