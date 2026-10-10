import math

import pytest

from ai_platform.machine_learning.evaluation import ModelEvaluationService


def accuracy(targets, predictions):
    return sum(
        expected == actual
        for expected, actual in zip(targets, predictions, strict=True)
    ) / len(targets)


def test_validate_metrics_returns_validated_mapping():
    metrics = {"accuracy": accuracy}

    result = ModelEvaluationService.validate_metrics(metrics)

    assert result == metrics
    assert result is not metrics


def test_validate_metrics_accepts_none():
    assert ModelEvaluationService.validate_metrics(None) == {}


def test_validate_metrics_rejects_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        ModelEvaluationService.validate_metrics([("accuracy", accuracy)])


def test_validate_metrics_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty strings"):
        ModelEvaluationService.validate_metrics({" ": accuracy})


def test_validate_metrics_rejects_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        ModelEvaluationService.validate_metrics({"accuracy": 123})


def test_evaluate_calculates_metrics():
    service = ModelEvaluationService()

    result = service.evaluate(
        targets=[1, 0, 1, 1],
        predictions=[1, 0, 0, 1],
        metrics={"accuracy": accuracy},
    )

    assert result == {"accuracy": 0.75}


def test_evaluate_supports_multiple_metrics():
    service = ModelEvaluationService()

    result = service.evaluate(
        targets=[1, 0],
        predictions=[1, 1],
        metrics={
            "accuracy": accuracy,
            "prediction_count": lambda _targets, predictions: len(predictions),
        },
    )

    assert result == {"accuracy": 0.5, "prediction_count": 2.0}


def test_evaluate_rejects_empty_metrics():
    with pytest.raises(ValueError, match="At least one metric"):
        ModelEvaluationService().evaluate([1], [1], {})


@pytest.mark.parametrize(
    "score",
    [
        "invalid",
        True,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evaluate_rejects_invalid_metric_scores(score):
    with pytest.raises(ValueError):
        ModelEvaluationService().evaluate(
            targets=[1],
            predictions=[1],
            metrics={"invalid_metric": lambda _targets, _predictions: score},
        )


def test_evaluate_propagates_metric_errors():
    def broken_metric(_targets, _predictions):
        raise RuntimeError("metric calculation failed")

    with pytest.raises(RuntimeError, match="metric calculation failed"):
        ModelEvaluationService().evaluate(
            targets=[1],
            predictions=[1],
            metrics={"broken": broken_metric},
        )
