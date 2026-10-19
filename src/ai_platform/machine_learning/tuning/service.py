"""Reusable hyperparameter optimization for machine-learning estimators."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sklearn.model_selection import RandomizedSearchCV


@dataclass(frozen=True)
class HyperparameterTuningResult:
    """Best fitted estimator and ranked cross-validation results."""

    best_estimator: object
    best_params: dict[str, object]
    best_score: float
    scoring: str
    cv: int
    candidates_evaluated: int
    leaderboard: tuple[dict[str, object], ...]


class HyperparameterTuningService:
    """Find promising estimator settings using reproducible CV search."""

    def tune(
        self,
        estimator: object,
        features: object,
        targets: object,
        *,
        param_distributions: Mapping[str, object],
        scoring: str | Callable[..., object] = "neg_mean_absolute_error",
        cv: int = 5,
        n_iter: int = 20,
        random_state: int | None = 42,
        n_jobs: int | None = None,
    ) -> HyperparameterTuningResult:
        """Search settings and refit the best estimator on all supplied data.

        Features and targets should contain training data only. Evaluate the
        final selected estimator separately on untouched test data.
        """
        if not callable(getattr(estimator, "fit", None)):
            raise TypeError("Estimator must expose a callable fit method.")

        if not isinstance(param_distributions, Mapping) or not param_distributions:
            raise ValueError("param_distributions must be a non-empty mapping.")

        parameters: dict[str, object] = {}
        for name, values in param_distributions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Hyperparameter names must be non-empty strings.")

            if isinstance(values, (str, bytes, Mapping)):
                raise TypeError(
                    f"Values for '{name}' must be a non-empty sequence "
                    "or a distribution exposing rvs()."
                )

            if callable(getattr(values, "rvs", None)):
                parameters[name] = values
                continue

            try:
                choices = list(values)
            except TypeError as exc:
                raise TypeError(
                    f"Values for '{name}' must be iterable or expose rvs()."
                ) from exc

            if not choices:
                raise ValueError(f"Values for '{name}' cannot be empty.")

            parameters[name] = choices

        for name, value in (("cv", cv), ("n_iter", n_iter)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            minimum = 2 if name == "cv" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")

        if isinstance(random_state, bool) or (
            random_state is not None and not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an integer or None.")
        if random_state is not None and random_state < 0:
            raise ValueError("random_state cannot be negative.")

        if isinstance(n_jobs, bool) or (
            n_jobs is not None and not isinstance(n_jobs, int)
        ):
            raise TypeError("n_jobs must be an integer or None.")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if isinstance(scoring, str):
            if not scoring.strip():
                raise ValueError("scoring cannot be empty.")
            scoring_name = scoring
        elif callable(scoring):
            scoring_name = getattr(scoring, "__name__", type(scoring).__name__)
        else:
            raise TypeError("scoring must be a non-empty string or callable.")

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameters,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(features, targets)

        score = float(search.best_score_)
        if not math.isfinite(score):
            raise ValueError("The best cross-validation score is not finite.")

        results = search.cv_results_
        indices = sorted(
            range(len(results["params"])),
            key=lambda i: (int(results["rank_test_score"][i]), i),
        )

        leaderboard = []
        for index in indices:
            mean_score = float(results["mean_test_score"][index])
            std_score = float(results["std_test_score"][index])
            fit_time = float(results["mean_fit_time"][index])

            leaderboard.append({
                "rank": int(results["rank_test_score"][index]),
                "params": dict(results["params"][index]),
                "mean_test_score": mean_score if math.isfinite(mean_score) else None,
                "std_test_score": std_score if math.isfinite(std_score) else None,
                "mean_fit_time": fit_time if math.isfinite(fit_time) else None,
            })

        return HyperparameterTuningResult(
            best_estimator=search.best_estimator_,
            best_params=dict(search.best_params_),
            best_score=score,
            scoring=scoring_name,
            cv=cv,
            candidates_evaluated=len(leaderboard),
            leaderboard=tuple(leaderboard),
        )
