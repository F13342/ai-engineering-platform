import asyncio

import httpx

from ai_platform.api.app import create_app


class FakeTrainingRunRepository:
    def __init__(self):
        self.records = [
            {
                "run_id": 1,
                "model_id": 10,
                "model_name": "platform-model",
                "model_type": "classifier",
                "model_version": "1.0.0",
                "status": "completed",
                "metrics": {"accuracy": 0.95},
                "parameters": {
                    "purpose": "platform",
                    "optimization": {
                        "best_params": {"model__alpha": 0.1},
                        "best_cv_score": -0.12,
                        "candidates_evaluated": 2,
                        "leaderboard": [
                            {
                                "rank": 1,
                                "params": {"model__alpha": 0.1},
                                "mean_test_score": -0.12,
                                "std_test_score": 0.01,
                                "mean_fit_time": 0.001,
                            },
                            {
                                "rank": 2,
                                "params": {"model__alpha": 1.0},
                                "mean_test_score": -0.2,
                                "std_test_score": 0.02,
                                "mean_fit_time": 0.001,
                            },
                        ],
                    },
                },
                "error_message": None,
                "started_at": "2026-10-10 10:00:00",
                "finished_at": "2026-10-10 10:01:00",
            },
            {
                "run_id": 2,
                "model_id": None,
                "model_name": "failed-model",
                "model_type": "classifier",
                "model_version": "1.0.0",
                "status": "failed",
                "metrics": {},
                "parameters": {},
                "error_message": "RuntimeError: training failed",
                "started_at": "2026-10-10 11:00:00",
                "finished_at": "2026-10-10 11:00:05",
            },
            {
                "run_id": 3,
                "model_id": None,
                "model_name": "pending-model",
                "model_type": "classifier",
                "model_version": "2.0.0",
                "status": "running",
                "metrics": {},
                "parameters": {},
                "error_message": None,
                "started_at": "2026-10-10 12:00:00",
                "finished_at": None,
            },
        ]

    def get_training_runs(
        self,
        *,
        status=None,
        model_id=None,
        limit=100,
    ):
        records = self.records

        if status is not None:
            records = [
                record for record in records
                if record["status"] == status
            ]

        if model_id is not None:
            records = [
                record for record in records
                if record["model_id"] == model_id
            ]

        return list(reversed(records))[:limit]

    def get_training_run(self, run_id):
        return next(
            (
                record for record in self.records
                if record["run_id"] == run_id
            ),
            None,
        )


def make_request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def make_app():
    repository = FakeTrainingRunRepository()
    app = create_app(training_run_repository=repository)
    return app, repository


def test_list_training_runs_returns_records():
    app, _ = make_app()

    response = make_request(app, "GET", "/v1/training-runs")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["run_id"] == 3
    assert data[1]["status"] == "failed"
    assert data[2]["status"] == "completed"


def test_list_training_runs_filters_by_status():
    app, _ = make_app()

    response = make_request(
        app,
        "GET",
        "/v1/training-runs",
        params={"status": "failed"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["run_id"] == 2
    assert data[0]["error_message"] == "RuntimeError: training failed"


def test_list_training_runs_filters_by_model_id():
    app, _ = make_app()

    response = make_request(
        app,
        "GET",
        "/v1/training-runs",
        params={"model_id": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["model_id"] == 10


def test_list_training_runs_applies_limit():
    app, _ = make_app()

    response = make_request(
        app,
        "GET",
        "/v1/training-runs",
        params={"limit": 1},
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_training_runs_rejects_invalid_status():
    app, _ = make_app()

    response = make_request(
        app,
        "GET",
        "/v1/training-runs",
        params={"status": "unknown"},
    )

    assert response.status_code == 422


def test_list_training_runs_rejects_out_of_range_limit():
    app, _ = make_app()

    response = make_request(
        app,
        "GET",
        "/v1/training-runs",
        params={"limit": 501},
    )

    assert response.status_code == 422


def test_get_training_run_returns_details():
    app, _ = make_app()

    response = make_request(app, "GET", "/v1/training-runs/1")

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == 1
    assert data["status"] == "completed"
    assert data["metrics"] == {"accuracy": 0.95}
    assert data["parameters"]["purpose"] == "platform"

    optimization = data["parameters"]["optimization"]
    assert optimization["best_params"] == {"model__alpha": 0.1}
    assert optimization["candidates_evaluated"] == 2
    assert len(optimization["leaderboard"]) == 2
    assert optimization["leaderboard"][0]["rank"] == 1
    assert optimization["leaderboard"][0]["mean_test_score"] == -0.12


def test_get_training_run_returns_404_for_unknown_id():
    app, _ = make_app()

    response = make_request(app, "GET", "/v1/training-runs/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Training run not found: 999"


def test_get_training_run_rejects_non_positive_id():
    app, _ = make_app()

    response = make_request(app, "GET", "/v1/training-runs/0")

    assert response.status_code == 422
