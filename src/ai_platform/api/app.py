"""HTTP API for the AI Engineering Platform."""

from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query

from ai_platform.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    ModelMetadataResponse,
    TrainingRunResponse,
)
from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.machine_learning.inference.runtime import (
    ModelInferenceRuntime,
)


def create_app(
    runtime: ModelInferenceRuntime | None = None,
    training_run_repository: TrainingRunRepository | None = None,
) -> FastAPI:
    """Create the API with injectable inference and training dependencies."""
    inference_runtime = (
        runtime if runtime is not None else ModelInferenceRuntime()
    )
    run_repository = (
        training_run_repository
        if training_run_repository is not None
        else TrainingRunRepository()
    )

    app = FastAPI(
        title="AI Engineering Platform API",
        version="0.1.0",
        description=(
            "API for model registry access, inference, and training run tracking."
        ),
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Report whether the API process is responding."""
        return {
            "status": "ok",
            "service": "ai-engineering-platform",
        }

    @app.get(
        "/v1/models",
        response_model=list[ModelMetadataResponse],
        tags=["models"],
    )
    def list_models() -> list[dict[str, object]]:
        """Return metadata for all registered models."""
        return inference_runtime.registry.get_models()

    @app.get(
        "/v1/models/{model_id}",
        response_model=ModelMetadataResponse,
        tags=["models"],
    )
    def get_model(
        model_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Return metadata for a registered model."""
        record = inference_runtime.registry.get_model(model_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}",
            )

        return record

    @app.post(
        "/v1/models/{model_id}/predict",
        response_model=InferenceResponse,
        tags=["inference"],
    )
    def predict(
        request: InferenceRequest,
        model_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Execute inference using a registered model."""
        try:
            return inference_runtime.predict(model_id, request.inputs)
        except LookupError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            status_code = (
                409 if "not ready" in str(exc).lower() else 422
            )
            raise HTTPException(
                status_code=status_code,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Model inference failed.",
            ) from exc

    @app.get(
        "/v1/training-runs",
        response_model=list[TrainingRunResponse],
        tags=["training"],
    )
    def list_training_runs(
        status: Literal["running", "completed", "failed"] | None = Query(
            default=None
        ),
        model_id: int | None = Query(default=None, gt=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, object]]:
        """List training attempts with optional filters."""
        return run_repository.get_training_runs(
            status=status,
            model_id=model_id,
            limit=limit,
        )

    @app.get(
        "/v1/training-runs/{run_id}",
        response_model=TrainingRunResponse,
        tags=["training"],
    )
    def get_training_run(
        run_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Return details for one training attempt."""
        record = run_repository.get_training_run(run_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training run not found: {run_id}",
            )

        return record

    return app


app = create_app()
