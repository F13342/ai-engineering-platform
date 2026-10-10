"""HTTP API for the AI Engineering Platform."""

from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query

from ai_platform.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    ActiveModelVersionRequest,
    ActiveModelVersionResponse,
    ModelMetadataResponse,
    ModelStatusUpdate,
    ModelVersionComparisonResponse,
    TrainingRunResponse,
)
from ai_platform.api.security import require_api_key
from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.machine_learning.inference.runtime import (
    ModelInferenceRuntime,
)
from ai_platform.machine_learning.versions import (
    ActiveModelVersionService,
    ModelVersionComparisonService,
)


def create_app(
    runtime: ModelInferenceRuntime | None = None,
    training_run_repository: TrainingRunRepository | None = None,
    version_comparison_service: ModelVersionComparisonService | None = None,
    active_version_service: ActiveModelVersionService | None = None,
) -> FastAPI:
    """Create the API with injectable inference and training dependencies."""
    active_versions = active_version_service
    if active_versions is None and runtime is not None:
        active_versions = getattr(runtime, "active_versions", None)
    if active_versions is None:
        active_versions = ActiveModelVersionService()

    inference_runtime = (
        runtime
        if runtime is not None
        else ModelInferenceRuntime(active_version_service=active_versions)
    )
    run_repository = (
        training_run_repository
        if training_run_repository is not None
        else TrainingRunRepository()
    )
    version_comparison = (
        version_comparison_service
        if version_comparison_service is not None
        else ModelVersionComparisonService()
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
        "/v1/models/by-name/{name}/versions",
        response_model=ModelVersionComparisonResponse,
        tags=["models"],
    )
    def compare_model_versions(
        name: str = Path(min_length=1, max_length=200),
    ) -> dict[str, object]:
        """Return comparison metadata for all versions sharing a model name."""
        try:
            return version_comparison.compare_versions(name)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

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

    @app.put(
        "/v1/models/{model_id}/status",
        response_model=ModelMetadataResponse,
        tags=["models"],
        dependencies=[Depends(require_api_key)],
    )
    def update_model_status(
        request: ModelStatusUpdate,
        model_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Enable or disable a model using the protected administrative API."""
        try:
            return inference_runtime.registry.set_model_status(
                model_id,
                request.status,
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Model status update failed.",
            ) from exc

    @app.get(
        "/v1/models/by-name/{name}/active",
        response_model=ActiveModelVersionResponse,
        tags=["models"],
    )
    def get_active_model_version(
        name: str = Path(min_length=1, max_length=200),
    ) -> dict[str, object]:
        """Return the currently selected version for a model name."""
        try:
            record = active_versions.get_active_version(name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"No active version configured for model: {name}",
            )

        return record

    @app.put(
        "/v1/models/by-name/{name}/active",
        response_model=ActiveModelVersionResponse,
        tags=["models"],
        dependencies=[Depends(require_api_key)],
    )
    def activate_model_version(
        request: ActiveModelVersionRequest,
        name: str = Path(min_length=1, max_length=200),
    ) -> dict[str, object]:
        """Activate a ready model version through the protected API."""
        try:
            return active_versions.activate_version(name, request.model_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Active model version update failed.",
            ) from exc

    @app.post(
        "/v1/models/by-name/{name}/predict",
        response_model=InferenceResponse,
        tags=["inference"],
    )
    def predict_active_model(
        request: InferenceRequest,
        name: str = Path(min_length=1, max_length=200),
    ) -> dict[str, object]:
        """Execute inference with the currently active model version."""
        try:
            return inference_runtime.predict_active(name, request.inputs)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
                detail="Active model inference failed.",
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
