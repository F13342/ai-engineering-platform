"""HTTP API for the AI Engineering Platform."""

from fastapi import FastAPI, HTTPException, Path

from ai_platform.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    ModelMetadataResponse,
)
from ai_platform.machine_learning.inference.runtime import (
    ModelInferenceRuntime,
)


def create_app(
    runtime: ModelInferenceRuntime | None = None,
) -> FastAPI:
    """Create the API application with an injectable inference runtime."""
    inference_runtime = (
        runtime if runtime is not None else ModelInferenceRuntime()
    )

    app = FastAPI(
        title="AI Engineering Platform API",
        version="0.1.0",
        description="API for model registry access and inference.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Report whether the API process is responding."""
        return {
            "status": "ok",
            "service": "ai-engineering-platform",
        }

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

    return app


app = create_app()
