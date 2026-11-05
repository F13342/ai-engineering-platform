"""HTTP API for the AI Engineering Platform."""

import os
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse
from pathlib import Path as FilePath

from ai_platform.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    LanguageModelGenerationRequest,
    LanguageModelGenerationResponse,
    DocumentGenerationRequest,
    ActiveModelVersionRequest,
    ActiveModelVersionResponse,
    ModelMetadataResponse,
    ModelStatusUpdate,
    ModelVersionComparisonResponse,
    TrainingRunResponse,
    ConversationCreateRequest,
    ConversationSummaryResponse,
    ConversationDetailResponse,
    ConversationMessage,
)
from ai_platform.document_generation import DocumentGenerationService
from ai_platform.conversations.service import ConversationService
from ai_platform.api.security import require_api_key
from ai_platform.machine_learning.language_model.generation import (
    LanguageModelGenerationService,
)
from ai_platform.machine_learning.language_model.ollama_provider import (
    OllamaGenerationService,
    OllamaServiceUnavailableError,
)
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


def _build_language_model_prompt(
    request: LanguageModelGenerationRequest,
    history: list[ConversationMessage] | None = None,
) -> str:
    """Build an Arabic-safe identity prompt with conversation context."""

    system_instructions = (
        "\u0623\u0646\u062a \u0627\u0644\u0645\u0633\u0627\u0639\u062f "
        "\u0627\u0644\u0630\u0643\u064a \u0627\u0644\u0645\u062d\u0644\u064a "
        "\u0627\u0644\u0645\u062f\u0645\u062c \u0641\u064a "
        "\u0645\u0646\u0635\u0629 AI Engineering Platform.\n"

        "\u0627\u0644\u0645\u0646\u0635\u0629 \u0645\u0634\u0631\u0648\u0639 "
        "\u0628\u0631\u0645\u062c\u064a \u0645\u0633\u062a\u0642\u0644 "
        "\u064a\u0645\u0644\u0643\u0647 \u0648\u064a\u0637\u0648\u0631\u0647 "
        "\u0635\u0627\u062d\u0628 \u0647\u0630\u0627 "
        "\u0627\u0644\u0645\u0634\u0631\u0648\u0639. "
        "\u0644\u064a\u0633\u062a \u0645\u0646\u062a\u062c\u064b\u0627 "
        "\u0645\u0646 Google \u0648\u0644\u0627 \u062a\u062a\u0628\u0639 "
        "Google \u0623\u0648 \u0623\u064a \u0634\u0631\u0643\u0629 "
        "\u0623\u062e\u0631\u0649.\n"

        "\u0642\u062f \u062a\u0633\u062a\u062e\u062f\u0645 "
        "\u0627\u0644\u0645\u0646\u0635\u0629 \u0646\u0645\u0648\u0630\u062c "
        "Gemma \u0627\u0644\u0645\u0637\u0648\u0631 \u0645\u0646 Google\u060c "
        "\u0644\u0643\u0646 \u0630\u0644\u0643 \u0644\u0627 \u064a\u0639\u0646\u064a "
        "\u0623\u0646 \u0627\u0644\u0645\u0646\u0635\u0629 "
        "\u0646\u0641\u0633\u0647\u0627 \u062a\u0627\u0628\u0639\u0629 "
        "\u0625\u0644\u0649 Google.\n"

        "\u0644\u0627 \u062a\u062f\u0651\u0639\u0650 \u0623\u0646\u0643 "
        "\u0645\u0648\u0638\u0641 \u0623\u0648 \u0645\u0645\u062b\u0644 "
        "\u0644\u0623\u064a \u0634\u0631\u0643\u0629. "
        "\u0625\u0630\u0627 \u0633\u064f\u0626\u0644\u062a "
        "\u0639\u0646 \u0647\u0648\u064a\u062a\u0643\u060c "
        "\u0641\u0642\u0644 \u0625\u0646\u0643 "
        "\u0627\u0644\u0645\u0633\u0627\u0639\u062f "
        "\u0627\u0644\u0645\u062d\u0644\u064a \u0644\u0645\u0646\u0635\u0629 "
        "AI Engineering Platform.\n"

        "\u0623\u062c\u0628 \u0628\u0644\u063a\u0629 "
        "\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645. "
        "\u0627\u0633\u062a\u062e\u062f\u0645 "
        "\u0633\u062c\u0644 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629 "
        "\u0644\u0641\u0647\u0645 \u0627\u0644\u0623\u0633\u0626\u0644\u0629 "
        "\u0627\u0644\u062a\u064a \u062a\u0634\u064a\u0631 "
        "\u0625\u0644\u0649 \u0631\u0633\u0627\u0626\u0644 "
        "\u0633\u0627\u0628\u0642\u0629.\n"

        "\u0644\u0627 \u062a\u062e\u062a\u0631\u0639 "
        "\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 "
        "\u0623\u0648 \u062a\u0641\u0627\u0635\u064a\u0644 "
        "\u0645\u0634\u0631\u0648\u0639\u0647. "
        "\u0625\u0630\u0627 \u0644\u0645 \u062a\u0631\u062f "
        "\u0645\u0639\u0644\u0648\u0645\u0629 \u0641\u064a "
        "\u0627\u0644\u0633\u064a\u0627\u0642\u060c "
        "\u0642\u0644 \u0625\u0646\u0643 "
        "\u0644\u0627 \u062a\u0639\u0631\u0641\u0647\u0627."
    )

    sections = [
        f"\u062a\u0639\u0644\u064a\u0645\u0627\u062a "
        f"\u0627\u0644\u0645\u0633\u0627\u0639\u062f:\n{system_instructions}"
    ]

    history_messages = request.history if history is None else history

    if history_messages:
        history_lines = [
            "\u0633\u062c\u0644 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629 "
            "\u0627\u0644\u0633\u0627\u0628\u0642\u0629:"
        ]

        for message in history_messages:
            speaker = (
                "\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645"
                if message.role == "user"
                else "\u0627\u0644\u0645\u0633\u0627\u0639\u062f"
            )
            history_lines.append(f"{speaker}: {message.content}")

        sections.append("\n".join(history_lines))

    sections.append(
        f"\u0631\u0633\u0627\u0644\u0629 "
        f"\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 "
        f"\u0627\u0644\u062d\u0627\u0644\u064a\u0629:\n{request.prompt}"
        f"\n\n\u0625\u062c\u0627\u0628\u0629 "
        f"\u0627\u0644\u0645\u0633\u0627\u0639\u062f:"
    )

    return "\n\n".join(sections)


def create_app(
    runtime: ModelInferenceRuntime | None = None,
    training_run_repository: TrainingRunRepository | None = None,
    version_comparison_service: ModelVersionComparisonService | None = None,
    active_version_service: ActiveModelVersionService | None = None,
    language_model_generation_service: LanguageModelGenerationService | None = None,
    document_generation_service: DocumentGenerationService | None = None,
    conversation_service: ConversationService | None = None,
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

    generation_service = language_model_generation_service
    if generation_service is None:
        provider_name = os.environ.get(
            "AI_PLATFORM_LLM_PROVIDER", "checkpoint"
        ).strip().lower()

        if provider_name == "ollama":
            generation_service = OllamaGenerationService()

        elif provider_name == "checkpoint":
            checkpoint_path = os.environ.get(
                "AI_PLATFORM_LANGUAGE_MODEL_CHECKPOINT"
            )
            if checkpoint_path:
                generation_service = LanguageModelGenerationService(
                    checkpoint_path,
                    device=os.environ.get(
                        "AI_PLATFORM_LANGUAGE_MODEL_DEVICE", "auto"
                    ),
                )

        elif provider_name != "disabled":
            raise ValueError(
                f"Unsupported local language model provider: {provider_name}"
            )

    document_service = document_generation_service
    if document_service is None:
        document_service = DocumentGenerationService()

    conversation_manager = (
        conversation_service
        if conversation_service is not None
        else ConversationService()
    )

    try:
        local_user_id = int(os.environ.get("AI_PLATFORM_LOCAL_USER_ID", "920"))
    except ValueError as exc:
        raise ValueError("AI_PLATFORM_LOCAL_USER_ID must be an integer.") from exc

    if local_user_id <= 0:
        raise ValueError("AI_PLATFORM_LOCAL_USER_ID must be positive.")

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


    @app.post(
        "/v1/conversations",
        response_model=ConversationSummaryResponse,
        status_code=201,
        tags=["conversations"],
        dependencies=[Depends(require_api_key)],
    )
    def create_conversation(
        request: ConversationCreateRequest,
    ) -> dict[str, object]:
        """Create a conversation for the configured local owner."""
        try:
            conversation_id = conversation_manager.create_conversation(
                user_id=local_user_id,
                title=request.title,
            )
            record = conversation_manager.get_conversation(
                local_user_id,
                conversation_id,
            )["conversation"]
            return {
                "conversation_id": record["conversation_id"],
                "title": record["title"],
                "created_at": record["created_at"],
            }
        except LookupError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local owner account is not initialized.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/v1/conversations",
        response_model=list[ConversationSummaryResponse],
        tags=["conversations"],
        dependencies=[Depends(require_api_key)],
    )
    def list_conversations() -> list[dict[str, object]]:
        """List conversations belonging to the configured local owner."""
        try:
            records = conversation_manager.list_conversations(local_user_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local owner account is not initialized.",
            ) from exc

        return [
            {
                "conversation_id": record["conversation_id"],
                "title": record["title"],
                "created_at": record["created_at"],
            }
            for record in records
        ]

    @app.get(
        "/v1/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
        tags=["conversations"],
        dependencies=[Depends(require_api_key)],
    )
    def get_conversation(
        conversation_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Retrieve one owned conversation and its messages."""
        try:
            result = conversation_manager.get_conversation(
                local_user_id,
                conversation_id,
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            ) from exc

        record = result["conversation"]
        messages = result["messages"]

        return {
            "conversation_id": record["conversation_id"],
            "title": record["title"],
            "created_at": record["created_at"],
            "messages": [
                {
                    "message_id": message["message_id"],
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                }
                for message in messages
            ],
        }

    @app.delete(
        "/v1/conversations/{conversation_id}",
        tags=["conversations"],
        dependencies=[Depends(require_api_key)],
    )
    def delete_conversation(
        conversation_id: int = Path(gt=0),
    ) -> dict[str, object]:
        """Delete one owned conversation and its messages."""
        try:
            conversation_manager.delete_conversation(
                local_user_id,
                conversation_id,
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            ) from exc

        return {
            "status": "deleted",
            "conversation_id": conversation_id,
        }

    @app.post(
        "/v1/language-model/generate",
        response_model=LanguageModelGenerationResponse,
        tags=["language-model"],
        dependencies=[Depends(require_api_key)],
    )
    def generate_language_model(
        request: LanguageModelGenerationRequest,
    ) -> dict[str, object]:
        """Generate text with the configured local language model."""
        if generation_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Language model generation is not configured. Configure the "
                    "local checkpoint or the local Ollama provider."
                ),
            )

        conversation_history = request.history

        if request.conversation_id is not None:
            try:
                stored_history = conversation_manager.get_recent_history(
                    local_user_id,
                    request.conversation_id,
                    limit=6,
                )
            except LookupError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                ) from exc

            conversation_history = [
                ConversationMessage(**message)
                for message in stored_history
            ]

        started_at = perf_counter()
        try:
            generated_text = generation_service.generate(
                prompt=_build_language_model_prompt(
                    request,
                    history=conversation_history,
                ),
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )
        except OllamaServiceUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="The configured language model checkpoint is unavailable.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Language model generation failed.",
            ) from exc

        if request.conversation_id is not None:
            try:
                conversation_manager.add_exchange(
                    user_id=local_user_id,
                    conversation_id=request.conversation_id,
                    user_content=request.prompt,
                    assistant_content=generated_text,
                )
            except LookupError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                ) from exc

        return {
            "text": generated_text,
            "generation_latency_ms": round(
                (perf_counter() - started_at) * 1000, 3
            ),
            "conversation_id": request.conversation_id,
        }


    @app.post(
        "/v1/documents/generate",
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def generate_document(
        request: DocumentGenerationRequest,
    ) -> FileResponse:
        """Generate a document and return it as a download."""
        try:
            generated = document_service.generate(
                title=request.title,
                content=request.content,
                output_format=request.output_format,
                language=request.language,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Document generation failed.",
            ) from exc

        return FileResponse(
            path=generated.path,
            media_type=generated.media_type,
            filename=generated.filename,
            headers={"Cache-Control": "no-store"},
        )


    @app.get(
        "/chat",
        response_class=FileResponse,
        include_in_schema=False,
    )
    def chat_page() -> FileResponse:
        """Serve the local chat interface."""
        page_path = FilePath(__file__).with_name("chat.html")
        if not page_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Chat page not found.",
            )
        return FileResponse(
            path=page_path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
