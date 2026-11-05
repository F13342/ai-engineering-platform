"""Request and response schemas for the AI Platform API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferenceRequest(BaseModel):
    """Input payload for model inference."""

    model_config = ConfigDict(extra="forbid")

    inputs: Any


class InferenceResponse(BaseModel):
    """Structured output from model inference."""

    model_id: int
    model_name: str
    model_version: str
    task_type: str
    predictions: Any
    inference_latency_ms: float = Field(ge=0)


class ModelMetadataResponse(BaseModel):
    """Public metadata for a registered model."""

    model_id: int
    name: str
    model_type: str
    version: str
    provider: str
    task_type: str
    framework: str
    artifact_uri: str | None
    status: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    created_at: str


class ModelStatusUpdate(BaseModel):
    """Validated request to enable or disable a registered model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "disabled"]


class TrainingRunResponse(BaseModel):
    """Public representation of a model training attempt."""

    model_config = ConfigDict(extra="forbid")

    run_id: int
    model_id: int | None
    model_name: str
    model_type: str
    model_version: str
    status: Literal["running", "completed", "failed"]
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    error_message: str | None
    started_at: str
    finished_at: str | None

class ModelVersionResponse(BaseModel):
    """Public metadata for one registered model version."""

    model_id: int
    version: str
    status: str
    model_type: str
    provider: str
    task_type: str
    framework: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    created_at: str


class ModelVersionComparisonResponse(BaseModel):
    """Comparison data for all versions sharing a model name."""

    model_name: str
    version_count: int = Field(ge=0)
    versions: list[ModelVersionResponse]


class ActiveModelVersionRequest(BaseModel):
    """Request to activate a registered model version."""

    model_config = ConfigDict(extra="forbid")

    model_id: int = Field(gt=0, strict=True)


class ActiveModelVersionResponse(BaseModel):
    """Public information about the active model version."""

    model_name: str
    model_id: int
    version: str
    status: str
    activated_at: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]


class ConversationMessage(BaseModel):
    """One validated message from the current conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content must not be blank.")
        return value


class LanguageModelGenerationRequest(BaseModel):
    """Validated request for local language-model generation."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = Field(default=None, gt=0)
    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=6,
    )
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class LanguageModelGenerationResponse(BaseModel):
    """Response containing generated text and elapsed generation time."""

    text: str
    generation_latency_ms: float = Field(ge=0)
    conversation_id: int | None = None


class ConversationCreateRequest(BaseModel):
    """Validated request for creating a local conversation."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="New conversation", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation title cannot be blank.")
        return value


class ConversationSummaryResponse(BaseModel):
    conversation_id: int
    title: str
    created_at: str


class StoredConversationMessageResponse(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: str


class ConversationDetailResponse(BaseModel):
    conversation_id: int
    title: str
    created_at: str
    messages: list[StoredConversationMessageResponse]


class DocumentGenerationRequest(BaseModel):
    """Validated request for generating a downloadable document."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20000)
    output_format: Literal["docx", "pdf", "pptx"]
    language: Literal["ar", "en"] = "en"

    @field_validator("title", "content")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field must not be blank.")
        return value
