"""Request and response schemas for the AI Platform API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
