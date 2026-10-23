from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    input: Any = Field(..., description="Inference input: text prompt, feature vector, base64 image or token list")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Inference hyperparameters")
    model: Optional[str] = Field(None, description="Optional target model identifier")
    version: Optional[str] = Field(None, description="Explicit target model version")
    priority: int = Field(1, ge=1, le=10, description="Priority level from 1 (standard) to 10 (highest)")
    device_preference: Optional[str] = Field("any", description="'cpu', 'gpu', or 'any'")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text for language model generation")
    max_tokens: int = Field(128, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stop: Optional[List[str]] = Field(None, description="List of stop words/tokens")


class PredictResponse(BaseModel):
    request_id: str
    model: str
    version: str
    prediction: Any
    latency_ms: float
    batch_size: int = 1
    device: str = "cpu"
    worker_id: Optional[str] = None


class JobCreateRequest(BaseModel):
    model: str
    version: Optional[str] = None
    input: Any
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10)


class JobResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="queued, running, completed, failed, cancelled")
    model: str
    version: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class ModelRegisterRequest(BaseModel):
    name: str
    version: str
    framework: str = "pytorch"
    device: str = "cpu"
    artifact_uri: Optional[str] = "local://models"
    description: Optional[str] = ""


class CanaryConfigRequest(BaseModel):
    weights: Dict[str, float] = Field(..., description="Version weights map summing to 1.0 (e.g. {'v1': 0.9, 'v2': 0.1})")


class ModelVersionInfo(BaseModel):
    version: str
    framework: str
    device: str
    status: str
    artifact_uri: str


class ModelInfo(BaseModel):
    name: str
    latest_version: str
    versions: List[str]
    status: str
    canary_weights: Optional[Dict[str, float]] = None


class MetricsSummary(BaseModel):
    requests_total: int
    requests_per_second: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    active_workers: int
    queue_depth: int


class HealthResponse(BaseModel):
    status: str
    api: str
    database: str
    redis: str
    active_workers: int
    loaded_models: List[str]
