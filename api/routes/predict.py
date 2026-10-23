import time
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.schemas.requests import PredictRequest, PredictResponse, GenerateRequest
from core.db.session import get_db
from api.core.auth import get_current_api_client
from api.core.rate_limiter import rate_limiter
from api.services.model_registry import model_registry_service
from api.services.usage_tracking import log_inference_usage
from worker.batcher.dynamic_batcher import dynamic_batcher

router = APIRouter(tags=["Inference"])


async def execute_inference(
    model: str,
    explicit_version: Optional[str],
    input_data: Any,
    parameters: Optional[Dict[str, Any]],
    client: Dict[str, Any],
    db: Session,
) -> PredictResponse:
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # 1. Resolve model version & device requirement (incorporates canary routing)
    resolved_version, device = model_registry_service.resolve_version(
        db=db,
        model_name=model,
        requested_version=explicit_version,
    )

    try:
        # 2. Submit to Dynamic Batcher for GPU/CPU forward pass
        prediction, batch_size, executed_device = await dynamic_batcher.submit_request(
            request_id=request_id,
            model=model,
            version=resolved_version,
            input_data=input_data,
            parameters=parameters,
        )

        latency_ms = round((time.time() - start_time) * 1000.0, 2)

        # 3. Asynchronously record usage telemetry & percentiles
        log_inference_usage(
            request_id=request_id,
            model=model,
            version=resolved_version,
            latency_ms=latency_ms,
            user_id=client.get("user_id"),
            device=executed_device,
            status_code=200,
        )

        return PredictResponse(
            request_id=request_id,
            model=model,
            version=resolved_version,
            prediction=prediction,
            latency_ms=latency_ms,
            batch_size=batch_size,
            device=executed_device,
        )

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000.0, 2)
        log_inference_usage(
            request_id=request_id,
            model=model,
            version=resolved_version,
            latency_ms=latency_ms,
            user_id=client.get("user_id"),
            device=device,
            status_code=500,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@router.post("/v1/models/{model}/predict", response_model=PredictResponse)
async def predict_model(
    model: str,
    request: PredictRequest,
    client: Dict[str, Any] = Depends(get_current_api_client),
    _: bool = Depends(rate_limiter),
    db: Session = Depends(get_db),
):
    """
    Execute synchronous prediction on the specified model.
    Utilizes dynamic batching and canary routing.
    """
    return await execute_inference(
        model=model,
        explicit_version=request.version,
        input_data=request.input,
        parameters=request.parameters,
        client=client,
        db=db,
    )


@router.post("/v1/models/{model}/versions/{version}/predict", response_model=PredictResponse)
async def predict_model_version(
    model: str,
    version: str,
    request: PredictRequest,
    client: Dict[str, Any] = Depends(get_current_api_client),
    _: bool = Depends(rate_limiter),
    db: Session = Depends(get_db),
):
    """
    Execute prediction pinned to an explicit model version.
    """
    return await execute_inference(
        model=model,
        explicit_version=version,
        input_data=request.input,
        parameters=request.parameters,
        client=client,
        db=db,
    )


@router.post("/v1/models/{model}/generate", response_model=PredictResponse)
async def generate_text(
    model: str,
    request: GenerateRequest,
    client: Dict[str, Any] = Depends(get_current_api_client),
    _: bool = Depends(rate_limiter),
    db: Session = Depends(get_db),
):
    """
    Text generation endpoint for LLM architectures with autoregressive token sampling.
    """
    parameters = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop": request.stop,
    }
    return await execute_inference(
        model=model,
        explicit_version=None,
        input_data={"prompt": request.prompt},
        parameters=parameters,
        client=client,
        db=db,
    )


@router.post("/v1/predict", response_model=PredictResponse)
async def predict_generic(
    request: PredictRequest,
    client: Dict[str, Any] = Depends(get_current_api_client),
    _: bool = Depends(rate_limiter),
    db: Session = Depends(get_db),
):
    """
    Universal prediction gateway routing based on request payload parameters.
    """
    target_model = request.model or "resnet"
    return await execute_inference(
        model=target_model,
        explicit_version=request.version,
        input_data=request.input,
        parameters=request.parameters,
        client=client,
        db=db,
    )
