from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.db.session import get_db
from core.schemas.requests import (
    ModelRegisterRequest,
    CanaryConfigRequest,
    ModelInfo,
)
from api.core.auth import get_current_api_client
from api.services.model_registry import model_registry_service

router = APIRouter(prefix="/v1/models", tags=["Model Registry"])


@router.get("", response_model=Dict[str, List[Dict[str, Any]]])
def list_models(
    db: Session = Depends(get_db),
    _: Dict[str, Any] = Depends(get_current_api_client),
):
    """
    List all available registered models, their active versions and canary statuses.
    """
    models = model_registry_service.list_models(db)
    return {"models": models}


@router.get("/{model}/versions")
def get_model_versions(
    model: str,
    db: Session = Depends(get_db),
    _: Dict[str, Any] = Depends(get_current_api_client),
):
    """
    Retrieve all registered versions and metadata for a specific model.
    """
    info = model_registry_service.get_model_versions(db, model)
    if not info.get("versions"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model}' not found in registry.",
        )
    return info


@router.post("/register")
def register_model(
    request: ModelRegisterRequest,
    db: Session = Depends(get_db),
    client: Dict[str, Any] = Depends(get_current_api_client),
):
    """
    Register a new model or model version into the platform registry.
    """
    entry = model_registry_service.register_model(
        db=db,
        name=request.name,
        version=request.version,
        framework=request.framework,
        device=request.device,
        artifact_uri=request.artifact_uri or "",
        description=request.description or "",
    )
    return {
        "status": "success",
        "message": f"Model '{request.name}' version '{request.version}' registered successfully.",
        "model": {
            "name": entry.name,
            "version": entry.version,
            "device": entry.device,
            "framework": entry.framework,
            "status": entry.status,
        },
    }


@router.post("/{model}/canary")
def set_canary_deployment(
    model: str,
    request: CanaryConfigRequest,
    db: Session = Depends(get_db),
    _: Dict[str, Any] = Depends(get_current_api_client),
):
    """
    Update canary traffic splitting weights for model versions (e.g. {'v1': 0.8, 'v2': 0.2}).
    Weights must sum to approximately 1.0.
    """
    total = sum(request.weights.values())
    if abs(total - 1.0) > 0.05:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Canary weights must sum to 1.0 (received {total})",
        )

    updated_weights = model_registry_service.set_canary_weights(db, model, request.weights)
    return {
        "status": "success",
        "model": model,
        "canary_weights": updated_weights,
    }
