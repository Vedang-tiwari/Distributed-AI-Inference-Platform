import uuid
import json
import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.schemas.requests import JobCreateRequest, JobResponse
from core.db.session import get_db, InferenceJob
from api.core.auth import get_current_api_client
from api.core.rate_limiter import rate_limiter
from worker.batcher.queue_manager import queue_manager

router = APIRouter(prefix="/v1/jobs", tags=["Asynchronous Jobs"])


@router.post("", response_model=JobResponse)
def submit_async_job(
    request: JobCreateRequest,
    client: Dict[str, Any] = Depends(get_current_api_client),
    _: bool = Depends(rate_limiter),
    db: Session = Depends(get_db),
):
    """
    Submit an asynchronous inference job for execution by distributed background workers.
    Returns immediately with a trackable job_id.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    job_data = {
        "job_id": job_id,
        "user_id": client.get("user_id"),
        "model": request.model,
        "version": request.version or "v1",
        "input": request.input,
        "parameters": request.parameters or {},
        "priority": request.priority,
        "status": "queued",
        "created_at": created_at,
        "result": None,
        "error": None,
    }

    # 1. Save to Database
    db_job = InferenceJob(
        id=job_id,
        user_id=client.get("user_id"),
        model=request.model,
        version=request.version or "v1",
        input_data=json.dumps(request.input),
        parameters=json.dumps(request.parameters or {}),
        status="queued",
    )
    db.add(db_job)
    db.commit()

    # 2. Push to Redis queue
    queue_manager.create_job(job_id, job_data)

    return JobResponse(
        job_id=job_id,
        status="queued",
        model=request.model,
        version=request.version or "v1",
        created_at=created_at,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(
    job_id: str,
    _: Dict[str, Any] = Depends(get_current_api_client),
    db: Session = Depends(get_db),
):
    """
    Poll the status and output of an asynchronous inference job.
    """
    # 1. Check Redis cache first
    cached_job = queue_manager.get_job(job_id)
    if cached_job:
        return JobResponse(
            job_id=cached_job.get("job_id", job_id),
            status=cached_job.get("status", "queued"),
            model=cached_job.get("model", "unknown"),
            version=cached_job.get("version"),
            created_at=cached_job.get("created_at", ""),
            completed_at=cached_job.get("completed_at"),
            result=cached_job.get("result"),
            error=cached_job.get("error"),
        )

    # 2. Fallback to Database
    db_job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    parsed_result = None
    if db_job.result:
        try:
            parsed_result = json.loads(db_job.result)
        except Exception:
            parsed_result = db_job.result

    return JobResponse(
        job_id=db_job.id,
        status=db_job.status,
        model=db_job.model,
        version=db_job.version,
        created_at=str(db_job.created_at),
        completed_at=str(db_job.completed_at) if db_job.completed_at else None,
        result=parsed_result,
        error=db_job.error,
    )
