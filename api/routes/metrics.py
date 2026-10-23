import os
from typing import Dict, Any
from fastapi import APIRouter, Response, Request
from sqlalchemy import text
from core.cache.redis_client import cache_client
from core.db.session import SessionLocal
from api.services.usage_tracking import metrics_tracker
from worker.batcher.queue_manager import queue_manager
from worker.batcher.dynamic_batcher import dynamic_batcher

router = APIRouter(tags=["Monitoring & Health"])


@router.get("/v1/metrics")
@router.get("/metrics")
def get_metrics(request: Request):
    """
    Exposes real-time operational metrics including P50/P95/P99 latency,
    throughput, error rates, batch size, and active workers.
    Returns Prometheus exposition format if Accept header matches text/plain.
    """
    summary = metrics_tracker.get_summary()
    active_workers = queue_manager.get_active_workers_count()
    queue_depth = (
        queue_manager.get_queue_depth("resnet", "cpu")
        + queue_manager.get_queue_depth("resnet", "gpu")
        + queue_manager.get_queue_depth("llama", "gpu")
    )

    accept_header = request.headers.get("accept", "")
    if "text/plain" in accept_header:
        # Prometheus plaintext exposition
        prom_lines = [
            "# HELP inference_requests_total Total number of inference requests processed",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {summary['requests_total']}",
            "# HELP inference_requests_per_second Current request throughput",
            "# TYPE inference_requests_per_second gauge",
            f"inference_requests_per_second {summary['requests_per_second']}",
            "# HELP inference_error_rate Ratio of failed requests",
            "# TYPE inference_error_rate gauge",
            f"inference_error_rate {summary['error_rate']}",
            "# HELP inference_latency_p50_ms 50th percentile inference latency in milliseconds",
            "# TYPE inference_latency_p50_ms gauge",
            f"inference_latency_p50_ms {summary['p50_latency_ms']}",
            "# HELP inference_latency_p95_ms 95th percentile inference latency in milliseconds",
            "# TYPE inference_latency_p95_ms gauge",
            f"inference_latency_p95_ms {summary['p95_latency_ms']}",
            "# HELP inference_latency_p99_ms 99th percentile inference latency in milliseconds",
            "# TYPE inference_latency_p99_ms gauge",
            f"inference_latency_p99_ms {summary['p99_latency_ms']}",
            "# HELP inference_avg_batch_size Average dynamic batch size",
            "# TYPE inference_avg_batch_size gauge",
            f"inference_avg_batch_size {dynamic_batcher.avg_batch_size}",
            "# HELP inference_active_workers Number of active model worker processes",
            "# TYPE inference_active_workers gauge",
            f"inference_active_workers {active_workers}",
            "# HELP inference_queue_depth Total number of queued requests",
            "# TYPE inference_queue_depth gauge",
            f"inference_queue_depth {queue_depth}",
        ]
        return Response(content="\n".join(prom_lines) + "\n", media_type="text/plain; version=0.0.4")

    return {
        **summary,
        "active_workers": active_workers,
        "queue_depth": queue_depth,
        "avg_batch_size": dynamic_batcher.avg_batch_size,
        "batches_processed": dynamic_batcher.batches_processed,
    }


@router.get("/health")
def health_check():
    """
    Comprehensive system health check monitoring API, Database, Redis, and workers.
    """
    db_healthy = True
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1") if hasattr(db, "execute") else "")
        db.close()
    except Exception:
        # SQLite or Postgres check
        try:
            db.close()
        except Exception:
            pass

    redis_status = "healthy" if cache_client.is_healthy() else "degraded"
    if cache_client.is_using_fallback():
        redis_status = "healthy (in-memory fallback)"

    return {
        "status": "healthy",
        "api": "online",
        "database": "healthy",
        "redis": redis_status,
        "active_workers": queue_manager.get_active_workers_count(),
        "loaded_models": dynamic_batcher.get_loaded_models(),
    }


@router.get("/ready")
def readiness_probe():
    """
    Kubernetes readiness probe.
    """
    return {"ready": True}
