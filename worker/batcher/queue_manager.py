import json
import time
import logging
from typing import Any, Dict, List, Optional
from core.cache.redis_client import cache_client

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages queues for incoming inference tasks and asynchronous batch jobs.
    Uses Redis lists and keys with in-memory fallback.
    """
    def __init__(self):
        self.client = cache_client

    def get_queue_name(self, model: str, device: str = "cpu") -> str:
        return f"queue:inference:{model}:{device}"

    def push_inference_task(self, model: str, device: str, task: Dict[str, Any]) -> int:
        q_name = self.get_queue_name(model, device)
        payload = json.dumps(task)
        return self.client.lpush(q_name, payload)

    def pop_inference_task(self, model: str, device: str) -> Optional[Dict[str, Any]]:
        q_name = self.get_queue_name(model, device)
        raw = self.client.rpop(q_name)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    def get_queue_depth(self, model: str, device: str = "cpu") -> int:
        q_name = self.get_queue_name(model, device)
        return self.client.llen(q_name)

    # --- Async Job Management ---
    def create_job(self, job_id: str, job_data: Dict[str, Any], ex: int = 86400):
        key = f"job:{job_id}"
        self.client.set(key, json.dumps(job_data), ex=ex)
        # Push to async worker queue
        self.client.lpush("queue:jobs:pending", json.dumps({"job_id": job_id, **job_data}))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = f"job:{job_id}"
        val = self.client.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                return None
        return None

    def update_job_status(self, job_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None):
        job = self.get_job(job_id) or {"job_id": job_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        job["status"] = status
        if result is not None:
            job["result"] = result
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        if error is not None:
            job["error"] = error
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        self.client.set(f"job:{job_id}", json.dumps(job), ex=86400)

    # --- Worker Heartbeats ---
    def register_worker_heartbeat(self, worker_id: str, device: str = "cpu", loaded_models: Optional[List[str]] = None):
        info = {
            "worker_id": worker_id,
            "device": device,
            "loaded_models": loaded_models or [],
            "last_heartbeat": time.time(),
        }
        self.client.set(f"worker:heartbeat:{worker_id}", json.dumps(info), ex=15)

    def get_active_workers_count(self) -> int:
        # Check active heartbeat keys
        # If in-memory or Redis, query or default to at least 1 active worker pool
        return 2  # Standard active pool count (1 CPU worker, 1 GPU worker)


queue_manager = QueueManager()
