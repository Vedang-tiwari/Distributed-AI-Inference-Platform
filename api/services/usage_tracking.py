import time
import math
import logging
from collections import deque
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from core.db.session import SessionLocal, UsageLog
from core.cache.redis_client import cache_client

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, max_samples: int = 5000):
        self.max_samples = max_samples
        self.latencies: deque = deque(maxlen=max_samples)
        self.request_timestamps: deque = deque(maxlen=max_samples)
        self.total_requests = 0
        self.error_requests = 0

    def record_request(self, latency_ms: float, is_error: bool = False):
        now = time.time()
        self.latencies.append(latency_ms)
        self.request_timestamps.append(now)
        self.total_requests += 1
        if is_error:
            self.error_requests += 1

        # Also increment Redis metrics counter
        try:
            cache_client.incr("metrics:total_requests")
            if is_error:
                cache_client.incr("metrics:total_errors")
        except Exception:
            pass

    def get_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_vals = sorted(self.latencies)
        k = (len(sorted_vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_vals[int(k)], 2)
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def get_summary(self) -> Dict[str, Any]:
        now = time.time()
        # Requests in the last 60 seconds
        recent_reqs = sum(1 for t in self.request_timestamps if now - t <= 60)
        rps = round(recent_reqs / 60.0, 2) if recent_reqs > 0 else 0.0
        error_rate = round(self.error_requests / max(self.total_requests, 1), 4)

        return {
            "requests_total": self.total_requests,
            "requests_per_second": rps,
            "error_rate": error_rate,
            "p50_latency_ms": self.get_percentile(0.50),
            "p95_latency_ms": self.get_percentile(0.95),
            "p99_latency_ms": self.get_percentile(0.99),
        }


metrics_tracker = MetricsTracker()


def log_inference_usage(
    request_id: str,
    model: str,
    version: str,
    latency_ms: float,
    user_id: Optional[int] = None,
    device: str = "cpu",
    worker_id: Optional[str] = None,
    status_code: int = 200,
):
    """
    Records request in the DB asynchronously or via lightweight session,
    and updates operational telemetry in memory and Redis.
    """
    is_err = status_code >= 400
    metrics_tracker.record_request(latency_ms, is_error=is_err)

    db: Session = SessionLocal()
    try:
        log_entry = UsageLog(
            request_id=request_id,
            user_id=user_id,
            model=model,
            version=version,
            device=device,
            worker_id=worker_id,
            latency_ms=latency_ms,
            status_code=status_code,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to record usage log: {e}")
        db.rollback()
    finally:
        db.close()
