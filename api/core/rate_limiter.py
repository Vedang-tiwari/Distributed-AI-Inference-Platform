import time
from typing import Dict, Any
from fastapi import HTTPException, status, Depends
from core.cache.redis_client import cache_client
from api.core.auth import get_current_api_client


class RateLimiter:
    """
    Fixed-window rate limiter with sub-second sliding expiration
    backed by Redis for low latency across distributed API gateway instances.
    """
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds

    async def __call__(self, client: Dict[str, Any] = Depends(get_current_api_client)):
        api_key = client.get("key", "anonymous")
        allowed_rpm = client.get("rate_limit_rpm", 100)

        # Generate a per-minute bucket key
        current_time = int(time.time())
        current_minute = current_time // self.window_seconds
        rate_key = f"rate:{api_key}:{current_minute}"

        # Increment request count
        count = cache_client.incr(rate_key)
        if count == 1:
            cache_client.expire(rate_key, self.window_seconds + 5)

        if count > allowed_rpm:
            retry_after = self.window_seconds - (current_time % self.window_seconds)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {count}/{allowed_rpm} req/min. Please retry in {retry_after}s.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(allowed_rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(current_time + retry_after),
                },
            )

        return True


rate_limiter = RateLimiter()
