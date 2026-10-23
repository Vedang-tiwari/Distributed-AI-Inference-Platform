import json
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from api.core.config import settings
from core.db.session import get_db, APIKey, User
from core.cache.redis_client import cache_client

security_bearer = HTTPBearer(auto_error=False)


async def get_current_api_client(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Validates API key from either Bearer token or X-API-Key header.
    Utilizes Redis caching to minimize database query overhead on high QPS.
    """
    if not settings.AUTH_ENABLED:
        return {
            "user_id": 1,
            "key": "disabled_auth_mode",
            "tier": "enterprise",
            "rate_limit_rpm": settings.PREMIUM_TIER_RPM,
        }

    raw_token: Optional[str] = None
    if auth_credentials and auth_credentials.credentials:
        raw_token = auth_credentials.credentials.strip()
    elif x_api_key:
        raw_token = x_api_key.strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide via 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 1. Check Redis cache for fast auth lookup
    cache_key = f"auth:key:{raw_token}"
    cached_data = cache_client.get(cache_key)
    if cached_data:
        try:
            return json.loads(cached_data)
        except Exception:
            pass

    # 2. Query Database
    api_key_record = db.query(APIKey).filter(APIKey.key == raw_token).first()
    if not api_key_record or not api_key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or deactivated API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_info = {
        "user_id": api_key_record.user_id,
        "key": api_key_record.key,
        "tier": api_key_record.tier,
        "rate_limit_rpm": api_key_record.rate_limit_rpm,
    }

    # Cache user auth info for 5 minutes
    cache_client.set(cache_key, json.dumps(user_info), ex=300)

    return user_info
