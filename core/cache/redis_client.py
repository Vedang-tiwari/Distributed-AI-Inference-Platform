import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional
from collections import defaultdict

from api.core.config import settings

logger = logging.getLogger(__name__)


class InMemoryFallbackCache:
    """Thread-safe in-memory Redis emulator for local tests and offline dev."""
    def __init__(self):
        self._lock = threading.Lock()
        self._store: Dict[str, Any] = {}
        self._expiries: Dict[str, float] = {}
        self._lists: Dict[str, List[Any]] = defaultdict(list)
        self._hashes: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def _purge_expired(self, key: str):
        if key in self._expiries and time.time() > self._expiries[key]:
            self._store.pop(key, None)
            self._expiries.pop(key, None)
            self._hashes.pop(key, None)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._purge_expired(key)
            val = self._store.get(key)
            return str(val) if val is not None else None

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._store[key] = str(value)
            if ex:
                self._expiries[key] = time.time() + ex
            elif key in self._expiries:
                self._expiries.pop(key, None)
            return True

    def delete(self, *keys: str) -> int:
        with self._lock:
            count = 0
            for k in keys:
                if k in self._store:
                    del self._store[k]
                    count += 1
                self._expiries.pop(k, None)
                self._hashes.pop(k, None)
            return count

    def incr(self, key: str) -> int:
        with self._lock:
            self._purge_expired(key)
            val = int(self._store.get(key, 0)) + 1
            self._store[key] = val
            return val

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key in self._store or key in self._hashes:
                self._expiries[key] = time.time() + seconds
                return True
            return False

    def lpush(self, key: str, *values: Any) -> int:
        with self._lock:
            for v in values:
                self._lists[key].insert(0, str(v))
            return len(self._lists[key])

    def rpop(self, key: str) -> Optional[str]:
        with self._lock:
            lst = self._lists.get(key)
            if lst and len(lst) > 0:
                return lst.pop()
            return None

    def llen(self, key: str) -> int:
        with self._lock:
            return len(self._lists.get(key, []))

    def hset(self, name: str, key: str, value: Any) -> int:
        with self._lock:
            self._purge_expired(name)
            self._hashes[name][key] = str(value)
            return 1

    def hget(self, name: str, key: str) -> Optional[str]:
        with self._lock:
            self._purge_expired(name)
            val = self._hashes[name].get(key)
            return str(val) if val is not None else None

    def hgetall(self, name: str) -> Dict[str, str]:
        with self._lock:
            self._purge_expired(name)
            return dict(self._hashes[name])

    def ping(self) -> bool:
        return True


class RedisClientWrapper:
    def __init__(self):
        self._client = None
        self._fallback = InMemoryFallbackCache()
        self._use_fallback = False
        self._initialize()

    def _initialize(self):
        try:
            import redis
            client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=1.5)
            client.ping()
            self._client = client
            self._use_fallback = False
            logger.info("Connected to Redis broker at %s", settings.REDIS_URL)
        except Exception as e:
            if settings.REDIS_FALLBACK_MEMORY:
                logger.warning("Redis connection failed (%s). Falling back to in-memory state.", e)
                self._use_fallback = True
                self._client = self._fallback
            else:
                raise e

    @property
    def client(self):
        if self._client is None:
            self._initialize()
        return self._client

    def is_healthy(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def is_using_fallback(self) -> bool:
        return self._use_fallback

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        return self.client.set(key, value, ex=ex)

    def delete(self, *keys: str) -> int:
        return self.client.delete(*keys)

    def incr(self, key: str) -> int:
        return self.client.incr(key)

    def expire(self, key: str, seconds: int) -> bool:
        return self.client.expire(key, seconds)

    def lpush(self, key: str, *values: Any) -> int:
        return self.client.lpush(key, *values)

    def rpop(self, key: str) -> Optional[str]:
        return self.client.rpop(key)

    def llen(self, key: str) -> int:
        return self.client.llen(key)

    def hset(self, name: str, key: str, value: Any) -> int:
        return self.client.hset(name, key, value)

    def hget(self, name: str, key: str) -> Optional[str]:
        return self.client.hget(name, key)

    def hgetall(self, name: str) -> Dict[str, str]:
        return self.client.hgetall(name)


cache_client = RedisClientWrapper()
