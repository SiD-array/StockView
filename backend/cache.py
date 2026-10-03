"""
Hybrid TTL Cache with Redis support and in-memory fallback.

Behavior:
  - If REDIS_URL is configured and reachable, data is stored in Redis so it
    survives process restarts and Render spin-downs.
  - If REDIS_URL is not set or Redis is temporarily down, it transparently
    falls back to an in-memory thread-safe TTL cache. The application never
    crashes due to cache issues.
"""

import pickle
import time
from threading import Lock
from typing import Any

from config import get_redis_url

_redis_client = None
_redis_checked = False
_redis_lock = Lock()


def get_redis_client():
    """Lazily initialize and return a shared Redis client, or None if unavailable."""
    global _redis_client, _redis_checked
    with _redis_lock:
        if _redis_checked:
            return _redis_client

        url = get_redis_url()
        if not url:
            _redis_checked = True
            _redis_client = None
            return None

        try:
            import redis  # type: ignore

            client = redis.from_url(
                url,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                decode_responses=False,  # raw bytes for pickle
            )
            # Test connection
            client.ping()
            _redis_client = client
            print("[cache] Connected to Redis successfully.")
        except Exception as exc:
            print(f"[cache] Could not connect to Redis ({exc}); using in-memory cache.")
            _redis_client = None

        _redis_checked = True
        return _redis_client


def is_redis_active() -> bool:
    """Return True if Redis is configured and active."""
    return get_redis_client() is not None


class MemoryTTLCache:
    """Thread-safe in-memory TTL cache with a maximum entry count."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 128):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                return None
            return value

    def get_stale(self, key: str) -> Any | None:
        """Return the last stored value for key, even if expired."""
        with self._lock:
            entry = self._store.get(key)
            return entry[1] if entry is not None else None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_size:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest_key]
            self._store[key] = (time.time() + self.ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class TTLCache:
    """
    Hybrid TTL Cache:
      - Uses Redis if REDIS_URL is reachable.
      - Uses MemoryTTLCache if Redis is unavailable.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 128, name: str = "default"):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.name = name
        self._memory = MemoryTTLCache(ttl_seconds=ttl_seconds, max_size=max_size)

    def _redis_key(self, key: str) -> str:
        return f"stockview:{self.name}:{key}"

    def get(self, key: str) -> Any | None:
        client = get_redis_client()
        if client is not None:
            try:
                raw = client.get(self._redis_key(key))
                if raw is not None:
                    payload = pickle.loads(raw)
                    if time.time() <= payload["expires_at"]:
                        return payload["value"]
                    return None
            except Exception as exc:
                print(f"[cache] Redis get error ({exc}); falling back to memory.")

        return self._memory.get(key)

    def get_stale(self, key: str) -> Any | None:
        """Return the last stored value, even if expired."""
        client = get_redis_client()
        if client is not None:
            try:
                raw = client.get(self._redis_key(key))
                if raw is not None:
                    payload = pickle.loads(raw)
                    return payload.get("value")
            except Exception as exc:
                print(f"[cache] Redis get_stale error ({exc}); falling back to memory.")

        return self._memory.get_stale(key)

    def set(self, key: str, value: Any) -> None:
        # Always update memory for fast local reads
        self._memory.set(key, value)

        client = get_redis_client()
        if client is not None:
            try:
                payload = {
                    "expires_at": time.time() + self.ttl_seconds,
                    "value": value,
                }
                raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                # Keep stale data in Redis for up to 7 days for fallback
                stale_ttl = max(self.ttl_seconds * 5, 86400 * 7)
                client.set(self._redis_key(key), raw, ex=stale_ttl)
            except Exception as exc:
                print(f"[cache] Redis set error ({exc}); saved to memory only.")

    def clear(self) -> None:
        self._memory.clear()
        client = get_redis_client()
        if client is not None:
            try:
                pattern = f"stockview:{self.name}:*"
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
            except Exception as exc:
                print(f"[cache] Redis clear error ({exc}).")
