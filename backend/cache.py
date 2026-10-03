"""
Hybrid TTL Cache with Redis support, strict retention limits, and automatic LRU eviction.

Features:
  1. Time-bounded retention: Every key has a hard expiration in Redis (default max 24 hours),
     so old data is automatically purged by Redis.
  2. LRU size cap: Each cache namespace enforces a maximum number of entries (max_size).
     When exceeded, the oldest accessed keys are proactively deleted.
  3. Memory safety guard: If Redis memory usage approaches max capacity (> 75%),
     it proactively evicts the oldest entries to prevent Out-Of-Memory errors.
  4. Resilient fallback: If Redis is absent or fails, falls back to memory safely.
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
    Hybrid TTL Cache with hard time bounds and proactive LRU eviction:
      - Uses Redis if REDIS_URL is reachable.
      - Uses MemoryTTLCache if Redis is unavailable.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 128,
        name: str = "default",
        max_stale_seconds: int | None = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.name = name
        # Maximum duration data can stay in Redis before being permanently purged
        # (defaults to 10x TTL, capped between 5 mins and 24 hours)
        self.max_stale_seconds = max_stale_seconds or max(
            min(self.ttl_seconds * 10, 86400),  # up to 24h
            300,  # at least 5 mins
        )
        self._memory = MemoryTTLCache(ttl_seconds=ttl_seconds, max_size=max_size)

    def _redis_key(self, key: str) -> str:
        return f"stockview:{self.name}:{key}"

    def _index_key(self) -> str:
        return f"stockview:idx:{self.name}"

    def _evict_oldest_in_redis(self, client) -> None:
        """Evict oldest entries if total keys in this namespace exceed max_size."""
        try:
            idx = self._index_key()
            current_count = client.zcard(idx)
            if current_count > self.max_size:
                excess = current_count - self.max_size
                # Pop the oldest accessed keys
                oldest_items = client.zpopmin(idx, excess)
                if oldest_items:
                    keys_to_del = [item[0] for item in oldest_items]
                    client.delete(*keys_to_del)
        except Exception as exc:
            print(f"[cache] Redis LRU eviction notice ({exc}).")

    def _guard_memory_limit(self, client, max_threshold: float = 0.75) -> None:
        """If Redis memory exceeds max_threshold (75%), proactively prune oldest keys."""
        try:
            info = client.info("memory")
            used = info.get("used_memory", 0)
            max_mem = info.get("maxmemory", 0)
            if max_mem > 0 and (used / max_mem) > max_threshold:
                print(f"[cache] Redis memory at {used / max_mem:.1%}; pruning oldest items.")
                idx = self._index_key()
                # Prune oldest 25% of entries in this namespace
                count = max(1, client.zcard(idx) // 4)
                oldest_items = client.zpopmin(idx, count)
                if oldest_items:
                    client.delete(*[item[0] for item in oldest_items])
        except Exception:
            pass

    def get(self, key: str) -> Any | None:
        client = get_redis_client()
        if client is not None:
            try:
                r_key = self._redis_key(key)
                raw = client.get(r_key)
                if raw is not None:
                    payload = pickle.loads(raw)
                    # Update LRU access score
                    client.zadd(self._index_key(), {r_key: time.time()})
                    if time.time() <= payload["expires_at"]:
                        return payload["value"]
                    return None
            except Exception as exc:
                print(f"[cache] Redis get error ({exc}); falling back to memory.")

        return self._memory.get(key)

    def get_stale(self, key: str) -> Any | None:
        """Return the last stored value, even if expired, provided it's within max_stale."""
        client = get_redis_client()
        if client is not None:
            try:
                r_key = self._redis_key(key)
                raw = client.get(r_key)
                if raw is not None:
                    payload = pickle.loads(raw)
                    client.zadd(self._index_key(), {r_key: time.time()})
                    return payload.get("value")
            except Exception as exc:
                print(f"[cache] Redis get_stale error ({exc}); falling back to memory.")

        return self._memory.get_stale(key)

    def set(self, key: str, value: Any) -> None:
        # 1. Update in-memory cache for instant local reads
        self._memory.set(key, value)

        client = get_redis_client()
        if client is not None:
            try:
                # 2. Check if memory is filling up before writing
                self._guard_memory_limit(client)

                r_key = self._redis_key(key)
                payload = {
                    "expires_at": time.time() + self.ttl_seconds,
                    "value": value,
                }
                raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

                # Set hard expiration on the key so it automatically drops off
                client.set(r_key, raw, ex=self.max_stale_seconds)

                # Record key in LRU index with current timestamp
                idx = self._index_key()
                client.zadd(idx, {r_key: time.time()})
                client.expire(idx, self.max_stale_seconds * 2)

                # 3. Enforce max_size cap by evicting oldest keys
                self._evict_oldest_in_redis(client)
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
                client.delete(self._index_key())
            except Exception as exc:
                print(f"[cache] Redis clear error ({exc}).")
