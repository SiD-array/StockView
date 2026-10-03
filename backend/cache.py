"""Simple in-memory TTL cache for stock data and trained models."""

import time
from threading import Lock
from typing import Any


class TTLCache:
    """Thread-safe TTL cache with a maximum entry count."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 64):
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
                # Keep the expired entry around so it can be served via
                # get_stale() if the upstream provider is unavailable.
                return None
            return value

    def get_stale(self, key: str) -> Any | None:
        """Return the last stored value for key, even if it has expired."""
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
