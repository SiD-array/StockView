"""
Tests for the hybrid TTL Cache (Memory + Redis).

Verifies in-memory TTL logic, stale fallback, max_size eviction,
and graceful Redis fallback when Redis is mocked or unavailable.
"""

import pickle
import time
from unittest.mock import MagicMock, patch

import pytest

import cache
from cache import MemoryTTLCache, TTLCache


def test_memory_cache_basic_and_expiry():
    c = MemoryTTLCache(ttl_seconds=1, max_size=10)
    c.set("k1", "val1")
    assert c.get("k1") == "val1"
    assert c.get_stale("k1") == "val1"

    # Simulate expiry by backdating timestamp
    exp, val = c._store["k1"]
    c._store["k1"] = (time.time() - 10, val)

    assert c.get("k1") is None
    assert c.get_stale("k1") == "val1"


def test_memory_cache_eviction():
    c = MemoryTTLCache(ttl_seconds=60, max_size=2)
    c.set("a", 1)
    time.sleep(0.01)
    c.set("b", 2)
    time.sleep(0.01)
    c.set("c", 3)  # should evict 'a'

    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_hybrid_cache_fallback_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache._redis_checked = False
    cache._redis_client = None

    c = TTLCache(ttl_seconds=60, name="test")
    c.set("foo", {"data": 123})

    assert c.get("foo") == {"data": 123}
    assert c.get_stale("foo") == {"data": 123}
    assert not cache.is_redis_active()


def test_hybrid_cache_uses_redis_when_available():
    mock_redis = MagicMock()
    # Mock ping
    mock_redis.ping.return_value = True

    with patch.object(cache, "get_redis_client", return_value=mock_redis):
        c = TTLCache(ttl_seconds=30, name="quote")
        c.set("AAPL", {"price": 150.0})

        # Verify Redis set was called with pickle payload
        assert mock_redis.set.called
        key_arg = mock_redis.set.call_args[0][0]
        assert key_arg == "stockview:quote:AAPL"

        # Mock Redis get returning a valid unexpired payload
        payload = {"expires_at": time.time() + 100, "value": {"price": 150.0}}
        mock_redis.get.return_value = pickle.dumps(payload)

        res = c.get("AAPL")
        assert res == {"price": 150.0}


def test_hybrid_cache_graceful_redis_failure():
    mock_redis = MagicMock()
    # Redis throws a connection error on get
    mock_redis.get.side_effect = ConnectionError("Redis down")

    with patch.object(cache, "get_redis_client", return_value=mock_redis):
        c = TTLCache(ttl_seconds=30, name="data")
        # Should write to memory
        c.set("sym", 999)
        # On get, redis fails, should smoothly return from memory
        assert c.get("sym") == 999
