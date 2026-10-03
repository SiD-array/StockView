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
        assert c.get("sym") == 999


def test_redis_hard_expiration_passed():
    mock_redis = MagicMock()
    mock_redis.zcard.return_value = 1
    mock_redis.info.return_value = {"used_memory": 100, "maxmemory": 1000}

    with patch.object(cache, "get_redis_client", return_value=mock_redis):
        c = TTLCache(ttl_seconds=30, max_stale_seconds=600, name="quote")
        c.set("AAPL", {"price": 100})

        # Check that set was called with ex=600
        assert mock_redis.set.called
        assert mock_redis.set.call_args[1]["ex"] == 600


def test_redis_lru_eviction():
    mock_redis = MagicMock()
    # Simulate count exceeding max_size (max_size=2, count=3)
    mock_redis.zcard.return_value = 3
    mock_redis.zpopmin.return_value = [("stockview:quote:OLD", 1000.0)]
    mock_redis.info.return_value = {"used_memory": 100, "maxmemory": 1000}

    with patch.object(cache, "get_redis_client", return_value=mock_redis):
        c = TTLCache(ttl_seconds=30, max_size=2, name="quote")
        c.set("NEW", {"price": 200})

        # zpopmin should have been called to pop 1 excess item
        assert mock_redis.zpopmin.called
        # And delete should have been called with that popped key
        mock_redis.delete.assert_any_call("stockview:quote:OLD")


def test_redis_memory_guard_triggers_pruning():
    mock_redis = MagicMock()
    # 80MB out of 100MB = 80% (exceeds 75% threshold)
    mock_redis.info.return_value = {"used_memory": 80_000_000, "maxmemory": 100_000_000}
    mock_redis.zcard.return_value = 8
    mock_redis.zpopmin.return_value = [("stockview:model:OLD1", 1.0), ("stockview:model:OLD2", 2.0)]

    with patch.object(cache, "get_redis_client", return_value=mock_redis):
        c = TTLCache(ttl_seconds=30, name="model")
        c._guard_memory_limit(mock_redis, max_threshold=0.75)

        # Should have pruned oldest entries
        assert mock_redis.zpopmin.called
        mock_redis.delete.assert_called_with("stockview:model:OLD1", "stockview:model:OLD2")

