"""
Redis utilities for production-grade session, rate limiting, deduplication, and locking.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import Any, Dict, Optional, ContextManager
import orjson

try:
    import redis
except Exception as e:  # pragma: no cover
    raise ImportError("redis package is required. Install with 'pip install redis'.") from e

logger = logging.getLogger(__name__)


_DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
_SESSION_TTL = int(os.getenv("SESSION_CACHE_TTL_SECONDS", "900"))  # 15 minutes default
_RL_WINDOW = int(os.getenv("RL_WINDOW_SECONDS", "60"))
_RL_MAX = int(os.getenv("RL_MAX_MESSAGES", "10"))
_DEDUPE_TTL = int(os.getenv("DEDUPE_TTL_SECONDS", "86400"))  # 24 hours
_ORDER_TTL = int(os.getenv("ORDER_TTL_SECONDS", "86400"))


_client: Optional["redis.Redis"] = None


def get_redis() -> "redis.Redis":
    """Return a singleton Redis client or raise if unavailable."""
    global _client
    if _client is not None:
        return _client
    try:
        _client = redis.from_url(_DEFAULT_REDIS_URL, decode_responses=True)
        # ping to validate
        _client.ping()
        # Use log_once to prevent duplicate logs across workers
        try:
            from .log_once import log_once_info
            log_once_info(logger, "redis_connected", "Redis connected: %s", _DEFAULT_REDIS_URL)
        except (ImportError, Exception):
            # Fallback to normal logging if log_once isn't available yet
            logger.info("Redis connected: %s", _DEFAULT_REDIS_URL)
        return _client
    except Exception as e:
        logger.critical("REDIS_FAILURE: Failed to connect to Redis at %s: %s", _DEFAULT_REDIS_URL, e)
        raise RuntimeError(f"Failed to connect to Redis at {_DEFAULT_REDIS_URL}") from e


class RedisLock(ContextManager["RedisLock"]):
    """Simple Redis-based distributed lock with token verification."""

    def __init__(self, key: str, ttl_seconds: float = 10.0, wait_timeout: float = 5.0):
        self._client = get_redis()
        self._key = f"lock:{key}"
        self._ttl_ms = int(ttl_seconds * 1000)
        self._wait_ms = int(wait_timeout * 1000)
        self._token = str(uuid.uuid4())
        self._acquired = False

    def __enter__(self) -> "RedisLock":
        if not self._client:
            raise RuntimeError("RedisLock requires Redis client")
        deadline = time.time() * 1000 + self._wait_ms
        while time.time() * 1000 < deadline:
            try:
                if self._client.set(self._key, self._token, nx=True, px=self._ttl_ms):
                    self._acquired = True
                    break
            except Exception as e:
                logger.critical("REDIS_FAILURE: RedisLock set failed: %s", e)
                raise
            time.sleep(0.05)
        if not self._acquired:
            raise TimeoutError(f"Failed to acquire RedisLock for {self._key} within {self._wait_ms}ms")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._client or not self._acquired:
            return
        # Lua script to release only if token matches
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end"
        )
        try:
            self._client.eval(script, 1, self._key, self._token)
        except Exception as e:
            logger.critical("REDIS_FAILURE: RedisLock release failed: %s", e)
            raise


class SessionCache:
    """JSON-based session cache in Redis with TTL."""

    def __init__(self):
        self._client = get_redis()
        self._ttl = _SESSION_TTL

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("SessionCache requires Redis client")
        try:
            raw = self._client.get(self._key(session_id))
            if not raw:
                return None
            # raw is str due to decode_responses=True; orjson.loads accepts str
            return orjson.loads(raw)
        except Exception as e:
            logger.critical("REDIS_FAILURE: SessionCache.get error: %s", e)
            raise

    def set(self, session_id: str, data: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        if not self._client:
            raise RuntimeError("SessionCache requires Redis client")
        try:
            ttl = ttl_seconds or self._ttl
            # orjson.dumps returns bytes; convert to str for redis with decode_responses=True
            payload = orjson.dumps(data, default=str).decode("utf-8")
            self._client.set(self._key(session_id), payload, ex=ttl)
        except Exception as e:
            logger.critical("REDIS_FAILURE: SessionCache.set error: %s", e)
            raise

    def invalidate(self, session_id: str) -> None:
        if not self._client:
            raise RuntimeError("SessionCache requires Redis client")
        try:
            self._client.delete(self._key(session_id))
        except Exception as e:
            logger.critical("REDIS_FAILURE: SessionCache.invalidate error: %s", e)
            raise


class RateLimiter:
    """Simple fixed-window rate limiter using Redis INCR + EXPIRE."""

    def __init__(self, window_seconds: int = _RL_WINDOW, max_messages: int = _RL_MAX, scope: str = "wa"):
        self._client = get_redis()
        self._window = window_seconds
        self._max = max_messages
        self._scope = scope

    def allow(self, key: str) -> bool:
        if not self._client:
            raise RuntimeError("RateLimiter requires Redis client")
        k = f"rl:{self._scope}:{key}"
        try:
            count = self._client.incr(k)
            if count == 1:
                self._client.expire(k, self._window)
            if count > self._max:
                return False
            return True
        except Exception as e:
            logger.critical("REDIS_FAILURE: RateLimiter error: %s", e)
            raise


class Deduplicator:
    """Reject duplicate message IDs within a TTL window using SETNX."""

    def __init__(self, ttl_seconds: int = _DEDUPE_TTL, scope: str = "wa"):
        self._client = get_redis()
        self._ttl = ttl_seconds
        self._scope = scope

    def is_new(self, message_id: str) -> bool:
        if not self._client:
            raise RuntimeError("Deduplicator requires Redis client")
        key = f"dedupe:{self._scope}:{message_id}"
        try:
            created = self._client.set(key, "1", nx=True, ex=self._ttl)
            return bool(created)
        except Exception as e:
            logger.critical("REDIS_FAILURE: Deduplicator error: %s", e)
            raise


class OrderGuard:
    """Ensure messages are processed in non-decreasing timestamp order per user."""

    def __init__(self, ttl_seconds: int = _ORDER_TTL, scope: str = "wa"):
        self._client = get_redis()
        self._ttl = ttl_seconds
        self._scope = scope

    def allow(self, user_key: str, ts: int) -> bool:
        if not self._client:
            raise RuntimeError("OrderGuard requires Redis client")
        key = f"order:{self._scope}:{user_key}"
        try:
            last = self._client.get(key)
            if last is not None:
                try:
                    last_i = int(last)
                    if ts < last_i:
                        return False
                except Exception as e:
                    logger.warning("OrderGuard: failed to parse last ts '%s': %s", last, e)
            pipe = self._client.pipeline(True)
            pipe.set(key, str(ts))
            pipe.expire(key, self._ttl)
            pipe.execute()
            return True
        except Exception as e:
            logger.critical("REDIS_FAILURE: OrderGuard error: %s", e)
            raise


def session_lock_key(session_id: str) -> str:
    return f"session:{session_id}"
