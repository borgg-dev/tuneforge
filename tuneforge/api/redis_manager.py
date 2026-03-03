"""
Redis connection management for TuneForge.

Provides centralized Redis operations for:
- Sliding-window rate limiting
- SSE pub/sub for real-time generation status
- Generation status caching
"""

import json
import time

import redis.asyncio as redis
from loguru import logger


class RedisManager:
    """Centralized Redis connection and operations manager."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._pool = redis.ConnectionPool.from_url(
            url, decode_responses=True, max_connections=20
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Separate client for pub/sub to avoid blocking the main connection
        self._pubsub_pool = redis.ConnectionPool.from_url(
            url, decode_responses=True, max_connections=10
        )
        safe_url = url.split("@")[-1] if "@" in url else url
        logger.info("Redis manager initialized: {}", safe_url)

    async def ping(self) -> bool:
        """Verify the Redis connection is alive."""
        return await self._client.ping()

    async def close(self) -> None:
        """Shut down Redis connections."""
        await self._client.aclose()
        await self._pool.disconnect()
        await self._pubsub_pool.disconnect()

    @property
    def client(self) -> redis.Redis:
        return self._client

    # --- Rate Limiting ---

    async def check_rate_limit(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, int]:
        """Sliding-window rate limiter using sorted sets.

        Returns (allowed, remaining_requests).
        """
        now = time.time()
        redis_key = f"ratelimit:{key}"

        pipe = self._client.pipeline(transaction=True)
        pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
        pipe.zadd(redis_key, {f"{now}:{id(pipe)}": now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, window_seconds + 1)

        results = await pipe.execute()
        current_count = results[2]

        if current_count > max_requests:
            # Remove the entry we just added since request is denied
            await self._client.zremrangebyscore(redis_key, now, now + 0.001)
            return False, 0
        return True, max(0, max_requests - current_count)

    # --- SSE Pub/Sub ---

    async def publish_status(
        self, request_id: str, status: str, **kwargs
    ) -> None:
        """Publish a generation status update to the SSE channel."""
        channel = f"sse:generation:{request_id}"
        data = {"status": status, "timestamp": time.time(), **kwargs}
        await self._client.publish(channel, json.dumps(data))

        # Also cache in a hash for poll-based status checks
        await self._client.hset(
            f"generation:{request_id}",
            mapping={"status": status, "updated_at": str(time.time())},
        )
        await self._client.expire(f"generation:{request_id}", 3600)

    async def subscribe(self, channel: str) -> redis.client.PubSub:
        """Subscribe to a Redis pub/sub channel."""
        pubsub_client = redis.Redis(connection_pool=self._pubsub_pool)
        pubsub = pubsub_client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

    # --- Generation Status Cache ---

    async def get_generation_status(self, request_id: str) -> dict | None:
        """Get cached generation status from Redis hash."""
        data = await self._client.hgetall(f"generation:{request_id}")
        return data or None
