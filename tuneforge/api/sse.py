"""
Server-Sent Events for real-time generation status updates.

Uses Redis pub/sub to receive status changes and streams them
to connected clients in SSE format.
"""

import asyncio
import json
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import Request
from loguru import logger

if TYPE_CHECKING:
    from tuneforge.api.redis_manager import RedisManager


class SSEManager:
    """Manages SSE connections backed by Redis pub/sub."""

    def __init__(self, redis_manager: "RedisManager") -> None:
        self._redis = redis_manager

    async def event_stream(
        self, request_id: str, request: Request
    ) -> AsyncGenerator[str, None]:
        """Yield SSE events for a generation request.

        Subscribes to Redis channel ``sse:generation:{request_id}``
        and yields events in SSE format until the generation
        completes, fails, or the client disconnects.
        """
        channel = f"sse:generation:{request_id}"
        pubsub = await self._redis.subscribe(channel)

        try:
            # Send initial connection event
            yield _format_event({"status": "connected", "request_id": request_id})

            # Listen for messages with timeout to check for disconnects
            while True:
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected for {}", request_id)
                    break

                try:
                    message = await asyncio.wait_for(
                        _next_message(pubsub), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent connection timeout
                    yield ": keepalive\n\n"
                    continue

                if message is None:
                    continue

                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield _format_event(data)

                    # Terminal states — stop streaming
                    if data.get("status") in ("completed", "failed"):
                        break

        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:
                pass


async def _next_message(pubsub) -> dict | None:
    """Get the next message from a pub/sub subscription."""
    async for message in pubsub.listen():
        if message["type"] == "message":
            return message
    return None


def _format_event(data: dict) -> str:
    """Format data as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"
