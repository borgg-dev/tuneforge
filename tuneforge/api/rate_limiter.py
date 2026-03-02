"""
Sliding-window rate limiter for the TuneForge API.

Enforces per-API-key request limits with automatic cleanup of expired entries.
"""

import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, HTTPException, Request, status
from loguru import logger

from tuneforge.api.auth import get_api_key


class RateLimiter:
    """In-memory sliding-window rate limiter.

    Tracks timestamps of requests per key and rejects requests that
    exceed the configured limit within the sliding window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300.0  # 5 minutes

    def is_allowed(self, key: str) -> bool:
        """Check if a request from *key* is within the rate limit.

        Returns True if the request is allowed, False if rate-limited.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[key]
            # Remove expired timestamps
            self._requests[key] = [t for t in timestamps if t > cutoff]
            timestamps = self._requests[key]

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)

            # Periodic cleanup of idle keys
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(cutoff)
                self._last_cleanup = now

        return True

    def remaining(self, key: str) -> int:
        """Return the number of remaining requests allowed for *key*."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            active = [t for t in self._requests.get(key, []) if t > cutoff]
        return max(0, self.max_requests - len(active))

    def reset_time(self, key: str) -> float:
        """Seconds until the oldest request in the window expires."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            active = [t for t in self._requests.get(key, []) if t > cutoff]
        if not active:
            return 0.0
        return max(0.0, active[0] + self.window_seconds - now)

    def _cleanup(self, cutoff: float) -> None:
        """Remove keys that have no requests within the current window."""
        dead_keys = [
            k for k, ts in self._requests.items() if not ts or ts[-1] <= cutoff
        ]
        for k in dead_keys:
            del self._requests[k]
        if dead_keys:
            logger.debug("Rate limiter cleaned up {} idle keys", len(dead_keys))


# Module-level singleton — created at import, reconfigured from settings at startup.
_limiter = RateLimiter()


def configure(max_requests: int = 60, window_seconds: int = 60) -> None:
    """Replace the global rate limiter with new limits."""
    global _limiter
    _limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)


async def check_rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
) -> str:
    """FastAPI dependency that enforces per-key rate limiting.

    Returns the API key on success; raises 429 on limit exceeded.
    """
    if not _limiter.is_allowed(api_key):
        retry_after = int(_limiter.reset_time(api_key)) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    return api_key
