"""
Usage tracking and billing for the TuneForge API.

Tracks per-API-key generation usage with in-memory aggregation
and optional persistence hooks.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock

from loguru import logger


@dataclass
class UsageStats:
    """Aggregated usage statistics for a billing period."""

    total_requests: int = 0
    total_tracks: int = 0
    total_seconds: float = 0.0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _UsageEntry:
    """Single generation event."""

    request_id: str
    num_tracks: int
    total_seconds: float
    timestamp: float  # monotonic


class UsageTracker:
    """Per-API-key usage tracker with sliding period windows."""

    def __init__(self) -> None:
        self._entries: dict[str, list[_UsageEntry]] = defaultdict(list)
        self._lock = Lock()
        self._start_time = time.time()

    def log_generation(
        self,
        api_key: str,
        request_id: str,
        num_tracks: int,
        total_seconds: float,
    ) -> None:
        """Record a generation event for *api_key*."""
        entry = _UsageEntry(
            request_id=request_id,
            num_tracks=num_tracks,
            total_seconds=total_seconds,
            timestamp=time.monotonic(),
        )
        with self._lock:
            self._entries[api_key].append(entry)
        logger.debug(
            "Usage logged: key={}… req={} tracks={} secs={:.1f}",
            api_key[:8],
            request_id,
            num_tracks,
            total_seconds,
        )

    def get_usage(self, api_key: str, period: str = "month") -> UsageStats:
        """Aggregate usage for *api_key* over the given period.

        Supported periods: ``"hour"``, ``"day"``, ``"month"``.
        """
        period_seconds = {"hour": 3600, "day": 86400, "month": 2592000}.get(period, 2592000)
        cutoff = time.monotonic() - period_seconds
        now_utc = datetime.now(timezone.utc)

        with self._lock:
            entries = self._entries.get(api_key, [])
            active = [e for e in entries if e.timestamp > cutoff]

        stats = UsageStats(
            total_requests=len(active),
            total_tracks=sum(e.num_tracks for e in active),
            total_seconds=sum(e.total_seconds for e in active),
            period_start=datetime.fromtimestamp(
                time.time() - period_seconds, tz=timezone.utc
            ),
            period_end=now_utc,
        )
        return stats

    def cleanup(self, max_age_seconds: float = 2592000) -> int:
        """Remove entries older than *max_age_seconds*.

        Returns the number of entries removed.
        """
        cutoff = time.monotonic() - max_age_seconds
        removed = 0
        with self._lock:
            for key in list(self._entries):
                before = len(self._entries[key])
                self._entries[key] = [e for e in self._entries[key] if e.timestamp > cutoff]
                removed += before - len(self._entries[key])
                if not self._entries[key]:
                    del self._entries[key]
        if removed:
            logger.debug("Usage tracker cleaned up {} stale entries", removed)
        return removed
