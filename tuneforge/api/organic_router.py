"""
Organic query router for TuneForge.

Routes product/API generation requests to subnet miners using the EMA
quality scores produced by the validator's challenge scoring pipeline.

Design principles:
- Organic queries do NOT affect miner scores or weights
- The challenge pipeline is the quality signal — EMA scores are the leaderboard
- Pick ONE miner per request (no fan-out) to avoid wasting compute
- Only miners with EMA >= MIN_EMA_THRESHOLD are eligible for organic traffic
- Load-balance across qualifying miners weighted by EMA score
- Track active (in-flight) requests per miner to avoid overloading
- On failure, fall back to the next-best available miner
"""

import asyncio
import copy
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Per-miner tracking (organic-only runtime stats)
# ---------------------------------------------------------------------------

@dataclass
class MinerStats:
    """Runtime stats tracked per miner for the organic path only."""

    uid: int
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    active_requests: int = 0  # in-flight right now

    @property
    def total_requests(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        total = self.total_requests
        return self.success_count / total if total > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.success_count, 1)


# ---------------------------------------------------------------------------
# Leaderboard snapshot (loaded from validator's output)
# ---------------------------------------------------------------------------

@dataclass
class MinerEMA:
    """EMA entry loaded from the validator's leaderboard snapshot."""

    uid: int
    ema: float
    weight: float
    rounds: int


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class OrganicQueryRouter:
    """Routes organic generation requests to the best available miner.

    Miner selection uses the EMA scores from the validator's challenge
    scoring pipeline — that's the quality leaderboard.

    Selection algorithm:
        1. Load leaderboard snapshot (EMA scores from challenges)
        2. Filter: only miners with EMA >= MIN_EMA_THRESHOLD
        3. Score: effective_score = ema / (1 + active_requests * load_penalty)
        4. Penalise miners with recent organic failures
        5. Pick the miner with the highest effective_score
        6. On failure, try the next-best miner (up to MAX_RETRIES)
    """

    # Minimum EMA score to be eligible for organic traffic.
    # Miners below this threshold haven't proven quality through challenges.
    MIN_EMA_THRESHOLD: float = 0.45

    # Maximum concurrent organic requests per miner.
    MAX_ACTIVE_PER_MINER: int = 3

    # How much each active request penalises the effective score.
    LOAD_PENALTY: float = 0.5

    # Penalty multiplier for miners with poor organic success rate.
    # effective_score *= success_rate ^ FAILURE_PENALTY_POWER
    FAILURE_PENALTY_POWER: float = 2.0

    # Maximum different miners to try before giving up.
    MAX_RETRIES: int = 3

    # How often to reload the leaderboard snapshot (seconds).
    SNAPSHOT_RELOAD_INTERVAL: float = 60.0

    def __init__(self, dendrite, metagraph, settings) -> None:
        self._dendrite = dendrite
        self._metagraph = metagraph
        self._settings = settings
        self._miner_stats: dict[int, MinerStats] = {}
        self._lock = asyncio.Lock()

        # Leaderboard snapshot state
        self._leaderboard: dict[int, MinerEMA] = {}
        self._snapshot_path = str(
            Path(settings.storage_path) / "leaderboard.json"
        )
        self._snapshot_last_loaded: float = 0.0
        self._snapshot_mtime: float = 0.0

        # Try to load immediately
        self._load_snapshot()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_metagraph(self, metagraph) -> None:
        """Update the metagraph reference on resync."""
        self._metagraph = metagraph

    async def generate(
        self, synapse, timeout: float = 120.0
    ) -> tuple[bytes | None, dict]:
        """Route a generation request to the best available miner.

        Returns (audio_bytes, metadata_dict) or (None, {}) on total failure.
        Tries up to MAX_RETRIES different miners sequentially.
        """
        # Refresh leaderboard if stale
        self._maybe_reload_snapshot()

        if not self._leaderboard:
            logger.warning("[ORGANIC] No leaderboard data — cannot route")
            return None, {}

        tried: set[int] = set()

        for attempt in range(self.MAX_RETRIES):
            uid = await self._pick_miner(exclude=tried)
            if uid is None:
                logger.warning(
                    "[ORGANIC] No eligible miners left (attempt {}/{})",
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                break

            tried.add(uid)
            hotkey = self._metagraph.hotkeys[uid]
            ema_entry = self._leaderboard.get(uid)
            logger.info(
                "[ORGANIC] Attempt {}/{}: UID {} ({}…) ema={:.4f}",
                attempt + 1,
                self.MAX_RETRIES,
                uid,
                hotkey[:16],
                ema_entry.ema if ema_entry else 0.0,
            )

            result = await self._query_miner(uid, synapse, timeout)
            if result[0] is not None:
                return result

        logger.warning(
            "[ORGANIC] All {} attempts exhausted, no valid audio",
            len(tried),
        )
        return None, {}

    # ------------------------------------------------------------------
    # Leaderboard snapshot loading
    # ------------------------------------------------------------------

    def _load_snapshot(self) -> None:
        """Load the leaderboard snapshot written by the validator."""
        path = Path(self._snapshot_path)
        if not path.exists():
            logger.debug("[ORGANIC] Leaderboard snapshot not found at {}", self._snapshot_path)
            return

        try:
            mtime = path.stat().st_mtime
            if mtime == self._snapshot_mtime:
                return  # File unchanged

            with open(path) as f:
                data = json.load(f)

            entries: dict[int, MinerEMA] = {}
            for uid_str, info in data.get("miners", {}).items():
                uid = int(uid_str)
                entries[uid] = MinerEMA(
                    uid=uid,
                    ema=info["ema"],
                    weight=info["weight"],
                    rounds=info["rounds"],
                )

            self._leaderboard = entries
            self._snapshot_mtime = mtime
            self._snapshot_last_loaded = time.monotonic()

            eligible = sum(
                1 for e in entries.values()
                if e.ema >= self.MIN_EMA_THRESHOLD
            )
            logger.info(
                "[ORGANIC] Leaderboard loaded: {} miners, {} with EMA >= {}",
                len(entries),
                eligible,
                self.MIN_EMA_THRESHOLD,
            )
        except Exception as exc:
            logger.warning("[ORGANIC] Failed to load leaderboard snapshot: {}", exc)

    def _maybe_reload_snapshot(self) -> None:
        """Reload the snapshot if enough time has passed."""
        now = time.monotonic()
        if now - self._snapshot_last_loaded >= self.SNAPSHOT_RELOAD_INTERVAL:
            self._load_snapshot()

    # ------------------------------------------------------------------
    # Miner selection
    # ------------------------------------------------------------------

    async def _pick_miner(self, exclude: set[int] | None = None) -> int | None:
        """Pick the single best miner using EMA scores and load balancing.

        Only miners that are:
        - In the leaderboard with EMA >= MIN_EMA_THRESHOLD
        - Currently serving (axon is_serving)
        - Not at max active requests
        - Not in the exclude set

        are eligible. Among those, pick the one with the highest
        load-adjusted effective score.
        """
        exclude = exclude or set()
        metagraph = self._metagraph
        if metagraph is None:
            return None

        best_uid: int | None = None
        best_score: float = -1.0

        for uid, entry in self._leaderboard.items():
            if uid in exclude:
                continue

            # Minimum EMA quality threshold
            if entry.ema < self.MIN_EMA_THRESHOLD:
                continue

            # Must be serving on the network
            try:
                if not metagraph.axons[uid].is_serving:
                    continue
            except (IndexError, AttributeError):
                continue

            stats = self._miner_stats.get(uid)

            # Skip miners at capacity
            if stats and stats.active_requests >= self.MAX_ACTIVE_PER_MINER:
                continue

            # Base score = EMA (the quality signal from challenges)
            base_score = entry.ema

            # Penalise miners with poor organic track record
            if stats and stats.total_requests >= 3:
                base_score *= stats.success_rate ** self.FAILURE_PENALTY_POWER

            # Load penalty: spread traffic across miners
            active = stats.active_requests if stats else 0
            effective_score = base_score / (1.0 + active * self.LOAD_PENALTY)

            if effective_score > best_score:
                best_score = effective_score
                best_uid = uid

        if best_uid is not None:
            entry = self._leaderboard[best_uid]
            stats = self._miner_stats.get(best_uid)
            logger.debug(
                "[ORGANIC] Selected UID {} — ema={:.4f}, "
                "active={}, effective_score={:.4f}",
                best_uid,
                entry.ema,
                stats.active_requests if stats else 0,
                best_score,
            )

        return best_uid

    # ------------------------------------------------------------------
    # Single-miner query
    # ------------------------------------------------------------------

    def _resolve_axon(self, uid: int):
        """Return the axon for *uid*, rewriting to 127.0.0.1 when the miner
        is co-located with this API server (avoids hairpin-NAT issues).

        Set ``TF_LOCAL_MINER_IPS`` (comma-separated) to list external IPs
        that should be rewritten.  If unset, no rewriting happens.
        """
        axon = self._metagraph.axons[uid]
        local_ips = os.environ.get("TF_LOCAL_MINER_IPS", "")
        if not local_ips:
            return axon
        rewrite_set = {ip.strip() for ip in local_ips.split(",") if ip.strip()}
        if axon.ip in rewrite_set:
            axon = copy.deepcopy(axon)
            logger.debug(
                "[ORGANIC] Rewriting axon IP for UID {} from {} → 127.0.0.1",
                uid, axon.ip,
            )
            axon.ip = "127.0.0.1"
        return axon

    async def _query_miner(
        self, uid: int, synapse, timeout: float
    ) -> tuple[bytes | None, dict]:
        """Send the generation request to a single miner and validate."""
        axon = self._resolve_axon(uid)

        # Track in-flight
        async with self._lock:
            stats = self._miner_stats.setdefault(uid, MinerStats(uid=uid))
            stats.active_requests += 1

        try:
            responses = await self._dendrite.forward(
                axons=[axon],
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )
        except Exception as exc:
            logger.error("[ORGANIC] Dendrite forward failed for UID {}: {}", uid, exc)
            await self._release_and_record(uid, success=False)
            return None, {}

        if not responses or responses[0] is None:
            logger.warning("[ORGANIC] Empty response from UID {}", uid)
            await self._release_and_record(uid, success=False)
            return None, {}

        resp = responses[0]

        try:
            audio_bytes = resp.deserialize()
        except Exception:
            logger.warning("[ORGANIC] Failed to deserialize audio from UID {}", uid)
            await self._release_and_record(uid, success=False)
            return None, {}

        if not audio_bytes or not self._quality_gate(
            audio_bytes, synapse.duration_seconds
        ):
            logger.warning("[ORGANIC] Audio from UID {} failed quality gate", uid)
            await self._release_and_record(uid, success=False)
            return None, {}

        # Success
        await self._release_and_record(uid, success=True, resp=resp)

        return audio_bytes, {
            "miner_uid": uid,
            "miner_hotkey": self._metagraph.hotkeys[uid],
            "generation_time_ms": resp.generation_time_ms or 0,
            "sample_rate": resp.sample_rate or self._settings.generation_sample_rate,
            "model_id": resp.model_id,
        }

    # ------------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------------

    def _quality_gate(self, audio_bytes: bytes, expected_duration: float) -> bool:
        """Quick sanity check on generated audio.

        Checks: non-empty, non-silent, reasonable duration, no extreme clipping.
        """
        if len(audio_bytes) < 1024:
            logger.warning("[QUALITY] Too small: {} bytes", len(audio_bytes))
            return False

        try:
            import numpy as np
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(audio_bytes))
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Non-silent
            rms = float(np.sqrt(np.mean(data**2)))
            if rms < 0.001:
                logger.warning("[QUALITY] Silent audio: rms={:.6f}", rms)
                return False

            # Duration check (within 50% of expected)
            actual_duration = len(data) / sr
            if (
                actual_duration < expected_duration * 0.5
                or actual_duration > expected_duration * 1.5
            ):
                logger.warning(
                    "[QUALITY] Duration mismatch: actual={:.1f}s expected={:.1f}s",
                    actual_duration, expected_duration,
                )
                return False

            # Clipping check
            peak = float(np.max(np.abs(data)))
            if peak > 0.99:
                logger.warning("[QUALITY] Clipping detected: peak={:.4f}", peak)
                return False

            logger.debug(
                "[QUALITY] Passed: duration={:.1f}s rms={:.4f} peak={:.4f}",
                actual_duration, rms, peak,
            )
            return True
        except Exception as exc:
            logger.warning("[QUALITY] Failed to analyze audio: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    async def _release_and_record(
        self, uid: int, success: bool, resp=None
    ) -> None:
        """Decrement active count and record success/failure."""
        async with self._lock:
            stats = self._miner_stats.setdefault(uid, MinerStats(uid=uid))
            stats.active_requests = max(0, stats.active_requests - 1)

            if success:
                stats.success_count += 1
                stats.total_latency_ms += (resp.generation_time_ms or 0) if resp else 0
                stats.last_success = time.monotonic()
            else:
                stats.failure_count += 1
                stats.last_failure = time.monotonic()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_eligible_miners(self) -> list[dict]:
        """Return eligible miners with their scores for monitoring."""
        result = []
        for uid, entry in sorted(self._leaderboard.items()):
            if entry.ema < self.MIN_EMA_THRESHOLD:
                continue
            try:
                serving = self._metagraph.axons[uid].is_serving
            except (IndexError, AttributeError):
                serving = False
            stats = self._miner_stats.get(uid)
            result.append({
                "uid": uid,
                "ema": round(entry.ema, 4),
                "rounds": entry.rounds,
                "serving": serving,
                "organic_success_rate": round(stats.success_rate, 3) if stats else None,
                "organic_avg_latency_ms": round(stats.avg_latency_ms, 1) if stats else None,
                "active_requests": stats.active_requests if stats else 0,
                "total_organic_requests": stats.total_requests if stats else 0,
            })
        return result
