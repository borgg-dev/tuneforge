"""
Miner leaderboard for TuneForge.

Tracks exponential moving average (EMA) scores per miner UID.
Applies a power-law curve so top performers receive disproportionately
more weight — all miners are scored, no threshold cutoff.
"""

import json
import tempfile
import threading
from pathlib import Path

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import (
    EMA_ALPHA,
    EMA_NEW_MINER_SEED,
    ELITE_K,
    ELITE_POOL,
    STEEPEN_POWER,
)

EMA_STATE_VERSION = 3


class MinerLeaderboard:
    """
    EMA-based miner score tracker with tiered power-law weighting.

    Miners are ranked by EMA and split into two tiers:
      - **Elite** (top K): share ``elite_pool`` fraction of total weight (default 80%)
      - **Remaining**: share ``1 - elite_pool`` fraction (default 20%)

    Within each tier, weight is distributed by ``ema ^ power``.
    This creates a highly competitive landscape — miners fight for top-K
    slots to earn the lion's share of incentive, mirroring organic routing
    where only top-ranked miners receive real user queries.

    Parameters:
        alpha: EMA smoothing factor (higher -> more responsive).
        power: Exponent for the power-law curve (2.0 = quadratic).
        elite_k: Number of miners in the elite tier.
        elite_pool: Fraction of total weight reserved for elite tier.
    """

    def __init__(
        self,
        alpha: float = EMA_ALPHA,
        power: float = STEEPEN_POWER,
        elite_k: int = ELITE_K,
        elite_pool: float = ELITE_POOL,
    ) -> None:
        self._alpha = alpha
        self._power = power
        self._elite_k = elite_k
        self._elite_pool = elite_pool

        self._ema: dict[int, float] = {}
        self._rounds: dict[int, int] = {}
        self._hotkeys: dict[int, str] = {}
        self._lock = threading.RLock()

        logger.info(
            f"Leaderboard: alpha={alpha}, power={power}, "
            f"elite_k={elite_k}, elite_pool={elite_pool}"
        )

    def update(self, uid: int, raw_score: float) -> None:
        """
        Update EMA for a miner.

        Args:
            uid: Miner UID.
            raw_score: Latest round score in [0, 1].
        """
        raw_score = float(np.clip(np.nan_to_num(raw_score, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))

        with self._lock:
            count = self._rounds.get(uid, 0)

            if uid not in self._ema:
                self._ema[uid] = EMA_NEW_MINER_SEED
            self._ema[uid] = self._alpha * raw_score + (1.0 - self._alpha) * self._ema[uid]

            self._rounds[uid] = count + 1

    def check_hotkey_changes(self, uid_to_hotkey: dict[int, str]) -> list[int]:
        """Detect UID recycling by comparing hotkeys against last known values.

        When a UID gets a new hotkey (new miner registered on that slot),
        its EMA and round count are reset to 0 so the new miner starts fresh.

        Args:
            uid_to_hotkey: Mapping of UID -> current hotkey from metagraph.

        Returns:
            List of UIDs that were reset due to hotkey change.
        """
        reset_uids = []
        with self._lock:
            for uid, hotkey in uid_to_hotkey.items():
                prev = self._hotkeys.get(uid)
                if prev is not None and prev != hotkey:
                    # Hotkey changed — new miner on this UID
                    old_ema = self._ema.get(uid, 0.0)
                    self._ema.pop(uid, None)
                    self._rounds.pop(uid, None)
                    reset_uids.append(uid)
                    logger.info(
                        f"UID {uid} hotkey changed ({prev[:8]}… → {hotkey[:8]}…), "
                        f"EMA reset from {old_ema:.4f} to 0.0"
                    )
                self._hotkeys[uid] = hotkey

            # Prune stale entries for UIDs no longer in the metagraph
            current_uids = set(uid_to_hotkey.keys())
            stale_uids = [u for u in self._ema if u not in current_uids]
            for uid in stale_uids:
                self._ema.pop(uid, None)
                self._rounds.pop(uid, None)
                self._hotkeys.pop(uid, None)
            if stale_uids:
                logger.info(f"Pruned {len(stale_uids)} stale UIDs from leaderboard: {stale_uids}")

        return reset_uids

    def get_weight(self, uid: int) -> float:
        """
        Get power-law weight for a miner.

        Weight = ema ^ power. All miners with EMA > 0 get weight.
        """
        with self._lock:
            ema = self._ema.get(uid, 0.0)
        if ema <= 0.0:
            return 0.0
        return float(ema ** self._power)

    def get_ema(self, uid: int) -> float:
        """Get raw EMA score for a miner."""
        with self._lock:
            return self._ema.get(uid, 0.0)

    def get_all_weights(self) -> dict[int, float]:
        """Get tiered power-law weights for all tracked miners.

        Top ``elite_k`` miners (by EMA) share ``elite_pool`` of total weight.
        Remaining miners share ``1 - elite_pool``.
        Within each tier, weight is proportional to ``ema ^ power``.
        """
        with self._lock:
            scored = {uid: ema for uid, ema in self._ema.items() if ema > 0}

        if not scored:
            return {}

        # Rank by EMA descending
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        elite_k = min(self._elite_k, len(ranked))
        elite = ranked[:elite_k]
        rest = ranked[elite_k:]

        def _distribute(miners: list, pool: float) -> dict[int, float]:
            if not miners:
                return {}
            raw = {uid: ema ** self._power for uid, ema in miners}
            total = sum(raw.values())
            if total <= 0:
                return {}
            return {uid: (w / total) * pool for uid, w in raw.items()}

        # If all miners fit in elite tier, they get 100%
        if not rest:
            return _distribute(elite, 1.0)

        weights = _distribute(elite, self._elite_pool)
        weights.update(_distribute(rest, 1.0 - self._elite_pool))
        return weights

    def get_all_uids(self) -> list[int]:
        """Get all tracked miner UIDs."""
        with self._lock:
            return list(self._ema.keys())

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the leaderboard state.

        Used to share EMA scores with the organic query router
        (which runs in a separate API server process).
        """
        with self._lock:
            all_weights = self.get_all_weights()
            ranked = sorted(self._ema.items(), key=lambda x: x[1], reverse=True)
            entries = {}
            for rank, (uid, ema) in enumerate(ranked):
                entries[str(uid)] = {
                    "ema": round(ema, 6),
                    "weight": round(all_weights.get(uid, 0.0), 6),
                    "rank": rank + 1,
                    "tier": "elite" if rank < self._elite_k else "rest",
                    "rounds": self._rounds.get(uid, 0),
                }
            return {
                "miners": entries,
                "elite_k": self._elite_k,
                "elite_pool": self._elite_pool,
            }

    def save_snapshot(self, path: str) -> None:
        """Write leaderboard snapshot to a JSON file.

        The organic query router reads this file to pick miners
        based on challenge-derived EMA quality scores.
        """
        data = self.snapshot()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file + rename
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(dest.parent), suffix=".tmp", delete=False
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
            Path(tmp_path).rename(dest)
            logger.debug("Leaderboard snapshot saved to {}", path)
        except Exception as exc:
            logger.warning("Failed to save leaderboard snapshot: {}", exc)

    def save_state(self, path: str) -> None:
        """Persist full EMA state to disk with atomic write and .bak backup.

        JSON format:
            {"version": 2, "alpha": ..., "power": ...,
             "ema": {"uid": value, ...}, "rounds": {"uid": count, ...}}
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "version": EMA_STATE_VERSION,
                "alpha": self._alpha,
                "power": self._power,
                "ema": {str(k): v for k, v in self._ema.items()},
                "rounds": {str(k): v for k, v in self._rounds.items()},
                "hotkeys": {str(k): v for k, v in self._hotkeys.items()},
            }

        # Create backup of existing file
        if dest.exists():
            bak = dest.with_suffix(dest.suffix + ".bak")
            try:
                bak.write_text(dest.read_text())
            except Exception as exc:
                logger.warning("Failed to create EMA state backup: {}", exc)

        # Atomic write via temp file + rename
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(dest.parent), suffix=".tmp", delete=False
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
            Path(tmp_path).rename(dest)
            logger.info("EMA state saved to {} ({} miners)", path, len(data["ema"]))
        except Exception as exc:
            logger.error("Failed to save EMA state: {}", exc)

    def load_state(self, path: str) -> bool:
        """Load EMA state from disk.

        Falls back to .bak if primary file is corrupt.
        Validates UIDs are non-negative and EMA values are in [0, 1].
        Accepts both v1 (old baseline format) and v2 state files.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        dest = Path(path)
        bak = dest.with_suffix(dest.suffix + ".bak")

        # Try primary file first, then backup
        for source, filepath in [("primary", dest), ("backup", bak)]:
            if not filepath.exists():
                continue
            try:
                raw = filepath.read_text()
                data = json.loads(raw)
                return self._apply_loaded_state(data, source)
            except Exception as exc:
                logger.warning("Failed to load EMA state from {} ({}): {}", filepath, source, exc)

        logger.info("No EMA state file found at {} — starting fresh", path)
        return False

    def _apply_loaded_state(self, data: dict, source: str) -> bool:
        """Apply loaded state data after validation.

        Args:
            data: Parsed JSON state data.
            source: Description of the source ("primary" or "backup").

        Returns:
            True if state was applied successfully.
        """
        if not isinstance(data, dict):
            logger.warning("EMA state ({}) is not a dict — skipping", source)
            return False

        version = data.get("version", 0)
        if version not in (1, 2, 3):
            logger.warning("EMA state ({}) version {} not supported — skipping", source, version)
            return False

        raw_ema = data.get("ema", {})
        raw_rounds = data.get("rounds", {})

        loaded_ema: dict[int, float] = {}
        loaded_rounds: dict[int, int] = {}
        skipped = 0

        for uid_str, value in raw_ema.items():
            try:
                uid = int(uid_str)
            except (ValueError, TypeError):
                skipped += 1
                continue

            if uid < 0:
                logger.warning("Skipping negative UID {} in EMA state ({})", uid, source)
                skipped += 1
                continue

            try:
                ema_val = float(value)
            except (ValueError, TypeError):
                skipped += 1
                continue

            if not (0.0 <= ema_val <= 1.0):
                logger.warning("Skipping UID {} with out-of-range EMA {} ({})", uid, ema_val, source)
                skipped += 1
                continue

            loaded_ema[uid] = ema_val

        for uid_str, count in raw_rounds.items():
            try:
                uid = int(uid_str)
                count_val = int(count)
                if uid >= 0 and count_val >= 0:
                    loaded_rounds[uid] = count_val
            except (ValueError, TypeError):
                continue

        # Load hotkeys (v2+ only, v1 files won't have them)
        loaded_hotkeys: dict[int, str] = {}
        for uid_str, hotkey in data.get("hotkeys", {}).items():
            try:
                uid = int(uid_str)
                if uid >= 0 and isinstance(hotkey, str) and hotkey:
                    loaded_hotkeys[uid] = hotkey
            except (ValueError, TypeError):
                continue

        with self._lock:
            self._ema = loaded_ema
            self._rounds = loaded_rounds
            self._hotkeys = loaded_hotkeys

        logger.info(
            "Loaded EMA state from {} ({}): {} miners, {} hotkeys, {} skipped",
            source, "primary" if source == "primary" else "backup",
            len(loaded_ema), len(loaded_hotkeys), skipped,
        )
        return True

    def summary(self) -> dict:
        """Produce leaderboard summary for logging."""
        if not self._ema:
            return {"total_miners": 0, "with_weight": 0, "elite_count": 0}

        all_weights = self.get_all_weights()
        with_weight = sum(1 for w in all_weights.values() if w > 0)
        emas = list(self._ema.values())
        weights = list(all_weights.values()) if all_weights else []

        ranked = sorted(self._ema.items(), key=lambda x: x[1], reverse=True)
        elite_count = min(self._elite_k, len(ranked))
        elite_weight = sum(all_weights.get(uid, 0) for uid, _ in ranked[:elite_count])

        return {
            "total_miners": len(self._ema),
            "with_weight": with_weight,
            "elite_count": elite_count,
            "elite_weight_share": round(elite_weight, 4) if weights else 0.0,
            "ema_mean": float(np.mean(emas)),
            "ema_max": float(np.max(emas)),
            "weight_mean": float(np.mean(weights)) if weights else 0.0,
            "weight_max": float(np.max(weights)) if weights else 0.0,
        }
