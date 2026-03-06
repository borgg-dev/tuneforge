"""
Miner leaderboard for TuneForge.

Tracks exponential moving average (EMA) scores per miner UID.
Applies a steepening function so only consistently good miners
receive significant weight.
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
    STEEPEN_BASELINE,
    STEEPEN_POWER,
)

EMA_STATE_VERSION = 1


class MinerLeaderboard:
    """
    EMA-based miner score tracker with steepening.

    Parameters:
        alpha: EMA smoothing factor (higher -> more responsive).
        steepen_baseline: EMA below this maps to weight 0.
        steepen_power: Exponent for the steepening curve.
    """

    def __init__(
        self,
        alpha: float = EMA_ALPHA,
        steepen_baseline: float = STEEPEN_BASELINE,
        steepen_power: float = STEEPEN_POWER,
    ) -> None:
        self._alpha = alpha
        self._baseline = steepen_baseline
        self._power = steepen_power

        self._ema: dict[int, float] = {}
        self._rounds: dict[int, int] = {}
        self._lock = threading.Lock()

        logger.info(
            f"Leaderboard: alpha={alpha}, "
            f"baseline={steepen_baseline}, power={steepen_power}"
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

    def get_weight(self, uid: int) -> float:
        """
        Get steepened weight for a miner.

        Returns 0.0 if the miner's EMA is below the baseline.
        """
        with self._lock:
            ema = self._ema.get(uid, 0.0)
        return self._steepen(ema)

    def get_ema(self, uid: int) -> float:
        """Get raw EMA score for a miner."""
        with self._lock:
            return self._ema.get(uid, 0.0)

    def get_all_weights(self) -> dict[int, float]:
        """Get steepened weights for all tracked miners."""
        with self._lock:
            uids = list(self._ema.keys())
        return {uid: self.get_weight(uid) for uid in uids}

    def get_all_uids(self) -> list[int]:
        """Get all tracked miner UIDs."""
        return list(self._ema.keys())

    def _steepen(self, ema: float) -> float:
        """
        Apply steepening function with soft floor.

        Uses a sigmoid transition around the baseline instead of a hard cliff.
        This prevents oscillation for miners near the threshold and reduces
        on-chain weight churn.

        Below (baseline - margin): weight approaches 0
        At baseline: weight is approximately 0.5 of the power-law value
        Above baseline: standard power-law steepening
        """
        # Soft sigmoid transition (width = 0.05 around baseline)
        sigmoid_width = 0.05
        sigmoid = 1.0 / (1.0 + np.exp(-(ema - self._baseline) / sigmoid_width))

        # Power-law component (normalized above baseline)
        if ema <= 0.0:
            return 0.0
        normalised = max(0.0, (ema - self._baseline) / (1.0 - self._baseline))
        power_component = float(normalised ** self._power)

        # Combine: sigmoid gates the power-law value
        return float(sigmoid * power_component)

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the leaderboard state.

        Used to share EMA scores with the organic query router
        (which runs in a separate API server process).
        """
        with self._lock:
            entries = {}
            for uid in self._ema:
                entries[str(uid)] = {
                    "ema": round(self._ema[uid], 6),
                    "weight": round(self._steepen(self._ema[uid]), 6),
                    "rounds": self._rounds.get(uid, 0),
                }
            return {
                "baseline": self._baseline,
                "miners": entries,
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
            {"version": 1, "alpha": ..., "baseline": ..., "power": ...,
             "ema": {"uid": value, ...}, "rounds": {"uid": count, ...}}
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            data = {
                "version": EMA_STATE_VERSION,
                "alpha": self._alpha,
                "baseline": self._baseline,
                "power": self._power,
                "ema": {str(k): v for k, v in self._ema.items()},
                "rounds": {str(k): v for k, v in self._rounds.items()},
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
        if version != EMA_STATE_VERSION:
            logger.warning("EMA state ({}) version {} != {} — skipping", source, version, EMA_STATE_VERSION)
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

        with self._lock:
            self._ema = loaded_ema
            self._rounds = loaded_rounds

        logger.info(
            "Loaded EMA state from {} ({}): {} miners, {} skipped",
            source, "primary" if source == "primary" else "backup",
            len(loaded_ema), skipped,
        )
        return True

    def summary(self) -> dict:
        """Produce leaderboard summary for logging."""
        if not self._ema:
            return {"total_miners": 0, "above_baseline": 0}

        above = sum(1 for uid in self._ema if self.get_weight(uid) > 0)
        emas = list(self._ema.values())
        weights = [self.get_weight(uid) for uid in self._ema]
        return {
            "total_miners": len(self._ema),
            "above_baseline": above,
            "ema_mean": float(np.mean(emas)),
            "ema_max": float(np.max(emas)),
            "weight_mean": float(np.mean(weights)) if weights else 0.0,
            "weight_max": float(np.max(weights)) if weights else 0.0,
        }
