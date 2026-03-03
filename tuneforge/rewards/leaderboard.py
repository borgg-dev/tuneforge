"""
Miner leaderboard for TuneForge.

Tracks exponential moving average (EMA) scores per miner UID.
Applies a steepening function so only consistently good miners
receive significant weight.
"""

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import (
    EMA_ALPHA,
    STEEPEN_BASELINE,
    STEEPEN_POWER,
)


class MinerLeaderboard:
    """
    EMA-based miner score tracker with steepening.

    Parameters:
        alpha: EMA smoothing factor (higher → more responsive).
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
        count = self._rounds.get(uid, 0)

        if uid not in self._ema:
            self._ema[uid] = raw_score
        else:
            self._ema[uid] = self._alpha * raw_score + (1.0 - self._alpha) * self._ema[uid]

        self._rounds[uid] = count + 1

    def get_weight(self, uid: int) -> float:
        """
        Get steepened weight for a miner.

        Returns 0.0 if the miner's EMA is below the baseline.
        """
        ema = self._ema.get(uid, 0.0)
        return self._steepen(ema)

    def get_ema(self, uid: int) -> float:
        """Get raw EMA score for a miner."""
        return self._ema.get(uid, 0.0)

    def get_all_weights(self) -> dict[int, float]:
        """Get steepened weights for all tracked miners."""
        return {uid: self.get_weight(uid) for uid in self._ema}

    def get_all_uids(self) -> list[int]:
        """Get all tracked miner UIDs."""
        return list(self._ema.keys())

    def _steepen(self, ema: float) -> float:
        """
        Apply steepening function.

        Maps EMA below baseline to 0, then raises the remainder
        to the given power to amplify high performers.
        """
        if ema <= self._baseline:
            return 0.0
        normalised = (ema - self._baseline) / (1.0 - self._baseline)
        return float(normalised ** self._power)

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the leaderboard state.

        Used to share EMA scores with the organic query router
        (which runs in a separate API server process).
        """
        entries = {}
        for uid in self._ema:
            entries[str(uid)] = {
                "ema": round(self._ema[uid], 6),
                "weight": round(self.get_weight(uid), 6),
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
        import json
        import tempfile
        from pathlib import Path

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
