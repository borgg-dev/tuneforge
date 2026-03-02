"""
Miner leaderboard for TuneForge.

Tracks exponential moving average (EMA) scores per miner UID.
Applies a steepening function so only consistently good miners
receive significant weight.
"""

import math

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import (
    EMA_ALPHA,
    EMA_WARMUP,
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
        self._warmup_rounds = math.ceil(2.0 / alpha - 1.0)

        self._ema: dict[int, float] = {}
        self._rounds: dict[int, int] = {}

        logger.info(
            f"Leaderboard: alpha={alpha}, warmup={self._warmup_rounds}, "
            f"baseline={steepen_baseline}, power={steepen_power}"
        )

    def update(self, uid: int, raw_score: float) -> None:
        """
        Update EMA for a miner.

        Args:
            uid: Miner UID.
            raw_score: Latest round score in [0, 1].
        """
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        count = self._rounds.get(uid, 0)

        if uid not in self._ema:
            self._ema[uid] = raw_score
        else:
            self._ema[uid] = self._alpha * raw_score + (1.0 - self._alpha) * self._ema[uid]

        self._rounds[uid] = count + 1

    def get_weight(self, uid: int) -> float:
        """
        Get steepened weight for a miner.

        Returns 0.0 for miners that haven't warmed up or whose
        EMA is below the baseline.
        """
        if not self.is_warmed_up(uid):
            return 0.0
        ema = self._ema.get(uid, 0.0)
        return self._steepen(ema)

    def get_ema(self, uid: int) -> float:
        """Get raw EMA score for a miner."""
        return self._ema.get(uid, 0.0)

    def is_warmed_up(self, uid: int) -> bool:
        """Check if miner has enough rounds for a stable EMA."""
        return self._rounds.get(uid, 0) >= self._warmup_rounds

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

    def summary(self) -> dict:
        """Produce leaderboard summary for logging."""
        if not self._ema:
            return {"total_miners": 0, "warmed_up": 0}

        warmed = sum(1 for uid in self._ema if self.is_warmed_up(uid))
        emas = list(self._ema.values())
        weights = [self.get_weight(uid) for uid in self._ema]
        return {
            "total_miners": len(self._ema),
            "warmed_up": warmed,
            "ema_mean": float(np.mean(emas)),
            "ema_max": float(np.max(emas)),
            "weight_mean": float(np.mean(weights)) if weights else 0.0,
            "weight_max": float(np.max(weights)) if weights else 0.0,
        }
