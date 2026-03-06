"""
Stereo quality scorer for TuneForge.

Assesses stereo field quality: width, phase coherence, mid/side balance.
Penalizes mono-disguised-as-stereo and anti-phase content.
"""

import numpy as np
from loguru import logger


class StereoQualityScorer:
    """Assess stereo field quality of generated audio."""

    WEIGHTS: dict[str, float] = {
        "stereo_width": 0.30,
        "phase_coherence": 0.35,
        "mid_side_balance": 0.35,
    }

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Score stereo quality. Expects 2-channel audio (shape: [samples, 2]).
        If mono (1-D or single channel), returns penalty scores."""
        try:
            if audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1):
                # Mono audio — mild penalty (many AI models are mono)
                return {
                    "stereo_width": 0.3,
                    "phase_coherence": 0.8,
                    "mid_side_balance": 0.3,
                }

            if audio.ndim == 2 and audio.shape[0] == 2:
                # Channels-first format
                left = audio[0].astype(np.float32)
                right = audio[1].astype(np.float32)
            elif audio.ndim == 2 and audio.shape[1] == 2:
                # Channels-last format
                left = audio[:, 0].astype(np.float32)
                right = audio[:, 1].astype(np.float32)
            else:
                return {k: 0.5 for k in self.WEIGHTS}

            return {
                "stereo_width": self._score_stereo_width(left, right),
                "phase_coherence": self._score_phase_coherence(left, right),
                "mid_side_balance": self._score_mid_side_balance(left, right),
            }

        except Exception as exc:
            logger.error("Stereo quality scoring failed: {}", exc)
            return {k: 0.5 for k in self.WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        total = sum(self.WEIGHTS[k] * scores.get(k, 0.0) for k in self.WEIGHTS)
        return float(np.clip(total, 0.0, 1.0))

    @staticmethod
    def _score_stereo_width(left: np.ndarray, right: np.ndarray) -> float:
        """Cross-correlation between L/R. Identical = mono disguised as stereo."""
        if len(left) == 0:
            return 0.0

        # Normalized cross-correlation at zero lag
        l_norm = left - np.mean(left)
        r_norm = right - np.mean(right)
        l_std = np.std(l_norm)
        r_std = np.std(r_norm)

        if l_std < 1e-8 or r_std < 1e-8:
            return 0.0

        correlation = float(np.mean(l_norm * r_norm) / (l_std * r_std))

        # Correlation ~1.0 = mono disguised as stereo -> score 0.2
        # Correlation ~0.3-0.7 = good stereo -> score 1.0
        # Correlation ~0.0 = uncorrelated -> score 0.5
        if correlation > 0.95:
            return 0.2  # Mono disguised as stereo
        elif 0.3 <= correlation <= 0.8:
            return 1.0  # Good stereo
        elif correlation < 0.1:
            return 0.5  # Uncorrelated (unusual)
        elif correlation > 0.8:
            # Linear ramp from 1.0 at 0.8 to 0.2 at 0.95
            return 1.0 - (correlation - 0.8) / 0.15 * 0.8
        else:
            # 0.1-0.3 range
            return 0.5 + (correlation - 0.1) / 0.2 * 0.5

    @staticmethod
    def _score_phase_coherence(left: np.ndarray, right: np.ndarray) -> float:
        """Detect anti-phase content. Mid/side energy analysis."""
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        mid_energy = float(np.sum(mid**2))
        side_energy = float(np.sum(side**2)) + 1e-10

        if mid_energy < 1e-10:
            return 0.0  # All anti-phase

        ratio = mid_energy / side_energy

        # Good: mid dominant (ratio > 2). Bad: side dominant (ratio < 1)
        if ratio >= 3.0:
            return 1.0
        elif ratio >= 1.0:
            return 0.5 + 0.5 * (ratio - 1.0) / 2.0
        else:
            return float(np.clip(ratio * 0.5, 0.0, 0.5))

    @staticmethod
    def _score_mid_side_balance(left: np.ndarray, right: np.ndarray) -> float:
        """Score the mid/side energy ratio for good stereo balance."""
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        mid_energy = float(np.sum(mid**2))
        side_energy = float(np.sum(side**2)) + 1e-10

        ratio = mid_energy / side_energy

        # Bell curve centered at 4.0 (mid ~4x side = typical good stereo)
        return float(np.clip(np.exp(-0.3 * (ratio - 4.0) ** 2), 0.0, 1.0))
