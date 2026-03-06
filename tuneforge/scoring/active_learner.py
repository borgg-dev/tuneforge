"""
Active learning for annotation task selection.

Selects the most informative A/B pairs for human annotation by
prioritizing pairs where the preference model is most uncertain
(predictions closest to 0.5).
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class ActiveLearner:
    """Select uncertain pairs for annotation to maximize training signal."""

    def __init__(self, preference_model, top_k: int = 3) -> None:
        self._preference = preference_model
        self._top_k = top_k

    def select_uncertain_pairs(
        self,
        audio_data: list[tuple[np.ndarray, int]],
        uids: list[int],
    ) -> list[tuple[int, int, float]]:
        """Select the most uncertain pairs from scored responses.

        Args:
            audio_data: List of (audio_array, sample_rate) per response.
            uids: Corresponding miner UIDs.

        Returns:
            List of (uid_a, uid_b, uncertainty) sorted by uncertainty descending.
            Limited to top_k pairs.
        """
        if len(audio_data) < 2:
            return []

        # Score each audio
        scores = []
        for audio, sr in audio_data:
            try:
                s = self._preference.score(audio, sr)
                scores.append(s)
            except Exception:
                scores.append(0.5)

        # Generate all pairs and compute uncertainty
        pairs = []
        n = len(scores)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(scores[i] - scores[j])
                uncertainty = 1.0 - diff  # Closer to 0 diff = more uncertain
                pairs.append((uids[i], uids[j], uncertainty))

        # Sort by uncertainty (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[: self._top_k]
