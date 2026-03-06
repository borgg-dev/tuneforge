"""
Annotator reliability tracking for TuneForge.

Tracks how often each annotator agrees with the majority vote,
weighting their future votes by reliability.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class AnnotatorReliabilityTracker:
    """Track and weight annotator reliability based on agreement with majority."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        min_reliability: float = 0.1,
        weighted_threshold: float = 0.6,
    ) -> None:
        self._alpha = ema_alpha
        self._min_reliability = min_reliability
        self._weighted_threshold = weighted_threshold
        self._reliability: dict[str, float] = {}
        self._vote_counts: dict[str, int] = {}

    def update(self, user_id: str, agreed_with_majority: bool) -> float:
        """Update reliability score for an annotator after a task resolves.

        Returns updated reliability.
        """
        current = self._reliability.get(user_id, 1.0)
        agreement = 1.0 if agreed_with_majority else 0.0
        new_reliability = self._alpha * agreement + (1 - self._alpha) * current
        self._reliability[user_id] = max(new_reliability, self._min_reliability)
        self._vote_counts[user_id] = self._vote_counts.get(user_id, 0) + 1
        return self._reliability[user_id]

    def get_reliability(self, user_id: str) -> float:
        """Get current reliability score (1.0 for unknown annotators)."""
        return self._reliability.get(user_id, 1.0)

    def weighted_majority_vote(
        self, votes: list[tuple[str, str]]
    ) -> str | None:
        """Compute reliability-weighted majority vote.

        Args:
            votes: List of (user_id, choice) where choice is "a" or "b".

        Returns:
            "a", "b", or None if no clear majority.
        """
        if not votes:
            return None

        weighted_a = sum(
            self.get_reliability(uid) for uid, choice in votes if choice == "a"
        )
        weighted_b = sum(
            self.get_reliability(uid) for uid, choice in votes if choice == "b"
        )
        total = weighted_a + weighted_b
        if total < 1e-8:
            return None

        if weighted_a / total > self._weighted_threshold:
            return "a"
        if weighted_b / total > self._weighted_threshold:
            return "b"
        return None

    def snapshot(self) -> dict:
        """Return serializable snapshot of reliability state."""
        return {
            "reliability": dict(self._reliability),
            "vote_counts": dict(self._vote_counts),
        }

    def load_snapshot(self, data: dict) -> None:
        """Restore reliability state from snapshot."""
        self._reliability = data.get("reliability", {})
        self._vote_counts = data.get("vote_counts", {})
