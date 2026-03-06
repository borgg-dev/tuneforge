"""Tests for the annotator reliability tracker."""

import pytest


class TestAnnotatorReliabilityTracker:
    """Tests for AnnotatorReliabilityTracker."""

    def _make_tracker(self, **kwargs):
        from tuneforge.scoring.annotator_reliability import AnnotatorReliabilityTracker

        return AnnotatorReliabilityTracker(**kwargs)

    def test_new_annotator_starts_at_one(self):
        tracker = self._make_tracker()
        assert tracker.get_reliability("user_new") == 1.0

    def test_reliability_decreases_after_disagreement(self):
        tracker = self._make_tracker(ema_alpha=0.1)
        initial = tracker.get_reliability("user1")
        updated = tracker.update("user1", agreed_with_majority=False)
        assert updated < initial

    def test_reliability_stays_high_after_agreement(self):
        tracker = self._make_tracker(ema_alpha=0.1)
        # Start at 1.0, agree -> EMA: 0.1*1 + 0.9*1.0 = 1.0
        updated = tracker.update("user1", agreed_with_majority=True)
        assert updated == pytest.approx(1.0)

    def test_reliability_recovers_after_agreement(self):
        tracker = self._make_tracker(ema_alpha=0.5)
        # Disagree: 0.5*0 + 0.5*1.0 = 0.5
        tracker.update("user1", agreed_with_majority=False)
        assert tracker.get_reliability("user1") == pytest.approx(0.5)
        # Agree: 0.5*1 + 0.5*0.5 = 0.75
        tracker.update("user1", agreed_with_majority=True)
        assert tracker.get_reliability("user1") == pytest.approx(0.75)

    def test_min_reliability_floor(self):
        tracker = self._make_tracker(ema_alpha=0.9, min_reliability=0.2)
        # Many disagreements should floor at min_reliability
        for _ in range(50):
            tracker.update("user_bad", agreed_with_majority=False)
        assert tracker.get_reliability("user_bad") >= 0.2

    def test_vote_counts_tracked(self):
        tracker = self._make_tracker()
        tracker.update("user1", True)
        tracker.update("user1", False)
        tracker.update("user1", True)
        snapshot = tracker.snapshot()
        assert snapshot["vote_counts"]["user1"] == 3

    def test_weighted_majority_vote_a_wins(self):
        tracker = self._make_tracker(weighted_threshold=0.6)
        # Two reliable voters for "a", one unreliable for "b"
        tracker._reliability["reliable1"] = 1.0
        tracker._reliability["reliable2"] = 1.0
        tracker._reliability["unreliable"] = 0.1

        votes = [("reliable1", "a"), ("reliable2", "a"), ("unreliable", "b")]
        result = tracker.weighted_majority_vote(votes)
        assert result == "a"

    def test_weighted_majority_vote_b_wins(self):
        tracker = self._make_tracker(weighted_threshold=0.6)
        tracker._reliability["r1"] = 1.0
        tracker._reliability["r2"] = 1.0
        tracker._reliability["u1"] = 0.1

        votes = [("r1", "b"), ("r2", "b"), ("u1", "a")]
        result = tracker.weighted_majority_vote(votes)
        assert result == "b"

    def test_weighted_majority_vote_no_clear_winner(self):
        tracker = self._make_tracker(weighted_threshold=0.6)
        tracker._reliability["u1"] = 1.0
        tracker._reliability["u2"] = 1.0

        votes = [("u1", "a"), ("u2", "b")]
        result = tracker.weighted_majority_vote(votes)
        assert result is None  # 50/50 split, neither > 0.6

    def test_weighted_majority_vote_empty(self):
        tracker = self._make_tracker()
        assert tracker.weighted_majority_vote([]) is None

    def test_snapshot_load_roundtrip(self):
        tracker1 = self._make_tracker()
        tracker1.update("alice", True)
        tracker1.update("alice", False)
        tracker1.update("bob", True)

        snap = tracker1.snapshot()

        tracker2 = self._make_tracker()
        tracker2.load_snapshot(snap)

        assert tracker2.get_reliability("alice") == tracker1.get_reliability("alice")
        assert tracker2.get_reliability("bob") == tracker1.get_reliability("bob")
        assert tracker2.snapshot() == snap

    def test_snapshot_empty_data(self):
        tracker = self._make_tracker()
        tracker.load_snapshot({})
        assert tracker.get_reliability("anyone") == 1.0
