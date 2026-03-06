"""Tests for the active learner module."""

import numpy as np
import pytest


class MockPreferenceModel:
    """Mock preference model that returns pre-configured scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self._call_idx = 0

    def score(self, audio: np.ndarray, sr: int) -> float:
        idx = self._call_idx
        self._call_idx += 1
        return self._scores[idx % len(self._scores)]


class TestActiveLearner:
    """Tests for ActiveLearner.select_uncertain_pairs."""

    def test_fewer_than_two_returns_empty(self):
        from tuneforge.scoring.active_learner import ActiveLearner

        mock = MockPreferenceModel([0.5])
        learner = ActiveLearner(mock, top_k=3)

        result = learner.select_uncertain_pairs(
            audio_data=[(np.zeros(1000, dtype=np.float32), 16000)],
            uids=[1],
        )
        assert result == []

    def test_empty_input_returns_empty(self):
        from tuneforge.scoring.active_learner import ActiveLearner

        mock = MockPreferenceModel([0.5])
        learner = ActiveLearner(mock, top_k=3)

        result = learner.select_uncertain_pairs(audio_data=[], uids=[])
        assert result == []

    def test_returns_top_k_most_uncertain(self):
        from tuneforge.scoring.active_learner import ActiveLearner

        # Scores: 0.1, 0.5, 0.9 for UIDs 10, 20, 30
        # Pairs and diffs:
        #   (10,20): |0.1-0.5| = 0.4, uncertainty = 0.6
        #   (10,30): |0.1-0.9| = 0.8, uncertainty = 0.2
        #   (20,30): |0.5-0.9| = 0.4, uncertainty = 0.6
        mock = MockPreferenceModel([0.1, 0.5, 0.9])
        learner = ActiveLearner(mock, top_k=2)

        audio = np.zeros(1000, dtype=np.float32)
        result = learner.select_uncertain_pairs(
            audio_data=[(audio, 16000), (audio, 16000), (audio, 16000)],
            uids=[10, 20, 30],
        )

        assert len(result) == 2
        # Top two uncertainties should be 0.6
        assert result[0][2] == pytest.approx(0.6)
        assert result[1][2] == pytest.approx(0.6)
        # Least uncertain pair (10,30) with uncertainty 0.2 should NOT be included
        uid_pairs = [(r[0], r[1]) for r in result]
        assert (10, 30) not in uid_pairs

    def test_top_k_limits_output(self):
        from tuneforge.scoring.active_learner import ActiveLearner

        # 4 responses -> 6 pairs, top_k=1 should return only 1
        mock = MockPreferenceModel([0.5, 0.5, 0.5, 0.5])
        learner = ActiveLearner(mock, top_k=1)

        audio = np.zeros(1000, dtype=np.float32)
        result = learner.select_uncertain_pairs(
            audio_data=[(audio, 16000)] * 4,
            uids=[1, 2, 3, 4],
        )

        assert len(result) == 1

    def test_identical_scores_max_uncertainty(self):
        from tuneforge.scoring.active_learner import ActiveLearner

        # All scores identical -> all diffs = 0 -> all uncertainties = 1.0
        mock = MockPreferenceModel([0.7, 0.7, 0.7])
        learner = ActiveLearner(mock, top_k=10)

        audio = np.zeros(1000, dtype=np.float32)
        result = learner.select_uncertain_pairs(
            audio_data=[(audio, 16000)] * 3,
            uids=[1, 2, 3],
        )

        assert len(result) == 3
        for _, _, uncertainty in result:
            assert uncertainty == pytest.approx(1.0)

    def test_scoring_exception_uses_fallback(self):
        """If scoring raises, the score defaults to 0.5."""
        from tuneforge.scoring.active_learner import ActiveLearner

        class FailingModel:
            def __init__(self):
                self._call = 0

            def score(self, audio, sr):
                self._call += 1
                if self._call == 1:
                    raise RuntimeError("boom")
                return 0.9

        learner = ActiveLearner(FailingModel(), top_k=5)
        audio = np.zeros(1000, dtype=np.float32)
        result = learner.select_uncertain_pairs(
            audio_data=[(audio, 16000), (audio, 16000)],
            uids=[1, 2],
        )

        # First scores 0.5 (fallback), second scores 0.9
        assert len(result) == 1
        assert result[0][2] == pytest.approx(0.6)  # 1.0 - |0.5 - 0.9|
