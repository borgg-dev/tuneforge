"""Tests for neural codec quality scoring metrics.

The EnCodec model is heavy, so we mock it to avoid downloads in CI.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tuneforge.scoring.neural_codec_quality import NeuralCodecQualityScorer


@pytest.fixture
def scorer():
    return NeuralCodecQualityScorer()


class TestNeuralCodecWeights:

    def test_weights_sum_to_one(self):
        total = sum(NeuralCodecQualityScorer.WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestNeuralCodecLoadFailure:

    def test_fallback_when_model_unavailable(self, scorer):
        """When EnCodec model fails to load, scores should default to 0.5."""
        with patch.object(scorer, "_load", return_value=False):
            scores = scorer.score(np.zeros(24000, dtype=np.float32), 24000)
            assert set(scores.keys()) == set(NeuralCodecQualityScorer.WEIGHTS.keys())
            for v in scores.values():
                assert v == 0.5

    def test_fallback_aggregate(self, scorer):
        """Fallback scores of 0.5 should aggregate to 0.5."""
        scores = {k: 0.5 for k in NeuralCodecQualityScorer.WEIGHTS}
        agg = scorer.aggregate(scores)
        assert agg == pytest.approx(0.5, abs=1e-6)


class TestNeuralCodecScoreKeys:

    def test_score_returns_correct_keys(self, scorer):
        """Even with mocked model, score() returns expected dict keys."""
        with patch.object(scorer, "_load", return_value=False):
            scores = scorer.score(np.random.randn(24000).astype(np.float32), 24000)
            assert set(scores.keys()) == {"reconstruction_quality", "codec_naturalness"}


class TestNeuralCodecAggregate:

    def test_aggregate_in_range(self, scorer):
        scores = {"reconstruction_quality": 0.8, "codec_naturalness": 0.6}
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_zero(self, scorer):
        scores = {k: 0.0 for k in NeuralCodecQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == 0.0

    def test_aggregate_max(self, scorer):
        scores = {k: 1.0 for k in NeuralCodecQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_weighted(self, scorer):
        """Verify weighting: reconstruction_quality=0.60, codec_naturalness=0.40."""
        scores = {"reconstruction_quality": 1.0, "codec_naturalness": 0.0}
        agg = scorer.aggregate(scores)
        assert agg == pytest.approx(0.60, abs=1e-6)


class TestNeuralCodecStaticMethods:

    def test_spectral_convergence_identical(self):
        """Identical signals should have convergence score of 1.0."""
        audio = np.random.randn(24000).astype(np.float32) * 0.5
        score = NeuralCodecQualityScorer._spectral_convergence(audio, audio, 24000)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_spectral_convergence_different(self):
        """Very different signals should score lower."""
        orig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        recon = np.random.randn(24000).astype(np.float32) * 0.5
        score = NeuralCodecQualityScorer._spectral_convergence(orig, recon, 24000)
        assert score < 0.8

    def test_mel_distance_identical(self):
        """Identical signals should have mel distance score of 1.0."""
        audio = np.random.randn(24000).astype(np.float32) * 0.3
        score = NeuralCodecQualityScorer._mel_distance_score(audio, audio, 24000)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_mel_distance_different(self):
        """Very different signals should score lower."""
        orig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        recon = np.random.randn(24000).astype(np.float32) * 0.5
        score = NeuralCodecQualityScorer._mel_distance_score(orig, recon, 24000)
        assert score < 0.9


class TestNeuralCodecExceptionHandling:

    def test_score_exception_returns_fallback(self, scorer):
        """If score() throws internally, return 0.5 fallback."""
        with patch.object(scorer, "_load", side_effect=Exception("boom")):
            scores = scorer.score(np.zeros(24000, dtype=np.float32), 24000)
            for v in scores.values():
                assert v == 0.5
