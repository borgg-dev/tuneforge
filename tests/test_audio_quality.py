"""Tests for audio quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.config.scoring_config import QUALITY_WEIGHTS
from tuneforge.scoring.audio_quality import AudioQualityScorer


@pytest.fixture
def scorer():
    return AudioQualityScorer()


class TestAudioQualityClipping:

    def test_clean_audio_high_score(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["clipping"] > 0.9

    def test_clipped_audio_low_score(self, scorer, sample_audio_clipped, sample_rate):
        scores = scorer.score(sample_audio_clipped, sample_rate)
        assert scores["clipping"] < 0.5


class TestAudioQualityDynamicRange:

    def test_complex_audio_has_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["dynamic_range"] > 0.0

    def test_silence_zero_range(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["dynamic_range"] == 0.0


def _librosa_features_available() -> bool:
    """Check if librosa spectral features work (may fail with numba/coverage conflict)."""
    try:
        import librosa
        import numpy as _np

        y = _np.zeros(4096, dtype=_np.float32)
        librosa.feature.spectral_centroid(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_spectral = pytest.mark.skipif(
    not _librosa_features_available(),
    reason="librosa spectral features unavailable (numba/coverage conflict)",
)


class TestAudioQualitySpectral:

    @_skip_spectral
    def test_noise_lower_spectral_quality(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        assert scores["spectral_quality"] < 0.8

    @_skip_spectral
    def test_tonal_higher_spectral_quality(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["spectral_quality"] > 0.3

    def test_spectral_quality_key_present(self, scorer, sample_audio_sine, sample_rate):
        """spectral_quality key is always present (0.0 if librosa fails)."""
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert "spectral_quality" in scores
        assert 0.0 <= scores["spectral_quality"] <= 1.0


class TestAudioQualityContentRatio:

    def test_silence_zero_content(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["content_ratio"] == 0.0

    def test_full_audio_high_content(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["content_ratio"] > 0.8


class TestAudioQualityAggregate:

    def test_aggregate_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_weights_sum_to_one(self):
        total = sum(QUALITY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_silence_low_aggregate(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg < 0.2

    def test_complex_audio_reasonable_aggregate(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg > 0.1
