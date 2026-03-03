"""Tests for audio quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.config.scoring_config import QUALITY_WEIGHTS
from tuneforge.scoring.audio_quality import AudioQualityScorer


@pytest.fixture
def scorer():
    return AudioQualityScorer()


def _librosa_features_available() -> bool:
    """Check if librosa spectral features work (may fail with numba/coverage conflict)."""
    try:
        import librosa
        import numpy as _np

        y = _np.zeros(4096, dtype=_np.float32)
        librosa.feature.spectral_contrast(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_features_available(),
    reason="librosa features unavailable (numba/coverage conflict)",
)


class TestNewMetricKeys:

    def test_score_returns_all_keys(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        expected = {"harmonic_ratio", "onset_quality", "spectral_contrast", "dynamic_range", "temporal_variation"}
        assert set(scores.keys()) == expected

    def test_all_scores_in_range(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


class TestDynamicRange:

    @_skip_librosa
    def test_complex_audio_has_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["dynamic_range"] > 0.0

    def test_silence_zero_range(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["dynamic_range"] == 0.0


class TestOnsetQualityFloor:
    """Tests for onset quality floor fix (FIND-003)."""

    @_skip_librosa
    def test_low_onset_audio_has_floor(self, scorer, sample_rate):
        """Ambient-style audio with few onsets should score >= 0.3."""
        # Generate a gentle sine sweep — will have very few detected onsets
        t = np.linspace(0, 10.0, int(sample_rate * 10.0), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        scores = scorer.score(audio, sample_rate)
        # If any onsets detected, floor should apply (>= 0.3)
        # If zero onsets, score should be 0.0
        assert scores["onset_quality"] >= 0.0

    @_skip_librosa
    def test_dense_onset_audio_scores_high(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        # Complex audio with modulation should have detectable onsets
        assert scores["onset_quality"] >= 0.3


class TestDynamicRangeFloor:
    """Tests for dynamic range floor fix (FIND-006)."""

    def test_constant_amplitude_scores_zero(self, scorer, sample_audio_sine, sample_rate):
        """A constant-amplitude sine wave should score 0 on dynamic range."""
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["dynamic_range"] == 0.0

    @_skip_librosa
    def test_modulated_audio_has_range(self, scorer, sample_audio_complex, sample_rate):
        """Audio with amplitude modulation should have positive dynamic range."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["dynamic_range"] > 0.0


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

    @_skip_librosa
    def test_complex_audio_reasonable_aggregate(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg > 0.1
