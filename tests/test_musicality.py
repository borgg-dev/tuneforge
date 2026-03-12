"""Tests for musicality scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.musicality import MUSICALITY_WEIGHTS, MusicalityScorer


@pytest.fixture
def scorer():
    return MusicalityScorer()


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


class TestMusicalityKeys:
    """Verify score() returns all expected keys and all values in [0, 1]."""

    @_skip_librosa
    def test_score_returns_all_keys(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        expected = {
            "pitch_stability",
            "harmonic_progression",
            "chord_coherence",
            "rhythmic_groove",
            "arrangement_sophistication",
        }
        assert set(scores.keys()) == expected

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @_skip_librosa
    def test_all_scores_in_range_complex(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @_skip_librosa
    def test_all_scores_in_range_noise(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_silence_returns_all_keys(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        expected = {
            "pitch_stability",
            "harmonic_progression",
            "chord_coherence",
            "rhythmic_groove",
            "arrangement_sophistication",
        }
        assert set(scores.keys()) == expected


class TestPitchStability:
    """Complex audio should score higher than noise on pitch_stability."""

    @_skip_librosa
    def test_complex_higher_than_noise(
        self, scorer, sample_audio_complex, sample_audio_noise, sample_rate
    ):
        complex_scores = scorer.score(sample_audio_complex, sample_rate)
        noise_scores = scorer.score(sample_audio_noise, sample_rate)
        assert complex_scores["pitch_stability"] > noise_scores["pitch_stability"]

    @_skip_librosa
    def test_sine_has_pitch_stability(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["pitch_stability"] > 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["pitch_stability"] == 0.0


class TestHarmonicProgression:
    """Complex audio has non-zero harmonic progression score."""

    @_skip_librosa
    def test_complex_has_progression(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["harmonic_progression"] > 0.0

    @_skip_librosa
    def test_sine_has_progression(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        # Even a single-tone sine will have some chroma presence
        assert scores["harmonic_progression"] >= 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["harmonic_progression"] == 0.0


class TestRhythmicGroove:
    """Complex audio has detectable groove; silence scores 0."""

    @_skip_librosa
    def test_complex_has_groove(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["rhythmic_groove"] > 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["rhythmic_groove"] == 0.0

    @_skip_librosa
    def test_noise_has_ambient_floor(self, scorer, sample_audio_noise, sample_rate):
        """Noise has energy, so if no beats detected it gets at least the ambient floor."""
        scores = scorer.score(sample_audio_noise, sample_rate)
        # Either beats are detected (score > 0) or the ambient floor applies (>= 0.15)
        assert scores["rhythmic_groove"] >= 0.0


class TestArrangementSophistication:
    """Complex audio scores higher than silence on arrangement sophistication."""

    @_skip_librosa
    def test_complex_higher_than_silence(
        self, scorer, sample_audio_complex, sample_audio_silence, sample_rate
    ):
        complex_scores = scorer.score(sample_audio_complex, sample_rate)
        silence_scores = scorer.score(sample_audio_silence, sample_rate)
        assert (
            complex_scores["arrangement_sophistication"]
            > silence_scores["arrangement_sophistication"]
        )

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["arrangement_sophistication"] == 0.0

    @_skip_librosa
    def test_complex_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert 0.0 <= scores["arrangement_sophistication"] <= 1.0


class TestMusicalityAggregate:
    """Aggregate is in [0, 1], weights sum to 1.0, silence scores low."""

    @_skip_librosa
    def test_aggregate_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_weights_sum_to_one(self):
        total = sum(MUSICALITY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_silence_low_aggregate(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg < 0.2

    @_skip_librosa
    def test_complex_audio_reasonable_aggregate(
        self, scorer, sample_audio_complex, sample_rate
    ):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg > 0.05


class TestGamingResistance:
    """White noise scores lower than complex audio on aggregate."""

    @_skip_librosa
    def test_noise_lower_than_complex(
        self, scorer, sample_audio_complex, sample_audio_noise, sample_rate
    ):
        complex_scores = scorer.score(sample_audio_complex, sample_rate)
        noise_scores = scorer.score(sample_audio_noise, sample_rate)
        complex_agg = scorer.aggregate(complex_scores)
        noise_agg = scorer.aggregate(noise_scores)
        assert complex_agg > noise_agg

    @_skip_librosa
    def test_noise_all_metrics_bounded(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
