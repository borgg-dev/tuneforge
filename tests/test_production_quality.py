"""Tests for production quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.production_quality import PRODUCTION_WEIGHTS, ProductionQualityScorer


@pytest.fixture
def scorer():
    return ProductionQualityScorer()


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


class TestProductionQualityKeys:
    """Verify score() returns all expected keys with values in [0, 1]."""

    def test_score_returns_all_keys(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        expected = {
            "spectral_balance",
            "frequency_fullness",
            "loudness_consistency",
            "dynamic_expressiveness",
            "stereo_quality",
        }
        assert set(scores.keys()) == expected

    def test_all_scores_in_range(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_complex_audio_scores_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_noise_scores_in_range(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


class TestSpectralBalance:
    """Tests for spectral balance scoring."""

    @_skip_librosa
    def test_complex_audio_has_balance(self, scorer, sample_audio_complex, sample_rate):
        """Complex audio with harmonics should have non-zero spectral balance."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["spectral_balance"] > 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        """Silence should score 0 on spectral balance."""
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["spectral_balance"] == 0.0


class TestFrequencyFullness:
    """Tests for frequency fullness scoring."""

    @_skip_librosa
    def test_complex_fills_more_than_sine(self, scorer, sample_audio_complex, sample_audio_sine, sample_rate):
        """Complex audio should fill more spectrum than a pure sine wave."""
        complex_scores = scorer.score(sample_audio_complex, sample_rate)
        sine_scores = scorer.score(sample_audio_sine, sample_rate)
        assert complex_scores["frequency_fullness"] > sine_scores["frequency_fullness"]

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        """Silence should score 0 on frequency fullness."""
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["frequency_fullness"] == 0.0


class TestLoudnessConsistency:
    """Tests for loudness consistency scoring."""

    @_skip_librosa
    def test_complex_audio_has_consistency(self, scorer, sample_audio_complex, sample_rate):
        """Complex audio with amplitude modulation should have non-zero loudness consistency."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["loudness_consistency"] > 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        """Silence should score 0 on loudness consistency."""
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["loudness_consistency"] == 0.0


class TestDynamicExpressiveness:
    """Tests for dynamic expressiveness scoring."""

    @_skip_librosa
    def test_complex_audio_has_expressiveness(self, scorer, sample_audio_complex, sample_rate):
        """Complex audio with modulation should have non-zero dynamic expressiveness."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["dynamic_expressiveness"] > 0.0

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        """Silence should score 0 on dynamic expressiveness."""
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["dynamic_expressiveness"] == 0.0


class TestProductionAggregate:
    """Tests for the aggregate production quality score."""

    def test_aggregate_range(self, scorer, sample_audio_complex, sample_rate):
        """Aggregate score should be in [0, 1]."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_weights_sum_to_one(self):
        """Production weights must sum to 1.0."""
        total = sum(PRODUCTION_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_silence_low_aggregate(self, scorer, sample_audio_silence, sample_rate):
        """Silence should produce a very low aggregate score."""
        scores = scorer.score(sample_audio_silence, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg < 0.2

    @_skip_librosa
    def test_complex_audio_reasonable_aggregate(self, scorer, sample_audio_complex, sample_rate):
        """Complex audio should produce a reasonable aggregate score."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg > 0.1


class TestGamingResistance:
    """White noise should not game the production quality scorer."""

    @_skip_librosa
    def test_noise_scores_lower_than_complex_on_loudness(self, scorer, sample_audio_noise, sample_audio_complex, sample_rate):
        """White noise should score lower than complex musical audio on loudness+dynamics."""
        noise_scores = scorer.score(sample_audio_noise, sample_rate)
        complex_scores = scorer.score(sample_audio_complex, sample_rate)
        # Compare on loudness_consistency + dynamic_expressiveness (production metrics
        # where musical audio should clearly beat noise). Frequency fullness favours
        # broadband noise over synthetic tones, which is expected.
        complex_prod = complex_scores["loudness_consistency"] + complex_scores["dynamic_expressiveness"]
        noise_prod = noise_scores["loudness_consistency"] + noise_scores["dynamic_expressiveness"]
        assert complex_prod > noise_prod
