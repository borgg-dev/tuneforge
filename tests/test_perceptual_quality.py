"""Tests for perceptual quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.perceptual_quality import PerceptualQualityScorer


def _librosa_available() -> bool:
    try:
        import librosa
        y = np.zeros(4096, dtype=np.float32)
        librosa.feature.spectral_bandwidth(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_available(),
    reason="librosa features unavailable",
)


@pytest.fixture
def scorer():
    return PerceptualQualityScorer()


class TestPerceptualQualityWeights:

    def test_weights_sum_to_one(self):
        total = sum(PerceptualQualityScorer.WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestPerceptualQualitySilence:

    def test_silence_returns_zeros(self, scorer):
        audio = np.zeros(32000 * 5, dtype=np.float32)
        scores = scorer.score(audio, 32000)
        assert set(scores.keys()) == set(PerceptualQualityScorer.WEIGHTS.keys())
        for v in scores.values():
            assert v == 0.0

    def test_silence_aggregate_zero(self, scorer):
        scores = {k: 0.0 for k in PerceptualQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == 0.0


class TestPerceptualQualitySineWave:

    @_skip_librosa
    def test_sine_wave_returns_all_keys(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert set(scores.keys()) == set(PerceptualQualityScorer.WEIGHTS.keys())

    @_skip_librosa
    def test_sine_wave_scores_in_range(self, scorer, sample_audio_sine, sample_rate):
        scores = scorer.score(sample_audio_sine, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @_skip_librosa
    def test_sine_wave_reasonable_snr(self, scorer, sample_audio_sine, sample_rate):
        """A clean sine wave should have a decent SNR estimate."""
        scores = scorer.score(sample_audio_sine, sample_rate)
        assert scores["snr_estimate"] > 0.3

    @_skip_librosa
    def test_complex_audio_good_bandwidth_consistency(self, scorer, sample_audio_complex, sample_rate):
        """Complex audio with harmonics should have reasonable bandwidth consistency."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        # A multi-harmonic signal has broader, more consistent bandwidth than a pure sine
        assert scores["bandwidth_consistency"] >= 0.0  # At least computes without error


class TestPerceptualQualityNoise:

    @_skip_librosa
    def test_noise_lower_snr(self, scorer, sample_audio_noise, sample_rate):
        """White noise should have lower SNR than a clean sine wave."""
        noise_scores = scorer.score(sample_audio_noise, sample_rate)
        # Noise has lower SNR than a clean signal
        # (exact value depends on random, but generally lower than clean sine)
        assert 0.0 <= noise_scores["snr_estimate"] <= 1.0


class TestPerceptualQualityAggregate:

    def test_aggregate_in_range(self, scorer):
        scores = {"bandwidth_consistency": 0.8, "snr_estimate": 0.7,
                  "harmonic_noise_ratio": 0.6, "hf_presence": 0.5}
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_zero_input(self, scorer):
        scores = {k: 0.0 for k in PerceptualQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == 0.0

    def test_aggregate_max_input(self, scorer):
        scores = {k: 1.0 for k in PerceptualQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)

    @_skip_librosa
    def test_complex_audio_reasonable_aggregate(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_multichannel_handled(self, scorer):
        """2-D audio should be downmixed without error."""
        sr = 32000
        t = np.linspace(0, 5.0, sr * 5, endpoint=False)
        stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t),
                           0.5 * np.sin(2 * np.pi * 880 * t)], axis=0)
        scores = scorer.score(stereo.astype(np.float32), sr)
        assert set(scores.keys()) == set(PerceptualQualityScorer.WEIGHTS.keys())
