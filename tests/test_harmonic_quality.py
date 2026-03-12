"""Tests for harmonic quality scoring metrics."""

import numpy as np
import pytest


def _librosa_available() -> bool:
    """Check if librosa works in this environment."""
    try:
        import librosa
        y = np.zeros(4096, dtype=np.float32)
        librosa.stft(y)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_available(),
    reason="librosa unavailable",
)

SAMPLE_RATE = 32_000


@pytest.fixture
def scorer():
    from tuneforge.scoring.harmonic_quality import HarmonicQualityScorer
    return HarmonicQualityScorer()


@pytest.fixture
def sine_audio():
    """Pure 440 Hz sine wave, 5 seconds."""
    t = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False)
    return (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def complex_audio():
    """Multi-harmonic audio with formant-like structure."""
    t = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False)
    audio = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 1320 * t)
        + 0.08 * np.sin(2 * np.pi * 1760 * t)
        + 0.04 * np.sin(2 * np.pi * 2200 * t)
    )
    return audio.astype(np.float32)


@pytest.fixture
def silence_audio():
    return np.zeros(int(SAMPLE_RATE * 5.0), dtype=np.float32)


class TestHarmonicQualityKeys:
    """score() returns all expected keys with values in [0, 1]."""

    @_skip_librosa
    def test_score_returns_expected_keys(self, scorer, sine_audio):
        from tuneforge.scoring.harmonic_quality import HARMONIC_WEIGHTS
        scores = scorer.score(sine_audio, SAMPLE_RATE)
        assert set(scores.keys()) == set(HARMONIC_WEIGHTS.keys())

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sine_audio):
        scores = scorer.score(sine_audio, SAMPLE_RATE)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @_skip_librosa
    def test_complex_scores_in_range(self, scorer, complex_audio):
        scores = scorer.score(complex_audio, SAMPLE_RATE)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_silence_returns_neutral(self, scorer, silence_audio):
        scores = scorer.score(silence_audio, SAMPLE_RATE)
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for silence, got {val}"


class TestHarmonicAggregate:
    """Aggregate computation: weights and bounds."""

    def test_weights_sum_to_one(self):
        from tuneforge.scoring.harmonic_quality import HARMONIC_WEIGHTS
        total = sum(HARMONIC_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    @_skip_librosa
    def test_aggregate_in_range(self, scorer, sine_audio):
        scores = scorer.score(sine_audio, SAMPLE_RATE)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_all_ones(self, scorer):
        from tuneforge.scoring.harmonic_quality import HARMONIC_WEIGHTS
        scores = {k: 1.0 for k in HARMONIC_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_all_zeros(self, scorer):
        from tuneforge.scoring.harmonic_quality import HARMONIC_WEIGHTS
        scores = {k: 0.0 for k in HARMONIC_WEIGHTS}
        assert scorer.aggregate(scores) == 0.0


class TestFormantStructure:
    """Formant structure metric produces meaningful scores."""

    @_skip_librosa
    def test_sine_vs_complex(self, scorer, sine_audio, complex_audio):
        """Complex audio with harmonics should produce a formant score."""
        sine_scores = scorer.score(sine_audio, SAMPLE_RATE)
        complex_scores = scorer.score(complex_audio, SAMPLE_RATE)
        # Both should produce valid scores
        assert 0.0 <= sine_scores["formant_structure"] <= 1.0
        assert 0.0 <= complex_scores["formant_structure"] <= 1.0
