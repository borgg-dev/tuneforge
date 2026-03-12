"""Tests for vocal quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.vocal_quality import VOCAL_WEIGHTS, VocalQualityScorer


# ---------------------------------------------------------------------------
# Helper: check if librosa works (may fail under numba/coverage conflicts)
# ---------------------------------------------------------------------------

def _librosa_available() -> bool:
    """Check if librosa HPSS / pyin works in this environment."""
    try:
        import librosa

        sr = 32000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        librosa.effects.hpss(y)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                     fmax=librosa.note_to_hz("C7"), sr=sr)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_available(),
    reason="librosa unavailable (numba/coverage conflict)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32_000


@pytest.fixture
def scorer():
    return VocalQualityScorer()


@pytest.fixture
def sample_rate():
    return SAMPLE_RATE


@pytest.fixture
def sample_audio_complex():
    """10-second multi-frequency audio with harmonics — approximates real music."""
    t = np.linspace(0, 10.0, int(SAMPLE_RATE * 10.0), endpoint=False)
    audio = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
        + 0.05 * np.sin(2 * np.pi * 1760 * t)
    )
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 2 * t)
    audio *= modulation
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_silence():
    """10-second silence at 32 kHz."""
    return np.zeros(int(SAMPLE_RATE * 10.0), dtype=np.float32)


@pytest.fixture
def sample_audio_short():
    """Very short audio (0.3 seconds) — below minimum duration."""
    t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ===========================================================================
# Test classes
# ===========================================================================


class TestVocalQualityKeys:
    """score() returns all expected keys with values in [0, 1]."""

    @_skip_librosa
    def test_score_returns_all_keys(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert set(scores.keys()) == set(VOCAL_WEIGHTS.keys())

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


class TestAggregateWeights:
    """Aggregate computation: weights and bounds."""

    def test_weights_sum_to_one(self):
        total = sum(VOCAL_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    @_skip_librosa
    def test_aggregate_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_all_half(self, scorer):
        """All-0.5 scores (neutral) should produce 0.5 aggregate."""
        scores = {k: 0.5 for k in VOCAL_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(0.5, abs=1e-6)

    def test_aggregate_all_ones(self, scorer):
        scores = {k: 1.0 for k in VOCAL_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)


class TestInstrumentalGenre:
    """Instrumental genres should return neutral (0.5) scores."""

    def test_ambient_returns_neutral(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate, genre="ambient")
        assert set(scores.keys()) == set(VOCAL_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for ambient, got {val}"

    def test_electronic_returns_neutral(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate, genre="electronic")
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for electronic, got {val}"

    def test_classical_returns_neutral(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate, genre="classical")
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for classical, got {val}"


class TestSilenceHandling:
    """Silence returns neutral 0.5 scores (vocal absence should not penalize)."""

    def test_silence_returns_neutral(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert set(scores.keys()) == set(VOCAL_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for silence, got {val}"


class TestShortAudio:
    """Audio shorter than 0.5s returns neutral 0.5 scores."""

    def test_short_audio_returns_neutral(self, scorer, sample_audio_short, sample_rate):
        scores = scorer.score(sample_audio_short, sample_rate)
        assert set(scores.keys()) == set(VOCAL_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.5, f"{key} should be 0.5 for short audio, got {val}"


class TestNonInstrumentalGenre:
    """Non-instrumental genre with complex audio returns non-0.5 scores."""

    @_skip_librosa
    def test_pop_genre_not_all_neutral(self, scorer, sample_audio_complex, sample_rate):
        """Pop genre (vocal_expected=True) with complex audio should produce
        at least one metric that differs from 0.5."""
        scores = scorer.score(sample_audio_complex, sample_rate, genre="pop")
        assert set(scores.keys()) == set(VOCAL_WEIGHTS.keys())
        # At least one metric should differ from the neutral 0.5
        assert any(
            abs(val - 0.5) > 0.01 for val in scores.values()
        ), f"Expected non-neutral scores for pop genre, got {scores}"

    @_skip_librosa
    def test_default_genre_not_all_neutral(self, scorer, sample_audio_complex, sample_rate):
        """Default (no genre) with complex audio should produce non-0.5 scores."""
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert any(
            abs(val - 0.5) > 0.01 for val in scores.values()
        ), f"Expected non-neutral scores for default genre, got {scores}"

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sample_audio_complex, sample_rate):
        """Scores for non-instrumental genre should still be in [0, 1]."""
        scores = scorer.score(sample_audio_complex, sample_rate, genre="rock")
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
