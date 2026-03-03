"""Tests for structural completeness scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.structural_completeness import (
    STRUCTURAL_WEIGHTS,
    StructuralCompletenessScorer,
)


# ---------------------------------------------------------------------------
# Helper: check if librosa chroma features work
# ---------------------------------------------------------------------------

def _librosa_features_available() -> bool:
    """Check if librosa chroma features work in this environment."""
    try:
        import librosa
        import numpy as _np

        y = _np.zeros(4096, dtype=_np.float32)
        librosa.feature.chroma_cqt(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_features_available(),
    reason="librosa features unavailable (numba/coverage conflict)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32_000


@pytest.fixture
def scorer():
    return StructuralCompletenessScorer()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestStructuralKeys:
    """score() returns all expected keys with values in [0, 1]."""

    @_skip_librosa
    def test_score_returns_all_keys(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert set(scores.keys()) == set(STRUCTURAL_WEIGHTS.keys())

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


class TestSilenceReturnsZeros:
    """Silence (< 1e-6 max amplitude) returns all zeros."""

    def test_silence_returns_all_zeros(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert set(scores.keys()) == set(STRUCTURAL_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.0, f"{key} should be 0.0 for silence, got {val}"


class TestShortAudioReturnsZeros:
    """Audio shorter than 2.0 seconds returns all zeros."""

    def test_short_audio_returns_zeros(self, scorer, sample_rate):
        """Audio of 1.5 seconds should yield all-zero scores."""
        t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
        short_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        scores = scorer.score(short_audio, sample_rate)
        assert set(scores.keys()) == set(STRUCTURAL_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.0, f"{key} should be 0.0 for short audio, got {val}"

    def test_very_short_audio_returns_zeros(self, scorer, sample_rate):
        """Audio of 0.5 seconds should yield all-zero scores."""
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        short_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        scores = scorer.score(short_audio, sample_rate)
        for key, val in scores.items():
            assert val == 0.0, f"{key} should be 0.0 for very short audio, got {val}"


class TestComplexAudioNonZero:
    """Complex audio (multi-frequency with harmonics) produces non-zero scores."""

    @_skip_librosa
    def test_complex_audio_has_nonzero_scores(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        # At least one metric should be non-zero for complex audio
        non_zero = [k for k, v in scores.items() if v > 0.0]
        assert len(non_zero) > 0, f"All scores are zero for complex audio: {scores}"

    @_skip_librosa
    def test_complex_audio_section_count_nonzero(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["section_count"] > 0.0

    @_skip_librosa
    def test_complex_audio_transition_smoothness_nonzero(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert scores["transition_smoothness"] > 0.0


class TestAggregate:
    """Aggregate computation: weights and bounds."""

    def test_weights_sum_to_one(self):
        total = sum(STRUCTURAL_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    @_skip_librosa
    def test_aggregate_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_all_zeros(self, scorer):
        scores = {k: 0.0 for k in STRUCTURAL_WEIGHTS}
        assert scorer.aggregate(scores) == 0.0

    def test_aggregate_all_ones(self, scorer):
        scores = {k: 1.0 for k in STRUCTURAL_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)

    def test_silence_low_aggregate(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg == 0.0

    @_skip_librosa
    def test_complex_audio_positive_aggregate(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert agg > 0.0
