"""Tests for the preference model bootstrap heuristic."""

import numpy as np
import pytest


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


@pytest.fixture
def model():
    from tuneforge.scoring.preference_model import PreferenceModel

    return PreferenceModel(model_path=None)  # bootstrap mode


class TestPreferenceModelBootstrap:
    """Verify the model runs in bootstrap mode when no model path is given."""

    def test_bootstrap_flag_set(self, model):
        assert model._bootstrap is True

    def test_no_head_loaded(self, model):
        assert model._head is None


@_skip_librosa
class TestHeuristicScoreRange:
    """All audio types produce scores in [0, 1]."""

    def test_sine_in_range(self, model, sample_audio_sine, sample_rate):
        score = model.score(sample_audio_sine, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_complex_in_range(self, model, sample_audio_complex, sample_rate):
        score = model.score(sample_audio_complex, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_noise_in_range(self, model, sample_audio_noise, sample_rate):
        score = model.score(sample_audio_noise, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_silence_in_range(self, model, sample_audio_silence, sample_rate):
        score = model.score(sample_audio_silence, sample_rate)
        assert 0.0 <= score <= 1.0


class TestHeuristicSilence:
    """Silence returns 0.0."""

    def test_silence_returns_zero(self, model, sample_audio_silence, sample_rate):
        score = model.score(sample_audio_silence, sample_rate)
        assert score == 0.0


@_skip_librosa
class TestHeuristicQualityOrdering:
    """Complex audio scores higher than noise; noise scores higher than silence."""

    def test_complex_beats_noise(
        self, model, sample_audio_complex, sample_audio_noise, sample_rate
    ):
        complex_score = model.score(sample_audio_complex, sample_rate)
        noise_score = model.score(sample_audio_noise, sample_rate)
        assert complex_score > noise_score, (
            f"Complex ({complex_score:.4f}) should score higher than noise ({noise_score:.4f})"
        )

    def test_noise_beats_silence(
        self, model, sample_audio_noise, sample_audio_silence, sample_rate
    ):
        noise_score = model.score(sample_audio_noise, sample_rate)
        silence_score = model.score(sample_audio_silence, sample_rate)
        assert noise_score > silence_score, (
            f"Noise ({noise_score:.4f}) should score higher than silence ({silence_score:.4f})"
        )


@_skip_librosa
class TestHeuristicNotTrivial:
    """Complex audio scores > 0.2 (the heuristic produces meaningful scores)."""

    def test_complex_above_threshold(
        self, model, sample_audio_complex, sample_rate
    ):
        score = model.score(sample_audio_complex, sample_rate)
        assert score > 0.2, (
            f"Complex audio score ({score:.4f}) should be > 0.2"
        )
