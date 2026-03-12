"""Tests for miner components: audio utils, prompt parser, blacklist/priority."""

import base64

import numpy as np
import pytest

from tuneforge.generation.audio_utils import AudioUtils
from tuneforge.generation.prompt_parser import PromptParser


def _librosa_resample_available() -> bool:
    """Check if librosa.resample works (may fail with numba/coverage conflict)."""
    try:
        import librosa
        librosa.resample(np.zeros(100, dtype=np.float32), orig_sr=32000, target_sr=16000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_resample_available(),
    reason="librosa resample unavailable (numba/coverage conflict)",
)


# ---------------------------------------------------------------------------
# AudioUtils
# ---------------------------------------------------------------------------

class TestAudioUtilsNormalize:

    def test_normalize_loud(self):
        audio = np.array([0.0, 2.0, -2.0, 1.0], dtype=np.float32)
        normed = AudioUtils.normalize(audio)
        assert np.max(np.abs(normed)) == pytest.approx(1.0, abs=1e-6)

    def test_normalize_silent(self):
        audio = np.zeros(100, dtype=np.float32)
        normed = AudioUtils.normalize(audio)
        assert np.allclose(normed, 0.0)

    def test_normalize_already_normalized(self):
        audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        normed = AudioUtils.normalize(audio)
        assert np.max(np.abs(normed)) == pytest.approx(1.0, abs=1e-6)


class TestAudioUtilsFade:

    def test_fade_applied(self, sample_audio_sine, sample_rate):
        faded = AudioUtils.fade_edges(sample_audio_sine, sample_rate, fade_ms=50)
        assert abs(faded[0]) < 0.01
        assert abs(faded[-1]) < 0.01
        # Check RMS of middle region is non-trivial (avoid zero-crossing artifacts)
        mid = len(faded) // 2
        mid_slice = faded[mid - 100 : mid + 100]
        assert np.sqrt(np.mean(mid_slice ** 2)) > 0.01

    def test_fade_preserves_length(self, sample_audio_sine, sample_rate):
        faded = AudioUtils.fade_edges(sample_audio_sine, sample_rate, fade_ms=50)
        assert len(faded) == len(sample_audio_sine)


class TestAudioUtilsWavRoundtrip:

    def test_wav_encode_decode(self, sample_audio_sine, sample_rate):
        wav_bytes = AudioUtils.to_wav_bytes(sample_audio_sine, sample_rate)
        assert len(wav_bytes) > 44

        decoded, sr = AudioUtils.from_wav_bytes(wav_bytes)
        assert sr == sample_rate
        assert len(decoded) == len(sample_audio_sine)
        assert np.allclose(decoded, sample_audio_sine, atol=5e-5)


class TestAudioUtilsBase64:

    def test_base64_roundtrip(self):
        data = b"test_audio_bytes_12345"
        encoded = AudioUtils.to_base64(data)
        decoded = AudioUtils.from_base64(encoded)
        assert decoded == data


class TestAudioUtilsDuration:

    def test_duration(self, sample_audio_sine, sample_rate):
        dur = AudioUtils.compute_duration(sample_audio_sine, sample_rate)
        assert dur == pytest.approx(10.0, abs=0.01)


class TestAudioUtilsRMS:

    def test_rms_sine(self, sample_audio_sine):
        rms = AudioUtils.compute_rms(sample_audio_sine)
        assert 0.5 < rms < 0.7

    def test_rms_silence(self, sample_audio_silence):
        rms = AudioUtils.compute_rms(sample_audio_silence)
        assert rms == 0.0


class TestAudioUtilsPeak:

    def test_peak(self, sample_audio_sine):
        peak = AudioUtils.compute_peak(sample_audio_sine)
        assert peak == pytest.approx(0.8, abs=0.01)


class TestAudioUtilsLimiter:

    def test_limiter_reduces_peaks(self):
        audio = np.array([0.0, 1.5, -1.5, 0.5], dtype=np.float32)
        limited = AudioUtils.apply_limiter(audio, threshold=0.95)
        assert np.max(np.abs(limited)) <= 1.0
        assert limited[0] == pytest.approx(0.0, abs=1e-6)
        assert limited[3] == pytest.approx(0.5, abs=1e-6)


class TestAudioUtilsResample:

    def test_resample_same_rate(self, sample_audio_sine, sample_rate):
        result = AudioUtils.resample(sample_audio_sine, sample_rate, sample_rate)
        assert np.array_equal(result, sample_audio_sine)

    @_skip_librosa
    def test_resample_different_rate(self, sample_audio_sine, sample_rate):
        target_sr = 48000
        result = AudioUtils.resample(sample_audio_sine, sample_rate, target_sr)
        expected_len = int(10.0 * target_sr)
        assert abs(len(result) - expected_len) <= 1


# ---------------------------------------------------------------------------
# PromptParser
# ---------------------------------------------------------------------------

class TestPromptParser:

    def test_basic_prompt(self):
        prompt = PromptParser.build_prompt(
            genre="lo-fi", mood="melancholic", tempo=75,
            instruments=["piano", "vinyl crackle"],
        )
        assert "lo-fi" in prompt.lower() or "lofi" in prompt.lower()
        assert "melancholic" in prompt.lower()
        assert "75 BPM" in prompt
        assert "piano" in prompt.lower()

    def test_text_only(self):
        prompt = PromptParser.build_prompt(text="ambient drone music")
        assert prompt == "ambient drone music"

    def test_genre_only(self):
        prompt = PromptParser.build_prompt(genre="jazz")
        assert "jazz" in prompt.lower()

    def test_mood_only(self):
        prompt = PromptParser.build_prompt(mood="happy")
        assert "happy" in prompt.lower() or "bright" in prompt.lower()

    def test_empty_returns_default(self):
        prompt = PromptParser.build_prompt()
        assert len(prompt) > 0

    def test_all_fields(self):
        prompt = PromptParser.build_prompt(
            text="with a catchy melody",
            genre="pop",
            mood="energetic",
            tempo=128,
            instruments=["synth", "drums", "bass"],
            key="C major",
            time_sig="4/4",
        )
        assert "128 BPM" in prompt
        assert "C major" in prompt
        assert "4/4" in prompt

    def test_multiple_genres_produce_different_prompts(self):
        genres = ["rock", "jazz", "ambient", "hip-hop"]
        prompts = [PromptParser.build_prompt(genre=g) for g in genres]
        assert len(set(prompts)) == len(genres)
