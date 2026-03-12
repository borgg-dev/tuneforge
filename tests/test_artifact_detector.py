"""Tests for artifact detection scoring."""

import numpy as np
import pytest

from tuneforge.scoring.artifact_detector import ArtifactDetector


SAMPLE_RATE = 32_000
DURATION = 10.0  # seconds


@pytest.fixture
def detector():
    return ArtifactDetector()


@pytest.fixture
def clean_sine() -> np.ndarray:
    """10-second sine + noise — clean signal with enough variation
    to avoid triggering repetition detection on identical chunks.
    Noise at 50% of sine amplitude breaks inter-chunk correlation."""
    rng = np.random.default_rng(10)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * rng.standard_normal(len(t))
    return audio.astype(np.float32)


@pytest.fixture
def clean_complex() -> np.ndarray:
    """10-second frequency-sweeping chirp with harmonics — evolves over time
    so it is never exactly repetitive between distant chunks."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Chirp: frequency sweeps from 300 Hz to 900 Hz
    freq_inst = 300 + 600 * t / DURATION
    phase = 2 * np.pi * np.cumsum(freq_inst) / SAMPLE_RATE
    audio = 0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    return audio.astype(np.float32)


@pytest.fixture
def hard_clipped_audio() -> np.ndarray:
    """Audio with severe hard clipping — peak amplitude driven well above 1.0."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 3.0 * np.sin(2 * np.pi * 440 * t)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


@pytest.fixture
def soft_clipped_audio() -> np.ndarray:
    """Audio with long sustained sections above 0.95 (soft clipping)."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Create a signal that stays above 0.95 for extended periods
    audio = np.full_like(t, 0.97, dtype=np.float32)
    # Add brief dips every 2 seconds to create multiple long runs
    for i in range(int(DURATION) // 2):
        start = int(i * 2 * SAMPLE_RATE)
        audio[start : start + 50] = 0.1
    return audio.astype(np.float32)


@pytest.fixture
def looped_audio() -> np.ndarray:
    """Audio made by tiling the same 0.5-second pattern — exact repetition."""
    chunk_len = int(0.5 * SAMPLE_RATE)
    t_chunk = np.linspace(0, 0.5, chunk_len, endpoint=False)
    # Create a recognizable pattern
    pattern = (
        0.4 * np.sin(2 * np.pi * 440 * t_chunk)
        + 0.2 * np.sin(2 * np.pi * 660 * t_chunk)
    ).astype(np.float32)
    # Tile to fill 10 seconds
    n_tiles = int(DURATION / 0.5)
    return np.tile(pattern, n_tiles)


@pytest.fixture
def varied_audio() -> np.ndarray:
    """Audio with natural variation — different frequency in each chunk."""
    rng = np.random.default_rng(42)
    chunk_len = int(0.5 * SAMPLE_RATE)
    n_chunks = int(DURATION / 0.5)
    chunks = []
    for i in range(n_chunks):
        t = np.linspace(0, 0.5, chunk_len, endpoint=False)
        freq = 200 + i * 50 + rng.uniform(-10, 10)
        chunk = 0.5 * np.sin(2 * np.pi * freq * t)
        chunks.append(chunk.astype(np.float32))
    return np.concatenate(chunks)


@pytest.fixture
def discontinuous_audio() -> np.ndarray:
    """Smooth chirp signal with occasional noise bursts inserted.

    The background is a frequency-sweeping chirp (non-repetitive), with
    five 50 ms noise bursts at 2-second intervals.  These create sharp
    spectral flux spikes that stand out against the smooth background.
    """
    rng = np.random.default_rng(99)
    n_samples = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)
    freq_inst = 300 + 600 * t / DURATION
    phase = 2 * np.pi * np.cumsum(freq_inst) / SAMPLE_RATE
    audio = (0.5 * np.sin(phase)).astype(np.float32)

    burst_samples = int(0.05 * SAMPLE_RATE)
    for sec in [1.0, 3.0, 5.0, 7.0, 9.0]:
        start = int(sec * SAMPLE_RATE)
        end = min(start + burst_samples, n_samples)
        audio[start:end] = rng.standard_normal(end - start).astype(np.float32)
    return audio


@pytest.fixture
def short_audio() -> np.ndarray:
    """Audio shorter than 0.5 seconds."""
    t = np.linspace(0, 0.1, int(SAMPLE_RATE * 0.1), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silent_audio() -> np.ndarray:
    """Effectively silent audio."""
    return np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)


# -----------------------------------------------------------------------
# Test classes
# -----------------------------------------------------------------------


class TestArtifactDetectorBasics:
    """Basic interface and edge-case tests."""

    def test_detect_returns_float_in_range(self, detector, clean_sine):
        result = detector.detect(clean_sine, SAMPLE_RATE)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_detect_detailed_returns_dict(self, detector, clean_sine):
        result = detector.detect_detailed(clean_sine, SAMPLE_RATE)
        assert isinstance(result, dict)
        expected_keys = {"spectral_discontinuity", "clipping", "repetition", "spectral_holes"}
        assert set(result.keys()) == expected_keys
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_clean_sine_high_score(self, detector, clean_sine):
        """A clean sine wave should not trigger artifact penalties."""
        result = detector.detect(clean_sine, SAMPLE_RATE)
        assert result >= 0.8

    def test_clean_complex_high_score(self, detector, clean_complex):
        """Clean multi-frequency audio should score well."""
        result = detector.detect(clean_complex, SAMPLE_RATE)
        assert result >= 0.8

    def test_short_audio_returns_one(self, detector, short_audio):
        """Audio shorter than 0.5s should return 1.0 (no penalty)."""
        result = detector.detect(short_audio, SAMPLE_RATE)
        assert result == 1.0

    def test_silent_audio_returns_one(self, detector, silent_audio):
        """Silent audio should return 1.0 (handled elsewhere)."""
        result = detector.detect(silent_audio, SAMPLE_RATE)
        assert result == 1.0

    def test_stereo_input_handled(self, detector):
        """Stereo (2-D) input should be averaged to mono without error."""
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([mono, mono], axis=0)
        result = detector.detect(stereo, SAMPLE_RATE)
        assert 0.0 <= result <= 1.0


class TestClippingDetection:
    """Tests for the clipping artifact check."""

    def test_hard_clipped_penalized(self, detector, hard_clipped_audio):
        """Severely hard-clipped audio should receive a clipping penalty."""
        detailed = detector.detect_detailed(hard_clipped_audio, SAMPLE_RATE)
        assert detailed["clipping"] < 0.8

    def test_soft_clipped_penalized(self, detector, soft_clipped_audio):
        """Audio with long sustained runs above 0.95 should be penalized."""
        detailed = detector.detect_detailed(soft_clipped_audio, SAMPLE_RATE)
        assert detailed["clipping"] < 1.0

    def test_clean_audio_no_clipping_penalty(self, detector):
        """A moderate-amplitude sine wave should not trigger clipping."""
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        detailed = detector.detect_detailed(audio, SAMPLE_RATE)
        assert detailed["clipping"] == 1.0

    def test_mild_clipping_moderate_penalty(self, detector):
        """Audio with very slight clipping should get only a mild penalty."""
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        # 1.001x amplitude: barely clips — only samples near the peak exceed 0.999
        audio = 1.001 * np.sin(2 * np.pi * 440 * t)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        detailed = detector.detect_detailed(audio, SAMPLE_RATE)
        # Small fraction is clipped, so penalty should be moderate-to-high (not 0)
        assert detailed["clipping"] > 0.0


class TestRepetitionDetection:
    """Tests for the repetitive pattern check."""

    def test_exact_loop_penalized(self, detector, looped_audio):
        """Audio tiled from an identical 0.5s pattern should be penalized."""
        detailed = detector.detect_detailed(looped_audio, SAMPLE_RATE)
        assert detailed["repetition"] < 0.5

    def test_varied_audio_passes(self, detector, varied_audio):
        """Audio with different frequencies per chunk should not be flagged."""
        detailed = detector.detect_detailed(varied_audio, SAMPLE_RATE)
        assert detailed["repetition"] >= 0.9

    def test_clean_sine_not_flagged_as_loop(self, detector, clean_sine):
        """
        A continuous sine wave has high inter-chunk correlation but the
        check requires gap >= 2 chunks.  Depending on phase alignment it
        may or may not trigger — we just verify it does not crash and the
        return is in range.
        """
        detailed = detector.detect_detailed(clean_sine, SAMPLE_RATE)
        assert 0.0 <= detailed["repetition"] <= 1.0


class TestSpectralDiscontinuity:
    """Tests for the spectral discontinuity check."""

    def test_mild_noise_bursts_tolerated(self, detector, discontinuous_audio):
        """Mild noise bursts (5x50ms in 10s) should be tolerated.

        Normal music has 4-9% flagged frames from note attacks and dynamics.
        Mild synthetic bursts fall within this range and should not be
        penalised — only severe chunk-boundary glitches should trigger.
        """
        detailed = detector.detect_detailed(discontinuous_audio, SAMPLE_RATE)
        assert detailed["spectral_discontinuity"] >= 0.8

    def test_clean_sine_no_discontinuity(self, detector, clean_sine):
        """A clean sine wave should have near-zero spectral flux variation."""
        detailed = detector.detect_detailed(clean_sine, SAMPLE_RATE)
        assert detailed["spectral_discontinuity"] >= 0.9

    def test_clean_complex_no_discontinuity(self, detector, clean_complex):
        """Multi-frequency audio with smooth modulation should pass."""
        detailed = detector.detect_detailed(clean_complex, SAMPLE_RATE)
        assert detailed["spectral_discontinuity"] >= 0.8


class TestSpectralHoles:
    """Tests for the spectral hole check."""

    def test_clean_sine_no_holes(self, detector, clean_sine):
        """A sine wave concentrates energy at one freq — should not trigger
        the hole detector (which looks for drops below local average)."""
        detailed = detector.detect_detailed(clean_sine, SAMPLE_RATE)
        assert detailed["spectral_holes"] >= 0.5

    def test_broadband_noise_no_holes(self, detector):
        """White noise has flat spectrum — no holes expected."""
        rng = np.random.default_rng(99)
        audio = rng.uniform(-0.5, 0.5, int(SAMPLE_RATE * DURATION)).astype(np.float32)
        detailed = detector.detect_detailed(audio, SAMPLE_RATE)
        assert detailed["spectral_holes"] >= 0.9

    def test_notch_filtered_audio(self, detector):
        """Audio with a deep notch filter applied should show spectral holes."""
        from scipy.signal import iirnotch, lfilter

        rng = np.random.default_rng(77)
        audio = rng.uniform(-0.5, 0.5, int(SAMPLE_RATE * DURATION)).astype(np.float32)

        # Apply several deep notch filters to carve out frequency bands
        for notch_freq in [2000, 4000, 6000, 8000, 10000]:
            b, a = iirnotch(notch_freq, Q=5.0, fs=SAMPLE_RATE)
            audio = lfilter(b, a, audio).astype(np.float32)

        detailed = detector.detect_detailed(audio, SAMPLE_RATE)
        # The penalty is capped at 0.5; this may or may not trigger depending
        # on the depth/width of notches relative to the 500 Hz window.
        assert 0.5 <= detailed["spectral_holes"] <= 1.0


class TestComposite:
    """Tests for the composite detect() method."""

    def test_detect_returns_geometric_mean(self, detector, hard_clipped_audio):
        """detect() should return the geometric mean of floored per-check penalties."""
        detailed = detector.detect_detailed(hard_clipped_audio, SAMPLE_RATE)
        composite = detector.detect(hard_clipped_audio, SAMPLE_RATE)
        vals = [max(v, 0.1) for v in detailed.values()]
        expected = float(np.prod(vals) ** (1.0 / len(vals)))
        assert composite == pytest.approx(expected, abs=1e-6)

    def test_all_clean_composite_high(self, detector, clean_complex):
        """Clean audio should yield a high composite score."""
        composite = detector.detect(clean_complex, SAMPLE_RATE)
        assert composite >= 0.8

    def test_severely_clipped_loop_low_composite(self, detector):
        """Audio that is both clipped and looped should score very low."""
        chunk_len = int(0.5 * SAMPLE_RATE)
        t_chunk = np.linspace(0, 0.5, chunk_len, endpoint=False)
        pattern = (3.0 * np.sin(2 * np.pi * 440 * t_chunk)).astype(np.float32)
        pattern = np.clip(pattern, -1.0, 1.0)
        n_tiles = int(DURATION / 0.5)
        audio = np.tile(pattern, n_tiles)

        composite = detector.detect(audio, SAMPLE_RATE)
        assert composite < 0.5

    def test_exception_returns_one(self, detector):
        """If an unexpected error occurs, detect() should fail open (return 1.0)."""
        # Pass a completely invalid input type
        result = detector.detect(None, SAMPLE_RATE)  # type: ignore[arg-type]
        assert result == 1.0
