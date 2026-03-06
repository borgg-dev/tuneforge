"""Tests for melody coherence scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.melody_coherence import MELODY_WEIGHTS, MelodyCoherenceScorer


# ---------------------------------------------------------------------------
# Helper: check if librosa pyin works (may fail under numba/coverage conflicts)
# ---------------------------------------------------------------------------

def _librosa_pyin_available() -> bool:
    """Check if librosa pyin works in this environment."""
    try:
        import librosa

        sr = 32000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                     fmax=librosa.note_to_hz("C7"), sr=sr)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_pyin_available(),
    reason="librosa pyin unavailable (numba/coverage conflict)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32_000


@pytest.fixture
def scorer():
    return MelodyCoherenceScorer()


@pytest.fixture
def sample_rate():
    return SAMPLE_RATE


@pytest.fixture
def sample_audio_melody():
    """
    A stepped sine wave walking through a simple major scale (C4-G4).

    Each note lasts 0.5 s for a total of 2.5 s, repeated twice (5.0 s total)
    to give the repetition metric something to work with.
    """
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4 D4 E4 F4 G4
    note_dur = 0.5
    sr = SAMPLE_RATE
    segments = []
    # Repeat the scale twice for repetition structure
    for _ in range(2):
        for freq in freqs:
            t = np.linspace(0, note_dur, int(sr * note_dur), endpoint=False)
            segments.append(0.6 * np.sin(2 * np.pi * freq * t))
    return np.concatenate(segments).astype(np.float32)


@pytest.fixture
def sample_audio_noise():
    """Random noise — no melodic content."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, int(SAMPLE_RATE * 5.0)).astype(np.float32)


@pytest.fixture
def sample_audio_silence():
    """Near-zero array — treated as silence."""
    return np.full(int(SAMPLE_RATE * 5.0), 1e-8, dtype=np.float32)


@pytest.fixture
def sample_audio_single_pitch():
    """Pure 440 Hz sine tone — single pitch, no melodic variation."""
    t = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False)
    return (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def sample_audio_repeated_melody():
    """A two-note pattern repeated many times — high repetition."""
    freqs = [261.63, 329.63]  # C4, E4
    note_dur = 0.25
    sr = SAMPLE_RATE
    segments = []
    for _ in range(20):
        for freq in freqs:
            t = np.linspace(0, note_dur, int(sr * note_dur), endpoint=False)
            segments.append(0.6 * np.sin(2 * np.pi * freq * t))
    return np.concatenate(segments).astype(np.float32)


# ===========================================================================
# Test classes
# ===========================================================================


class TestMelodyKeys:
    """score() returns all expected keys with values in [0, 1]."""

    @_skip_librosa
    def test_score_returns_all_keys(self, scorer, sample_audio_melody, sample_rate):
        scores = scorer.score(sample_audio_melody, sample_rate)
        assert set(scores.keys()) == set(MELODY_WEIGHTS.keys())

    @_skip_librosa
    def test_all_scores_in_range(self, scorer, sample_audio_melody, sample_rate):
        scores = scorer.score(sample_audio_melody, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_silence_returns_all_zeros(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert set(scores.keys()) == set(MELODY_WEIGHTS.keys())
        for key, val in scores.items():
            assert val == 0.0, f"{key} should be 0.0 for silence, got {val}"


class TestIntervalQuality:
    """Interval quality: musical melodies beat noise; silence = 0."""

    @_skip_librosa
    def test_melody_scores_higher_than_noise(
        self, scorer, sample_audio_melody, sample_audio_noise, sample_rate
    ):
        melody_scores = scorer.score(sample_audio_melody, sample_rate)
        noise_scores = scorer.score(sample_audio_noise, sample_rate)
        assert melody_scores["interval_quality"] >= noise_scores["interval_quality"]

    def test_silence_scores_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["interval_quality"] == 0.0

    @_skip_librosa
    def test_melody_interval_quality_positive(
        self, scorer, sample_audio_melody, sample_rate
    ):
        scores = scorer.score(sample_audio_melody, sample_rate)
        assert scores["interval_quality"] > 0.0


class TestContourQuality:
    """Contour quality: smooth melodies score higher than random; noise scores low."""

    @_skip_librosa
    def test_melody_has_contour(self, scorer, sample_audio_melody, sample_rate):
        scores = scorer.score(sample_audio_melody, sample_rate)
        assert scores["contour_quality"] > 0.0

    @_skip_librosa
    def test_noise_contour_low(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        # Noise may produce some spurious phrases but should not score high
        assert scores["contour_quality"] < 0.9

    def test_short_audio_returns_zero(self, scorer, sample_rate):
        """Audio shorter than 0.5s returns 0 for all metrics."""
        short = np.sin(np.linspace(0, 0.3, int(sample_rate * 0.3))).astype(np.float32)
        scores = scorer.score(short, sample_rate)
        assert scores["contour_quality"] == 0.0

    @_skip_librosa
    def test_smooth_vs_random_pitch(self, scorer, sample_rate):
        """Smooth stepwise melody should score differently from random jumps."""
        sr = sample_rate
        # Smooth: gentle ascending scale
        smooth_freqs = [261.63, 277.18, 293.66, 311.13, 329.63]
        smooth_segments = []
        for freq in smooth_freqs * 4:
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            smooth_segments.append(0.6 * np.sin(2 * np.pi * freq * t))
        smooth_audio = np.concatenate(smooth_segments).astype(np.float32)

        # Random: large pitch jumps
        rng = np.random.default_rng(123)
        random_freqs = rng.uniform(200, 2000, 20)
        random_segments = []
        for freq in random_freqs:
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            random_segments.append(0.6 * np.sin(2 * np.pi * freq * t))
        random_audio = np.concatenate(random_segments).astype(np.float32)

        smooth_scores = scorer.score(smooth_audio, sr)
        random_scores = scorer.score(random_audio, sr)

        # They should produce different contour quality scores
        # (not both ~1.0 as with the old buggy implementation)
        assert smooth_scores["contour_quality"] != pytest.approx(
            random_scores["contour_quality"], abs=0.05
        ), (
            f"Smooth ({smooth_scores['contour_quality']:.3f}) and random "
            f"({random_scores['contour_quality']:.3f}) should differ"
        )


class TestRepetitionStructure:
    """Repetition structure: repeated melody scores higher than random noise."""

    @_skip_librosa
    def test_repeated_melody_has_repetition(
        self, scorer, sample_audio_repeated_melody, sample_rate
    ):
        scores = scorer.score(sample_audio_repeated_melody, sample_rate)
        assert scores["repetition_structure"] > 0.0

    @_skip_librosa
    def test_noise_has_low_repetition(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        # Noise is random — unlikely to produce high repetition score
        assert scores["repetition_structure"] < 0.8

    def test_very_short_audio_zero(self, scorer, sample_rate):
        """Audio under 2 seconds should score 0 for repetition."""
        sr = sample_rate
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        scores = scorer.score(audio, sr)
        # repetition_structure needs at least 2 seconds / 2 windows
        # The overall scorer may still return a value from other metrics,
        # but the raw repetition metric internally requires >= 2s.
        # We test via the internal method directly.
        import librosa
        rep = MelodyCoherenceScorer._score_repetition_structure(audio, sr, librosa)
        assert rep == 0.0


class TestMelodicMemorability:
    """Memorability: simple scale moderate; random noise low."""

    @_skip_librosa
    def test_scale_has_moderate_memorability(
        self, scorer, sample_audio_melody, sample_rate
    ):
        scores = scorer.score(sample_audio_melody, sample_rate)
        # A simple scale should have moderate memorability (not 0, not 1)
        assert 0.0 < scores["melodic_memorability"] <= 1.0

    @_skip_librosa
    def test_noise_memorability_low(self, scorer, sample_audio_noise, sample_rate):
        scores = scorer.score(sample_audio_noise, sample_rate)
        # Random noise has dispersed pitches — low memorability
        assert scores["melodic_memorability"] < 0.8

    def test_silence_memorability_zero(self, scorer, sample_audio_silence, sample_rate):
        scores = scorer.score(sample_audio_silence, sample_rate)
        assert scores["melodic_memorability"] == 0.0


class TestAggregate:
    """Aggregate computation: weights and bounds."""

    def test_weights_sum_to_one(self):
        total = sum(MELODY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    @_skip_librosa
    def test_aggregate_in_range(self, scorer, sample_audio_melody, sample_rate):
        scores = scorer.score(sample_audio_melody, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_all_zeros(self, scorer):
        scores = {k: 0.0 for k in MELODY_WEIGHTS}
        assert scorer.aggregate(scores) == 0.0

    def test_aggregate_all_ones(self, scorer):
        scores = {k: 1.0 for k in MELODY_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)


class TestGamingResistance:
    """Trivially engineered signals should not score 1.0 on every metric."""

    @_skip_librosa
    def test_single_pitch_not_perfect(
        self, scorer, sample_audio_single_pitch, sample_rate
    ):
        """A pure sine tone (single pitch, zero variation) must not ace everything."""
        scores = scorer.score(sample_audio_single_pitch, sample_rate)
        # At least one metric should be noticeably below 1.0
        assert any(v < 0.9 for v in scores.values()), (
            f"Pure sine tone scored too high on all metrics: {scores}"
        )

    @_skip_librosa
    def test_single_pitch_memorability_penalised(
        self, scorer, sample_audio_single_pitch, sample_rate
    ):
        """Single-pitch audio should have low pitch entropy, hurting memorability."""
        scores = scorer.score(sample_audio_single_pitch, sample_rate)
        # Low entropy (one note) maps to low entropy_score via bell curve
        assert scores["melodic_memorability"] < 0.9
