"""Tests for chord coherence scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.chord_coherence import (
    ChordCoherenceScorer,
    CHORD_TEMPLATES,
    _COF_INDEX,
)


def _librosa_available() -> bool:
    try:
        import librosa
        y = np.zeros(4096, dtype=np.float32)
        librosa.feature.chroma_cqt(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_available(),
    reason="librosa features unavailable",
)


@pytest.fixture
def scorer():
    return ChordCoherenceScorer()


class TestChordTemplates:

    def test_24_templates(self):
        """Should have 12 major + 12 minor = 24 templates."""
        assert len(CHORD_TEMPLATES) == 24

    def test_templates_normalized(self):
        """Each template should be unit-normalized."""
        for name, tmpl in CHORD_TEMPLATES.items():
            assert tmpl.shape == (12,)
            assert np.linalg.norm(tmpl) == pytest.approx(1.0, abs=1e-6), f"{name} not normalized"

    def test_c_major_template(self):
        """C major: C(0), E(4), G(7) should be non-zero."""
        tmpl = CHORD_TEMPLATES["C_major"]
        assert tmpl[0] > 0  # C
        assert tmpl[4] > 0  # E
        assert tmpl[7] > 0  # G
        # Other positions should be zero
        for i in [1, 2, 3, 5, 6, 8, 9, 10, 11]:
            assert tmpl[i] == 0.0


class TestMatchChord:

    def test_c_major_chroma(self):
        """A chroma vector with energy at C, E, G should match C_major."""
        chroma = np.zeros(12)
        chroma[0] = 1.0  # C
        chroma[4] = 1.0  # E
        chroma[7] = 1.0  # G
        chord, confidence = ChordCoherenceScorer._match_chord(chroma)
        assert chord == "C_major"
        assert confidence > 0.9

    def test_a_minor_chroma(self):
        """A chroma vector with energy at A, C, E should match A_minor."""
        chroma = np.zeros(12)
        chroma[9] = 1.0   # A
        chroma[0] = 1.0   # C
        chroma[4] = 1.0   # E
        chord, confidence = ChordCoherenceScorer._match_chord(chroma)
        assert chord == "A_minor"
        assert confidence > 0.9

    def test_zero_chroma(self):
        """Zero vector should return 'none' with 0 confidence."""
        chroma = np.zeros(12)
        chord, confidence = ChordCoherenceScorer._match_chord(chroma)
        assert chord == "none"
        assert confidence == 0.0


class TestTransitionQuality:

    def test_same_chord(self):
        score = ChordCoherenceScorer._transition_quality("C_major", "C_major")
        assert score == 0.7

    def test_relative_major_minor(self):
        """Same root, different mode should score 1.0."""
        score = ChordCoherenceScorer._transition_quality("C_major", "C_minor")
        assert score == 1.0

    def test_adjacent_cof(self):
        """C -> G is adjacent on the circle of fifths."""
        score = ChordCoherenceScorer._transition_quality("C_major", "G_major")
        assert score == 1.0

    def test_distant_cof(self):
        """C -> F# is far on the circle of fifths."""
        score = ChordCoherenceScorer._transition_quality("C_major", "F#_major")
        assert score <= 0.2

    def test_none_chord(self):
        score = ChordCoherenceScorer._transition_quality("none", "C_major")
        assert score == 0.3


class TestScoreVariety:

    def test_five_chords_30s_optimal(self):
        """5 unique chords in 30s should be near-optimal."""
        chords = ["C_major", "G_major", "A_minor", "F_major", "D_minor"]
        score = ChordCoherenceScorer._score_variety(chords, 30.0)
        assert score > 0.9

    def test_one_chord_low_variety(self):
        """A single chord repeated should score low."""
        chords = ["C_major"] * 10
        score = ChordCoherenceScorer._score_variety(chords, 30.0)
        assert score < 0.5

    def test_many_chords_lower_variety(self):
        """Too many chords (far from 5) should score lower."""
        chords = [f"note_{i}" for i in range(15)]  # 15 unique
        score = ChordCoherenceScorer._score_variety(chords, 30.0)
        assert score < 0.5


class TestScoreClarity:

    def test_high_confidence(self):
        score = ChordCoherenceScorer._score_clarity([0.9, 0.85, 0.95])
        assert score == pytest.approx(1.0, abs=0.1)

    def test_low_confidence(self):
        score = ChordCoherenceScorer._score_clarity([0.5, 0.5, 0.5])
        assert score == pytest.approx(0.0, abs=0.05)

    def test_empty(self):
        score = ChordCoherenceScorer._score_clarity([])
        assert score == 0.0


class TestChordCoherenceWeights:

    def test_weights_sum_to_one(self):
        total = sum(ChordCoherenceScorer.WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestChordCoherenceInterface:

    def test_short_audio_returns_zeros(self, scorer):
        """Audio < 2s should return zeros."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 32000)).astype(np.float32)
        scores = scorer.score(audio, 32000)
        assert set(scores.keys()) == set(ChordCoherenceScorer.WEIGHTS.keys())
        for v in scores.values():
            assert v == 0.0

    @_skip_librosa
    def test_score_returns_all_keys(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        assert set(scores.keys()) == set(ChordCoherenceScorer.WEIGHTS.keys())

    @_skip_librosa
    def test_scores_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    @_skip_librosa
    def test_aggregate_in_range(self, scorer, sample_audio_complex, sample_rate):
        scores = scorer.score(sample_audio_complex, sample_rate)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0
