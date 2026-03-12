"""Tests for tuneforge.scoring.attribute_verifier."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tuneforge.scoring.attribute_verifier import (
    AttributeVerifier,
    _cof_distance,
    _parse_key,
    _relative_root,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synapse(**kwargs):
    """Build a minimal synapse-like object with the given attributes."""
    return SimpleNamespace(**kwargs)


def _c_major_chroma():
    """
    Return a (12, T) chroma matrix whose summed distribution peaks at C,
    producing a clear C-major detection.
    """
    # C-major profile emphasises C(0), E(4), G(7)
    dist = np.array([10.0, 0.5, 1.0, 0.5, 5.0, 2.0, 0.5, 6.0, 0.5, 1.0, 0.5, 1.0])
    # Broadcast into (12, 10) so librosa-like shape
    return dist[:, None] * np.ones((1, 10))


def _a_minor_chroma():
    """Chroma distribution that correlates best with A minor."""
    # A-minor profile emphasises A(9), C(0), E(4)
    dist = np.array([5.0, 0.5, 1.0, 0.5, 4.0, 1.0, 0.5, 2.0, 1.0, 10.0, 0.5, 1.0])
    return dist[:, None] * np.ones((1, 10))


def _f_sharp_major_chroma():
    """Chroma distribution that correlates best with F# major (distant from C)."""
    # F# major emphasises F#(6), A#(10), C#(1)
    dist = np.array([0.5, 5.0, 0.5, 0.5, 0.5, 1.0, 10.0, 0.5, 0.5, 1.0, 6.0, 0.5])
    return dist[:, None] * np.ones((1, 10))


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestParseKey:
    def test_basic(self):
        assert _parse_key("C major") == ("C", "major")
        assert _parse_key("A minor") == ("A", "minor")

    def test_enharmonic(self):
        assert _parse_key("Db major") == ("C#", "major")
        assert _parse_key("Bb minor") == ("A#", "minor")

    def test_invalid(self):
        assert _parse_key("X major") is None
        assert _parse_key("C") is None
        assert _parse_key("C lydian") is None

    def test_case_insensitive_root(self):
        assert _parse_key("c major") == ("C", "major")


class TestCofDistance:
    def test_same_note(self):
        assert _cof_distance("C", "C") == 0

    def test_adjacent(self):
        assert _cof_distance("C", "G") == 1
        assert _cof_distance("C", "F") == 1

    def test_opposite(self):
        assert _cof_distance("C", "F#") == 6


class TestRelativeRoot:
    def test_c_major(self):
        assert _relative_root("C", "major") == "A"

    def test_a_minor(self):
        assert _relative_root("A", "minor") == "C"


# ---------------------------------------------------------------------------
# Tempo verification
# ---------------------------------------------------------------------------

class TestVerifyTempo:
    def test_exact_tempo(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        with patch("librosa.beat.beat_track", return_value=(np.array([120.0]), None)):
            score = verifier.verify_tempo(audio, 22050, 120.0)
        assert score == 1.0

    def test_half_time_match(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        with patch("librosa.beat.beat_track", return_value=(np.array([60.0]), None)):
            score = verifier.verify_tempo(audio, 22050, 120.0)
        assert score == 1.0  # 60 * 2 = 120

    def test_far_off(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        # 300 BPM detected vs 80 requested: candidates are 300, 150, 600
        # closest is 150, diff=70, tolerance=8, max_diff=40 → clipped to 0.0
        with patch("librosa.beat.beat_track", return_value=(np.array([300.0]), None)):
            score = verifier.verify_tempo(audio, 22050, 80.0)
        assert score == 0.0

    def test_failure_returns_neutral(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)

        with patch("librosa.beat.beat_track", side_effect=RuntimeError("boom")):
            score = verifier.verify_tempo(audio, 22050, 120.0)
        assert score == 0.5


# ---------------------------------------------------------------------------
# Key verification
# ---------------------------------------------------------------------------

class TestVerifyKey:
    def test_verify_key_exact_match(self):
        """C major chroma + 'C major' request -> 1.0."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        with patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_key(audio, 22050, "C major")
        assert score == 1.0

    def test_verify_key_relative(self):
        """C major detected when A minor requested -> 0.8 (relative key)."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        # Return C-major-like chroma so detection finds C major
        with patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_key(audio, 22050, "A minor")
        assert score == 0.8

    def test_verify_key_no_match(self):
        """F# major detected when C major requested -> 0.0 (CoF dist 6)."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        with patch("librosa.feature.chroma_cqt", return_value=_f_sharp_major_chroma()):
            score = verifier.verify_key(audio, 22050, "C major")
        assert score == 0.0

    def test_verify_key_cof_distance_1(self):
        """C major detected when G major requested -> 0.5 (CoF dist 1)."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)

        with patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_key(audio, 22050, "G major")
        assert score == 0.5

    def test_verify_key_unparseable_returns_neutral(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)
        assert verifier.verify_key(audio, 22050, "nonsense") == 0.5

    def test_verify_key_failure_returns_neutral(self):
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)

        with patch("librosa.feature.chroma_cqt", side_effect=RuntimeError("boom")):
            score = verifier.verify_key(audio, 22050, "C major")
        assert score == 0.5


# ---------------------------------------------------------------------------
# Instrument verification
# ---------------------------------------------------------------------------

class TestVerifyInstruments:
    def test_all_detected(self):
        mock_clap = MagicMock()
        mock_clap.score.return_value = 0.6  # above 0.25 threshold
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050 * 5).astype(np.float32)

        score = verifier.verify_instruments(audio, 22050, ["piano", "drums"])
        assert score == 1.0
        assert mock_clap.score.call_count == 2

    def test_partial_detection(self):
        mock_clap = MagicMock()
        # piano detected, drums not
        mock_clap.score.side_effect = [0.6, 0.1]
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050 * 5).astype(np.float32)

        score = verifier.verify_instruments(audio, 22050, ["piano", "drums"])
        assert score == 0.5

    def test_none_detected(self):
        mock_clap = MagicMock()
        mock_clap.score.return_value = 0.1  # below threshold
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050 * 5).astype(np.float32)

        score = verifier.verify_instruments(audio, 22050, ["piano", "drums"])
        assert score == 0.0

    def test_empty_instruments_returns_neutral(self):
        verifier = AttributeVerifier(clap_scorer=MagicMock())
        audio = np.random.randn(22050).astype(np.float32)
        assert verifier.verify_instruments(audio, 22050, []) == 0.5

    def test_no_clap_returns_neutral(self):
        verifier = AttributeVerifier(clap_scorer=None)
        audio = np.random.randn(22050).astype(np.float32)
        assert verifier.verify_instruments(audio, 22050, ["piano"]) == 0.5

    def test_clap_prompt_format(self):
        mock_clap = MagicMock()
        mock_clap.score.return_value = 0.6
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050).astype(np.float32)

        verifier.verify_instruments(audio, 22050, ["violin"])
        mock_clap.score.assert_called_once()
        call_args = mock_clap.score.call_args
        assert call_args[0][2] == "music featuring violin"


# ---------------------------------------------------------------------------
# verify_all integration
# ---------------------------------------------------------------------------

class TestVerifyAll:
    def test_verify_all_proportional_reweighting(self):
        """When only tempo is set, the full score should equal the tempo score."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)
        synapse = _make_synapse(tempo_bpm=120.0)

        with patch("librosa.beat.beat_track", return_value=(np.array([120.0]), None)):
            score = verifier.verify_all(audio, 22050, synapse)
        assert score == 1.0

    def test_verify_all_no_attributes(self):
        """No attributes specified -> return 0.5 (neutral)."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)
        synapse = _make_synapse()  # no attributes
        assert verifier.verify_all(audio, 22050, synapse) == 0.5

    def test_verify_all_tempo_and_key(self):
        """Both tempo and key set; check weighted average."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)
        synapse = _make_synapse(tempo_bpm=120.0, key_signature="C major")

        with patch("librosa.beat.beat_track", return_value=(np.array([120.0]), None)), \
             patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_all(audio, 22050, synapse)

        # Both perfect: (0.4 * 1.0 + 0.3 * 1.0) / (0.4 + 0.3) = 1.0
        assert score == pytest.approx(1.0)

    def test_verify_all_all_three(self):
        """All three attributes set; verify proportional weighting."""
        mock_clap = MagicMock()
        mock_clap.score.return_value = 0.6  # all instruments detected
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050 * 5).astype(np.float32)
        synapse = _make_synapse(
            tempo_bpm=120.0,
            key_signature="C major",
            instruments=["piano"],
        )

        with patch("librosa.beat.beat_track", return_value=(np.array([120.0]), None)), \
             patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_all(audio, 22050, synapse)

        # All perfect: (0.4*1.0 + 0.3*1.0 + 0.3*1.0) / 1.0 = 1.0
        assert score == pytest.approx(1.0)

    def test_verify_all_only_key(self):
        """When only key is set, score equals key score."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)
        synapse = _make_synapse(key_signature="C major")

        with patch("librosa.feature.chroma_cqt", return_value=_c_major_chroma()):
            score = verifier.verify_all(audio, 22050, synapse)
        assert score == pytest.approx(1.0)

    def test_verify_all_only_instruments(self):
        """When only instruments set, score equals instruments score."""
        mock_clap = MagicMock()
        mock_clap.score.side_effect = [0.6, 0.1]  # 1 of 2 detected
        verifier = AttributeVerifier(clap_scorer=mock_clap)
        audio = np.random.randn(22050 * 5).astype(np.float32)
        synapse = _make_synapse(instruments=["piano", "drums"])

        score = verifier.verify_all(audio, 22050, synapse)
        assert score == pytest.approx(0.5)

    def test_verify_all_empty_key_string_ignored(self):
        """An empty key_signature string should be ignored."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)
        synapse = _make_synapse(key_signature="", instruments=[])
        assert verifier.verify_all(audio, 22050, synapse) == 0.5

    def test_verify_all_empty_instruments_ignored(self):
        """An empty instruments list should be ignored."""
        verifier = AttributeVerifier()
        audio = np.random.randn(22050).astype(np.float32)
        synapse = _make_synapse(instruments=[])
        assert verifier.verify_all(audio, 22050, synapse) == 0.5
