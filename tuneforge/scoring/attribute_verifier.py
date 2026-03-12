"""
Musical attribute verifier for TuneForge.

Verifies that generated audio matches the musical attributes specified in
the challenge (tempo, key, instruments).  Uses librosa for tempo and key
detection, and CLAP zero-shot classification for instrument detection.
"""

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}
_CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Sub-metric weights
_WEIGHTS = {"tempo": 0.40, "key": 0.30, "instruments": 0.30}


def _rotate(profile: list[float], n: int) -> list[float]:
    """Rotate a profile list by *n* positions to the right."""
    n = n % len(profile)
    return profile[-n:] + profile[:-n] if n else list(profile)


def _parse_key(key_str: str) -> tuple[str, str] | None:
    """
    Parse a key string like "C major", "Db minor", "F# major" into
    (canonical_root, mode) or None on failure.
    """
    parts = key_str.strip().split()
    if len(parts) < 2:
        return None

    root_raw = parts[0].capitalize()
    # Handle e.g. "c#" → "C#"
    if len(root_raw) > 1:
        root_raw = root_raw[0].upper() + root_raw[1:]

    mode = parts[1].lower()
    if mode not in ("major", "minor"):
        return None

    # Enharmonic normalization
    root = _ENHARMONIC.get(root_raw, root_raw)
    if root not in _NOTE_NAMES:
        return None

    return root, mode


def _cof_distance(note_a: str, note_b: str) -> int:
    """Return the minimum circle-of-fifths distance between two notes."""
    if note_a not in _CIRCLE_OF_FIFTHS or note_b not in _CIRCLE_OF_FIFTHS:
        return 12  # impossible distance → will map to 0.0
    idx_a = _CIRCLE_OF_FIFTHS.index(note_a)
    idx_b = _CIRCLE_OF_FIFTHS.index(note_b)
    d = abs(idx_a - idx_b)
    return min(d, 12 - d)


def _relative_root(root: str, mode: str) -> str:
    """
    Return the root of the relative key.
    Major → relative minor root is 9 semitones up (same as 3 semitones down).
    Minor → relative major root is 3 semitones up.
    """
    idx = _NOTE_NAMES.index(root)
    if mode == "major":
        return _NOTE_NAMES[(idx + 9) % 12]
    else:
        return _NOTE_NAMES[(idx + 3) % 12]


class AttributeVerifier:
    """Verify that generated audio matches challenge musical attributes."""

    def __init__(self, clap_scorer=None):
        self._clap = clap_scorer

    # ------------------------------------------------------------------
    # Tempo
    # ------------------------------------------------------------------

    def verify_tempo(self, audio: np.ndarray, sr: int, requested_bpm: float) -> float:
        """
        Verify that the audio tempo matches the requested BPM.

        Uses ``librosa.beat.beat_track()`` to estimate the dominant tempo.
        A tolerance of +/-10% of the requested BPM is applied, and both
        double-time and half-time interpretations are tested to account for
        common beat-tracker errors.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            requested_bpm: Target tempo from the challenge.

        Returns:
            Score in [0, 1].  1.0 = tempo matches; 0.0 = far off.
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            # librosa may return a scalar or a 1-element array
            detected_bpm = float(np.atleast_1d(tempo)[0])

            if detected_bpm <= 0 or requested_bpm <= 0:
                return 0.5  # cannot determine -- return neutral

            tolerance = requested_bpm * 0.10  # +/-10%

            # Candidate tempos: detected, half-time, double-time
            candidates = [detected_bpm, detected_bpm / 2.0, detected_bpm * 2.0]

            best_diff = min(abs(c - requested_bpm) for c in candidates)

            if best_diff <= tolerance:
                return 1.0

            # Graceful linear decay up to +/-50% off
            max_diff = requested_bpm * 0.50
            score = 1.0 - (best_diff - tolerance) / (max_diff - tolerance)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.debug(f"Tempo verification failed: {exc}")
            return 0.5  # neutral on failure

    # ------------------------------------------------------------------
    # Key detection
    # ------------------------------------------------------------------

    def verify_key(self, audio: np.ndarray, sr: int, requested_key: str) -> float:
        """
        Detect the musical key of *audio* and compare to *requested_key*.

        Uses chroma CQT features correlated against Krumhansl-Kessler key
        profiles.  Scoring is based on circle-of-fifths distance between
        the detected and requested keys.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            requested_key: e.g. "C major", "Db minor".

        Returns:
            Score in [0, 1].
        """
        try:
            import librosa

            parsed = _parse_key(requested_key)
            if parsed is None:
                logger.debug(f"Could not parse requested key: {requested_key}")
                return 0.5

            req_root, req_mode = parsed

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            # Compute chroma and sum across time
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_dist = chroma.sum(axis=1)  # shape (12,)

            # Correlate against all 24 keys
            best_corr = -np.inf
            det_root_idx = 0
            det_mode = "major"

            for shift in range(12):
                for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
                    rotated = _rotate(profile, shift)
                    corr = float(np.corrcoef(chroma_dist, rotated)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        det_root_idx = shift
                        det_mode = mode

            det_root = _NOTE_NAMES[det_root_idx]

            # --- Scoring ---
            if det_root == req_root and det_mode == req_mode:
                return 1.0

            # Relative major/minor check
            rel_root = _relative_root(req_root, req_mode)
            rel_mode = "minor" if req_mode == "major" else "major"
            if det_root == rel_root and det_mode == rel_mode:
                return 0.8

            # Circle-of-fifths distance (ignore mode)
            dist = _cof_distance(det_root, req_root)
            if dist == 1:
                return 0.5
            if dist == 2:
                return 0.3

            return 0.0

        except Exception as exc:
            logger.debug(f"Key verification failed: {exc}")
            return 0.5

    # ------------------------------------------------------------------
    # Instrument detection
    # ------------------------------------------------------------------

    def verify_instruments(self, audio: np.ndarray, sr: int, instruments: list[str]) -> float:
        """
        Detect whether requested instruments are present using CLAP
        zero-shot classification.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            instruments: List of instrument names, e.g. ["piano", "drums"].

        Returns:
            Score in [0, 1].  Fraction of instruments detected.
        """
        if not instruments or self._clap is None:
            return 0.5

        try:
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            detected_count = 0
            for instrument in instruments:
                clap_score = self._clap.score(audio, sr, f"music featuring {instrument}")
                if clap_score > 0.25:
                    detected_count += 1

            return detected_count / len(instruments)
        except Exception as exc:
            logger.debug(f"Instrument verification failed: {exc}")
            return 0.5

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def verify_all(self, audio: np.ndarray, sr: int, synapse) -> float:
        """
        Aggregate attribute verification across all specified challenge attributes.

        Only attributes that were actually set in the challenge are verified.
        Weights are proportionally re-distributed among present attributes.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            synapse: ``MusicGenerationSynapse`` carrying the challenge parameters.

        Returns:
            Aggregate attribute score in [0, 1].
        """
        active: dict[str, float] = {}

        # --- Tempo ---
        tempo_bpm = getattr(synapse, "tempo_bpm", None)
        if tempo_bpm is not None and tempo_bpm > 0:
            active["tempo"] = self.verify_tempo(audio, sr, float(tempo_bpm))

        # --- Key ---
        key_sig = getattr(synapse, "key_signature", None)
        if key_sig is not None and isinstance(key_sig, str) and key_sig.strip():
            active["key"] = self.verify_key(audio, sr, key_sig)

        # --- Instruments ---
        instruments = getattr(synapse, "instruments", None)
        if instruments is not None and isinstance(instruments, list) and len(instruments) > 0:
            active["instruments"] = self.verify_instruments(audio, sr, instruments)

        if not active:
            return 0.5  # no attributes specified -- neutral

        # Proportional re-weighting
        total_weight = sum(_WEIGHTS[k] for k in active)
        weighted_sum = sum(_WEIGHTS[k] * v for k, v in active.items())

        return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
