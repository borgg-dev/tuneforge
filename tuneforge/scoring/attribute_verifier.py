"""
Musical attribute verifier for TuneForge.

Verifies that generated audio matches the musical attributes specified in
the challenge (tempo, key, time signature, instruments).  Only tempo
verification is currently implemented as it is the most reliably detectable
attribute via signal analysis.
"""

import numpy as np
from loguru import logger


class AttributeVerifier:
    """Verify that generated audio matches challenge musical attributes."""

    def verify_tempo(self, audio: np.ndarray, sr: int, requested_bpm: float) -> float:
        """
        Verify that the audio tempo matches the requested BPM.

        Uses ``librosa.beat.beat_track()`` to estimate the dominant tempo.
        A tolerance of ±10% of the requested BPM is applied, and both
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
                return 0.5  # cannot determine — return neutral

            tolerance = requested_bpm * 0.10  # ±10%

            # Candidate tempos: detected, half-time, double-time
            candidates = [detected_bpm, detected_bpm / 2.0, detected_bpm * 2.0]

            best_diff = min(abs(c - requested_bpm) for c in candidates)

            if best_diff <= tolerance:
                return 1.0

            # Graceful linear decay up to ±50% off
            max_diff = requested_bpm * 0.50
            score = 1.0 - (best_diff - tolerance) / (max_diff - tolerance)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.debug(f"Tempo verification failed: {exc}")
            return 0.5  # neutral on failure

    def verify_all(self, audio: np.ndarray, sr: int, synapse) -> float:
        """
        Aggregate attribute verification across all specified challenge attributes.

        Only attributes that were actually set in the challenge are verified.
        Currently only tempo is verified as it is the most reliably detectable
        attribute via signal analysis alone.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            synapse: ``MusicGenerationSynapse`` carrying the challenge parameters.

        Returns:
            Aggregate attribute score in [0, 1].
        """
        scores: list[float] = []

        # --- Tempo ---
        tempo_bpm = getattr(synapse, "tempo_bpm", None)
        if tempo_bpm is not None and tempo_bpm > 0:
            scores.append(self.verify_tempo(audio, sr, float(tempo_bpm)))

        # Future attributes (key_signature, time_signature, instruments) can be
        # added here as reliable detection methods become available.

        if not scores:
            return 1.0  # nothing to verify — full score

        return float(np.clip(np.mean(scores), 0.0, 1.0))
