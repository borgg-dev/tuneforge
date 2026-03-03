"""
Audio quality scorer for TuneForge.

Analyses musical properties of generated audio to produce per-metric and
aggregate quality scores.  All metrics target musical content rather than
raw signal properties so that professionally produced music scores well and
trivially engineered signals do not.

Genre-aware: accepts an optional genre string to adjust targets via
``GenreProfile`` (e.g. electronic has lower dynamic range targets).
"""

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import QUALITY_WEIGHTS, SILENCE_THRESHOLD
from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile


class AudioQualityScorer:
    """Assess audio quality using musically-meaningful signal analysis."""

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """
        Compute per-metric quality scores.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware target adjustment.

        Returns:
            Dict with keys: harmonic_ratio, onset_quality, spectral_contrast,
            dynamic_range, temporal_variation.  All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")

            return {
                "harmonic_ratio": self._score_harmonic_ratio(audio, librosa),
                "onset_quality": self._score_onset_quality(audio, sr, librosa, profile),
                "spectral_contrast": self._score_spectral_contrast(audio, sr, librosa),
                "dynamic_range": self._score_dynamic_range(audio, sr, librosa, profile),
                "temporal_variation": self._score_temporal_variation(audio, sr, librosa),
            }
        except Exception as exc:
            logger.error(f"Audio quality scoring failed: {exc}")
            return {k: 0.0 for k in QUALITY_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in QUALITY_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_harmonic_ratio(audio: np.ndarray, librosa) -> float:
        """
        Score based on harmonic-to-percussive energy ratio.

        Musical audio contains a mix of harmonic and percussive content.
        Pure noise has neither; pure sine waves have only harmonic content.
        The ideal range (~0.3-0.7 harmonic ratio) is scored highest.
        """
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.sum(harmonic ** 2))
            percussive_energy = float(np.sum(percussive ** 2))
            total_energy = harmonic_energy + percussive_energy + 1e-8

            if total_energy < 1e-8:
                return 0.0

            ratio = harmonic_energy / total_energy
            # Score peaks at 0.5, falls off toward 0.0 and 1.0
            # Using a bell-shaped curve centred at 0.5
            score = 1.0 - abs(ratio - 0.5) / 0.5
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_onset_quality(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on onset density.

        Musical audio has rhythmic onsets; pure noise or drones have very
        few.  The onset density ceiling is genre-aware (electronic = 8/sec,
        ambient = 1/sec).  A floor of 0.3 is applied for any audio with
        detectable onsets to avoid penalizing genres where low onset density
        is a legitimate musical characteristic.
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="time")
            n_onsets = len(onsets)
            if n_onsets == 0:
                return 0.0
            # Genre-aware ceiling for onset density
            ceiling = profile.onset_density_ceiling
            raw = n_onsets / (duration * ceiling)
            score = 0.3 + 0.7 * min(raw, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_spectral_contrast(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on spectral contrast across frequency bands.

        Musical audio has varied spectral content with clear peaks and
        troughs.  Engineered or spectrally flat signals score lower.
        """
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            # contrast shape: (n_bands+1, n_frames)
            mean_contrast = float(np.mean(contrast))
            # Typical musical audio has mean contrast ~20-40 dB; cap at 40 dB
            score = mean_contrast / 40.0
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_dynamic_range(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on loudness range (LRA-like).

        Computes the difference between the 95th and 5th percentile of
        frame-level RMS in dB.  Target range is genre-aware (electronic
        ~6 dB, classical ~14 dB).  Score peaks at the genre target and
        drops for both extremes.
        """
        try:
            hop_length = 512
            frame_len = 2048
            rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_length)[0]
            rms = rms[rms > 1e-8]
            if len(rms) < 2:
                return 0.0

            rms_db = 20.0 * np.log10(rms + 1e-10)
            lra = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

            # Signals with near-zero dynamic range are not music
            if lra < 1.0:
                return 0.0

            # Genre-aware target for dynamic range
            target = profile.dynamic_range_target
            tolerance = 5.0  # ±5 dB around target = full score range
            score = 1.0 - abs(lra - target) / (target + tolerance)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_temporal_variation(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on chromatic feature variation over time.

        Uses autocorrelation of chroma features to reward music that
        evolves while maintaining coherence.  Moderate self-similarity
        (0.3-0.7) scores highest; perfect repetition (1.0) or total
        randomness (0.0) both score lower.
        """
        try:
            hop_length = 512
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
            if chroma.shape[1] < 4:
                return 0.0

            # Column-normalise chroma
            norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
            chroma_norm = chroma / norms

            n_frames = chroma_norm.shape[1]
            mid = n_frames // 2
            first_half = chroma_norm[:, :mid]
            second_half = chroma_norm[:, mid: mid + first_half.shape[1]]
            if first_half.shape[1] == 0:
                return 0.0

            # Mean cosine similarity between first and second half
            similarity = float(np.mean(np.sum(first_half * second_half, axis=0)))
            similarity = float(np.clip(similarity, 0.0, 1.0))

            # Score peaks at 0.5 similarity (evolving but coherent)
            score = 1.0 - abs(similarity - 0.5) / 0.5
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
