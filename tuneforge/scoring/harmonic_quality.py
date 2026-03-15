"""
Harmonic quality scorer for TuneForge.

Evaluates harmonic characteristics of generated music using spectral
analysis.  Three complementary metrics capture different facets of
harmonic quality:

* **vocal_presence** -- Vocal-range energy ratio in the harmonic signal.
* **vocal_clarity** -- Spectral flatness in the vocal band (low = clear).
* **formant_structure** -- Natural formant peak structure in 300-3500 Hz.

Genre-aware: instrumental genres (ambient, electronic, classical) receive
a neutral 0.5 score on all metrics so that vocal absence does not penalize
genuinely instrumental music.
"""

import numpy as np
from loguru import logger



# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

HARMONIC_WEIGHTS: dict[str, float] = {
    "vocal_presence": 0.35,
    "vocal_clarity": 0.30,
    "formant_structure": 0.35,
}


class HarmonicQualityScorer:
    """Assess harmonic quality of generated audio."""

    # Minimum audio duration (seconds) to attempt analysis
    _MIN_DURATION: float = 0.5
    # Amplitude below which audio is considered silence
    _SILENCE_THRESHOLD: float = 1e-6

    def score(
        self,
        audio: np.ndarray,
        sr: int,
        genre: str = "",
        vocals_requested: bool = False,
    ) -> dict[str, float]:
        """
        Compute per-metric harmonic quality scores.

        Args:
            audio: Waveform array (1-D or 2-D).
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware scoring.
            vocals_requested: If True, always evaluate vocals regardless of genre.

        Returns:
            Dict with keys matching ``HARMONIC_WEIGHTS``.  All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # --- Prompt gate: only score vocals when explicitly requested ---
            if not vocals_requested:
                return {k: 0.5 for k in HARMONIC_WEIGHTS}

            # --- Edge-case guards (neutral, not zero) ---
            if np.max(np.abs(audio)) < self._SILENCE_THRESHOLD:
                return {k: 0.5 for k in HARMONIC_WEIGHTS}
            if len(audio) / sr < self._MIN_DURATION:
                return {k: 0.5 for k in HARMONIC_WEIGHTS}

            # Pre-compute HPSS harmonic signal (shared across metrics)
            harmonic = librosa.effects.hpss(audio)[0]

            return {
                "vocal_presence": self._score_vocal_presence(harmonic, sr, librosa),
                "vocal_clarity": self._score_vocal_clarity(harmonic, sr, librosa),
                "formant_structure": self._score_formant_structure(harmonic, sr, librosa),
            }
        except Exception as exc:
            logger.error(f"Harmonic quality scoring failed: {exc}")
            return {k: 0.5 for k in HARMONIC_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate harmonic quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in HARMONIC_WEIGHTS.items():
            total += scores.get(metric, 0.5) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_vocal_presence(harmonic: np.ndarray, sr: int, librosa) -> float:
        """
        Detect vocal-range activity in the HPSS harmonic signal.

        Computes the STFT of the harmonic component and measures the energy
        ratio of the vocal band (300 Hz -- 4 kHz) vs total energy.  A bell
        curve centered at 0.35 rewards moderate vocal presence.  A floor of
        0.3 is applied for non-instrumental genres where vocals are optional.
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(harmonic, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            vocal_mask = (freqs >= 300.0) & (freqs <= 4000.0)

            total_energy = float(np.sum(S ** 2))
            if total_energy < 1e-10:
                return 0.3

            vocal_energy = float(np.sum(S[vocal_mask, :] ** 2))
            ratio = vocal_energy / total_energy

            # Bell curve centered at 0.35 (moderate vocal presence)
            score = float(np.exp(-8.0 * (ratio - 0.35) ** 2))

            # Apply floor of 0.3 for non-instrumental genres
            score = max(score, 0.3)

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.3

    @staticmethod
    def _score_vocal_clarity(harmonic: np.ndarray, sr: int, librosa) -> float:
        """
        Spectral flatness in the vocal band (300 Hz -- 4 kHz).

        Clear vocals have a peaked (non-flat) spectrum, while muddy or
        noisy audio has high spectral flatness.  Flatness is computed as
        the ratio of geometric mean to arithmetic mean of magnitudes in
        the vocal band.

        Score: ``1.0 - min(flatness / 0.3, 1.0)`` -- lower flatness = higher score.
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(harmonic, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            vocal_mask = (freqs >= 300.0) & (freqs <= 4000.0)
            vocal_S = S[vocal_mask, :]

            if vocal_S.size == 0:
                return 0.5

            # Mean magnitude per vocal bin across time
            mean_magnitudes = np.mean(vocal_S, axis=1)

            if np.any(mean_magnitudes < 0):
                return 0.5

            # Compute spectral flatness: geometric mean / arithmetic mean
            arithmetic_mean = float(np.mean(mean_magnitudes))
            if arithmetic_mean < 1e-10:
                return 0.5

            # Geometric mean via log to avoid overflow
            log_magnitudes = np.log(mean_magnitudes + 1e-10)
            geometric_mean = float(np.exp(np.mean(log_magnitudes)))

            flatness = geometric_mean / arithmetic_mean

            score = 1.0 - min(flatness / 0.3, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _score_formant_structure(harmonic: np.ndarray, sr: int, librosa) -> float:
        """
        Analyze spectral envelope in 300-3500 Hz for natural formant peaks.

        Natural vocal/harmonic audio typically has 2-3 formant peaks in
        this range.  Uses smoothed spectral envelope and peak detection.
        A bell curve centered at 2.5 peaks rewards natural formant structure.
        """
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks

            n_fft = 2048
            S = np.abs(librosa.stft(harmonic, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Focus on formant range 300-3500 Hz
            formant_mask = (freqs >= 300.0) & (freqs <= 3500.0)
            formant_S = S[formant_mask, :]

            if formant_S.size == 0:
                return 0.5

            # Average spectral envelope across time
            spectral_envelope = np.mean(formant_S, axis=1)

            if len(spectral_envelope) < 5:
                return 0.5

            # Smooth the envelope to find broad formant peaks
            # Sigma proportional to number of bins for consistent smoothing
            sigma = max(1.0, len(spectral_envelope) / 15.0)
            smoothed = gaussian_filter1d(spectral_envelope, sigma=sigma)

            # Find peaks in smoothed envelope
            # Prominence threshold relative to envelope range
            envelope_range = float(np.max(smoothed) - np.min(smoothed))
            if envelope_range < 1e-10:
                return 0.5

            prominence = envelope_range * 0.1
            peaks, _ = find_peaks(smoothed, prominence=prominence)

            n_peaks = len(peaks)

            # Bell curve centered at 2.5 peaks (2-3 is natural)
            score = float(np.exp(-1.5 * (n_peaks - 2.5) ** 2))

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5
