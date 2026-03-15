"""
Vocal quality scorer for TuneForge.

Evaluates vocal characteristics of generated music using harmonic signal
analysis.  Four complementary metrics capture different facets of vocal
quality:

* **vocal_presence** — Vocal-range energy ratio in the harmonic signal.
* **vocal_clarity** — Spectral flatness in the vocal band (low = clear).
* **pitch_consistency** — f0 jitter stability via pyin on harmonic signal.
* **harmonic_richness** — Ratio of harmonic energy to total energy.

Genre-aware: instrumental genres (ambient, electronic, classical) receive
a neutral 0.5 score on all metrics so that vocal absence does not penalize
genuinely instrumental music.
"""

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

VOCAL_WEIGHTS: dict[str, float] = {
    "vocal_presence": 0.25,
    "vocal_clarity": 0.25,
    "pitch_consistency": 0.25,
    "harmonic_richness": 0.25,
}


class VocalQualityScorer:
    """Assess vocal quality of generated audio."""

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
        Compute per-metric vocal quality scores.

        Args:
            audio: Waveform array (1-D or 2-D).
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware scoring.
            vocals_requested: If True, always evaluate vocals regardless of genre.

        Returns:
            Dict with keys matching ``VOCAL_WEIGHTS``.  All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # --- Edge-case guards (neutral, not zero) ---
            if np.max(np.abs(audio)) < self._SILENCE_THRESHOLD:
                return {k: 0.5 for k in VOCAL_WEIGHTS}
            if len(audio) / sr < self._MIN_DURATION:
                return {k: 0.5 for k in VOCAL_WEIGHTS}

            # Pre-compute HPSS harmonic signal (shared across metrics)
            harmonic = librosa.effects.hpss(audio)[0]

            # --- Prompt gate ---
            if not vocals_requested:
                # Check if unwanted vocals are present; penalize if so
                presence = self._detect_vocal_presence(harmonic, sr, librosa)
                if presence > 0.4:
                    # Vocals detected but not requested — penalize proportionally
                    penalty = max(0.05, 0.5 - presence)
                    return {k: penalty for k in VOCAL_WEIGHTS}
                return {k: 0.5 for k in VOCAL_WEIGHTS}

            return {
                "vocal_presence": self._score_vocal_presence(harmonic, sr, librosa),
                "vocal_clarity": self._score_vocal_clarity(harmonic, sr, librosa),
                "pitch_consistency": self._score_pitch_consistency(harmonic, sr, librosa),
                "harmonic_richness": self._score_harmonic_richness(audio, harmonic, sr, librosa),
            }
        except Exception as exc:
            logger.error(f"Vocal quality scoring failed: {exc}")
            return {k: 0.5 for k in VOCAL_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate vocal quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in VOCAL_WEIGHTS.items():
            total += scores.get(metric, 0.5) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Vocal detection (for unwanted-vocal penalty)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vocal_presence(harmonic: np.ndarray, sr: int, librosa) -> float:
        """Return raw vocal-band energy ratio (0-1). Higher = more vocal content."""
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(harmonic, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            vocal_mask = (freqs >= 300.0) & (freqs <= 4000.0)
            total_energy = float(np.sum(S ** 2))
            if total_energy < 1e-10:
                return 0.0
            vocal_energy = float(np.sum(S[vocal_mask, :] ** 2))
            return vocal_energy / total_energy
        except Exception:
            return 0.0

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

        Score: ``1.0 - min(flatness / 0.3, 1.0)`` — lower flatness = higher score.
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
    def _score_pitch_consistency(harmonic: np.ndarray, sr: int, librosa) -> float:
        """
        f0 stability in the harmonic signal via pyin.

        Musical vocals have moderate frame-to-frame pitch jitter; off-key
        or erratic singing produces high jitter.  The standard deviation of
        consecutive f0 differences (in semitones) is scored with a bell
        curve centered at 1.5 semitones.
        """
        try:
            fmin = librosa.note_to_hz("C2")
            fmax = librosa.note_to_hz("C7")

            f0, voiced_flag, _ = librosa.pyin(
                harmonic, fmin=fmin, fmax=fmax, sr=sr,
            )

            voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(voiced) < 2:
                return 0.5

            # Convert consecutive f0 differences to semitones
            semitone_diffs = np.abs(12.0 * np.log2(voiced[1:] / (voiced[:-1] + 1e-10) + 1e-10))

            if len(semitone_diffs) == 0:
                return 0.5

            std_jitter = float(np.std(semitone_diffs))

            # Bell curve centered at 1.5 semitones std
            score = float(np.exp(-2.0 * (std_jitter - 1.5) ** 2))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _score_harmonic_richness(audio: np.ndarray, harmonic: np.ndarray, sr: int, librosa) -> float:
        """
        Vocal harmonics quality: ratio of harmonic energy to total energy.

        Rich vocals have strong harmonic structure.  Uses
        ``librosa.effects.harmonic`` energy vs total audio energy as a
        proxy.  Scored via bell curve centered at 0.5.
        """
        try:
            # Compute harmonic component energy via librosa.effects.harmonic
            harmonic_only = librosa.effects.harmonic(y=audio)

            total_energy = float(np.sum(audio ** 2))
            if total_energy < 1e-10:
                return 0.5

            harmonic_energy = float(np.sum(harmonic_only ** 2))
            ratio = harmonic_energy / total_energy

            # Bell curve centered at 0.5
            score = float(np.exp(-8.0 * (ratio - 0.5) ** 2))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5
