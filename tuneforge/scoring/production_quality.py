"""
Production quality scorer for TuneForge.

Measures audio engineering and mastering quality using metrics that
professional audio engineers assess: spectral balance, frequency
utilization, loudness standards, and dynamic expressiveness.

Genre-aware: accepts an optional genre string to adjust targets via
``GenreProfile`` (e.g. electronic music expects tighter loudness).
"""

import numpy as np
from loguru import logger

from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile
from tuneforge.scoring.stereo_quality import StereoQualityScorer


def librosa_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using librosa (module-level helper for static methods)."""
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


PRODUCTION_WEIGHTS: dict[str, float] = {
    "spectral_balance": 0.25,
    "frequency_fullness": 0.20,
    "loudness_consistency": 0.20,
    "dynamic_expressiveness": 0.15,
    "stereo_quality": 0.20,
}


class ProductionQualityScorer:
    """Assess production/mastering quality of generated audio."""

    def __init__(self) -> None:
        self._stereo = StereoQualityScorer()

    def score(self, audio: np.ndarray, sr: int, genre: str = "", raw_audio: np.ndarray | None = None) -> dict[str, float]:
        """
        Compute per-metric production quality scores.

        Args:
            audio: 1-D or 2-D float waveform.
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware target adjustment.
            raw_audio: Optional multichannel audio for stereo analysis.
                       If provided and 2-channel, used for stereo quality scoring.

        Returns:
            Dict with keys: spectral_balance, frequency_fullness,
            loudness_consistency, dynamic_expressiveness, stereo_quality.
            All values in [0, 1].
        """
        try:
            import librosa

            # Compute stereo quality before downmixing
            if raw_audio is not None and raw_audio.ndim == 2:
                stereo_scores = self._stereo.score(raw_audio, sr, genre=genre)
                stereo_quality = self._stereo.aggregate(stereo_scores)
            else:
                stereo_quality = 0.3  # Below-neutral default when no multichannel data

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")

            return {
                "spectral_balance": self._score_spectral_balance(audio, sr, librosa, profile),
                "frequency_fullness": self._score_frequency_fullness(audio, sr, librosa),
                "loudness_consistency": self._score_loudness_consistency(audio, sr, librosa, profile),
                "dynamic_expressiveness": self._score_dynamic_expressiveness(audio, sr, librosa, profile),
                "stereo_quality": stereo_quality,
            }
        except Exception as exc:
            logger.error(f"Production quality scoring failed: {exc}")
            return {k: 0.0 for k in PRODUCTION_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate production quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in PRODUCTION_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_spectral_balance(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on spectral energy distribution across frequency bands.

        Well-produced music has a balanced distribution of energy across
        sub-bass, bass, low-mid, high-mid, presence, and brilliance bands.
        The coefficient of variation (std/mean) of band energies is used
        to measure balance.  A bell curve centered at CV ~ 1.0 rewards
        moderate variation (natural for music) while penalizing both
        flat/unnatural spectra (CV too low) and unbalanced spectra (CV too high).
        """
        try:
            # Check for near-silence
            if np.max(np.abs(audio)) < 1e-6:
                return 0.0

            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Define frequency bands (Hz)
            bands = [
                (20, 60),       # sub-bass
                (60, 250),      # bass
                (250, 2000),    # low-mid
                (2000, 6000),   # high-mid
                (6000, 12000),  # presence
                (12000, sr / 2),  # brilliance
            ]

            band_energies = []
            for low, high in bands:
                # Only include bands that fall within the Nyquist frequency
                if low >= sr / 2:
                    continue
                high = min(high, sr / 2)
                mask = (freqs >= low) & (freqs < high)
                if not np.any(mask):
                    continue
                energy = float(np.sqrt(np.mean(S[mask, :] ** 2)))
                band_energies.append(energy)

            if len(band_energies) < 2:
                return 0.0

            band_energies = np.array(band_energies)
            mean_energy = float(np.mean(band_energies))

            if mean_energy < 1e-10:
                return 0.0

            cv = float(np.std(band_energies) / mean_energy)

            # Bell curve centered at genre-specific CV target
            cv_target = profile.spectral_balance_cv_target
            score = 1.0 - abs(cv - cv_target) / 1.5
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_frequency_fullness(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on how much of the frequency spectrum is actively used.

        Well-produced music fills more of the audible spectrum than thin
        or tinny audio.  Frequency bins with energy above a noise floor
        (-60 dB relative to peak) are counted as active.  A fullness ratio
        of 60% or more earns a perfect score.  Pure sine waves score low;
        rich, full mixes score high.
        """
        try:
            # Check for silence
            if np.max(np.abs(audio)) < 1e-6:
                return 0.0

            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft))

            # Average magnitude spectrum across time
            mean_spectrum = np.mean(S, axis=1)

            peak = float(np.max(mean_spectrum))
            if peak < 1e-10:
                return 0.0

            # Convert to dB relative to peak
            spectrum_db = 20.0 * np.log10(mean_spectrum / peak + 1e-10)

            # Count bins above noise floor (-60 dB)
            noise_floor = -60.0
            active_bins = int(np.sum(spectrum_db > noise_floor))
            total_bins = len(mean_spectrum)

            if total_bins == 0:
                return 0.0

            fullness_ratio = active_bins / total_bins

            # 60% utilization = perfect score
            score = fullness_ratio / 0.6
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _k_weight_filter(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply ITU-R BS.1770-4 K-weighting filter for perceptual loudness.

        Two cascaded biquad stages:
        1. High-shelf pre-filter (+4 dB above 1.5 kHz, head acoustic model)
        2. RLB high-pass (38 Hz, revised low-frequency weighting)

        Resamples to 48 kHz internally so standard filter coefficients apply.
        """
        from scipy.signal import sosfilt

        # Resample to 48 kHz for standard ITU coefficients
        if sr != 48000:
            audio_48k = librosa_resample(audio.astype(np.float32), sr, 48000)
        else:
            audio_48k = audio.astype(np.float32)

        # Stage 1: High-shelf pre-filter (ITU-R BS.1770-4, 48 kHz)
        sos_stage1 = np.array([[
            1.53512485958697, -2.69169618940638, 1.19839281085285,
            1.0, -1.69065929318241, 0.73248077421585,
        ]])
        # Stage 2: RLB high-pass (ITU-R BS.1770-4, 48 kHz)
        sos_stage2 = np.array([[
            1.0, -2.0, 1.0,
            1.0, -1.99004745483398, 0.99007225036621,
        ]])

        sos = np.vstack([sos_stage1, sos_stage2])
        return sosfilt(sos, audio_48k)

    @staticmethod
    def _compute_integrated_lufs(audio: np.ndarray, sr: int) -> float:
        """
        Compute integrated LUFS (ITU-R BS.1770-4).

        Returns LUFS value (typically -5 to -50 for music).
        """
        weighted = ProductionQualityScorer._k_weight_filter(audio, sr)
        mean_square = float(np.mean(weighted ** 2))
        if mean_square < 1e-20:
            return -70.0
        return -0.691 + 10.0 * np.log10(mean_square)

    @staticmethod
    def _compute_short_term_lufs(audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute short-term LUFS in 3-second windows with 1-second hop."""
        weighted = ProductionQualityScorer._k_weight_filter(audio, sr)
        effective_sr = 48000  # always 48k after K-weighting
        window_samples = int(3.0 * effective_sr)
        hop_samples = int(1.0 * effective_sr)

        lufs_values: list[float] = []
        for start in range(0, len(weighted) - window_samples + 1, hop_samples):
            window = weighted[start: start + window_samples]
            ms = float(np.mean(window ** 2))
            if ms < 1e-20:
                lufs_values.append(-70.0)
            else:
                lufs_values.append(-0.691 + 10.0 * np.log10(ms))

        return np.array(lufs_values) if lufs_values else np.array([-70.0])

    @staticmethod
    def _score_loudness_consistency(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on LUFS loudness measurement (ITU-R BS.1770-4).

        Two sub-scores combined equally:

        * **Integrated LUFS** — how close the overall loudness is to the
          genre-specific target (e.g. -14 LUFS for pop, -23 for classical).
          Uses K-weighted perceptual loudness instead of raw RMS.
        * **Short-term LUFS consistency** — standard deviation of 3-second
          LUFS windows scored via bell curve at genre target.  Measures
          mastering consistency.

        Returns 0.0 for silence or very short audio.
        """
        try:
            from tuneforge.config.scoring_config import LUFS_TOLERANCE

            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            if np.max(np.abs(audio)) < 1e-6:
                return 0.0

            # --- Integrated LUFS score ---
            integrated = ProductionQualityScorer._compute_integrated_lufs(audio, sr)

            # Genre-aware LUFS target
            target = profile.lufs_target
            lufs_dev = abs(integrated - target)

            if lufs_dev <= LUFS_TOLERANCE:
                integrated_score = 1.0
            else:
                # Linear ramp from tolerance to tolerance * 3
                integrated_score = 1.0 - (lufs_dev - LUFS_TOLERANCE) / (LUFS_TOLERANCE * 2.0)
                integrated_score = float(np.clip(integrated_score, 0.0, 1.0))

            # --- Short-term LUFS consistency ---
            if duration < 3.0:
                # Too short for short-term analysis; use integrated only
                return integrated_score

            st_lufs = ProductionQualityScorer._compute_short_term_lufs(audio, sr)
            # Filter out floor values
            st_lufs = st_lufs[st_lufs > -69.0]

            if len(st_lufs) < 2:
                return integrated_score

            std_lufs = float(np.std(st_lufs))

            # Bell curve centered at genre-specific loudness std target
            loudness_target = profile.loudness_std_target
            consistency_score = 1.0 - abs(std_lufs - loudness_target) / 8.0
            consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

            return 0.5 * integrated_score + 0.5 * consistency_score
        except Exception:
            return 0.0

    @staticmethod
    def _score_dynamic_expressiveness(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on micro-dynamic variation within musical phrases.

        Musical audio has intentional swells, crescendos, and decrescendos.
        The first derivative of the RMS envelope captures these rate-of-change
        dynamics.  Moderate variation in the derivative indicates expressive
        performance; flat derivative = robotic/lifeless; erratic derivative
        = noise/artifacts.
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            # Check for silence
            if np.max(np.abs(audio)) < 1e-6:
                return 0.0

            # Fine resolution RMS envelope
            frame_length = 1024
            hop_length = 512
            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            if len(rms) < 4:
                return 0.0

            # Filter out near-silence frames
            if np.max(rms) < 1e-8:
                return 0.0

            # Compute first derivative of RMS envelope
            derivative = np.diff(rms)

            if len(derivative) < 2:
                return 0.0

            deriv_std = float(np.std(derivative))

            # Normalize by the mean RMS level to make the target scale-invariant
            mean_rms = float(np.mean(rms))
            if mean_rms < 1e-8:
                return 0.0

            # Normalized derivative std
            norm_deriv_std = deriv_std / mean_rms

            # Target: moderate expressiveness
            # Genre-aware calibrated target for normalized derivative std
            target = profile.dynamic_expressiveness_target

            score = float(np.exp(-50.0 * (norm_deriv_std - target) ** 2))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
