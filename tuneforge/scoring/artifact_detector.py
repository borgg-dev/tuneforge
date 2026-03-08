"""
Artifact detector for TuneForge.

Detects common AI music generation artifacts — spectral discontinuities,
clipping, repetitive loops, and spectral holes — and returns a penalty
multiplier in [0, 1] where 1.0 means no artifacts and 0.0 means severe
artifacts.
"""

import numpy as np
from loguru import logger


class ArtifactDetector:
    """Detect AI-generation artifacts in audio and return a penalty multiplier."""

    def detect(self, audio: np.ndarray, sr: int) -> float:
        """
        Run all artifact checks and return the worst-case penalty.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.

        Returns:
            Penalty multiplier in [0, 1].  1.0 = no artifacts, 0.0 = severe.
        """
        try:
            detailed = self.detect_detailed(audio, sr)
            # Use geometric mean instead of min — one bad check shouldn't
            # zero out the entire score. This is more fair across model types.
            # Floor each check at 0.1 so a single zero doesn't kill the score.
            vals = [max(v, 0.1) for v in detailed.values()]
            geo_mean = float(np.prod(vals) ** (1.0 / len(vals)))
            logger.debug(f"Artifact penalties: {detailed} → {geo_mean:.3f}")
            return geo_mean
        except Exception as exc:
            logger.error(f"Artifact detection failed: {exc}")
            return 1.0

    def detect_detailed(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Run all artifact checks and return per-check penalties.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.

        Returns:
            Dict mapping check name to penalty in [0, 1].
        """
        try:
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # Too short to assess
            duration = len(audio) / sr
            if duration < 0.5:
                return {
                    "spectral_discontinuity": 1.0,
                    "clipping": 1.0,
                    "repetition": 1.0,
                    "spectral_holes": 1.0,
                }

            # Silent audio — handled by other scorers
            if np.max(np.abs(audio)) < 1e-6:
                return {
                    "spectral_discontinuity": 1.0,
                    "clipping": 1.0,
                    "repetition": 1.0,
                    "spectral_holes": 1.0,
                }

            return {
                "spectral_discontinuity": self._check_spectral_discontinuity(audio, sr),
                "clipping": self._check_clipping(audio, sr),
                "repetition": self._check_repetition(audio, sr),
                "spectral_holes": self._check_spectral_holes(audio, sr),
            }
        except Exception as exc:
            logger.error(f"Artifact detection (detailed) failed: {exc}")
            return {
                "spectral_discontinuity": 1.0,
                "clipping": 1.0,
                "repetition": 1.0,
                "spectral_holes": 1.0,
            }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_spectral_discontinuity(audio: np.ndarray, sr: int) -> float:
        """
        Detect spectral discontinuities via frame-to-frame spectral flux.

        AI generators can produce spectral "jumps" between generation chunks.
        We flag frames where flux exceeds median + 3*MAD as discontinuities,
        using robust statistics so outliers do not inflate the threshold.

        Returns:
            Penalty in [0, 1].
        """
        try:
            n_fft = 2048
            hop_length = 512

            # Compute STFT magnitude
            from scipy.signal import stft as _stft

            _, _, Zxx = _stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            S = np.abs(Zxx)

            if S.shape[1] < 3:
                return 1.0

            # Frame-to-frame spectral flux (L2 norm of differences)
            diffs = np.diff(S, axis=1)
            flux = np.linalg.norm(diffs, axis=0)

            if len(flux) < 3:
                return 1.0

            # Use median + MAD for robust outlier detection (mean + std
            # is inflated by the very outliers we want to detect).
            median_flux = np.median(flux)
            mad = np.median(np.abs(flux - median_flux))

            if mad < 1e-10:
                return 1.0

            threshold = median_flux + 3.0 * 1.4826 * mad  # 1.4826 scales MAD to std
            flagged = np.sum(flux > threshold)
            total = len(flux)
            ratio = flagged / total

            # Normal music has 4-9% flagged frames due to note attacks,
            # crescendos, etc.  Only penalise when the ratio clearly exceeds
            # what natural music produces (>15%).
            if ratio <= 0.10:
                penalty = 1.0
            elif ratio >= 0.40:
                penalty = 0.0
            else:
                penalty = 1.0 - (ratio - 0.10) / (0.40 - 0.10)
            return float(np.clip(penalty, 0.0, 1.0))
        except Exception:
            return 1.0

    @staticmethod
    def _check_clipping(audio: np.ndarray, sr: int) -> float:
        """
        Detect hard and soft clipping.

        Hard clipping: samples with |value| > 0.999.
        Soft clipping: runs of >100 consecutive samples above 0.95 amplitude.

        Returns:
            Penalty in [0, 1].
        """
        try:
            total_samples = len(audio)
            if total_samples == 0:
                return 1.0

            # Hard clipping
            hard_clip_count = int(np.sum(np.abs(audio) > 0.999))

            # Soft clipping: consecutive runs above 0.95
            above_threshold = np.abs(audio) > 0.95
            soft_clip_count = 0

            if np.any(above_threshold):
                # Find runs of True values
                padded = np.concatenate(([False], above_threshold, [False]))
                edges = np.diff(padded.astype(int))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                run_lengths = ends - starts

                # Count samples in runs longer than 100
                long_runs = run_lengths[run_lengths > 100]
                soft_clip_count = int(np.sum(long_runs))

            clipping_samples = hard_clip_count + soft_clip_count
            clipping_ratio = clipping_samples / total_samples

            # Penalty: 1.0 for ratio < 0.001, linear ramp to 0.0 at ratio >= 0.05
            if clipping_ratio < 0.001:
                return 1.0
            elif clipping_ratio >= 0.05:
                return 0.0
            else:
                # Linear interpolation: ratio 0.001 -> 1.0, ratio 0.05 -> 0.0
                penalty = 1.0 - (clipping_ratio - 0.001) / (0.05 - 0.001)
                return float(np.clip(penalty, 0.0, 1.0))
        except Exception:
            return 1.0

    @staticmethod
    def _check_repetition(audio: np.ndarray, sr: int) -> float:
        """
        Detect repetitive (looped) audio segments.

        Segments audio into 2-second non-overlapping chunks and checks
        normalized cross-correlation between non-adjacent pairs.  Uses
        a high correlation threshold (>0.98) to only flag near-identical
        waveforms (copy-paste loops), not natural musical repetition
        like verse/chorus structure or steady rhythms.

        Returns:
            Penalty in [0, 1].
        """
        try:
            chunk_len = int(2.0 * sr)
            if chunk_len == 0:
                return 1.0

            n_chunks = len(audio) // chunk_len
            if n_chunks < 3:
                return 1.0

            chunks = [audio[i * chunk_len : (i + 1) * chunk_len] for i in range(n_chunks)]

            # Normalize each chunk
            norm_chunks = []
            for chunk in chunks:
                norm = np.linalg.norm(chunk)
                if norm < 1e-8:
                    norm_chunks.append(chunk)
                else:
                    norm_chunks.append(chunk / norm)

            # Check non-adjacent pairs (gap >= 1 chunk = 2 seconds)
            repeated_chunks = set()
            for i in range(n_chunks):
                for j in range(i + 2, n_chunks):
                    corr = float(np.dot(norm_chunks[i], norm_chunks[j]))
                    if corr > 0.98:
                        repeated_chunks.add(i)
                        repeated_chunks.add(j)

            repeat_ratio = len(repeated_chunks) / n_chunks

            # Penalty: 1.0 if no repeats, linear ramp to 0.0 if >50% repeated
            if repeat_ratio <= 0.0:
                return 1.0
            elif repeat_ratio >= 0.50:
                return 0.0
            else:
                penalty = 1.0 - repeat_ratio / 0.50
                return float(np.clip(penalty, 0.0, 1.0))
        except Exception:
            return 1.0

    @staticmethod
    def _check_spectral_holes(audio: np.ndarray, sr: int) -> float:
        """
        Detect spectral holes — frequency bands with anomalously low energy.

        Computes mean power spectrum, smooths it, then looks for bins where
        power drops >30 dB below the local average (window of ~500 Hz).
        Some spectral gaps are natural, so the penalty is capped at 0.5.

        Returns:
            Penalty in [0.5, 1.0].
        """
        try:
            from scipy.ndimage import uniform_filter1d

            n_fft = 2048
            hop_length = 512

            from scipy.signal import stft as _stft

            _, _, Zxx = _stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            S = np.abs(Zxx)

            # Mean power spectrum (average |STFT|^2 across time)
            mean_power = np.mean(S ** 2, axis=1)

            if len(mean_power) < 10:
                return 1.0

            # Smooth with a window corresponding to ~500 Hz
            freq_resolution = sr / n_fft
            window_bins = max(int(500.0 / freq_resolution), 3)
            smoothed = uniform_filter1d(mean_power, size=window_bins)

            # Avoid log of zero
            smoothed = np.maximum(smoothed, 1e-20)
            mean_power = np.maximum(mean_power, 1e-20)

            # Find bins where power drops more than 30 dB below local average
            ratio_db = 10.0 * np.log10(mean_power / smoothed)
            hole_bins = np.sum(ratio_db < -30.0)
            total_bins = len(mean_power)
            hole_ratio = hole_bins / total_bins

            # Penalty: 1.0 if hole_ratio < 0.02, ramp to 0.5 if hole_ratio > 0.10
            if hole_ratio < 0.02:
                return 1.0
            elif hole_ratio >= 0.10:
                return 0.5
            else:
                penalty = 1.0 - 0.5 * (hole_ratio - 0.02) / (0.10 - 0.02)
                return float(np.clip(penalty, 0.5, 1.0))
        except Exception:
            return 1.0
