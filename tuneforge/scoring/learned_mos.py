"""
Learned Mean Opinion Score (MOS) scorer for TuneForge.

Estimates perceptual audio quality using signal-processing heuristics that
correlate with human quality judgements.  Unlike individual signal-level
scorers, this module combines multi-resolution spectral analysis, codec
robustness testing, loudness consistency, and harmonic richness into a
single composite quality estimate.

No pre-trained neural models are required -- only numpy, scipy, and librosa.
"""

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Module-level weights (must sum to 1.0)
# ---------------------------------------------------------------------------
LEARNED_MOS_WEIGHTS: dict[str, float] = {
    "waveform_quality": 0.35,
    "codec_robustness": 0.25,
    "perceptual_loudness_consistency": 0.20,
    "harmonic_richness": 0.20,
}

# ---------------------------------------------------------------------------
# Empirical reference statistics for "good music" mel-spectrograms.
# Derived from analysis of professionally mastered tracks.
# ---------------------------------------------------------------------------
_REF_SPECTRAL_FLATNESS: float = 0.08  # geometric/arithmetic mean ratio
_REF_SPECTRAL_BANDWIDTH_NORM: float = 0.35  # normalised to Nyquist
_REF_CENTROID_CV: float = 0.25  # coefficient of variation of centroid

# Minimum audio duration (seconds) to attempt scoring
_MIN_DURATION: float = 0.25


class LearnedMOSScorer:
    """Estimate perceptual quality via multi-resolution signal analysis."""

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Compute per-metric learned MOS sub-scores.

        Args:
            audio: 1-D or 2-D float waveform.
            sr:    Sample rate in Hz.

        Returns:
            Dict with keys matching ``LEARNED_MOS_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            import librosa
            from scipy import signal as scipy_signal

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            duration = len(audio) / max(sr, 1)
            if duration < _MIN_DURATION:
                return {k: 0.0 for k in LEARNED_MOS_WEIGHTS}

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.001:
                return {k: 0.0 for k in LEARNED_MOS_WEIGHTS}

            return {
                "waveform_quality": self._score_waveform_quality(
                    audio, sr, librosa,
                ),
                "codec_robustness": self._score_codec_robustness(
                    audio, sr, librosa, scipy_signal,
                ),
                "perceptual_loudness_consistency": self._score_loudness_consistency(
                    audio, sr, scipy_signal,
                ),
                "harmonic_richness": self._score_harmonic_richness(
                    audio, sr, librosa,
                ),
            }
        except Exception as exc:
            logger.error("Learned MOS scoring failed: {}", exc)
            return {k: 0.0 for k in LEARNED_MOS_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate quality score in [0, 1].
        """
        total = sum(
            LEARNED_MOS_WEIGHTS[k] * scores.get(k, 0.0)
            for k in LEARNED_MOS_WEIGHTS
        )
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Sub-metric: waveform quality (multi-resolution STFT)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_waveform_quality(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Multi-resolution mel-spectrogram analysis.

        Computes mel-spectrograms at three FFT sizes and measures spectral
        flatness, normalised bandwidth, and centroid variability at each
        resolution.  The per-resolution scores are averaged and compared
        against empirical reference statistics for well-mastered music.
        """
        try:
            n_ffts = (512, 1024, 2048)
            resolution_scores: list[float] = []

            for n_fft in n_ffts:
                hop = n_fft // 4
                n_mels = min(128, n_fft // 4)

                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
                )
                mel_db = librosa.power_to_db(mel_spec + 1e-10, ref=np.max)

                # --- spectral flatness (per-frame geometric / arithmetic) ---
                flatness = librosa.feature.spectral_flatness(
                    y=audio, n_fft=n_fft, hop_length=hop,
                )[0]
                mean_flat = float(np.mean(flatness))
                # Good music: moderate flatness (not white noise, not pure tone)
                flat_score = float(np.exp(
                    -50.0 * (mean_flat - _REF_SPECTRAL_FLATNESS) ** 2
                ))

                # --- normalised spectral bandwidth ---
                bw = librosa.feature.spectral_bandwidth(
                    y=audio, sr=sr, n_fft=n_fft, hop_length=hop,
                )[0]
                nyquist = sr / 2.0
                norm_bw = float(np.mean(bw)) / nyquist if nyquist > 0 else 0.0
                bw_score = float(np.exp(
                    -20.0 * (norm_bw - _REF_SPECTRAL_BANDWIDTH_NORM) ** 2
                ))

                # --- centroid coefficient of variation ---
                centroid = librosa.feature.spectral_centroid(
                    y=audio, sr=sr, n_fft=n_fft, hop_length=hop,
                )[0]
                mean_c = float(np.mean(centroid))
                if mean_c > 1e-8:
                    cv = float(np.std(centroid)) / mean_c
                else:
                    cv = 1.0
                # Moderate variation is good; too stable or too erratic is bad
                cv_score = float(np.exp(
                    -15.0 * (cv - _REF_CENTROID_CV) ** 2
                ))

                # --- mel-spectrum shape score ---
                # Good music has a characteristic declining energy profile
                mean_mel = np.mean(mel_db, axis=1)  # [n_mels]
                if len(mean_mel) > 1:
                    # Normalise to [0, 1] range
                    mel_range = float(np.ptp(mean_mel))
                    shape_score = float(np.clip(mel_range / 60.0, 0.0, 1.0))
                else:
                    shape_score = 0.0

                res_score = 0.25 * flat_score + 0.25 * bw_score + 0.25 * cv_score + 0.25 * shape_score
                resolution_scores.append(res_score)

            score = float(np.mean(resolution_scores))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Sub-metric: codec robustness
    # ------------------------------------------------------------------

    @staticmethod
    def _score_codec_robustness(
        audio: np.ndarray, sr: int, librosa, scipy_signal,
    ) -> float:
        """
        Low-bitrate codec robustness test.

        Resamples to 16 kHz, applies a low-pass filter at 8 kHz, and
        measures spectral divergence between the original and degraded
        versions.  High-quality audio degrades gracefully; AI artefacts
        often amplify under bandwidth reduction.
        """
        try:
            target_sr = 16000
            if sr != target_sr:
                degraded = librosa.resample(
                    audio.astype(np.float32), orig_sr=sr, target_sr=target_sr,
                )
            else:
                degraded = audio.copy()

            # Low-pass Butterworth at 8 kHz (Nyquist of degraded signal)
            nyq = target_sr / 2.0
            cutoff = 8000.0 / nyq
            # Clamp cutoff to valid range for butter
            cutoff = min(cutoff, 0.99)
            if cutoff > 0.01:
                b, a = scipy_signal.butter(4, cutoff, btype="low")
                degraded = scipy_signal.filtfilt(b, a, degraded).astype(np.float32)

            # Resample degraded back to original rate for comparison
            if target_sr != sr:
                degraded_up = librosa.resample(
                    degraded, orig_sr=target_sr, target_sr=sr,
                )
            else:
                degraded_up = degraded

            # Align lengths
            min_len = min(len(audio), len(degraded_up))
            if min_len < 1:
                return 0.0
            ref = audio[:min_len]
            deg = degraded_up[:min_len]

            # Spectral divergence via STFT magnitude
            n_fft = 1024
            hop = 256
            S_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
            S_deg = np.abs(librosa.stft(deg, n_fft=n_fft, hop_length=hop))

            # Align frame counts (resampling can cause off-by-one)
            min_frames = min(S_ref.shape[1], S_deg.shape[1])
            S_ref = S_ref[:, :min_frames]
            S_deg = S_deg[:, :min_frames]

            # Log-spectral distance (mean over frames)
            eps = 1e-10
            lsd = np.mean(np.sqrt(
                np.mean((np.log10(S_ref + eps) - np.log10(S_deg + eps)) ** 2, axis=0)
            ))

            # Lower LSD = more robust = higher score
            # Empirical range: good music ~0.1-0.3, poor ~0.5+
            score = float(np.exp(-4.0 * lsd))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Sub-metric: perceptual loudness consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _score_loudness_consistency(
        audio: np.ndarray, sr: int, scipy_signal,
    ) -> float:
        """
        A-weighted loudness contour smoothness.

        Computes loudness in overlapping windows (400 ms window, 200 ms hop)
        using A-weighting, then measures the jitter (standard deviation of
        frame-to-frame loudness differences).  Professional masters have
        smooth contours; AI music often has micro-fluctuations.
        """
        try:
            # --- A-weighting filter ---
            # Design A-weighting via analog prototype poles/zeros
            # Simplified: use a peaking filter that approximates A-weighting
            # for the purpose of perceptual loudness estimation
            nyq = sr / 2.0
            if nyq < 100:
                return 0.0

            # A-weighting approximation: high-pass at 500 Hz + low-pass at 10 kHz
            hp_freq = min(500.0 / nyq, 0.99)
            lp_freq = min(10000.0 / nyq, 0.99)

            if hp_freq > 0.01:
                b_hp, a_hp = scipy_signal.butter(2, hp_freq, btype="high")
                weighted = scipy_signal.filtfilt(b_hp, a_hp, audio)
            else:
                weighted = audio.copy()

            if lp_freq > 0.01 and lp_freq < 0.99:
                b_lp, a_lp = scipy_signal.butter(2, lp_freq, btype="low")
                weighted = scipy_signal.filtfilt(b_lp, a_lp, weighted)

            weighted = weighted.astype(np.float64)

            # --- windowed loudness ---
            win_samples = int(0.4 * sr)  # 400 ms
            hop_samples = int(0.2 * sr)  # 200 ms

            if win_samples < 1 or hop_samples < 1:
                return 0.0

            loudness_db: list[float] = []
            for start in range(0, len(weighted) - win_samples + 1, hop_samples):
                frame = weighted[start : start + win_samples]
                rms = float(np.sqrt(np.mean(frame ** 2)))
                if rms > 1e-10:
                    loudness_db.append(20.0 * np.log10(rms))
                else:
                    loudness_db.append(-100.0)

            if len(loudness_db) < 3:
                return 0.0

            loudness_arr = np.array(loudness_db)

            # Remove effectively silent frames for jitter calculation
            active_mask = loudness_arr > -60.0
            active = loudness_arr[active_mask]
            if len(active) < 3:
                return 0.0

            # Jitter = std of consecutive loudness differences
            diffs = np.diff(active)
            jitter = float(np.std(diffs))

            # Lower jitter = smoother = higher score
            # Empirical: professional ~0.5-2.0 dB jitter, poor ~4+ dB
            score = float(np.exp(-0.5 * jitter))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Sub-metric: harmonic richness
    # ------------------------------------------------------------------

    @staticmethod
    def _score_harmonic_richness(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Measure harmonic series richness and envelope smoothness.

        For detected fundamental frequencies, counts the number of
        audible harmonics and measures the smoothness of the harmonic
        amplitude envelope.  Natural instruments produce rich harmonic
        series with characteristic rolloff; AI audio often has either
        too few harmonics or unnatural harmonic ratios.
        """
        try:
            # Use pyin for robust fundamental frequency estimation
            # Analyse a representative segment (up to 10 seconds)
            max_samples = min(len(audio), sr * 10)
            segment = audio[:max_samples]

            f0, voiced_flag, _ = librosa.pyin(
                segment,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C6")),
                sr=sr,
                hop_length=512,
            )

            # Get voiced frames with valid f0
            valid_mask = voiced_flag & ~np.isnan(f0)
            valid_f0 = f0[valid_mask]

            if len(valid_f0) < 3:
                # No pitched content detected -- could be percussion-only
                return 0.3

            # Use median f0 as representative fundamental
            median_f0 = float(np.median(valid_f0))
            if median_f0 < 20.0:
                return 0.0

            # Compute magnitude spectrum of the full segment
            n_fft = 4096
            S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=512))
            # Average magnitude across time
            mag = np.mean(S, axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # --- Count detectable harmonics ---
            max_harmonic = int(min((sr / 2.0) / median_f0, 20))
            harmonic_amplitudes: list[float] = []
            noise_floor = float(np.percentile(mag[mag > 0], 10)) if np.any(mag > 0) else 1e-10

            for h in range(1, max_harmonic + 1):
                target_freq = median_f0 * h
                if target_freq >= sr / 2.0:
                    break

                # Find the bin closest to the harmonic frequency
                bin_idx = int(np.argmin(np.abs(freqs - target_freq)))

                # Search a small window around the expected bin for the peak
                window = max(1, int(n_fft * 10.0 / sr))  # ~10 Hz window
                lo = max(0, bin_idx - window)
                hi = min(len(mag), bin_idx + window + 1)
                peak_amp = float(np.max(mag[lo:hi]))

                if peak_amp > noise_floor * 2.0:
                    harmonic_amplitudes.append(peak_amp)

            n_harmonics = len(harmonic_amplitudes)

            # Score for number of harmonics (more = richer, up to ~12)
            count_score = float(np.clip(n_harmonics / 12.0, 0.0, 1.0))

            # --- Harmonic envelope smoothness ---
            if n_harmonics >= 3:
                log_amps = np.log10(np.array(harmonic_amplitudes) + 1e-10)
                # Fit a line (natural rolloff is approximately linear in log)
                indices = np.arange(len(log_amps), dtype=np.float64)
                coeffs = np.polyfit(indices, log_amps, 1)
                fitted = np.polyval(coeffs, indices)
                residual = float(np.std(log_amps - fitted))

                # Lower residual = smoother envelope = higher score
                # Empirical: natural instruments ~0.1-0.3, synthetic ~0.5+
                envelope_score = float(np.exp(-3.0 * residual))
            else:
                envelope_score = 0.3

            score = 0.5 * count_score + 0.5 * envelope_score
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0
