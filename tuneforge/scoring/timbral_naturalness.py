"""
Timbral naturalness scorer for TuneForge.

Detects the most audible flaw in AI-generated music: metallic/synthetic
timbre that lacks the organic qualities of natural instruments and
professional recordings.  Measures spectral envelope shape, harmonic
decay patterns, transient behavior, temporal amplitude envelopes, and
spectral flux consistency.

Genre-aware: accepts an optional genre string to adjust targets via
``GenreProfile`` (e.g. electronic music tolerates flatter spectra).

Uses only numpy, scipy, and librosa — no heavy ML models — so scoring
remains fast enough for real-time validation.
"""

import numpy as np
from loguru import logger
from scipy.signal import lfilter

from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile


TIMBRAL_WEIGHTS: dict[str, float] = {
    "spectral_envelope_naturalness": 0.30,
    "harmonic_decay_quality": 0.25,
    "transient_naturalness": 0.20,
    "temporal_envelope_quality": 0.15,
    "spectral_flux_consistency": 0.10,
}


class TimbralNaturalnessScorer:
    """Assess timbral naturalness of generated audio."""

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """
        Compute per-metric timbral naturalness scores.

        Args:
            audio: 1-D or 2-D float waveform.
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware target adjustment.

        Returns:
            Dict with keys matching ``TIMBRAL_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            if len(audio) < sr * 0.1:
                return {k: 0.0 for k in TIMBRAL_WEIGHTS}

            if np.max(np.abs(audio)) < 1e-6:
                return {k: 0.0 for k in TIMBRAL_WEIGHTS}

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")

            return {
                "spectral_envelope_naturalness": self._score_spectral_envelope_naturalness(
                    audio, sr, librosa, profile,
                ),
                "harmonic_decay_quality": self._score_harmonic_decay_quality(
                    audio, sr, librosa, profile,
                ),
                "transient_naturalness": self._score_transient_naturalness(
                    audio, sr, librosa, profile,
                ),
                "temporal_envelope_quality": self._score_temporal_envelope_quality(
                    audio, sr, librosa,
                ),
                "spectral_flux_consistency": self._score_spectral_flux_consistency(
                    audio, sr, librosa,
                ),
            }
        except Exception as exc:
            logger.error(f"Timbral naturalness scoring failed: {exc}")
            return {k: 0.0 for k in TIMBRAL_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate timbral naturalness score in [0, 1].
        """
        total = 0.0
        for metric, weight in TIMBRAL_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_spectral_envelope_naturalness(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Score based on spectral envelope shape vs. natural instrument profiles.

        Natural instruments exhibit roughly -3 dB/octave spectral tilt
        (pink noise-like).  AI audio often has flatter or irregular tilt.
        The smoothed spectral envelope is extracted via cepstral smoothing,
        then spectral tilt is measured by linear regression in log-frequency
        space.  Score is based on proximity to natural roll-off, with
        genre-aware adjustment (electronic genres tolerate flatter spectra).
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            mean_spectrum = np.mean(S, axis=1)

            if np.max(mean_spectrum) < 1e-10:
                return 0.0

            # Cepstral smoothing: keep only the first N cepstral coefficients
            # to get a smooth spectral envelope
            log_spectrum = np.log(mean_spectrum + 1e-10)
            cepstrum = np.fft.irfft(log_spectrum)
            n_cepstral = 30  # smooth envelope — keep low quefrency
            cepstrum_windowed = np.zeros_like(cepstrum)
            cepstrum_windowed[:n_cepstral] = cepstrum[:n_cepstral]
            if len(cepstrum_windowed) > n_cepstral:
                # Preserve symmetry for real-valued result
                cepstrum_windowed[-n_cepstral + 1:] = cepstrum[-n_cepstral + 1:]
            smoothed_log = np.fft.rfft(cepstrum_windowed).real
            # Ensure same length as original spectrum
            smoothed_log = smoothed_log[:len(mean_spectrum)]

            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Work in log-frequency space, skip DC bin
            valid = freqs > 50.0  # ignore sub-bass for tilt measurement
            if np.sum(valid) < 10:
                return 0.0

            log_freqs = np.log2(freqs[valid])
            envelope_db = smoothed_log[valid]

            # Linear regression: envelope_db = slope * log_freqs + intercept
            # slope in dB/octave
            A = np.vstack([log_freqs, np.ones(len(log_freqs))]).T
            result = np.linalg.lstsq(A, envelope_db, rcond=None)
            slope = result[0][0]  # dB per octave in log domain

            # Convert log-magnitude slope to approximate dB/octave
            # log_spectrum is in nepers; multiply by 20/ln(10) for dB
            slope_db_per_octave = slope * (20.0 / np.log(10.0))

            # Natural target: approximately -3 dB/octave
            # Electronic genres: tolerate flatter spectra (-1 to -2 dB/oct)
            family = profile.family
            if family in ("electronic", "hip-hop"):
                target_tilt = -1.5
                tolerance = 3.0
            elif family in ("ambient",):
                target_tilt = -2.0
                tolerance = 3.0
            else:
                target_tilt = -3.0
                tolerance = 3.5

            deviation = abs(slope_db_per_octave - target_tilt)

            # Also measure envelope irregularity (residual from linear fit)
            predicted = result[0][0] * log_freqs + result[0][1]
            residuals = envelope_db - predicted
            irregularity = float(np.std(residuals))

            # Combine tilt score and regularity score
            tilt_score = max(0.0, 1.0 - deviation / tolerance)

            # Irregularity: natural instruments have some, but not excessive
            # One-sided: only penalize high irregularity (> 1.5)
            irregularity_penalty = max(0.0, (irregularity - 1.5) / 3.0)
            regularity_score = max(0.0, 1.0 - irregularity_penalty)

            score = 0.7 * tilt_score + 0.3 * regularity_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_harmonic_decay_quality(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Score based on harmonic amplitude decay patterns.

        Natural instruments have smooth harmonic decay: higher harmonics
        generally have lower amplitude, with the decay rate depending on
        the instrument.  AI audio often has unnatural harmonic amplitude
        distributions — harmonics that are too uniform, missing, or
        erratically distributed.

        Extracts the harmonic series from STFT, measures how monotonically
        harmonic amplitudes decay, and scores based on smoothness of the
        decay curve.
        """
        try:
            # Estimate fundamental frequency using autocorrelation on a
            # stable segment (middle portion of the audio)
            mid_start = len(audio) // 4
            mid_end = 3 * len(audio) // 4
            segment = audio[mid_start:mid_end]

            if len(segment) < sr * 0.1:
                return 0.5  # neutral if too short

            # Use librosa pitch tracking to get median f0
            f0, voiced_flag, _ = librosa.pyin(
                segment, fmin=50.0, fmax=2000.0, sr=sr,
                frame_length=2048,
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) < 3:
                # No clear pitch — could be percussive or noise-based
                # Score based on overall spectral smoothness instead
                return TimbralNaturalnessScorer._spectral_smoothness_fallback(
                    audio, sr, librosa,
                )

            median_f0 = float(np.median(f0_valid))
            if median_f0 < 50.0:
                return 0.5

            # Extract harmonic amplitudes from the magnitude spectrum
            n_fft = 4096
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            mean_spectrum = np.mean(S, axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Measure amplitude at each harmonic (up to 12 harmonics)
            n_harmonics = min(12, int((sr / 2) / median_f0))
            if n_harmonics < 3:
                return 0.5

            harmonic_amps = []
            freq_resolution = freqs[1] - freqs[0]
            for h in range(1, n_harmonics + 1):
                target_freq = median_f0 * h
                if target_freq >= sr / 2:
                    break
                # Find nearest frequency bin and take peak in +-2 bins
                idx = int(round(target_freq / freq_resolution))
                lo = max(0, idx - 2)
                hi = min(len(mean_spectrum), idx + 3)
                amp = float(np.max(mean_spectrum[lo:hi]))
                harmonic_amps.append(amp)

            if len(harmonic_amps) < 3:
                return 0.5

            harmonic_amps = np.array(harmonic_amps)

            # Normalize to fundamental
            if harmonic_amps[0] < 1e-10:
                return 0.0
            harmonic_amps_norm = harmonic_amps / harmonic_amps[0]

            # Measure monotonic decay quality:
            # For each consecutive pair, check if amplitude decreases
            n_pairs = len(harmonic_amps_norm) - 1
            if n_pairs < 2:
                return 0.5

            decay_violations = 0
            for i in range(n_pairs):
                if harmonic_amps_norm[i + 1] > harmonic_amps_norm[i] * 1.1:
                    # Allow 10% tolerance for minor non-monotonicity
                    decay_violations += 1

            monotonicity_score = 1.0 - (decay_violations / n_pairs)

            # Measure smoothness of the decay curve (low jitter = natural)
            log_amps = np.log10(harmonic_amps_norm + 1e-10)
            second_diff = np.diff(log_amps, n=2)
            jitter = float(np.std(second_diff))

            # One-sided floor: only penalize high jitter (> 0.3)
            jitter_penalty = max(0.0, (jitter - 0.3) / 1.0)
            smoothness_score = max(0.0, 1.0 - jitter_penalty)

            # Genre adjustment: electronic can have more synthetic harmonics
            if profile.family in ("electronic", "hip-hop"):
                # Be more lenient
                monotonicity_score = 0.3 + 0.7 * monotonicity_score
                smoothness_score = 0.3 + 0.7 * smoothness_score

            score = 0.5 * monotonicity_score + 0.5 * smoothness_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _spectral_smoothness_fallback(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Fallback for percussive/unpitched audio: score spectral smoothness.

        Natural percussive sounds still have smooth spectral envelopes.
        AI artifacts tend to produce jagged spectra.
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            mean_spectrum = np.mean(S, axis=1)

            if np.max(mean_spectrum) < 1e-10:
                return 0.0

            # Measure spectral smoothness as inverse of first-derivative variance
            log_spec = np.log10(mean_spectrum + 1e-10)
            first_diff = np.diff(log_spec)
            roughness = float(np.std(first_diff))

            # One-sided: penalize roughness above 0.15
            if roughness <= 0.15:
                return 1.0
            score = max(0.0, 1.0 - (roughness - 0.15) / 0.5)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_transient_naturalness(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Score based on spectral behavior during note attacks.

        Natural transients have specific spectral spread patterns during
        the attack phase (first 20-50ms after onset).  AI audio often has
        either too-sharp or too-smooth transients with unnatural spectral
        evolution.

        Detects onset frames, analyzes spectral evolution during attack
        windows, and scores based on attack sharpness and spectral spread.
        Uses a one-sided minimum floor: good transients score >= 0.6.
        """
        try:
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, units="frames", hop_length=512,
            )

            if len(onset_frames) < 2:
                # Too few transients to evaluate — neutral score
                return 0.6

            n_fft = 2048
            hop_length = 512
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

            # Attack window: ~20-50ms after onset
            # At hop_length=512, sr=44100: each frame ~11.6ms
            # So 2-4 frames after onset covers roughly 23-46ms
            attack_frames = max(2, int(0.04 * sr / hop_length))

            attack_spreads = []
            attack_sharpness_values = []

            for onset in onset_frames:
                if onset + attack_frames >= S.shape[1]:
                    continue
                if onset < 1:
                    continue

                # Spectral centroid spread during attack
                attack_region = S[:, onset:onset + attack_frames]
                pre_onset = S[:, max(0, onset - 1):onset]

                if attack_region.shape[1] == 0 or pre_onset.shape[1] == 0:
                    continue

                # Spectral centroid of attack vs pre-onset
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

                # Mean spectrum during attack
                attack_mean = np.mean(attack_region, axis=1)
                pre_mean = np.mean(pre_onset, axis=1)

                attack_total = np.sum(attack_mean)
                pre_total = np.sum(pre_mean)

                if attack_total < 1e-10 or pre_total < 1e-10:
                    continue

                # Spectral centroid
                attack_centroid = float(np.sum(freqs * attack_mean) / attack_total)
                pre_centroid = float(np.sum(freqs * pre_mean) / pre_total)

                # Spectral spread (bandwidth) during attack
                attack_spread = float(
                    np.sqrt(np.sum(((freqs - attack_centroid) ** 2) * attack_mean) / attack_total)
                )
                attack_spreads.append(attack_spread)

                # Attack sharpness: energy ratio (attack / pre-onset)
                energy_ratio = attack_total / (pre_total + 1e-10)
                attack_sharpness_values.append(energy_ratio)

            if len(attack_spreads) < 2:
                return 0.6

            # --- Spectral spread during transients ---
            # Natural instruments: moderate spread (500-3000 Hz std)
            # AI: often too narrow (pure tone attacks) or too wide (noise bursts)
            mean_spread = float(np.mean(attack_spreads))

            # One-sided minimum floor: score >= 0.6 for reasonable spread
            if 300.0 <= mean_spread <= 4000.0:
                spread_score = 1.0
            elif mean_spread < 300.0:
                spread_score = 0.6 + 0.4 * (mean_spread / 300.0)
            else:
                spread_score = max(0.6, 1.0 - (mean_spread - 4000.0) / 6000.0)

            # --- Attack sharpness consistency ---
            # Natural music: attacks vary in sharpness but are consistent within
            # instrument groups.  Measure CV of attack energy ratios.
            sharpness_arr = np.array(attack_sharpness_values)
            if len(sharpness_arr) > 1 and np.mean(sharpness_arr) > 1e-10:
                sharpness_cv = float(np.std(sharpness_arr) / np.mean(sharpness_arr))
            else:
                sharpness_cv = 0.0

            # One-sided: penalize only very high CV (> 1.5 = chaotic attacks)
            if sharpness_cv <= 1.5:
                sharpness_score = 1.0
            else:
                sharpness_score = max(0.6, 1.0 - (sharpness_cv - 1.5) / 3.0)

            # Genre adjustment: electronic allows sharper, more uniform transients
            if profile.family in ("electronic", "hip-hop"):
                spread_score = 0.3 + 0.7 * spread_score
                sharpness_score = 0.3 + 0.7 * sharpness_score

            score = 0.6 * spread_score + 0.4 * sharpness_score
            # Enforce minimum floor of 0.6 for any audio with valid transients
            score = max(0.6, score) if len(attack_spreads) >= 2 else score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_temporal_envelope_quality(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Score based on ADSR-like amplitude envelope patterns.

        Natural sounds exhibit attack-decay-sustain-release envelopes with
        organic amplitude modulation.  AI audio often has flat sustain
        with no natural decay, or unnatural amplitude jumps.

        Detects note-like segments between onsets and measures the presence
        of natural amplitude shaping within each segment.
        """
        try:
            # RMS envelope at fine resolution
            hop_length = 256
            frame_length = 1024
            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length,
            )[0]

            if len(rms) < 10:
                return 0.0

            if np.max(rms) < 1e-8:
                return 0.0

            # Detect onsets to segment into note-like regions
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, units="frames", hop_length=hop_length,
            )

            if len(onset_frames) < 2:
                # No clear note segmentation; measure overall envelope variation
                cv = float(np.std(rms) / (np.mean(rms) + 1e-10))
                # One-sided: penalize very flat envelopes (cv < 0.1)
                if cv >= 0.1:
                    return min(1.0, 0.6 + 0.4 * min(cv / 0.5, 1.0))
                return max(0.0, cv / 0.1 * 0.6)

            # Analyze each note segment for ADSR-like characteristics
            segment_scores = []

            for i in range(len(onset_frames)):
                seg_start = onset_frames[i]
                seg_end = onset_frames[i + 1] if i + 1 < len(onset_frames) else len(rms)

                seg_rms = rms[seg_start:seg_end]
                if len(seg_rms) < 4:
                    continue

                # Normalize segment
                seg_max = float(np.max(seg_rms))
                if seg_max < 1e-10:
                    continue
                seg_norm = seg_rms / seg_max

                # --- Attack detection ---
                # Find peak position (should be in first ~30% for natural attack)
                peak_pos = int(np.argmax(seg_norm))
                rel_peak_pos = peak_pos / len(seg_norm)

                # Natural: peak in first third; AI: sometimes peak at start or
                # no clear peak (flat)
                if rel_peak_pos < 0.35:
                    attack_score = 1.0
                elif rel_peak_pos < 0.6:
                    attack_score = 0.7
                else:
                    attack_score = 0.4

                # --- Decay/Release detection ---
                # After peak, amplitude should generally decrease (with variation)
                post_peak = seg_norm[peak_pos:]
                if len(post_peak) >= 3:
                    # Measure overall downward trend after peak
                    x = np.arange(len(post_peak), dtype=np.float64)
                    coeffs = np.polyfit(x, post_peak, 1)
                    slope = coeffs[0]

                    # Negative slope = natural decay; flat/positive = artificial
                    if slope < -0.001:
                        decay_score = min(1.0, 0.7 + 0.3 * min(abs(slope) / 0.01, 1.0))
                    elif slope < 0.001:
                        # Flat sustain — somewhat unnatural
                        decay_score = 0.4
                    else:
                        # Rising after peak — quite unnatural
                        decay_score = 0.2
                else:
                    decay_score = 0.5

                # --- Micro-variation (amplitude modulation) ---
                # Natural sounds have subtle amplitude fluctuations during sustain
                if len(seg_norm) >= 6:
                    high_freq = np.diff(seg_norm)
                    micro_var = float(np.std(high_freq))
                    # One-sided: penalize only if very flat (< 0.01)
                    if micro_var >= 0.01:
                        modulation_score = min(1.0, 0.6 + 0.4 * min(micro_var / 0.05, 1.0))
                    else:
                        modulation_score = max(0.0, micro_var / 0.01 * 0.6)
                else:
                    modulation_score = 0.5

                seg_score = 0.3 * attack_score + 0.4 * decay_score + 0.3 * modulation_score
                segment_scores.append(seg_score)

            if not segment_scores:
                return 0.0

            return float(np.clip(np.mean(segment_scores), 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_spectral_flux_consistency(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Score based on consistency of spectral evolution over time.

        Natural music has consistent spectral flux patterns — not too
        static (unchanging spectrum) and not too chaotic (random spectral
        jumps).  Measures the coefficient of variation (CV) of spectral
        flux over time.

        One-sided: penalizes only below a minimum threshold (too static),
        as chaotic flux is already caught by other metrics.
        """
        try:
            n_fft = 2048
            hop_length = 512
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

            if S.shape[1] < 4:
                return 0.0

            # Spectral flux: L2 norm of frame-to-frame spectral difference
            diff = np.diff(S, axis=1)
            flux = np.sqrt(np.sum(diff ** 2, axis=0))

            if len(flux) < 3:
                return 0.0

            mean_flux = float(np.mean(flux))
            if mean_flux < 1e-10:
                return 0.0

            std_flux = float(np.std(flux))
            cv = std_flux / mean_flux

            # One-sided minimum floor scoring:
            # CV < 0.2 = too static (synthetic drone-like)
            # CV >= 0.2 = natural range, score rises to 1.0
            # Very high CV (> 2.0) gets mild penalty for chaos
            if cv < 0.05:
                score = 0.1
            elif cv < 0.2:
                # Linear ramp from 0.1 to 0.8
                score = 0.1 + 0.7 * ((cv - 0.05) / 0.15)
            elif cv <= 2.0:
                # Good range — full or near-full score
                score = 0.8 + 0.2 * min((cv - 0.2) / 0.3, 1.0)
            else:
                # Mild penalty for extreme chaos
                score = max(0.5, 1.0 - (cv - 2.0) / 4.0)

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
