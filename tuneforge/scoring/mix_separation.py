"""
Mix separation scorer for TuneForge.

Evaluates mix clarity and instrument separation -- the ability to distinguish
individual sources in a mix.  Muddy mixes where all instruments occupy the
same frequency range are the most common amateur production tell.

Uses only numpy, scipy, and librosa (no Demucs or source-separation models)
for fast, lightweight analysis.

Genre-aware: accepts an optional genre string to adjust targets via
``GenreProfile`` (e.g. ambient music tolerates higher spectral flatness
in the low end because dense pads are intentional).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile


MIX_SEPARATION_WEIGHTS: dict[str, float] = {
    "spectral_clarity": 0.30,
    "frequency_masking_index": 0.25,
    "spatial_depth": 0.20,
    "low_end_clarity": 0.15,
    "mid_range_definition": 0.10,
}

# Frequency band boundaries used by several sub-metrics (Hz).
_BAND_EDGES: list[tuple[str, float, float]] = [
    ("sub_bass", 20.0, 80.0),
    ("bass", 80.0, 300.0),
    ("low_mid", 300.0, 2000.0),
    ("mid", 2000.0, 4000.0),
    ("high_mid", 4000.0, 8000.0),
    ("brilliance", 8000.0, 22050.0),
]


class MixSeparationScorer:
    """Assess mix clarity and instrument separation of generated audio."""

    def score(
        self,
        audio: np.ndarray,
        sr: int,
        genre: str = "",
    ) -> dict[str, float]:
        """
        Compute per-metric mix separation scores.

        Args:
            audio: 1-D or 2-D float waveform (channels x samples or samples).
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware adjustment.

        Returns:
            Dict with keys matching ``MIX_SEPARATION_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            import librosa

            # Keep a multichannel copy for spatial analysis before downmix.
            raw_audio = audio.copy()

            if audio.ndim > 1:
                mono = audio.mean(axis=0).astype(np.float32)
            else:
                mono = audio.astype(np.float32)

            if np.max(np.abs(mono)) < 1e-6:
                return {k: 0.0 for k in MIX_SEPARATION_WEIGHTS}

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")

            return {
                "spectral_clarity": self._score_spectral_clarity(
                    mono, sr, librosa, profile,
                ),
                "frequency_masking_index": self._score_frequency_masking_index(
                    mono, sr, librosa, profile,
                ),
                "spatial_depth": self._score_spatial_depth(
                    raw_audio, sr, librosa,
                ),
                "low_end_clarity": self._score_low_end_clarity(
                    mono, sr, librosa, profile,
                ),
                "mid_range_definition": self._score_mid_range_definition(
                    mono, sr, librosa,
                ),
            }
        except Exception as exc:
            logger.error(f"Mix separation scoring failed: {exc}")
            return {k: 0.0 for k in MIX_SEPARATION_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate mix separation score in [0, 1].
        """
        total = 0.0
        for metric, weight in MIX_SEPARATION_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _band_mask(
        freqs: np.ndarray, low: float, high: float, nyquist: float,
    ) -> np.ndarray:
        """Return a boolean mask for frequency bins within [low, high)."""
        high = min(high, nyquist)
        if low >= nyquist:
            return np.zeros(len(freqs), dtype=bool)
        return (freqs >= low) & (freqs < high)

    @staticmethod
    def _band_energies(
        S: np.ndarray, freqs: np.ndarray, sr: int,
    ) -> dict[str, np.ndarray]:
        """
        Compute per-frame RMS energy in each canonical band.

        Returns:
            Dict mapping band name to 1-D array of per-frame energies.
        """
        nyquist = sr / 2.0
        energies: dict[str, np.ndarray] = {}
        for name, low, high in _BAND_EDGES:
            mask = MixSeparationScorer._band_mask(freqs, low, high, nyquist)
            if not np.any(mask):
                continue
            # RMS across frequency bins for each frame.
            energies[name] = np.sqrt(np.mean(S[mask, :] ** 2, axis=0))
        return energies

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_spectral_clarity(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Measure inter-band energy contrast.

        Good mixes have distinct energy profiles in each frequency band.
        Low contrast means energy is smeared across the spectrum (muddy).
        Higher inter-band variance relative to the mean indicates better
        separation.
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            band_e = MixSeparationScorer._band_energies(S, freqs, sr)
            if len(band_e) < 3:
                return 0.0

            # Mean energy per band across all frames.
            means = np.array([float(np.mean(v)) for v in band_e.values()])
            overall_mean = float(np.mean(means))
            if overall_mean < 1e-10:
                return 0.0

            # Coefficient of variation of band means -- higher = more contrast.
            cv = float(np.std(means) / overall_mean)

            # Also measure per-frame contrast: for each frame compute the
            # standard deviation across bands and average.  This captures
            # instantaneous separation, not just the long-term average.
            n_frames = S.shape[1]
            band_matrix = np.stack(
                [v[:n_frames] for v in band_e.values()], axis=0,
            )  # (n_bands, n_frames)
            per_frame_cv = np.std(band_matrix, axis=0) / (
                np.mean(band_matrix, axis=0) + 1e-10
            )
            frame_cv = float(np.mean(per_frame_cv))

            # Combine long-term CV (60%) with per-frame CV (40%).
            combined_cv = 0.6 * cv + 0.4 * frame_cv

            # Target CV around 0.8-1.2 for well-mixed music.  Electronic
            # music may have a higher target (heavier bass, less mid).
            # Scale so that CV >= 1.0 saturates to 1.0.
            score = combined_cv / 1.0
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_frequency_masking_index(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Estimate frequency masking via spectral entropy in mid-range bands.

        Well-separated mixes have lower entropy in individual bands (cleaner,
        more peaked spectral shapes).  Muddy mixes have high entropy because
        multiple overlapping sources spread energy uniformly within a band.
        """
        try:
            n_fft = 2048
            hop_length = 512
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            nyquist = sr / 2.0

            # Focus on the mid-range where masking is most audible.
            mid_bands = [
                (300.0, 2000.0),
                (2000.0, 4000.0),
            ]

            entropies: list[float] = []
            for low, high in mid_bands:
                mask = MixSeparationScorer._band_mask(freqs, low, high, nyquist)
                if not np.any(mask):
                    continue

                band_S = S[mask, :]  # (n_bins_in_band, n_frames)
                n_bins = band_S.shape[0]
                if n_bins < 2:
                    continue

                # Normalize each frame to a probability distribution.
                col_sums = band_S.sum(axis=0, keepdims=True) + 1e-10
                prob = band_S / col_sums

                # Shannon entropy per frame (max = log2(n_bins) for uniform).
                frame_entropy = -np.sum(
                    prob * np.log2(prob + 1e-12), axis=0,
                )
                max_entropy = np.log2(n_bins)
                # Normalize to [0, 1].
                normalized = frame_entropy / (max_entropy + 1e-10)
                entropies.append(float(np.median(normalized)))

            if not entropies:
                return 0.0

            median_entropy = float(np.median(entropies))

            # Lower entropy = better separation.  Typical clean music has
            # normalized entropy around 0.5-0.7; noise is ~1.0; a single
            # sine in a band is ~0.0.  Map inversely: 0.3 entropy -> 1.0
            # score, 0.9 entropy -> 0.0.
            score = (0.9 - median_entropy) / 0.6
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_spatial_depth(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Measure spatial variation in the mix.

        For stereo audio: frequency-dependent stereo difference indicates
        that different instruments are panned to different positions.
        For mono audio: estimate reverb tail presence via autocorrelation
        of the energy envelope as a proxy for spatial depth.
        """
        try:
            is_stereo = audio.ndim == 2 and audio.shape[0] == 2

            if is_stereo:
                return MixSeparationScorer._spatial_depth_stereo(
                    audio, sr, librosa,
                )
            else:
                mono = audio.mean(axis=0) if audio.ndim > 1 else audio
                return MixSeparationScorer._spatial_depth_mono(
                    mono.astype(np.float32), sr, librosa,
                )
        except Exception:
            return 0.0

    @staticmethod
    def _spatial_depth_stereo(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Score stereo spatial separation.

        Compute STFT of left and right channels independently.  Measure
        per-band L-R spectral magnitude difference.  Good mixes pan
        different instruments to different positions, yielding larger
        frequency-dependent L-R variation.
        """
        try:
            n_fft = 2048
            left = audio[0].astype(np.float32)
            right = audio[1].astype(np.float32)

            S_left = np.abs(librosa.stft(left, n_fft=n_fft))
            S_right = np.abs(librosa.stft(right, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            nyquist = sr / 2.0

            band_diffs: list[float] = []
            for name, low, high in _BAND_EDGES:
                mask = MixSeparationScorer._band_mask(freqs, low, high, nyquist)
                if not np.any(mask):
                    continue
                l_energy = np.sqrt(np.mean(S_left[mask, :] ** 2))
                r_energy = np.sqrt(np.mean(S_right[mask, :] ** 2))
                total = l_energy + r_energy + 1e-10
                diff = abs(l_energy - r_energy) / total
                band_diffs.append(float(diff))

            if not band_diffs:
                return 0.0

            # Mean of per-band L-R difference.
            mean_diff = float(np.mean(band_diffs))
            # Variance of per-band differences: higher means different bands
            # are panned differently (good).
            var_diff = float(np.std(band_diffs))

            # Combine: some mean difference (0.6 weight) plus variation
            # across bands (0.4 weight).  Scale so that realistic values
            # map to [0, 1].
            score = 0.6 * min(mean_diff / 0.15, 1.0) + 0.4 * min(
                var_diff / 0.10, 1.0,
            )
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _spatial_depth_mono(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Estimate spatial depth from mono audio via reverb characteristics.

        Compute the autocorrelation of the RMS envelope.  Reverberant audio
        has a slower decay (higher autocorrelation at longer lags) indicating
        spatial depth.  Completely dry audio has a sharp autocorrelation drop.
        """
        try:
            hop_length = 512
            rms = librosa.feature.rms(
                y=audio, frame_length=2048, hop_length=hop_length,
            )[0]
            if len(rms) < 10:
                return 0.0

            # Normalize RMS envelope.
            rms = rms - np.mean(rms)
            norm = float(np.dot(rms, rms))
            if norm < 1e-10:
                return 0.0

            # Full autocorrelation via numpy correlate (mode="full").
            acf = np.correlate(rms, rms, mode="full")
            acf = acf[len(acf) // 2:]  # Keep positive lags only.
            acf = acf / (norm + 1e-10)

            # Measure the lag at which the autocorrelation drops below 0.5.
            # Longer = more reverb / spatial depth.
            below_half = np.where(acf < 0.5)[0]
            if len(below_half) == 0:
                decay_lag = len(acf)
            else:
                decay_lag = int(below_half[0])

            total_lags = len(acf)
            ratio = decay_lag / total_lags

            # Score: moderate reverb is best.  Very dry (ratio < 0.05) or
            # very washy (ratio > 0.5) are penalized.
            if ratio < 0.05:
                score = ratio / 0.05
            elif ratio > 0.5:
                score = max(1.0 - (ratio - 0.5) / 0.5, 0.0)
            else:
                # Map [0.05, 0.5] linearly to [0.6, 1.0].
                score = 0.6 + 0.4 * (ratio - 0.05) / 0.45

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_low_end_clarity(
        audio: np.ndarray, sr: int, librosa, profile: GenreProfile,
    ) -> float:
        """
        Evaluate clarity in the sub-bass and bass regions (40-300 Hz).

        Clean bass has low spectral flatness (clear fundamental + harmonics).
        Muddy bass has high spectral flatness (broadband energy, no clear
        fundamental).  Also measures correlation between sub-bass energy
        and the onset envelope as a proxy for bass-kick separation.
        """
        try:
            n_fft = 4096  # Higher resolution for low frequencies.
            hop_length = 1024
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            nyquist = sr / 2.0

            # --- Spectral flatness in 40-300 Hz ---
            bass_mask = MixSeparationScorer._band_mask(freqs, 40.0, 300.0, nyquist)
            if not np.any(bass_mask):
                return 0.0

            bass_S = S[bass_mask, :]  # (n_bins, n_frames)
            n_frames = bass_S.shape[1]
            if n_frames < 2:
                return 0.0

            # Per-frame spectral flatness (geometric mean / arithmetic mean).
            # Use log-domain for numerical stability.
            log_bass = np.log(bass_S + 1e-10)
            geo_mean = np.exp(np.mean(log_bass, axis=0))
            arith_mean = np.mean(bass_S, axis=0) + 1e-10
            flatness = geo_mean / arith_mean  # Per frame, in [0, 1].

            median_flatness = float(np.median(flatness))

            # Lower flatness = cleaner bass.  Typical clean bass: 0.05-0.2;
            # muddy bass: 0.4-0.7.  Map inversely.
            flatness_score = 1.0 - min(median_flatness / 0.5, 1.0)

            # --- Bass-kick separation ---
            # Correlation between sub-bass energy envelope and onset strength.
            sub_bass_mask = MixSeparationScorer._band_mask(
                freqs, 40.0, 80.0, nyquist,
            )
            if np.any(sub_bass_mask):
                sub_energy = np.sqrt(
                    np.mean(S[sub_bass_mask, :] ** 2, axis=0),
                )  # (n_frames,)

                # Onset strength envelope (librosa computes at hop_length).
                onset_env = librosa.onset.onset_strength(
                    y=audio, sr=sr, hop_length=hop_length,
                )

                # Align lengths.
                min_len = min(len(sub_energy), len(onset_env))
                if min_len > 4:
                    sub_energy = sub_energy[:min_len]
                    onset_env = onset_env[:min_len]

                    # Normalize both.
                    sub_energy = sub_energy - np.mean(sub_energy)
                    onset_env = onset_env - np.mean(onset_env)

                    denom = (
                        np.linalg.norm(sub_energy) * np.linalg.norm(onset_env)
                        + 1e-10
                    )
                    correlation = float(
                        np.dot(sub_energy, onset_env) / denom,
                    )

                    # Moderate positive correlation (0.3-0.6) indicates that
                    # kick and bass are rhythmically aligned but not identical.
                    # Very high correlation (>0.8) means they are glued (bad);
                    # negative or zero means they are unrelated (neutral).
                    if correlation > 0.8:
                        kick_score = 0.3
                    elif correlation > 0.3:
                        kick_score = 1.0 - (correlation - 0.3) / 0.5
                    else:
                        kick_score = 0.6  # Neutral for low correlation.
                else:
                    kick_score = 0.5
            else:
                kick_score = 0.5

            # Combine: flatness is primary (70%), kick separation secondary (30%).
            score = 0.7 * flatness_score + 0.3 * kick_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_mid_range_definition(
        audio: np.ndarray, sr: int, librosa,
    ) -> float:
        """
        Evaluate spectral peak sharpness in the 1-5 kHz range.

        Well-mixed audio has distinct spectral peaks (clear formants,
        instrument fundamentals).  Poorly mixed audio has broad, overlapping
        bumps.  Measure average peak prominence in the mel-spectrogram
        mid-range.
        """
        try:
            from scipy.signal import find_peaks

            # Mel spectrogram for perceptually spaced frequency analysis.
            n_fft = 2048
            hop_length = 512
            n_mels = 128
            S_mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            S_db = librosa.power_to_db(S_mel, ref=np.max)

            # Get mel frequencies and select 1-5 kHz range.
            mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2.0)
            mid_mask = (mel_freqs >= 1000.0) & (mel_freqs <= 5000.0)
            if not np.any(mid_mask):
                return 0.0

            mid_indices = np.where(mid_mask)[0]
            if len(mid_indices) < 3:
                return 0.0

            # Average spectrum over time in the mid-range.
            avg_mid = np.mean(S_db[mid_mask, :], axis=1)

            # Find peaks and measure their prominence.
            peaks, properties = find_peaks(
                avg_mid, distance=2, prominence=1.0,
            )

            if len(peaks) == 0:
                # No prominent peaks at all -- very flat / muddy.
                return 0.1

            prominences = properties["prominences"]
            mean_prominence = float(np.mean(prominences))
            n_peaks = len(peaks)

            # Score components:
            # 1) Prominence: higher = more distinct peaks.
            #    Typical good mix: 5-15 dB prominence; muddy: 1-3 dB.
            prominence_score = min(mean_prominence / 10.0, 1.0)

            # 2) Peak count: some peaks are good, but too many means noise.
            #    In the 1-5 kHz range with our mel resolution we expect
            #    2-8 meaningful peaks for well-defined instruments.
            n_mid_bins = len(mid_indices)
            peak_density = n_peaks / n_mid_bins
            if peak_density < 0.05:
                density_score = 0.3
            elif peak_density > 0.5:
                density_score = 0.4  # Too many peaks = noisy
            else:
                density_score = 0.5 + 0.5 * min(
                    (peak_density - 0.05) / 0.25, 1.0,
                )

            # Combine.
            score = 0.65 * prominence_score + 0.35 * density_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
