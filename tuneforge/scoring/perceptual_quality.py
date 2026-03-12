"""
Perceptual quality scorer for TuneForge.

Lightweight spectral-based MOS (Mean Opinion Score) estimator.
Uses spectral characteristics that correlate with human perception
of audio quality without requiring a separate neural model.
"""

import numpy as np
from loguru import logger


class PerceptualQualityScorer:
    """Estimate perceptual audio quality via spectral analysis."""

    WEIGHTS: dict[str, float] = {
        "bandwidth_consistency": 0.30,
        "snr_estimate": 0.30,
        "harmonic_noise_ratio": 0.20,
        "hf_presence": 0.20,
    }

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """Compute perceptual quality sub-scores."""
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.001:
                return {k: 0.0 for k in self.WEIGHTS}

            return {
                "bandwidth_consistency": self._score_bandwidth_consistency(audio, sr, librosa),
                "snr_estimate": self._score_snr(audio, sr, librosa),
                "harmonic_noise_ratio": self._score_hnr(audio, librosa),
                "hf_presence": self._score_hf_presence(audio, sr, librosa),
            }
        except Exception as exc:
            logger.error("Perceptual quality scoring failed: {}", exc)
            return {k: 0.0 for k in self.WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        total = sum(self.WEIGHTS[k] * scores.get(k, 0.0) for k in self.WEIGHTS)
        return float(np.clip(total, 0.0, 1.0))

    @staticmethod
    def _score_bandwidth_consistency(audio, sr, librosa) -> float:
        """Spectral bandwidth coefficient of variation. Consistent = good."""
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        mean_bw = float(np.mean(bw))
        if mean_bw < 1e-8:
            return 0.0
        cv = float(np.std(bw)) / mean_bw
        return float(np.clip(1.0 - cv / 0.5, 0.0, 1.0))

    @staticmethod
    def _score_snr(audio, sr, librosa) -> float:
        """Estimate SNR from STFT magnitude distribution."""
        S = np.abs(librosa.stft(audio))
        flat = S.flatten()
        n = len(flat)
        if n < 4:
            return 0.0
        sorted_flat = np.sort(flat)
        signal_power = float(np.mean(sorted_flat[-n // 4 :] ** 2))
        noise_power = float(np.mean(sorted_flat[: n // 4] ** 2)) + 1e-10
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr_db / 40.0, 0.0, 1.0))

    @staticmethod
    def _score_hnr(audio, librosa) -> float:
        """Harmonic-to-noise ratio."""
        harmonic = librosa.effects.harmonic(y=audio)
        residual = audio[: len(harmonic)] - harmonic
        h_energy = float(np.sum(harmonic**2))
        r_energy = float(np.sum(residual**2)) + 1e-10
        hnr = 10 * np.log10(h_energy / r_energy)
        return float(np.clip(hnr / 20.0, 0.0, 1.0))

    @staticmethod
    def _score_hf_presence(audio, sr, librosa) -> float:
        """High-frequency energy presence (4kHz+). Good music has some."""
        # If sample rate is too low to represent 4kHz, return neutral
        if sr < 8000:
            return 0.5
        S = np.abs(librosa.stft(audio, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hf_mask = freqs >= 4000
        if not np.any(hf_mask):
            return 0.5
        hf_energy = float(np.sum(S[hf_mask, :] ** 2))
        total_energy = float(np.sum(S**2)) + 1e-10
        hf_ratio = hf_energy / total_energy
        # Bell curve centered at 0.2 (good HF content)
        return float(np.clip(np.exp(-8.0 * (hf_ratio - 0.2) ** 2), 0.0, 1.0))
